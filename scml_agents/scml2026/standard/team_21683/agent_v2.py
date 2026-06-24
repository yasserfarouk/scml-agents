#!/usr/bin/env python
"""RobustV2: AS0(=PenguinAgent改良版)の枠組みを土台に、価格戦略と subset 選択を
改善して AS0 を上回ることを狙うエージェント。

AS0/Penguin から継承する「強い土台」:
- StdSyncAgent 直接ベースで全生産レベルを同一ロジックで扱う。
- productivity=0.7。需要は per-step 精密会計（total_supplies_at/total_sales_at）。
  買: day_production - inventory - total_supplies_at(step)
  売: min(n_lines, day_production + inventory) - total_sales_at(step)
- 当日オファーは「需要に最も近い量」を与える subset を採用（threshold で許容）。
- 将来オファーは t+1/t+2/t+3 に 50/30/20% で分散、各 needs/3。
- 適応的 threshold（在庫・時間）、終盤の積極化、パートナー成功率追跡（AS0 由来）。

AS0 への改善（ここで AS0 を上回ることを狙う）:
1. **価格**: AS0 は固定倍率(0.7/1.2)の粗い譲歩。RobustV2 は交渉の relative_time に応じた
   譲歩カーブ（序盤は強気、締切に向けて floor まで譲歩）でより良い価格を取りに行く。
2. **subset 選択の価格考慮**: 量が同程度に良い候補が複数あるとき、price_score 合計が高い
   （＝より得な価格の）subset を選ぶ。量マッチングは保ちつつ利益率を上げる。
3. **安全弁**: powerset は相手を上位 N 件に制限し、O(2^n) ハングを防ぐ。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import chain, combinations, repeat

from negmas import SAOResponse, ResponseType
from numpy.random import choice
from scml.std import StdSyncAgent
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE

__all__ = ["SuperimagentZ", "RobustV2"]

# powerset を計算する相手数の上限（O(2^n) のハング/暴走を防ぐ）
_MAX_POWERSET_PARTNERS = 12


def distribute(q: int, n: int) -> list[int]:
    if n <= 0:
        return []
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class SuperimagentZ(StdSyncAgent):
    """提出エージェント（AS0 改良版）。

    2025勢・同一ワールド評価で AS0 を全指標(min/中央値/平均)＋低分散で上回った版。
    詳細は WORKLOG.md / README.md を参照。
    """

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7,
                 edge_exogenous=True, last_buy_exo=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = 1 if threshold is None else threshold
        self._base_ptoday = ptoday
        self._productivity = productivity
        # エッジレベルで「外生契約側」の当日需要を尊重するか。
        # AS0/Penguin は全レベル一律で 0.7×n_lines を需要とするため、第一レベルで
        # 買う必要が無い(needed_supplies=0)のに買い付けようとする弱点がある。
        # True なら外生側の当日需要を needed_supplies/needed_sales で上限制限する。
        self._edge_exogenous = edge_exogenous
        # 最終レベルの買い需要を外生販売義務(needed_supplies)に合わせるか。
        # 診断で最終レベルは買い不足(inv_in が AS0 より小)→shortfall が判明したため。
        self._last_buy_exo = last_buy_exo

        # パートナー成功率（AS0 由来）
        self.partner_negotiations = defaultdict(int)
        self.partner_successes = defaultdict(int)
        self.partner_success_rate = defaultdict(lambda: 0.5)

        # モード（AS0 由来）
        self.aggressive_mode = False
        self.conservative_mode = False
        self.last_balances: list[float] = []

    # ------------------------------------------------------------------ #
    # ステップごとの適応（AS0 由来の閾値/モード制御）
    # ------------------------------------------------------------------ #
    def step(self):
        super().step()
        base_threshold = self.awi.n_lines * 0.1

        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if inventory_ratio < 0.3:
            self._threshold = max(1, int(base_threshold * 1.5))
        elif inventory_ratio > 0.8:
            self._threshold = max(1, int(base_threshold * 0.7))
        else:
            self._threshold = max(1, int(base_threshold))

        time_left = (self.awi.n_steps - self.awi.current_step) / max(1, self.awi.n_steps)
        self.aggressive_mode = False
        if time_left < 0.3:
            self._threshold = int(self._threshold * 1.8)
            self.aggressive_mode = True
        elif time_left < 0.6:
            self._threshold = int(self._threshold * 1.3)

        # 利益トレンドで保守/積極を切替（AS0 由来）
        balance = float(getattr(self.awi, "current_balance", 0) or 0)
        self.last_balances.append(balance)
        if len(self.last_balances) > 5:
            self.last_balances.pop(0)
        if len(self.last_balances) >= 3:
            trend = sum(self.last_balances[-3:]) / 3 - sum(self.last_balances[:2]) / 2
            if trend < -50:
                self.conservative_mode = True
            elif trend > 50:
                self.conservative_mode = False

    def get_effective_ptoday(self) -> float:
        if self.aggressive_mode:
            return min(0.9, self._base_ptoday + 0.15)
        if self.conservative_mode:
            return max(0.5, self._base_ptoday - 0.1)
        return self._base_ptoday

    # ------------------------------------------------------------------ #
    # パートナー成功率
    # ------------------------------------------------------------------ #
    def update_partner_performance(self, partner, success: bool):
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
        self.partner_success_rate[partner] = (
            self.partner_successes[partner] / self.partner_negotiations[partner]
        )

    def select_partners_by_performance(self, partners, ratio=None) -> list:
        if not partners:
            return []
        if ratio is None:
            ratio = self.get_effective_ptoday()
        scored = sorted(partners, key=lambda p: self.partner_success_rate[p], reverse=True)
        k = max(1, int(len(partners) * ratio))
        selected = scored[:k]
        remaining = scored[k:]
        if remaining and len(selected) < len(partners):  # 探索
            extra = min(2, len(remaining), len(partners) - len(selected))
            selected = selected + random.sample(remaining, extra)
        return selected

    # ------------------------------------------------------------------ #
    # 需要計算（AS0/Penguin の per-step 精密会計）
    # ------------------------------------------------------------------ #
    def supply_need_at(self, step: int) -> int:
        awi = self.awi
        day_production = awi.n_lines * self._productivity
        need = int(day_production - awi.current_inventory_input - awi.total_supplies_at(step))
        if step == awi.current_step:
            # 第一レベルは原料が外生（買う必要なし）。外生需要で上限制限し過剰買付を防ぐ。
            if self._edge_exogenous and awi.is_first_level:
                need = min(need, max(0, int(awi.needed_supplies)))
            # 最終レベルは外生販売義務を満たすために買う。当日の買い需要を needed_supplies に合わせる。
            elif self._last_buy_exo and awi.is_last_level:
                need = max(need, max(0, int(awi.needed_supplies)))
        return need

    def sales_need_at(self, step: int) -> int:
        awi = self.awi
        day_production = awi.n_lines * self._productivity
        need = int(
            max(
                0,
                min(awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(step),
            )
        )
        # 最終レベルは製品が外生（売る必要なし）。当日は外生需要で上限制限し過剰売却を防ぐ。
        if self._edge_exogenous and step == awi.current_step and awi.is_last_level:
            need = min(need, max(0, int(awi.needed_sales)))
        return need

    def needs_at(self, step: int, partner) -> int:
        if self.is_supplier(partner):
            return self.supply_need_at(step)
        if self.is_consumer(partner):
            return self.sales_need_at(step)
        return 0

    # ------------------------------------------------------------------ #
    # 提案
    # ------------------------------------------------------------------ #
    def first_proposals(self):
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()
        first_dict = {}
        fut_sup, fut_con = [], []
        for k, q in distribution.items():
            if q > 0:
                first_dict[k] = (q, s, self.best_price(k))  # 初回は強気
            elif self.is_supplier(k):
                fut_sup.append(k)
            elif self.is_consumer(k):
                fut_con.append(k)
        response = dict(first_dict)
        response |= self.future_supplie_offer(self.select_partners_by_performance(fut_sup))
        response |= self.future_consume_offer(self.select_partners_by_performance(fut_con))
        return response

    def counter_all(self, offers, states):
        response = {}
        awi = self.awi
        for all_partners, selling in (
            (list(self.awi.my_suppliers), False),
            (list(self.awi.my_consumers), True),
        ):
            if not all_partners:
                continue
            needs = self.sales_need_at(awi.current_step) if selling else self.supply_need_at(awi.current_step)
            if selling and awi.total_sales_at(awi.current_step) > awi.n_lines:
                needs = 0

            partners = {p for p in all_partners if p in offers and offers[p] is not None}
            current_offers, future_offers = {}, {}
            for p in partners:
                o = offers[p]
                if not self.is_valid_price(o[UNIT_PRICE], p):
                    continue
                if o[TIME] == awi.current_step:
                    current_offers[p] = o
                else:
                    future_offers[p] = o

            # --- 将来オファーの受諾（per-step 需要内なら受ける） ---
            dup = [0 for _ in range(awi.n_steps + 1)]
            for p, o in future_offers.items():
                t = o[TIME]
                if 0 <= t <= awi.n_steps and o[QUANTITY] + dup[t - 1] <= self.needs_at(t, p):
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, o)
                    dup[t - 1] += o[QUANTITY]
                    self.update_partner_performance(p, True)

            # --- 当日オファーの subset 選択（量マッチング + 価格考慮タイブレーク） ---
            current_partners = set(current_offers.keys())
            best = self.best_current_subset(current_partners, current_offers, needs)

            flag = 0
            if best is not None and needs > 0:
                partner_ids = list(best)
                response.update(
                    {k: SAOResponse(ResponseType.ACCEPT_OFFER, current_offers[k]) for k in partner_ids}
                )
                for k in partner_ids:
                    self.update_partner_performance(k, True)
                others = list((current_partners.difference(partner_ids)) - set(response.keys()))
                for k in others:
                    self.update_partner_performance(k, False)
                os_s = [x for x in others if self.is_supplier(x)]
                os_c = [x for x in others if self.is_consumer(x)]
                for k, x in self.future_supplie_offer(os_s).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(os_c).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                flag = 1

            if flag != 1:
                other_partners = {
                    p for p in all_partners
                    if p not in response and p in self.negotiators.keys()
                }
                distribution = self.distribute_todays_needs(other_partners)
                fut_sup, fut_con = [], []
                for k, q in distribution.items():
                    if q > 0:
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, awi.current_step, self.concede_price(k, states.get(k))),
                        )
                    elif self.is_supplier(k):
                        fut_sup.append(k)
                    elif self.is_consumer(k):
                        fut_con.append(k)
                for k, x in self.future_supplie_offer(fut_sup).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(fut_con).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
        return response

    # ------------------------------------------------------------------ #
    # subset 選択: 量を需要に近づけつつ、同程度なら価格の良い方を選ぶ
    # ------------------------------------------------------------------ #
    def best_current_subset(self, current_partners, current_offers, needs):
        if not current_partners or needs <= 0:
            return None
        # 価格の良い相手を優先して powerset の対象を上限に制限（ハング防止）
        ranked = sorted(
            current_partners,
            key=lambda p: (self.price_score(p, current_offers[p]), self.partner_success_rate[p]),
            reverse=True,
        )[:_MAX_POWERSET_PARTNERS]

        threshold = self._threshold * (2.0 if self.aggressive_mode else 1.0)
        best_plus = None  # (key, subset) for offered>=needs
        best_minus = None  # for offered<needs
        best_plus_key = None
        best_minus_key = None
        for subset in powerset(ranked):
            if not subset:
                continue
            offered = sum(current_offers[p][QUANTITY] for p in subset)
            diff = abs(offered - needs)
            # パートナー品質と価格でタイブレーク（AS0 は量のみ → ここが改善点）
            quality = sum(self.partner_success_rate[p] for p in subset) / len(subset)
            price = sum(self.price_score(p, current_offers[p]) for p in subset) / len(subset)
            adj_diff = diff * (2.0 - quality)
            if offered - needs >= 0:
                key = (adj_diff, -price)  # diff 小 → 価格良(price 大)
                if best_plus_key is None or key < best_plus_key:
                    best_plus_key, best_plus = key, subset
            else:
                key = (adj_diff, -price)
                if best_minus_key is None or key < best_minus_key:
                    best_minus_key, best_minus = key, subset

        # 超過側が threshold 内なら採用、なければ不足側の最良
        if best_plus is not None and best_plus_key[0] <= threshold:
            return set(best_plus)
        if best_minus is not None:
            return set(best_minus)
        return None

    # ------------------------------------------------------------------ #
    # 価格
    # ------------------------------------------------------------------ #
    def concede_price(self, partner, state):
        """交渉の relative_time に応じた譲歩カーブ（AS0 の固定倍率を改善）。

        序盤は best_price（強気）、締切に向けて floor まで線形に譲歩する。
        floor は AS0 と同等水準（売: 0.7*max, 買: 1.2*min）。
        """
        nmi = self.get_nmi(partner)
        if nmi is None:
            return self.best_price(partner)
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        r = state.relative_time if state is not None else 0.5
        # 成功率の高い相手には少し早めに譲歩（取引成立を優先）
        r = min(1.0, r + 0.1 * self.partner_success_rate[partner])
        if self.is_consumer(partner):  # 売り: maxp から floor(0.7*max) へ
            floor = max(0.7 * maxp, minp)
            return max(minp, maxp - r * (maxp - floor))
        else:  # 買い: minp から ceil(1.2*min) へ
            ceil = min(1.2 * minp, maxp)
            return min(maxp, minp + r * (ceil - minp))

    def price_score(self, partner, offer) -> float:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return 0.0
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        span = max(1e-9, maxp - minp)
        if self.is_supplier(partner):  # 買い: 安いほど良い
            return (maxp - offer[UNIT_PRICE]) / span
        return (offer[UNIT_PRICE] - minp) / span  # 売り: 高いほど良い

    def is_valid_price(self, price, partner) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            issues = self.awi.current_output_issues if self.is_consumer(partner) else self.awi.current_input_issues
        else:
            issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return price >= minp
        if self.is_supplier(partner):
            return price <= maxp
        return False

    def best_price(self, partner):
        nmi = self.get_nmi(partner)
        if nmi is None:
            issues = self.awi.current_input_issues if self.is_supplier(partner) else self.awi.current_output_issues
            issue = issues[UNIT_PRICE]
        else:
            issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self.is_supplier(partner) else pmax

    # ------------------------------------------------------------------ #
    # 需要分配 / 将来オファー（AS0/Penguin と同様、成功率で相手選択）
    # ------------------------------------------------------------------ #
    def distribute_todays_needs(self, partners=None) -> dict:
        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))
        sup = [p for p in partners if self.is_supplier(p)]
        con = [p for p in partners if self.is_consumer(p)]
        awi = self.awi
        supply_needs = self.supply_need_at(awi.current_step)
        sales_needs = self.sales_need_at(awi.current_step)
        if sup and supply_needs > 0:
            response |= self._distribute_over(self.select_partners_by_performance(sup), supply_needs)
        if con and sales_needs > 0 and awi.total_sales_at(awi.current_step) <= awi.n_lines:
            response |= self._distribute_over(self.select_partners_by_performance(con), sales_needs)
        return response

    def _distribute_over(self, partners, needs) -> dict:
        response = dict(zip(partners, repeat(0)))
        if not partners:
            return response
        partners = list(partners)
        random.shuffle(partners)
        partners = partners[: max(1, int(self.get_effective_ptoday() * len(partners)))]
        n = len(partners)
        if 0 < needs < n:
            partners = random.sample(partners, random.randint(1, needs))
            n = len(partners)
        if n > 0:
            response |= dict(zip(partners, distribute(needs, n)))
        return response

    def _future_offer(self, partner_list, need_fn) -> dict:
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response
        ranked = self.select_partners_by_performance(partner_list, ratio=1.0)
        groups = [
            (1, ranked[: int(len(ranked) * 0.5)]),
            (2, ranked[int(len(ranked) * 0.5): int(len(ranked) * 0.8)]),
            (3, ranked[int(len(ranked) * 0.8):]),
        ]
        for offset, group in groups:
            t = s + offset
            if t >= n or not group:
                continue
            step_needs = int(max(0, need_fn(t)) / 3)
            if step_needs <= 0:
                continue
            for k, q in zip(group, distribute(step_needs, len(group))):
                if q > 0:
                    response[k] = (q, t, self.best_price(k))
        return response

    def future_supplie_offer(self, partner_list) -> dict:
        return self._future_offer(partner_list, self.supply_need_at)

    def future_consume_offer(self, partner_list) -> dict:
        awi = self.awi
        # 売りは total_sales_at <= n_lines の step のみ
        def need_fn(t):
            if awi.total_sales_at(t) > awi.n_lines:
                return 0
            return self.sales_need_at(t)
        return self._future_offer(partner_list, need_fn)

    # ------------------------------------------------------------------ #
    def is_supplier(self, partner) -> bool:
        return partner in self.awi.my_suppliers

    def is_consumer(self, partner) -> bool:
        return partner in self.awi.my_consumers

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        for p in partners:
            if p != self.id:
                self.update_partner_performance(p, False)


# 後方互換エイリアス（実験コードは RobustV2 を参照しているため）
RobustV2 = SuperimagentZ


if __name__ == "__main__":
    import sys
    from helpers.runner import run

    run([SuperimagentZ], sys.argv[1] if len(sys.argv) > 1 else "std")
