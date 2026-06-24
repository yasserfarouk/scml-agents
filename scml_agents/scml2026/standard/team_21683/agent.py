#!/usr/bin/env python
"""SCML 2024 Standard 向け 堅牢再構築エージェント。

設計思想（旧エージェントの最下位敗北＝過学習/破滅的試合への対策）:

1. エッジ(第一/最終)レベルは、ライブラリ保守の堅牢な ``SyncRandomStdAgent``
   ベースをそのまま使う。エッジは外生契約のショックに晒されるため、独自の
   攻めた生産ターゲットでコミットし過ぎると shortfall/在庫費で大損する。
   旧版が ``target_productivity`` を 0.85〜0.95 まで上げていたのが破滅の元。

2. 中間レベルのみ自作ロジックを使う。中間は外生契約が無いため内部生産ターゲット
   が必要だが、ターゲットは控えめ(既定 0.6)に保つ。

3. 全レベル共通のガードレール（破滅回避の核心）:
   - 価格安全: 明確に損になるオファーは絶対に受けない/出さない。
   - コミット上限: 売り過ぎ(shortfall)・買い過ぎ(在庫)を防ぐため、
     今日＋将来の合計コミット量を需要+小さなバッファ以内に厳格に制限する。
   - 将来契約は控えめ: 遠い将来ほど割引し、在庫圧があれば更に絞る。

4. 特定の相手アーキタイプに勝つための調整はしない（汎化重視）。相手の評価は
   「需要マッチング + 価格安全 + 取引成功率(軽いタイブレーク)」のみで行う。
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

from negmas.sao import ResponseType, SAOResponse
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.std.agents import SyncRandomStdAgent


def powerset(iterable):
    items = list(iterable)
    return (s for r in range(len(items) + 1) for s in combinations(items, r))


def split_evenly(quantity: int, partners: list[str]) -> dict[str, int]:
    """数量を相手に均等配分する。"""
    if quantity <= 0 or not partners:
        return {p: 0 for p in partners}
    base, rem = divmod(int(quantity), len(partners))
    return {p: base + (1 if i < rem else 0) for i, p in enumerate(partners)}


class RobustImaAgent(SyncRandomStdAgent):
    """需要マッチング + 価格安全 + コミット上限を備えた堅牢な std エージェント。

    エッジレベルは堅牢ベース、中間レベルは保守的な自作ロジックで動く。
    """

    def __init__(
        self,
        *args,
        # 中間レベルの内部生産ターゲット（控えめ）
        productivity: float = 0.6,
        # 今日の需要に割り当てる相手の割合（残りは将来オファー）
        today_ratio: float = 0.7,
        # 需要を超えて許容するバッファ（小さいほど安全）
        overfill: float = 0.10,
        # 価格の基本譲歩幅（span に対する割合）
        concession: float = 0.16,
        # subset 採用に必要な効用の最小改善幅
        min_margin: float = 0.0,
        # 中間レベルで自作ロジックを使うか（False で純・堅牢ベースになる＝アブレーション用）
        use_custom: bool = True,
        # エッジ(第一/最終)レベルのベースパラメータを上書きするか。
        # 同一ワールドでの公平比較(stability v2)で tune_edges=True + 自作中間が
        # vanilla や SyncRandom を上回ったため既定 True。
        # (注: 12 config 級では min が run 間で激しく振れるため、この判断は
        #  より大きな config 数での再確認が望ましい。)
        tune_edges: bool = True,
        **kwargs,
    ):
        # エッジレベルで使うベースの控えめなチューニング。
        if tune_edges:
            edge_defaults = dict(
                today_target_productivity=0.35,
                future_target_productivity=0.35,
                today_concentration=0.15,
                future_concentration=0.65,
                today_concession_exp=2.0,
                future_concession_exp=4.0,
                future_min_price=0.25,
                prioritize_near_future=True,
                prioritize_far_future=False,
                pfuture=0.12,
            )
            edge_defaults.update(kwargs)
            super().__init__(*args, **edge_defaults)
        else:
            super().__init__(*args, **kwargs)

        self.productivity = productivity
        self.today_ratio = today_ratio
        self.overfill = overfill
        self.concession = concession
        self.min_margin = min_margin
        self._use_custom = use_custom

        # 取引成功率（軽いタイブレーク用）
        self.successes: dict[str, int] = defaultdict(int)
        self.trials: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # 戦略ディスパッチ
    # ------------------------------------------------------------------ #
    def use_custom_logic(self) -> bool:
        """中間レベルでのみ自作ロジックを使う。エッジは堅牢ベース。

        ``use_custom=False`` で自作ロジックを完全に無効化（純・堅牢ベース）でき、
        自作ロジックの寄与を測るアブレーションに使う。
        """
        return self._use_custom and self.awi.is_middle_level

    def first_proposals(self):
        if not self.use_custom_logic():
            return super().first_proposals()
        offers: dict[str, tuple] = {}
        negotiators = set(self.negotiators.keys())
        for partners, selling in (
            (list(self.awi.my_suppliers), False),
            (list(self.awi.my_consumers), True),
        ):
            partners = [p for p in partners if p in negotiators]
            ranked = self.ranked_partners(partners)
            today_n = max(1, int(len(ranked) * self.today_ratio)) if ranked else 0
            today_partners = ranked[:today_n]
            future_partners = ranked[today_n:]
            q_today = self.current_need(selling)
            for p, q in split_evenly(q_today, today_partners).items():
                offers[p] = self.make_offer(p, self.awi.current_step, q)
            offers.update(self.future_offers(future_partners, selling, committed=q_today))
        return {p: offers.get(p, self.unneeded_offer()) for p in self.negotiators.keys()}

    def counter_all(self, offers, states):
        if not self.use_custom_logic():
            return super().counter_all(offers, states)

        responses: dict[str, SAOResponse] = {}
        for partners, selling in (
            (list(self.awi.my_suppliers), False),
            (list(self.awi.my_consumers), True),
        ):
            active = [p for p in partners if p in offers and offers[p] is not None]
            if not active:
                continue

            current = [p for p in active if offers[p][TIME] == self.awi.current_step]
            future = [p for p in active if offers[p][TIME] > self.awi.current_step]

            # --- 将来オファーの受諾（コミット上限つき） ---
            remaining_future: dict[int, int] = defaultdict(int)
            for p in sorted(future, key=self.partner_score, reverse=True):
                offer = offers[p]
                t = offer[TIME]
                limit = self.future_accept_limit(t, selling)
                room = max(0, self.allowed_quantity(limit) - remaining_future[t])
                if self.offer_is_safe(p, offer, states.get(p)) and 0 < offer[QUANTITY] <= room:
                    responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    remaining_future[t] += offer[QUANTITY]

            # --- 今日のオファーの subset 受諾（効用最大 + 需要近接） ---
            accepted_current = self.best_current_subset(current, offers, states, selling)
            for p in accepted_current:
                responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])

            # --- 残りの相手へカウンター ---
            rejected = [p for p in active if p not in responses]
            ranked = self.ranked_partners(rejected)
            today_n = max(1, int(len(ranked) * self.today_ratio)) if ranked else 0
            used_today = sum(offers[p][QUANTITY] for p in accepted_current)
            q_today = max(0, self.current_need(selling) - used_today)
            for p, q in split_evenly(q_today, ranked[:today_n]).items():
                responses[p] = SAOResponse(
                    ResponseType.REJECT_OFFER, self.make_offer(p, self.awi.current_step, q)
                )
            for p, offer in self.future_offers(
                ranked[today_n:], selling, committed=q_today + used_today
            ).items():
                responses[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        # 触れていない相手は不要オファーで応答
        for p in offers:
            if p not in responses:
                responses[p] = SAOResponse(ResponseType.REJECT_OFFER, self.unneeded_offer())
        return responses

    # ------------------------------------------------------------------ #
    # subset 選択（効用ベース、ガードレール内）
    # ------------------------------------------------------------------ #
    def best_current_subset(self, partners, offers, states, selling) -> set[str]:
        candidates = [p for p in partners if self.offer_is_safe(p, offers[p], states.get(p))]
        if not candidates:
            return set()
        # 価格の良い順に上位12件に絞り、組合せ爆発を防ぐ
        candidates = sorted(
            candidates, key=lambda p: self.price_score(p, offers[p]), reverse=True
        )[:12]
        need = self.current_need(selling)
        cap = self.allowed_quantity(need)
        try:
            baseline_utility = float(self.ufun.from_offers({}))
        except Exception:
            baseline_utility = float("-inf")

        best: set[str] = set()
        best_key = (baseline_utility, float("-inf"), float("-inf"), float("-inf"))
        for subset in powerset(candidates):
            q = sum(offers[p][QUANTITY] for p in subset)
            if q <= 0 or q > cap:  # コミット上限を厳守
                continue
            diff = abs(need - q)
            over = max(0, q - need)
            try:
                utility = float(self.ufun.from_offers({p: offers[p] for p in subset}))
            except Exception:
                utility = sum(self.price_score(p, offers[p]) for p in subset)
            price = sum(self.price_score(p, offers[p]) for p in subset)
            key = (utility, -diff, -over, price)
            if key > best_key:
                best_key = key
                best = set(subset)
        # 効用が「何も受けない」を上回らなければ受けない（損回避）
        if best_key[0] <= baseline_utility + self.min_margin:
            return set()
        return best

    # ------------------------------------------------------------------ #
    # 将来オファー生成
    # ------------------------------------------------------------------ #
    def future_offers(self, partners, selling, committed: int = 0) -> dict[str, tuple]:
        partners = self.ranked_partners(partners)
        if not partners:
            return {}
        offers: dict[str, tuple] = {}
        weights = (0.5, 0.3, 0.2)  # t+1, t+2, t+3
        start = 0
        for offset, weight in enumerate(weights, start=1):
            t = self.awi.current_step + offset
            if t >= self.awi.n_steps:
                continue
            end = (
                len(partners)
                if offset == len(weights)
                else start + max(1, int(len(partners) * weight))
            )
            group = partners[start:end]
            start = end
            group_need = self.future_accept_limit(t, selling)
            for p, q in split_evenly(group_need, group).items():
                offers[p] = self.make_offer(p, t, q)
        return offers

    # ------------------------------------------------------------------ #
    # 需要計算（保守的ターゲット）
    # ------------------------------------------------------------------ #
    def current_need(self, selling: bool) -> int:
        """今日確保したい量。中間レベルは在庫を踏まえた生産ターゲット。"""
        if selling:
            if self.awi.is_first_level:
                return max(0, int(self.awi.needed_sales))
            # 中間: 既存在庫の範囲で生産ターゲット分を売る
            target = int(self.awi.n_lines * self.target_productivity())
            raw_stock = max(0, int(self.awi.current_inventory_input))
            sell_target = min(self.awi.n_lines, min(target, raw_stock))
            return max(int(self.awi.needed_sales), sell_target)
        if self.awi.is_last_level:
            return max(0, int(self.awi.needed_supplies))
        # 中間: 生産ターゲットに必要な原料を、現在庫を差し引いて買う
        target = int(self.awi.n_lines * self.target_productivity())
        stock = max(0, int(self.awi.current_inventory_input))
        return max(0, max(int(self.awi.needed_supplies), target) - stock)

    def need_at(self, t: int, selling: bool) -> int:
        if t == self.awi.current_step:
            return self.current_need(selling)
        needed_supplies, needed_sales = self.estimate_future_needs()
        needs = needed_sales if selling else needed_supplies
        return max(0, int(needs.get(t, 0)))

    def target_productivity(self) -> float:
        """中間レベルの生産ターゲット。在庫圧で軽く調整するが攻めすぎない。"""
        stock_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if stock_ratio > 0.8:
            return min(0.8, self.productivity + 0.1)
        if stock_ratio < 0.2:
            return max(0.4, self.productivity - 0.1)
        return self.productivity

    def allowed_quantity(self, need: int) -> int:
        """コミット上限。需要 + 小さなバッファ。"""
        return max(0, int(need * (1 + self.overfill)))

    def future_accept_limit(self, t: int, selling: bool) -> int:
        """将来契約の受諾/提案の上限。遠い将来・在庫圧で割引。"""
        need = self.need_at(t, selling)
        if need <= 0:
            return 0
        days_ahead = t - self.awi.current_step
        if days_ahead <= 0:
            return need
        remaining = max(1, self.awi.n_steps - self.awi.current_step)
        stock_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        factor = 1.0
        if days_ahead > 3:
            factor *= 0.75
        if days_ahead / remaining > 0.5:
            factor *= 0.75
        # 在庫が少ないのに将来売る、在庫過多なのに将来買うのは絞る
        if selling and stock_ratio < 0.3:
            factor *= 0.75 if days_ahead == 1 else 0.50
        if not selling and stock_ratio > 0.8:
            factor *= 0.75 if days_ahead == 1 else 0.50
        return max(0, int(need * factor))

    # ------------------------------------------------------------------ #
    # 価格ロジック（安全第一）
    # ------------------------------------------------------------------ #
    def make_offer(self, partner: str, t: int, quantity: int):
        if quantity <= 0:
            return self.unneeded_offer()
        nmi = self._nmi(partner)
        if not nmi:
            return self.unneeded_offer()
        qissue = nmi.issues[QUANTITY]
        tissue = nmi.issues[TIME]
        q = max(qissue.min_value, min(qissue.max_value, int(quantity)))
        t = max(tissue.min_value, min(tissue.max_value, int(t)))
        return (q, t, self.smart_price(partner, t))

    def unneeded_offer(self):
        return (0, self.awi.current_step, 0) if self.awi.allow_zero_quantity else None

    def smart_price(self, partner: str, t: int) -> int:
        nmi = self._nmi(partner)
        if not nmi:
            return 0
        pmin = nmi.issues[UNIT_PRICE].min_value
        pmax = nmi.issues[UNIT_PRICE].max_value
        span = pmax - pmin
        sigma = self.partner_score(partner)  # 成功率が高い相手には少し譲る
        concession = self.role_concession(partner)
        if self.is_supplier(partner):  # 買い: 安く
            return int(pmin + concession * (1 - sigma) * span + 0.5)
        return int(pmax - concession * (1 - sigma) * span + 0.5)  # 売り: 高く

    def offer_is_safe(self, partner: str, offer, state) -> bool:
        """価格が損にならない範囲かを判定（破滅回避の核心）。"""
        if offer is None or offer[QUANTITY] <= 0:
            return False
        nmi = self._nmi(partner)
        if not nmi:
            return False
        pmin = nmi.issues[UNIT_PRICE].min_value
        pmax = nmi.issues[UNIT_PRICE].max_value
        price = offer[UNIT_PRICE]
        if price < pmin or price > pmax:
            return False
        threshold = self.accept_price(partner, offer, state)
        if self.is_consumer(partner):  # 売り相手: 閾値以上で売る
            return price >= threshold
        return price <= threshold  # 買い相手: 閾値以下で買う

    def accept_price(self, partner: str, offer, state) -> float:
        nmi = self._nmi(partner)
        pmin = nmi.issues[UNIT_PRICE].min_value
        pmax = nmi.issues[UNIT_PRICE].max_value
        span = pmax - pmin
        sigma = self.partner_score(partner)
        r = state.relative_time if state is not None else 0.0
        if offer[TIME] > self.awi.current_step:
            r *= 0.35  # 将来契約は再交渉余地があるので譲歩を抑える
        concession = self.role_concession(partner)
        slack = min(0.35, concession * (1 - sigma) + 0.22 * r)
        if self.is_supplier(partner):
            return pmin + slack * span
        return pmax - slack * span

    def role_concession(self, partner: str) -> float:
        if self.awi.is_middle_level:
            return max(self.concession, 0.18)
        return self.concession

    def price_score(self, partner: str, offer) -> float:
        nmi = self._nmi(partner)
        if not nmi:
            return 0.0
        pmin = nmi.issues[UNIT_PRICE].min_value
        pmax = nmi.issues[UNIT_PRICE].max_value
        span = max(1, pmax - pmin)
        if self.is_supplier(partner):  # 買い: 安いほど良い
            return (pmax - offer[UNIT_PRICE]) / span
        return (offer[UNIT_PRICE] - pmin) / span  # 売り: 高いほど良い

    # ------------------------------------------------------------------ #
    # 相手評価（軽いタイブレークのみ）
    # ------------------------------------------------------------------ #
    def ranked_partners(self, partners) -> list[str]:
        partners = list(partners)
        partners.sort(key=lambda p: (self.partner_score(p), random.random()), reverse=True)
        return partners

    def partner_score(self, partner: str) -> float:
        return (self.successes[partner] + 1) / (self.trials[partner] + 2)

    # ------------------------------------------------------------------ #
    # ヘルパ
    # ------------------------------------------------------------------ #
    def _nmi(self, partner: str):
        try:
            return self.get_nmi(partner)
        except KeyError:
            return None

    def is_supplier(self, partner: str) -> bool:
        return partner in self.awi.my_suppliers

    def is_consumer(self, partner: str) -> bool:
        return partner in self.awi.my_consumers

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        for key in ("buyer", "seller"):
            partner = contract.annotation.get(key)
            if partner and partner != self.id:
                self.successes[partner] += 1
                self.trials[partner] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        for p in partners:
            if p != self.id:
                self.trials[p] += 1


if __name__ == "__main__":
    import sys

    from helpers.runner import run

    run([RobustImaAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
