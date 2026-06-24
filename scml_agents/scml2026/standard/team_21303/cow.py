#!/usr/bin/env python


from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import chain, combinations, repeat
import math

from negmas import *
from numpy.random import choice

from scml.std import *

__all__ = ["COW"]


def distribute(q: int, n: int) -> list[int]:
    """一定数を n 個のバケットに分配するユーティリティ。

    q: 分配すべき合計数
    n: バケット数

    戻り値は各バケットに割り当てる整数のリスト。
    q < n の場合は 0 と 1 を混ぜて各バケットに少なくとも0または1を割り当てる。
    q == n の場合はすべて 1。
    q > n の場合はランダムに (q-n) 個を追加で振り分ける。
    """
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    """与えたイテラブルの全部分集合を生成する（順序は保証しない）。

    小さい集合の全探索に使うユーティリティ。
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class COW(StdSyncAgent):
    
    """
    主要な特徴:
    - パートナー評価は成功率(success_rate)と成立価格のカタログ比(price_score)の重み和で算出。
    - スコア上位のパートナーに優先割当を行い、残りはランダムに一部割り当てる。
    - 初回提案とカウンター提案で異なる価格戦略を持つ。
    """

    def __init__(
        self,
        *args,
        threshold=1,
        ptoday=0.85,
        productivity=0.9,
        weight_success=0.6,
        weight_price=0.4,
        **kwargs,
    ):
        """コンストラクタ

        引数:
        - threshold: 受諾判定の閾値（大きいほど厳しくなる）
        - ptoday: 当日選択割合（0..1）
        - productivity: 1ラインあたりの生産性（需要見積もりに使う）
        - weight_success: スコア計算における成功率の重み
        - weight_price: スコア計算における価格スコアの重み
        残りの `*args, **kwargs` は親クラス `StdSyncAgent` にそのまま渡す。
        """
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1
        self._threshold = threshold
        self._base_ptoday = ptoday
        self._productivity = productivity

        # 交渉の回数／成立数を保持して成功率を算出するための構造
        self.partner_negotiations = defaultdict(int)
        self.partner_successes = defaultdict(int)

        # 成立価格の履歴（平均価格を算出して価格スコアに使う）
        self.partner_success_prices = defaultdict(list)
        # 成立取引の合計金額と合計取引個数（加重平均用）
        self.partner_success_total_amount = defaultdict(float)
        self.partner_success_total_quantity = defaultdict(int)
        # 成立取引の詳細履歴（(unit_price, quantity, step)）を保持してレンジ推定に使う
        self.partner_success_records = defaultdict(list)

        # 最近の利益推移を簡易追跡して戦略（aggressive/conservative）を切替える
        self.last_profits = []
        self.aggressive_mode = False
        self.conservative_mode = False

        # 各パートナーへのカウンター提案回数を追跡
        self.counter_offer_counts = defaultdict(int)

        

        # スコア計算の重み
        self.weight_success = weight_success
        self.weight_price = weight_price

    

    def step(self):
        """毎ステップの更新処理。

        - 動的に閾値を調整して受諾基準を時間や在庫状況に応じて変化させる。
        - 最近の利益推移から保守/攻撃モードをトグルする。
        """
        super().step()

        base_threshold = max(2.0, self.awi.n_lines * 0.2)

        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if inventory_ratio < 0.3:
            self._threshold = max(1, int(base_threshold * 1.5))
        elif inventory_ratio > 0.8:
            self._threshold = max(1, int(base_threshold * 0.7))
        else:
            self._threshold = max(1, int(base_threshold))

        time_left = (self.awi.n_steps - self.awi.current_step) / self.awi.n_steps
        if time_left < 0.3:
            self._threshold = int(self._threshold * 1.8)
            self.aggressive_mode = True
        elif time_left < 0.6:
            self._threshold = int(self._threshold * 1.3)

        current_balance = getattr(self.awi, "current_balance", 0)
        if len(self.last_profits) > 0:
            current_balance = current_balance - sum(self.last_profits)

        self.last_profits.append(current_balance)
        if len(self.last_profits) > 5:
            self.last_profits.pop(0)

        if len(self.last_profits) >= 3:
            recent_trend = (
                sum(self.last_profits[-3:]) / 3 - sum(self.last_profits[:2]) / 2
            )
            if recent_trend < -50:
                self.conservative_mode = True
            elif recent_trend > 50:
                self.conservative_mode = False

        

    def update_partner_performance(self, partner, success, price=None, quantity=None):
        """交渉結果を記録する低レベル関数。

        - partner: 相手識別子
        - success: 成立したかどうか（True/False）
        - price: 成立価格（ある場合は浮動小数として保存）
        この生データは `compute_partner_score` で集約して評価に使われる。
        """
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
            if price is not None:
                try:
                    self.partner_success_prices[partner].append(float(price))
                except Exception:
                    pass
            # quantity が与えられていれば合計金額/合計数量を更新する（加重平均用）
            if quantity is not None:
                try:
                    q = int(quantity)
                    amt = float(price) * q if price is not None else 0.0
                    self.partner_success_total_amount[partner] += amt
                    self.partner_success_total_quantity[partner] += q
                except Exception:
                    pass
            # 履歴として詳細を残す（後で最小/最大などを推定するため）
            try:
                if price is not None and quantity is not None:
                    rec_step = getattr(self.awi, "current_step", None)
                    self.partner_success_records[partner].append((float(price), int(quantity), rec_step))
                    # 履歴の長さを制限（直近200件）
                    if len(self.partner_success_records[partner]) > 200:
                        self.partner_success_records[partner] = self.partner_success_records[partner][-200:]
            except Exception:
                pass

    

    def get_effective_ptoday(self):
        """現在の戦略モードに応じた `ptoday` を返す。"""
        base_ptoday = self._base_ptoday
        if self.aggressive_mode:
            return min(0.9, base_ptoday + 0.15)
        elif self.conservative_mode:
            return max(0.5, base_ptoday - 0.1)
        else:
            return base_ptoday

    def compute_partner_score(self, partner):
        """パートナーの妥協者スコアを 0.0〜1.0 で返す。

        - 成功率(success_rate)は成立数/交渉回数で計算（データ不足時は0.5を用いる）。
        - 価格スコア(price_score)は過去成立価格の平均とカタログの最小/最大を比較して正規化する。
        - 最終スコアは重み付けした線形和を重み合計で正規化し、0..1 にクランプする。
        """
        neg = self.partner_negotiations.get(partner, 0)
        succ = self.partner_successes.get(partner, 0)
        success_rate = succ / neg if neg > 0 else 0.5

        prices = self.partner_success_prices.get(partner, [])
        total_qty = self.partner_success_total_quantity.get(partner, 0)
        total_amt = self.partner_success_total_amount.get(partner, 0.0)
        avg_price = ""
        minp = ""
        maxp = ""
        price_score = 0.5

        # まず平均価格を算出（加重平均優先）
        if total_qty > 0:
            try:
                avg_price = float(total_amt) / int(total_qty)
            except Exception:
                avg_price = float(total_amt) / max(1, int(total_qty))
        elif prices:
            avg_price = float(sum(prices)) / len(prices)

        # 履歴や全体観測からレンジを推定して正規化する（NMI は使用しない）
        if avg_price != "":
            # 優先: パートナー固有の観測レンジ
            recs = self.partner_success_records.get(partner, [])
            prices_obs = [r[0] for r in recs if r and r[0] is not None]

            if prices_obs:
                minp = float(min(prices_obs))
                maxp = float(max(prices_obs))
            else:
                # 次に全パートナーの観測から推定
                all_prices = []
                for lst in self.partner_success_records.values():
                    all_prices.extend([r[0] for r in lst if r and r[0] is not None])
                if all_prices:
                    minp = float(min(all_prices))
                    maxp = float(max(all_prices))
                else:
                    # 最後に±10% の簡易推定
                    try:
                        minp = float(avg_price) * 0.9
                        maxp = float(avg_price) * 1.1
                    except Exception:
                        minp = ""
                        maxp = ""

            if minp != "" and maxp != "":
                if minp > maxp:
                    minp, maxp = maxp, minp
                margin = maxp - minp
                if margin <= 0:
                    price_score = 0.5
                else:
                    if self.is_consumer(partner):
                        price_score = (avg_price - minp) / margin
                    else:
                        price_score = (maxp - avg_price) / margin
                    price_score = max(0.0, min(1.0, float(price_score)))
            else:
                price_score = 0.5
        else:
            price_score = 0.5

        total_weight = self.weight_success + self.weight_price if (self.weight_success + self.weight_price) != 0 else 1.0
        total = (self.weight_success * success_rate + self.weight_price * price_score) / total_weight
        total = max(0.0, min(1.0, float(total)))

        return total

    def select_partners_by_score(self, partners, ratio=None):
        """パートナー群からスコア上位を選択して返す。

        - partners: 候補リスト
        - ratio: 選択割合（None の場合は get_effective_ptoday() を使う）
        - 上位から一定数を選び、残りから少数をランダムで追加する。
        """
        if not partners:
            return []
        if ratio is None:
            ratio = self.get_effective_ptoday()
        scored = [(p, self.compute_partner_score(p)) for p in partners]
        scored.sort(key=lambda x: x[1], reverse=True)
        select_count = max(1, int(len(partners) * ratio))
        selected = [p for p, s in scored[:select_count]]
        remaining = [p for p, s in scored[select_count:]]
        if remaining and len(selected) < len(partners):
            extra_count = min(2, len(remaining), len(partners) - len(selected))
            selected.extend(random.sample(remaining, extra_count))
        return selected

    def select_partners_by_performance(self, partners, ratio=None):
        """後方互換用ラッパー。内部ではスコア選択を利用する。"""
        return self.select_partners_by_score(partners, ratio=ratio)

    def first_proposals(self):
        """初回提案を作成して返す。

        - 当日必要量を分配し、初回価格は `smart_price(..., is_first_proposal=True)` を使う。
        - 将来オファー候補は `future_supplie_offer` / `future_consume_offer` で生成。
        """
        
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

        first_dict = dict()
        future_supplie_partner = []
        future_consume_partner = []

        for k, q in distribution.items():
            if q > 0:
                price = self.smart_price(k, is_first_proposal=True)
                first_dict[k] = (q, s, price)
            elif self.is_supplier(k):
                future_supplie_partner.append(k)
            elif self.is_consumer(k):
                future_consume_partner.append(k)

        response = dict()
        response |= first_dict

        top_suppliers = self.select_partners_by_performance(future_supplie_partner)
        top_consumers = self.select_partners_by_performance(future_consume_partner)

        response |= self.future_supplie_offer(top_suppliers)
        response |= self.future_consume_offer(top_consumers)

        return response

    def counter_all(self, offers, states):
        """受信したオファーに対する応答を生成するメインロジック。

        - current_step のオファーは部分集合全探索で最適組合せを選ぶ（needs に近いもの）。
        - future_step のオファーは条件に合えば受諾する。
        - 受諾した場合は `update_partner_performance` で成立データを記録する。
        """
        response = dict()
        awi = self.awi

        for _edge_needs, all_partners, _issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            day_production = self.awi.n_lines * self._productivity
            needs = 0
            if self.is_supplier(all_partners[0]):
                needs = int(
                    day_production
                    - awi.current_inventory_input
                    - awi.total_supplies_at(awi.current_step)
                )
            elif self.is_consumer(all_partners[0]):
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = int(
                        max(
                            0,
                            min(
                                self.awi.n_lines,
                                day_production + awi.current_inventory_input,
                            )
                            - awi.total_sales_at(awi.current_step),
                        )
                    )

            partners = {_ for _ in all_partners if _ in offers.keys()}

            current_step_offers = dict()
            future_step_offers = dict()

            for p in partners:
                if offers[p] is None:
                    continue
                if offers[p][TIME] == self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    current_step_offers[p] = offers[p]
                elif offers[p][TIME] != self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    future_step_offers[p] = offers[p]

            current_step_partners = { _ for _ in partners if _ in current_step_offers.keys() }

            # 将来オファーを条件に合えば受諾する
            duplicate_list = [0 for _ in range(awi.n_steps)]
            for p, x in future_step_offers.items():
                step = future_step_offers[p][TIME]
                if step <= awi.n_steps:
                    if future_step_offers[p][QUANTITY] + duplicate_list[step - 1] <= self.needs_at(step, p):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, future_step_offers[p])
                        duplicate_list[step - 1] += future_step_offers[p][QUANTITY]
                        try:
                            price = future_step_offers[p][UNIT_PRICE]
                        except Exception:
                            price = None
                        try:
                            qty = future_step_offers[p][QUANTITY]
                        except Exception:
                            qty = None
                        self.update_partner_performance(p, True, price, qty)

            # 現在ステップの候補から最適集合を選ぶ（全探索）
            # 組み合わせ爆発を回避するため、動的計画法で部分集合選択を行う。
            # DP: (sum_quantity, count) -> (total_score, set(partners)) を保持し、
            # 各合計数量ごとに最高のスコア和を採用する。
            partners_list = list(current_step_partners)
            offered_qs = [current_step_offers[p][QUANTITY] for p in partners_list]
            scores = [self.compute_partner_score(p) for p in partners_list]

            dp = {(0, 0): (0.0, set())}
            for idx, p in enumerate(partners_list):
                q = offered_qs[idx]
                sc = scores[idx]
                # 現在の辞書を固定して反復（新しい状態を同じイテレーションで再利用しない）
                current_items = list(dp.items())
                for (s, c), (tot_score, sel) in current_items:
                    ns = s + q
                    nc = c + 1
                    new_score = tot_score + sc
                    existing = dp.get((ns, nc))
                    if existing is None or new_score > existing[0]:
                        dp[(ns, nc)] = (new_score, sel | {p})

            best_plus_set = None
            best_plus_diff = float("inf")
            best_minus_set = None
            best_minus_diff = float("inf")

            for (s, c), (tot_score, sel) in dp.items():
                if c == 0:
                    continue
                diff = abs(s - needs)
                avg_score = tot_score / c if c > 0 else 0.0
                adjusted_diff = diff * (2.0 - avg_score)

                if s - needs >= 0:
                    if adjusted_diff < best_plus_diff and needs > 0:
                        best_plus_diff = adjusted_diff
                        best_plus_set = sel
                else:
                    if adjusted_diff < best_minus_diff and s > 0:
                        best_minus_diff = adjusted_diff
                        best_minus_set = sel

            has_accept_offer = True
            if (
                best_plus_set is not None
                and best_plus_diff <= self._threshold * (2.0 if self.aggressive_mode else 1.0)
                and len(best_plus_set) > 0
            ):
                best_ids = best_plus_set
            elif (
                best_minus_set is not None and len(best_minus_set) > 0
            ):
                best_ids = best_minus_set
            else:
                has_accept_offer = False

            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = best_ids
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))

                response.update({ k: SAOResponse(ResponseType.ACCEPT_OFFER, current_step_offers[k]) for k in partner_ids })

                for k in partner_ids:
                    try:
                        price = current_step_offers[k][UNIT_PRICE]
                    except Exception:
                        price = None
                    try:
                        qty = current_step_offers[k][QUANTITY]
                    except Exception:
                        qty = None
                    self.update_partner_performance(k, True, price, qty)

                for k in others:
                    self.update_partner_performance(k, False)

                others_s = []
                others_c = []
                for x in others:
                    if self.is_supplier(x):
                        others_s.append(x)
                    if self.is_consumer(x):
                        others_c.append(x)

                others_s_dict = dict()
                others_s_dict |= self.future_supplie_offer(others_s)
                others_c_dict = dict()
                others_c_dict |= self.future_consume_offer(others_c)

                for k, x in others_s_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in others_c_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                flag = 1

            if flag != 1:
                other_partners = { _ for _ in all_partners if _ not in response.keys() and _ in self.negotiators.keys() }
                distribution = self.distribute_todays_needs(other_partners)
                future_supplie_partner = []
                future_consume_partner = []

                for k, q in distribution.items():
                    if q > 0:
                        price = self.smart_price(k, is_counter_offer=True)
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                        try:
                            self.counter_offer_counts[k] += 1
                        except Exception:
                            pass
                    elif self.is_supplier(k):
                        future_supplie_partner.append(k)
                    elif self.is_consumer(k):
                        future_consume_partner.append(k)

                future_supplie_offer_dict = dict()
                future_supplie_offer_dict |= self.future_supplie_offer(future_supplie_partner)
                future_consume_offer_dict = dict()
                future_consume_offer_dict |= self.future_consume_offer(future_consume_partner)

                for k, x in future_supplie_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in future_consume_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    def smart_price(self, partner, is_first_proposal=False, is_counter_offer=False):
        """価格を返すユーティリティ。

        - is_first_proposal=True: 初回提案用の価格を返す
        - is_counter_offer=True: カウンター提案用の価格を返す
        戻り値は float（価格）か None。
        """
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        margin = maxp - minp

        # ファーストオファーは妥協度に応じて価格を決定
        if is_first_proposal:
            comp = 0.5
            try:
                comp = float(self.compute_partner_score(partner))
            except Exception:
                pass
            # 売り（相手が消費者）の場合: 妥協度が高いほど高い価格
            if self.is_consumer(partner):
                if comp >= 0.8:
                    price = float(maxp)
                elif comp >= 0.4:
                    price = float(maxp) * 0.95
                else:
                    price = float(maxp) * 0.9
            else:
                # 買い（相手が供給者）の場合は逆向き
                if comp >= 0.8:
                    price = float(minp)
                elif comp >= 0.4:
                    price = float(minp) * 1.05
                else:
                    price = float(minp) * 1.1

            try:
                price = max(float(minp), min(float(maxp), price))
            except Exception:
                pass
            return price

        # カウンターオファーの段階的譲歩: (maxp-minp)*0.3 をステップごとに適用
        if is_counter_offer:
            cnt = self.counter_offer_counts.get(partner, 0)
            # cnt が 0 の場合は最初のカウンター（セカンドオファー）を指すため +1
            stage = cnt + 1
            step = margin * 0.3
            concession = min(margin, step * stage)
            if self.is_consumer(partner):
                # 売り手側: 価格を下げて譲歩
                new_price = maxp - concession
                return max(minp, min(maxp, new_price))
            else:
                # 買い手側: 価格を上げて譲歩
                new_price = minp + concession
                return max(minp, min(maxp, new_price))

        # デフォルト: 中間的な既存ロジックを維持
        base_concession = 0.5
        urgency_bonus = 0.1 if self.aggressive_mode else 0.0
        concession = min(1.0, base_concession + urgency_bonus)
        if self.is_consumer(partner):
            new_price = minp * 1.1 - margin * concession
            return max(new_price, minp)
        else:
            new_price = maxp * 0.9 + margin * concession
            return min(new_price, maxp)

    def is_valid_price(self, price, partner):
        """相手の価格が仕様（min/max）を満たすか判定する。"""
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return price >= minp
        elif self.is_supplier(partner):
            return price <= maxp
        else:
            return False

    def needs_at(self, step, partner):
        """指定ステップにおけるそのパートナーに対する需要量を返す。"""
        need = 0
        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        if self.is_supplier(partner):
            need = int(
                day_production - awi.current_inventory_input - awi.total_supplies_at(step)
            )
        elif self.is_consumer(partner):
            need = int(
                max(
                    0,
                    min(self.awi.n_lines, day_production + awi.current_inventory_input)
                    - awi.total_sales_at(step),
                )
            )
        return need

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """当日必要量をパートナー群に分配して辞書で返す。

        partners が None の場合は交渉相手全員を対象とする。
        上位パートナーには select_partners_by_score を使って優先配分する。
        """
        if partners is None:
            partners = self.negotiators.keys()

        response = dict(zip(partners, repeat(0)))

        supplie_partners = []
        consume_partners = []
        for x in partners:
            if self.is_supplier(x):
                supplie_partners.append(x)
            elif self.is_consumer(x):
                consume_partners.append(x)

        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        supplie_needs = int(
            day_production - awi.current_inventory_input - awi.total_supplies_at(awi.current_step)
        )
        consume_needs = int(
            max(
                0,
                min(self.awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step),
            )
        )

        if len(supplie_partners) > 0 and supplie_needs > 0:
            selected_suppliers = self.select_partners_by_score(supplie_partners)
            response |= self.distribute_todays_supplie_consume_needs(selected_suppliers, supplie_needs)

        # L0 特別処理: L0 から L1 へ出すオファーはシステムから流入した数の 1.5 倍を提示し、
        # まず全ての相手に 3 個ずつ割り当ててから主要パートナーへ余りを配分する。
        if (
            len(consume_partners) > 0
            and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= self.awi.n_lines
        ):
            # L0 の場合は特別な配分ルールを適用
            if getattr(self.awi, "level", None) == 0:
                raw_inflow = getattr(self.awi, "current_exogenous_input_quantity", None)
                try:
                    base_inflow = int(raw_inflow) if raw_inflow is not None and int(raw_inflow) > 0 else 20
                except Exception:
                    base_inflow = 20
                # オファーは流入量の1.5倍（交渉拒否を見越す）
                total_offer = max(0, int(math.ceil(1.5 * base_inflow)))
                # 初期配分: まず全員に3個ずつ
                l0_resp = dict(zip(consume_partners, repeat(0)))
                for p in consume_partners:
                    if total_offer <= 0:
                        break
                    give = min(3, total_offer)
                    l0_resp[p] = give
                    total_offer -= give

                # 余りを主要パートナーに配分
                if total_offer > 0:
                    majors = self.select_partners_by_score(consume_partners)
                    if not majors:
                        majors = list(consume_partners)
                    # スコア比例で配分
                    scores = [max(0.0001, self.compute_partner_score(p)) for p in majors]
                    ssum = sum(scores)
                    for i, p in enumerate(majors):
                        if total_offer <= 0:
                            break
                        if ssum > 0:
                            alloc = int((scores[i] / ssum) * total_offer)
                        else:
                            alloc = total_offer // len(majors)
                        if alloc > 0:
                            l0_resp[p] = l0_resp.get(p, 0) + alloc
                            total_offer -= alloc

                    # まだ余りがあればトップから一つずつ配る
                    idx = 0
                    while total_offer > 0 and majors:
                        tgt = majors[idx % len(majors)]
                        l0_resp[tgt] = l0_resp.get(tgt, 0) + 1
                        total_offer -= 1
                        idx += 1

                response |= l0_resp
            else:
                selected_consumers = self.select_partners_by_score(consume_partners)
                response |= self.distribute_todays_supplie_consume_needs(selected_consumers, consume_needs)

        return response

    def distribute_todays_supplie_consume_needs(self, partners, needs) -> dict[str, int]:
        """選択済パートナーに対して具体的な数量割当を行う。

        - 上位を優先して最低1を配り、残りを順位に応じた重みで配分する。
        - partners は降順にソート済みであることを想定する。
        """
        response = dict(zip(partners, repeat(0)))
        if not partners:
            return response
        # 上位パートナーを優先するため、スコア順でソートしてから上位割合だけを使う
        effective_ptoday = self.get_effective_ptoday()
        partners = sorted(partners, key=lambda p: self.compute_partner_score(p), reverse=True)
        partners = partners[: max(1, int(effective_ptoday * len(partners)))]
        n_partners = len(partners)

        if n_partners > 0:
            if needs < n_partners:
                for i in range(needs):
                    response[partners[i]] = 1
            else:
                for p in partners:
                    response[p] = 1

                remaining_needs = needs - n_partners
                if remaining_needs > 0:
                    weights = list(range(n_partners, 0, -1))
                    total_weight = sum(weights)

                    allocated = 0
                    for i, p in enumerate(partners):
                        if i == n_partners - 1:
                            assign = remaining_needs - allocated
                        else:
                            assign = int((weights[i] / total_weight) * remaining_needs)

                        response[p] += assign
                        allocated += assign

        return response

    def future_supplie_offer(self, partner_list):
        """将来ステップ向けの供給オファー候補を作成して返す。"""
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = dict()

        if not partner_list:
            return response

        sorted_partners = self.select_partners_by_score(partner_list, ratio=1.0)

        step1_list = sorted_partners[: int(len(sorted_partners) * 0.5)]
        step2_list = sorted_partners[int(len(sorted_partners) * 0.5) : int(len(sorted_partners) * 0.8)]
        step3_list = sorted_partners[int(len(sorted_partners) * 0.8) :]

        p = awi.n_lines * self._productivity

        for step_offset, partners in [(1, step1_list), (2, step2_list), (3, step3_list)]:
            if s + step_offset < n and partners:
                step_needs = int(
                    max(
                        0,
                        (p - awi.current_inventory_input - awi.total_supplies_at(s + step_offset)) / 3,
                    )
                )
                if step_needs > 0:
                    distribution = dict(zip(partners, distribute(step_needs, len(partners))))
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = (q, s + step_offset, self.best_price(k))

        return response

    def future_consume_offer(self, partner_list):
        """将来ステップ向けの消費オファー候補を作成して返す。"""
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = dict()

        if not partner_list:
            return response

        sorted_partners = self.select_partners_by_score(partner_list, ratio=1.0)

        step1_list = sorted_partners[: int(len(sorted_partners) * 0.5)]
        step2_list = sorted_partners[int(len(sorted_partners) * 0.5) : int(len(sorted_partners) * 0.8)]
        step3_list = sorted_partners[int(len(sorted_partners) * 0.8) :]

        p = awi.n_lines * self._productivity

        for step_offset, partners in [(1, step1_list), (2, step2_list), (3, step3_list)]:
            if (
                s + step_offset < n
                and awi.total_sales_at(s + step_offset) <= self.awi.n_lines
                and partners
            ):
                step_needs = int(
                    max(
                        0,
                        min(self.awi.n_lines, p + awi.current_inventory_input) - awi.total_sales_at(s + step_offset),
                    )
                    / 3
                )
                if step_needs > 0:
                    distribution = dict(zip(partners, distribute(step_needs, len(partners))))
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = (q, s + step_offset, self.best_price(k))

        return response

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        """その相手にとっての最良価格（供給者なら最小、消費者なら最大）を返す。"""
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmax*0.95 if self.is_supplier(partner) else pmin*1.05

    def price(self, partner):
        """後方互換用: カウンター価格を返すラッパー。"""
        return self.smart_price(partner, is_counter_offer=True)

    

 