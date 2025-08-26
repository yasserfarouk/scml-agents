#!/usr/bin/env python


from __future__ import annotations

import random
from collections import defaultdict, Counter

# required for development
from scml.std import *

# required for typing
from negmas import *

from itertools import chain, combinations, repeat

from numpy.random import choice

__all__ = ["AS0"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    """部分集合を返すメソッド"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class AS0(StdSyncAgent):
    """改良 SAT4 エージェント"""

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1
        self._threshold = threshold
        self._base_ptoday = ptoday
        self._productivity = productivity

        # 実用的改善のための追加属性
        self.partner_success_rate = defaultdict(lambda: 0.5)
        self.partner_negotiations = defaultdict(int)
        self.partner_successes = defaultdict(int)
        self.price_concessions = defaultdict(float)
        self.last_profits = []
        self.aggressive_mode = False
        self.conservative_mode = False

    def step(self):
        super().step()

        # 動的閾値更新（元の仕組みを改善）
        base_threshold = self.awi.n_lines * 0.1

        # 在庫状況による調整
        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if inventory_ratio < 0.3:  # 在庫不足
            self._threshold = max(1, int(base_threshold * 1.5))
        elif inventory_ratio > 0.8:  # 在庫過多
            self._threshold = max(1, int(base_threshold * 0.7))
        else:
            self._threshold = max(1, int(base_threshold))

        # 時間プレッシャーによる調整
        time_left = (self.awi.n_steps - self.awi.current_step) / self.awi.n_steps
        if time_left < 0.3:  # 終盤
            self._threshold = int(self._threshold * 1.8)
            self.aggressive_mode = True
        elif time_left < 0.6:  # 中盤
            self._threshold = int(self._threshold * 1.3)

        # パフォーマンス追跡と戦略調整
        current_balance = getattr(self.awi, "current_balance", 0)
        if len(self.last_profits) > 0:
            profit_change = current_balance - sum(self.last_profits)
        else:
            profit_change = 0

        self.last_profits.append(current_balance)
        if len(self.last_profits) > 5:
            self.last_profits.pop(0)

        # 成績が悪い場合は保守的モードに
        if len(self.last_profits) >= 3:
            recent_trend = (
                sum(self.last_profits[-3:]) / 3 - sum(self.last_profits[:2]) / 2
            )
            if recent_trend < -50:  # 利益が下降傾向
                self.conservative_mode = True
            elif recent_trend > 50:  # 利益が上昇傾向
                self.conservative_mode = False

    def update_partner_performance(self, partner, success):
        """パートナーの成功率を更新"""
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1

        # 成功率計算
        self.partner_success_rate[partner] = (
            self.partner_successes[partner] / self.partner_negotiations[partner]
        )

    def get_effective_ptoday(self):
        """状況に応じたptoday値を計算"""
        base_ptoday = self._base_ptoday

        if self.aggressive_mode:
            return min(0.9, base_ptoday + 0.15)  # より多くのパートナーと取引
        elif self.conservative_mode:
            return max(0.5, base_ptoday - 0.1)  # より少ないパートナーと取引
        else:
            return base_ptoday

    def select_partners_by_performance(self, partners, ratio=None):
        """パフォーマンスに基づくパートナー選択"""
        if not partners:
            return []

        if ratio is None:
            ratio = self.get_effective_ptoday()

        # 成功率でソート
        scored_partners = [(p, self.partner_success_rate[p]) for p in partners]
        scored_partners.sort(key=lambda x: x[1], reverse=True)

        # 上位を選択
        select_count = max(1, int(len(partners) * ratio))
        selected = [p[0] for p in scored_partners[:select_count]]

        # 残りからもランダムに少し選択（探索のため）
        remaining = [p[0] for p in scored_partners[select_count:]]
        if remaining and len(selected) < len(partners):
            extra_count = min(2, len(remaining), len(partners) - len(selected))
            selected.extend(random.sample(remaining, extra_count))

        return selected

    def first_proposals(self):
        """初回提案"""
        partners = self.negotiators.keys()
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

        first_dict = dict()
        future_supplie_partner = []
        future_consume_partner = []

        for k, q in distribution.items():
            if q > 0:
                # 状況に応じた価格戦略
                price = self.smart_price(k, is_first_proposal=True)
                first_dict[k] = (q, s, price)
            elif self.is_supplier(k):
                future_supplie_partner.append(k)
            elif self.is_consumer(k):
                future_consume_partner.append(k)

        response = dict()
        response |= first_dict

        # パフォーマンスの良いパートナーを優先
        top_suppliers = self.select_partners_by_performance(future_supplie_partner)
        top_consumers = self.select_partners_by_performance(future_consume_partner)

        response |= self.future_supplie_offer(top_suppliers)
        response |= self.future_consume_offer(top_consumers)

        return response

    def counter_all(self, offers, states):
        """改善応答システム"""
        response = dict()
        awi = self.awi

        for edge_needs, all_partners, issues in [
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

            current_step_partners = {
                _ for _ in partners if _ in current_step_offers.keys()
            }

            # 将来オファーの処理（元のロジックを保持）
            duplicate_list = [0 for _ in range(awi.n_steps)]
            for p, x in future_step_offers.items():
                step = future_step_offers[p][TIME]
                if step <= awi.n_steps:
                    if future_step_offers[p][QUANTITY] + duplicate_list[
                        step - 1
                    ] <= self.needs_at(step, p):
                        response[p] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, future_step_offers[p]
                        )
                        duplicate_list[step - 1] += future_step_offers[p][QUANTITY]
                        self.update_partner_performance(p, True)

            # 改善された現在ステップオファーの最適化
            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))

            # パフォーマンス重み付きの最適化
            for i, partner_ids in enumerate(plist):
                if not partner_ids:
                    continue

                offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)

                # パートナーの質を考慮したボーナス
                partner_quality_bonus = sum(
                    self.partner_success_rate[p] for p in partner_ids
                ) / len(partner_ids)
                adjusted_diff = diff * (
                    2.0 - partner_quality_bonus
                )  # 質の良いパートナーは差を小さく評価

                if offered - needs >= 0:
                    if adjusted_diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = adjusted_diff, i
                else:
                    if adjusted_diff < best_minus_diff and offered > 0:
                        best_minus_diff, best_index_minus = adjusted_diff, i

            has_accept_offer = True
            if (
                best_plus_diff
                <= self._threshold * (2.0 if self.aggressive_mode else 1.0)
                and len(plist[best_index_plus]) > 0
            ):
                best_indx = best_index_plus
            elif len(plist[best_index_minus]) > 0:
                best_indx = best_index_minus
            else:
                has_accept_offer = False

            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = plist[best_indx]
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))

                # 選択されたパートナーの成功を記録
                response.update(
                    {
                        k: SAOResponse(
                            ResponseType.ACCEPT_OFFER, current_step_offers[k]
                        )
                        for k in partner_ids
                    }
                )

                for k in partner_ids:
                    self.update_partner_performance(k, True)

                # 拒否されたパートナーの記録
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
                other_partners = {
                    _
                    for _ in all_partners
                    if _ not in response.keys() and _ in self.negotiators.keys()
                }
                distribution = self.distribute_todays_needs(other_partners)
                future_supplie_partner = []
                future_consume_partner = []

                for k, q in distribution.items():
                    if q > 0:
                        # 改善された価格戦略
                        price = self.smart_price(k, is_counter_offer=True)
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, price),
                        )
                    elif self.is_supplier(k):
                        future_supplie_partner.append(k)
                    elif self.is_consumer(k):
                        future_consume_partner.append(k)

                future_supplie_offer_dict = dict()
                future_supplie_offer_dict |= self.future_supplie_offer(
                    future_supplie_partner
                )
                future_consume_offer_dict = dict()
                future_consume_offer_dict |= self.future_consume_offer(
                    future_consume_partner
                )

                for k, x in future_supplie_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in future_consume_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    def smart_price(self, partner, is_first_proposal=False, is_counter_offer=False):
        """改善された価格戦略"""
        nmi = self.get_nmi(partner)
        issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value

        # パートナーの成功率を考慮
        success_rate = self.partner_success_rate[partner]

        if self.is_consumer(partner):
            # 顧客への価格
            if is_first_proposal:
                # 初回は強気（元のbest_price）
                return maxp
            else:
                # カウンターオファーでは譲歩
                base_concession = 0.7
                # 成功率の高いパートナーにはより良い価格
                partner_bonus = success_rate * 0.15
                # 緊急時はさらに譲歩
                urgency_bonus = 0.1 if self.aggressive_mode else 0.0

                final_rate = base_concession - partner_bonus - urgency_bonus
                return max(maxp * final_rate, minp)
        else:
            # 仕入先への価格
            if is_first_proposal:
                # 初回は強気（元のbest_price）
                return minp
            else:
                # カウンターオファーでは譲歩
                base_markup = 1.2
                # 成功率の高いパートナーからはより高く購入
                partner_penalty = success_rate * 0.15
                # 緊急時はさらに譲歩
                urgency_penalty = 0.1 if self.aggressive_mode else 0.0

                final_rate = base_markup + partner_penalty + urgency_penalty
                return min(minp * final_rate, maxp)

    # 元のメソッドを保持（基本ロジックは変更しない）
    def is_valid_price(self, price, partner):
        nmi = self.get_nmi(partner)
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
        need = 0
        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        if self.is_supplier(partner):
            need = int(
                day_production
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
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
        """パフォーマンス重視の需要分配"""
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
            day_production
            - awi.current_inventory_input
            - awi.total_supplies_at(awi.current_step)
        )
        consume_needs = int(
            max(
                0,
                min(self.awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step),
            )
        )

        if len(supplie_partners) > 0 and supplie_needs > 0:
            # パフォーマンスの良いパートナーを優先選択
            selected_suppliers = self.select_partners_by_performance(supplie_partners)
            response |= self.distribute_todays_supplie_consume_needs(
                selected_suppliers, supplie_needs
            )

        if (
            len(consume_partners) > 0
            and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= self.awi.n_lines
        ):
            selected_consumers = self.select_partners_by_performance(consume_partners)
            response |= self.distribute_todays_supplie_consume_needs(
                selected_consumers, consume_needs
            )

        return response

    def distribute_todays_supplie_consume_needs(
        self, partners, needs
    ) -> dict[str, int]:
        response = dict(zip(partners, repeat(0)))
        if not partners:
            return response

        random.shuffle(partners)
        effective_ptoday = self.get_effective_ptoday()
        partners = partners[: max(1, int(effective_ptoday * len(partners)))]
        n_partners = len(partners)

        if needs < n_partners and n_partners > 0:
            partners = random.sample(
                partners, random.randint(1, min(needs, n_partners))
            )
            n_partners = len(partners)

        if n_partners > 0:
            response |= dict(zip(partners, distribute(needs, n_partners)))

        return response

    def future_supplie_offer(self, partner_list):
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = dict()

        if not partner_list:
            return response

        # パフォーマンスでソートしてから分配
        sorted_partners = self.select_partners_by_performance(partner_list, ratio=1.0)

        step1_list = sorted_partners[: int(len(sorted_partners) * 0.5)]
        step2_list = sorted_partners[
            int(len(sorted_partners) * 0.5) : int(len(sorted_partners) * 0.8)
        ]
        step3_list = sorted_partners[int(len(sorted_partners) * 0.8) :]

        p = awi.n_lines * self._productivity

        for step_offset, partners in [
            (1, step1_list),
            (2, step2_list),
            (3, step3_list),
        ]:
            if s + step_offset < n and partners:
                step_needs = int(
                    max(
                        0,
                        (
                            p
                            - awi.current_inventory_input
                            - awi.total_supplies_at(s + step_offset)
                        )
                        / 3,
                    )
                )
                if step_needs > 0:
                    distribution = dict(
                        zip(partners, distribute(step_needs, len(partners)))
                    )
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = (q, s + step_offset, self.best_price(k))

        return response

    def future_consume_offer(self, partner_list):
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = dict()

        if not partner_list:
            return response

        # パフォーマンスでソートしてから分配
        sorted_partners = self.select_partners_by_performance(partner_list, ratio=1.0)

        step1_list = sorted_partners[: int(len(sorted_partners) * 0.5)]
        step2_list = sorted_partners[
            int(len(sorted_partners) * 0.5) : int(len(sorted_partners) * 0.8)
        ]
        step3_list = sorted_partners[int(len(sorted_partners) * 0.8) :]

        p = awi.n_lines * self._productivity

        for step_offset, partners in [
            (1, step1_list),
            (2, step2_list),
            (3, step3_list),
        ]:
            if (
                s + step_offset < n
                and awi.total_sales_at(s + step_offset) <= self.awi.n_lines
                and partners
            ):
                step_needs = int(
                    max(
                        0,
                        min(self.awi.n_lines, p + awi.current_inventory_input)
                        - awi.total_sales_at(s + step_offset),
                    )
                    / 3
                )
                if step_needs > 0:
                    distribution = dict(
                        zip(partners, distribute(step_needs, len(partners)))
                    )
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = (q, s + step_offset, self.best_price(k))

        return response

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner):
        """従来のprice関数（後方互換性のため保持）"""
        return self.smart_price(partner, is_counter_offer=True)
