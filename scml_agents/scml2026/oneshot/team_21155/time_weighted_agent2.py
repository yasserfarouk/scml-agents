# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
import traceback
from collections import defaultdict
from itertools import chain, combinations

from negmas import Contract, ResponseType, SAOResponse

from scml.oneshot import *
from scml.std import *

__all__ = ["TimeWeightedAgent"]


def powerset(iterable):
    """すべての組み合わせ（べき集合）を生成するジェネレータ"""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


class TimeWeightedAgent(OneShotSyncAgent):
    def __init__(
        self,
        *args,
        equal: bool = True,
        over_buying: float = 0.20,
        quantity_price_balance: float = 0.85,
        buyer_QP_score_threshold: float = 0.6,
        seller_QP_score_threshold: float = 0.6,
        decay_lambda: float = 2.0,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.over_buying = over_buying
        self.quantity_price_balance = quantity_price_balance
        self.buyer_QP_score_threshold = buyer_QP_score_threshold
        self.seller_QP_score_threshold = seller_QP_score_threshold
        self.decay_lambda = decay_lambda
        super().__init__(*args, **kwargs)

    def init(self):
        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.time_weighted_quantity = defaultdict(float)
        self.is_seller = self.awi.my_suppliers == ["SELLER"]
        return super().init()

    def _ordered_partners_by_time_weight(self, partners: list[str]) -> list[str]:
        """交渉成立ごとの時間重みスコアが高い相手から並べる。"""
        if not partners:
            return []

        offset = self.awi.current_step % len(partners)
        rotated = partners[offset:] + partners[:offset]
        return sorted(
            rotated,
            key=lambda p: self.time_weighted_quantity[p],
            reverse=True,
        )

    def _distribute_by_time_weight(
        self,
        q: int,
        partners: list[str],
        *,
        mx: int | None = None,
    ) -> dict[str, int]:
        """スコアが高い順に、上から1個ずつ周回して数量を割り当てる。"""
        q = int(q)
        dist = {p: 0 for p in partners}

        if q <= 0 or not partners:
            return dist

        if mx is not None:
            q = min(q, mx * len(partners))

        ordered_partners = self._ordered_partners_by_time_weight(partners)
        remaining = q

        while remaining > 0:
            moved = False
            for p in ordered_partners:
                if remaining <= 0:
                    break
                if mx is not None and dist[p] >= mx:
                    continue
                dist[p] += 1
                remaining -= 1
                moved = True
            if not moved:
                break

        return dist

    def _distribute_robust_needs(self, needs, partners, issues, is_selling: bool):
        """不足分をq_min/q_max制約内で、ペナルティが小さい合計数量に配分する。"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}

        needs = int(needs)
        q_min = max(0, int(issues[QUANTITY].min_value))
        q_max = max(q_min, int(issues[QUANTITY].max_value))
        allocations = {p: 0 for p in partners}

        if q_max <= 0:
            return allocations

        n = len(partners)
        shortfall_penalty = self.awi.current_shortfall_penalty
        disposal_cost = self.awi.current_disposal_cost

        def feasible(total_q: int) -> bool:
            if total_q == 0:
                return True
            if total_q < 0:
                return False
            if q_min == 0:
                return total_q <= n * q_max

            min_count = math.ceil(total_q / q_max)
            max_count = min(n, total_q // q_min)
            return min_count <= max_count

        def mismatch_penalty(total_q: int) -> float:
            shortage = max(0, needs - total_q)
            surplus = max(0, total_q - needs)
            if is_selling:
                return shortage * disposal_cost + surplus * shortfall_penalty
            return shortage * shortfall_penalty + surplus * disposal_cost

        max_total = n * q_max
        feasible_totals = [q for q in range(max_total + 1) if feasible(q)]
        if not feasible_totals:
            return allocations

        target_total = min(
            feasible_totals,
            key=lambda q: (mismatch_penalty(q), abs(needs - q), q > needs, q),
        )
        if target_total <= 0:
            return allocations

        ordered_partners = self._ordered_partners_by_time_weight(partners)

        if q_min > 0:
            feasible_counts = [
                k
                for k in range(1, n + 1)
                if k * q_min <= target_total <= k * q_max
            ]
            active_count = max(feasible_counts) if feasible_counts else 0
            if active_count == 0:
                return allocations
            active_partners = ordered_partners[:active_count]
            for p in active_partners:
                allocations[p] = q_min
            remaining = target_total - active_count * q_min
        else:
            active_partners = ordered_partners
            remaining = target_total

        while remaining > 0:
            moved = False
            for p in active_partners:
                if remaining <= 0:
                    break
                if allocations[p] >= q_max:
                    continue
                allocations[p] += 1
                remaining -= 1
                moved = True
            if not moved:
                break

        return allocations

    def distribute_needs(
        self,
        t: float,
        mx: int | None = None,
        equal: bool | None = None,
        allow_zero: bool | None = None,
        concentrated: bool = False,
        concentrated_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """必要数量を時間重みスコア順に1個ずつ分配する。"""
        if allow_zero is None:
            allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners, issues in [
            (self.awi.needed_supplies, self.awi.my_suppliers, self.awi.current_input_issues),
            (self.awi.needed_sales, self.awi.my_consumers, self.awi.current_output_issues),
        ]:
            partners = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)

            if needs <= 0:
                dist.update({p: 0 for p in partners})
                continue

            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)

            q_max = max(0, int(issues[QUANTITY].max_value))
            base_mx = 3 if mx is None else int(mx)
            needed_mx = math.ceil(adjusted_needs / max(1, len(partners)))
            dynamic_mx = min(q_max, max(base_mx, needed_mx))

            dist.update(
                self._distribute_by_time_weight(
                    adjusted_needs, partners, mx=dynamic_mx
                )
            )

        return dist

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        quantity = int(contract.agreement["quantity"])

        if partner_id in self.total_agreed_quantity:
            self.total_agreed_quantity[partner_id] += quantity

        t_relative = self.awi.current_step / max(1, self.awi.n_steps)
        weight = math.exp(self.decay_lambda * t_relative)
        self.time_weighted_quantity[partner_id] += quantity * weight

    def first_proposals(self):
        distribution = self.distribute_needs(
            t=0,
            mx=3,
            equal=True,
            allow_zero=False,
        )

        proposals = {}

        for k, q in distribution.items():
            if q <= 0 and not self.awi.allow_zero_quantity:
                proposals[k] = None
                continue

            is_selling = k in self.awi.my_consumers
            issues = self.awi.current_output_issues if is_selling else self.awi.current_input_issues

            p = self._target_price(issues, k, 0.0)

            proposals[k] = (q, self.awi.current_step, p)

        return proposals

    def _target_price(self, issues, partner, t_relative):
        pmin = int(issues[UNIT_PRICE].min_value)
        pmax = int(issues[UNIT_PRICE].max_value)
        is_selling = partner in self.awi.my_consumers

        if pmin == pmax:
            return pmin

        ideal_price = pmax if is_selling else pmin
        concede_price = pmin if is_selling else pmax

        # 💡 2段階（ステップ型）の確率戦略
        threshold_time = 0.50

        if t_relative < threshold_time:
            ideal_probability = 0.50  # 前半は 50% の確率で最高値
        else:
            ideal_probability = 0.25  # 💡 終盤は 25% に低下（確実な数量確保を優先）

        if random.random() < ideal_probability:
            return ideal_price
        else:
            return concede_price

    # 3. 受諾戦略(静的αによるペナルティ回避)
    def counter_all(self, offers, states):
        try:
            response = {}
            decision_time = (
                min(state.relative_time for state in states.values())
                if states
                else self.awi.current_step / max(1, self.awi.n_steps)
            )
            price_time = (
                sum(state.relative_time for state in states.values()) / len(states)
                if states
                else self.awi.current_step / max(1, self.awi.n_steps)
            )
            
            future_partners = {
                k for k, v in offers.items()
                if v is not None and v[TIME] != self.awi.current_step
            }
            offers = {
                k: v for k, v in offers.items()
                if v is not None and v[TIME] == self.awi.current_step
            }

            for needs, partners, issues, is_selling in [
                (self.awi.needed_supplies, self.awi.my_suppliers, self.awi.current_input_issues, False),
                (self.awi.needed_sales, self.awi.my_consumers, self.awi.current_output_issues, True),
            ]:
                active = {p for p in partners if p in offers}
                if not active:
                    continue

                if needs <= 0:
                    for k in active:
                        response[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    continue

                adjusted_needs = int(needs * (1 + self.over_buying) if not is_selling else needs)
                
                # Step 2: 役割分業に基づく「固定α」の適用 (数量絶対主義)
                static_alpha = 0.85 # 価格でえり好みせず、ペナルティ回避を重視

                best_QP_score = -float("inf")
                best_quantity_diff = 0
                best_set = None

                for subset in powerset(active):
                    if not subset:
                        continue
                    offered_quantity = sum(offers[p][QUANTITY] for p in subset)
                    quantity_diff = offered_quantity - needs
                    total_price = sum(offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in subset)

                    penalty = 0
                    if is_selling:
                        if offered_quantity > needs:
                            penalty += (offered_quantity - needs) * self.awi.current_shortfall_penalty
                        if offered_quantity < needs:
                            penalty += (needs - offered_quantity) * self.awi.current_disposal_cost
                    else:
                        if offered_quantity < needs:
                            penalty += (needs - offered_quantity) * self.awi.current_shortfall_penalty
                        if offered_quantity > needs:
                            penalty += (offered_quantity - needs) * self.awi.current_disposal_cost

                    if is_selling:
                        P_score = total_price - penalty
                    else:
                        P_score = -(total_price + penalty)

                    if P_score > 0:
                        p_norm = 0.5 + (P_score / 1000.0)
                    else:
                        p_norm = 0.5 - (abs(P_score) / 1000.0)
                    p_norm = max(0.0, min(1.0, p_norm))

                    q_norm = 1.0 - abs(quantity_diff) / needs if needs != 0 else 0
                    q_norm = max(0.0, min(1.0, q_norm))

                    # 固定αでスコアを計算
                    QP_score = (static_alpha * q_norm) + ((1.0 - static_alpha) * p_norm)

                    if QP_score > best_QP_score:
                        best_QP_score = QP_score
                        best_quantity_diff = quantity_diff
                        best_set = subset
                    elif QP_score == best_QP_score:
                        if (is_selling and quantity_diff < best_quantity_diff) or (not is_selling and quantity_diff > best_quantity_diff and quantity_diff <= 0):
                            best_quantity_diff = quantity_diff
                            best_set = subset

                # Step 3: 時間による動的β (ハードルの低下)
                base_threshold = self.seller_QP_score_threshold if is_selling else self.buyer_QP_score_threshold
                adjusted_threshold = base_threshold * (1 - 0.5 * decision_time)
                accepted = set(best_set) if best_set and best_QP_score >= adjusted_threshold else set()

                if accepted:
                    accepted_q = sum(offers[k][QUANTITY] for k in accepted)
                    for k in accepted:
                        response[k] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    rejected = active - accepted
                    shortage = needs - accepted_q
                else:
                    rejected = active
                    shortage = needs

                side_future_partners = {p for p in partners if p in future_partners}
                others = [
                    p
                    for p in partners
                    if p in rejected or p in side_future_partners
                ]
                
                # 💡 Step 4: 不足分のカウンターオファー 
                if shortage > 0 and others:
                    # 💡 修正: オウム返しを完全に廃止し、最初からRobust配分でペナルティ最小化を図る
                    allocations = self._distribute_robust_needs(
                        shortage,
                        others,
                        issues,
                        is_selling,
                    )
                    
                    for k, q in allocations.items():
                        if q > 0:
                            price = self._target_price(issues, k, price_time)
                            response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                        else:
                            response[k] = (
                                SAOResponse(ResponseType.END_NEGOTIATION, None)
                                if not self.awi.allow_zero_quantity
                                else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, issues[UNIT_PRICE].rand()))
                            )
                else:
                    for k in others:
                        response[k] = (
                            SAOResponse(ResponseType.END_NEGOTIATION, None)
                            if not self.awi.allow_zero_quantity
                            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, issues[UNIT_PRICE].rand()))
                        )

            return response

        except Exception as e:
            traceback.print_exc()
            return {p: SAOResponse(ResponseType.REJECT_OFFER, offer) for p, offer in offers.items() if offer is not None}
