#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from negmas import SAOResponse, ResponseType

from scml.oneshot import *

import math


# required for typing

# required for development
from scml.oneshot import OneShotSyncAgent

# required for typing
from negmas import Contract

import random

__all__ = ["CostAverseAgent"]


def log_message(*args, **kwargs):
    pass
    # print(message, **kwargs)


def distribute_evenly(total: int, n: int) -> list[int]:
    """
    totalをn個に均等に分配する関数

    :param int total: 分配するものの総数
    :param int n: 分配する場所(or 相手)の数
    :return: 分配されたリスト
    """
    # 分配相手が0以下の時は、分配できないので0を返す
    if n <= 0:
        distributeion = [0] * n
        log_message("zero distribution!!!!!!!!!!!!!!!!!!!!!!")
        return distributeion

    # ゼロでない場合
    base_value = total // n
    # 余りを計算します
    remainder = total % n

    # 基本値をリストに追加し、余りがある限り1を加えていきます
    distribution = [base_value + (1 if i < remainder else 0) for i in range(n)]
    # リストをシャッフル(この状態だと必ず初めの方に余りが入る)
    random.shuffle(distribution)

    return distribution


def distribute(q: int, n: int) -> list[int]:
    from numpy.random import choice
    from collections import Counter

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


from itertools import chain, combinations


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    入力されたiterableから冪集合を求める関数

    :param iterable: 冪集合を求めたいiterable
    :return: 冪集合
    :rtype: iterable
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class CostAverseAgent(OneShotSyncAgent):
    _total_first_propose = 0
    _total_agreements = 1
    _is_after_first_proposal = True
    _first_proposal_accepted = dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_proposal_accepted.clear()
        pass

    def before_step(self):
        log_message("============current_step:", self.awi.current_step, end="")
        if self.awi.is_first_level:
            log_message(" This is first level=============")
        elif self.awi.is_last_level:
            log_message(" This is last level============")
        else:
            log_message(" This is second level============")
        log_message(
            "catalog price:",
            self.awi.catalog_prices,
            "production cost:",
            self.awi.profile.cost,
        )
        log_message(
            "first needed supplies:",
            self.awi.needed_supplies,
            "first needed sales:",
            self.awi.needed_sales,
        )
        return super().before_step()

    def distribute_needs_evenly(self) -> dict[str, int]:
        dist = dict()
        for i, (needs, all_partners) in enumerate(
            [
                (self.awi.needed_supplies, self.awi.my_suppliers),
                (self.awi.needed_sales, self.awi.my_consumers),
            ]
        ):
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            partners = len(partner_ids)
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                continue
            needs_multiplier = 1.3
            if self.awi.is_last_level:
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute_evenly(int(needs * needs_multiplier), partners),
                        )
                    )
                )
            else:
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute_evenly(int(needs / needs_multiplier), partners),
                        )
                    )
                )
        return dist

    def distribute_needs_to_mynegotiator(
        self, partner_ids: list[str]
    ) -> dict[str, int]:
        dist = dict()
        for i, (needs, all_partners) in enumerate(
            [
                (self.awi.needed_supplies, self.awi.my_suppliers),
                (self.awi.needed_sales, self.awi.my_consumers),
            ]
        ):
            num_partners = len(partner_ids)
            all_partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            num_all_partners = len(all_partner_ids)
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * num_partners)))
                continue
            needs_multiplier = 1
            dist.update(
                dict(
                    zip(all_partner_ids, distribute(num_all_partners, num_all_partners))
                )
            )
            if self.awi.is_last_level:
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute(int(needs * needs_multiplier), num_partners),
                        )
                    )
                )
                return dist
            else:
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute(int(needs / needs_multiplier), num_partners),
                        )
                    )
                )
                return dist
        return dist

    def first_proposals(self):
        current_step, price = self._step_and_price(best_price=False)
        distribution = self.distribute_needs_evenly()
        d = {
            k: (quantity, current_step, price) if quantity > 0 else None
            for k, quantity in distribution.items()
        }
        log_message("my first proposal:", d)
        self._total_first_propose += len(d)
        self._is_after_first_proposal = True
        return d

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        buyer = contract.annotation["buyer"]
        seller = contract.annotation["seller"]
        caller = contract.annotation["caller"]
        if self.awi.is_first_level:
            log_message("agreement with", buyer)
        else:
            log_message("agreement with", seller)
            if self._is_after_first_proposal:
                if seller in self._first_proposal_accepted:
                    self._first_proposal_accepted[seller] += 1
                else:
                    self._first_proposal_accepted[seller] = 1
        if self._is_after_first_proposal:
            self._total_agreements += 1
        return super().on_negotiation_success(contract, mechanism)

    def counter_all(self, offers, states):
        self._is_after_first_proposal = False
        shortfall_loss, disposal_loss = self.calculate_losses(offers)
        shorfall_disposal_ratio = (
            shortfall_loss / disposal_loss if disposal_loss != 0 else float("inf")
        )
        log_message(
            f"\n\n▲▲▲▲shorfall_disposal_ratio▲▲▲▲: {shorfall_disposal_ratio:8.3f}\n\n"
        )
        log_message(
            "current disposal cost:",
            self.awi.current_disposal_cost,
            "current shortfall penalty:",
            self.awi.current_shortfall_penalty,
        )
        log_message("offers:", offers)
        log_message(
            "needed_supplies:",
            self.awi.needed_supplies,
            "needed_sales:",
            self.awi.needed_sales,
        )
        response = dict()
        for needs, all_partners, issues in [
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
            current_time = min([_.relative_time for _ in states.values()])
            current_round = round(current_time * 21)
            log_message("---Round", current_round, "---")
            price = issues[UNIT_PRICE].rand()
            partners = {_ for _ in all_partners if _ in offers.keys()}
            plist = list(powerset(partners))
            best_diff, best_indx = float("inf"), -1
            min_shortfall, min_sf_index = 10, -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                shortfall = needs - offered
                estimated_cost = self._calc_estimated_cost(offered, needs)
                utility_value = self.ufun.from_offers(
                    tuple(offers[k] for k in partner_ids),
                    tuple([self.awi.is_first_level] * len(partner_ids)),
                    False,
                    False,
                )
                diff = -utility_value
                if shortfall < min_shortfall and shortfall >= 0:
                    min_shortfall, min_sf_index = shortfall, i
                if diff < best_diff:
                    best_diff, best_indx = diff, i
            log_message(
                "min_shortfall:",
                min_shortfall,
                "min_shorfall_offers:",
                plist[min_sf_index],
            )
            time_threshold = 7
            partner_ids = plist[min_sf_index]
            others = list(partners.difference(partner_ids))
            if min_shortfall == 0 and needs > 0:
                log_message("expected 0 diff agreement")
            best_partner_ids = plist[best_indx]
            best_partber_excess = (
                sum(offers[p][QUANTITY] for p in best_partner_ids) - needs
            )
            log_message("best_partner_excess:", best_partber_excess)
            best_others = list(partners.difference(best_partner_ids))
            if self.awi.is_last_level:
                th = self._current_threshold(
                    current_round, increase_speed=shorfall_disposal_ratio / 2
                )
            elif self.awi.is_first_level:
                if shorfall_disposal_ratio != 0:
                    th = self._current_threshold(
                        current_round, increase_speed=1 / (2 * shorfall_disposal_ratio)
                    )
                else:
                    th = self._current_threshold(current_round, increase_speed=0.5)
            log_message("now th =", th)
            if min_shortfall == 0:
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
            elif 0 <= best_partber_excess <= th:
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in best_partner_ids
                } | {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for k in best_others
                }
            elif current_round > time_threshold and len(partner_ids) > 0:
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in best_partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
            elif min_shortfall <= 1:
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                excess_needs = needs - offered
                if excess_needs < 0:
                    response |= {
                        k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for k in partners
                    }
                else:
                    distribution = dict(
                        zip(others, distribute_evenly(int(excess_needs), len(others)))
                    )
                    response |= {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    } | {
                        k: SAOResponse(ResponseType.REJECT_OFFER, None)
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                        for k, q in distribution.items()
                    }
                continue
            else:
                distribution = self.distribute_needs_to_mynegotiator(partners)
                response.update(
                    {
                        k: SAOResponse(ResponseType.REJECT_OFFER, None)
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                        for k, q in distribution.items()
                    }
                )
        return response

    def _calc_estimated_cost(self, offerd: int, needs: int):
        shortage = 0
        is_first_level = self.awi.is_first_level
        if is_first_level:
            shortage = offerd - needs
        elif self.awi.is_last_level:
            shortage = needs - offerd
        if shortage < 0:
            disposal_cost = self.awi.current_disposal_cost
            return -disposal_cost * shortage
        elif shortage > 0:
            short_fall_penalty = self.awi.current_shortfall_penalty
            return short_fall_penalty * shortage
        else:
            return 0

    def _current_threshold(self, round: int, increase_speed: int = 0.5) -> int:
        if increase_speed < 0:
            increase_speed = 0.5
        log_message(
            "increase speed:",
            increase_speed,
            "float return value:",
            increase_speed * (round - 1),
        )
        return max(0, math.floor(increase_speed * (round - 1)))

    def _step_and_price(self, best_price=False):
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

    def step(self):
        log_message(
            "last needed supplies:",
            self.awi.needed_supplies,
            "last needed sales:",
            self.awi.needed_sales,
        )
        sumsfp = (
            self.awi.current_shortfall_penalty * self.awi.needed_supplies
            if self.awi.needed_supplies > 0
            else 0 - self.awi.current_shortfall_penalty * self.awi.needed_sales
            if self.awi.needed_sales < 0
            else 0
        )
        sumdc = (
            -self.awi.current_disposal_cost * self.awi.needed_supplies
            if self.awi.needed_supplies < 0
            else 0 + self.awi.current_disposal_cost * self.awi.needed_sales
            if self.awi.needed_sales > 0
            else 0
        )
        log_message(f"🔳shortfall penalty: {sumsfp:3.3f}, disposal cost: {sumdc:3.3f}")
        log_message("first_proposal_accepted:", self._first_proposal_accepted)

    def print_all_utility(self, offers: dict[str, tuple], world=None) -> None:
        if not offers:
            log_message("[print_all_utility] offers が空のため計算をスキップ")
            return
        sample_offer = next(iter(offers.values()))
        _, step, price = sample_offer
        qty_list, util_list = [], []
        log_message("── Estimated utility table (qty = 0‥12) ──")
        for qty in range(0, 13):
            virtual_offer = (qty, step, price)
            u = self.ufun.from_offers(
                (virtual_offer,), (self.awi.is_first_level,), False, False
            )
            qty_list.append(qty)
            util_list.append(u)
        min_u = min(util_list)
        max_u = max(util_list)
        abs_max = max(abs(min_u), abs(max_u), 1e-5)
        scale = 40
        log_message("── Utility vs Quantity (negative supported) ──")
        if self.awi.is_first_level:
            log_message("seller")
        else:
            log_message("buyer")
        for q, u in zip(qty_list, util_list):
            left_len = int(max(-u, 0) / abs_max * scale)
            right_len = int(max(u, 0) / abs_max * scale)
            left_bar = "█" * left_len
            right_bar = "█" * right_len
            log_message(f"qty {q:2d}: {u:8.3f} | {left_bar:>40}|{right_bar:<40}")
        short_fall_loss = util_list[1] - util_list[0]
        disposal_loss = util_list[-2] - util_list[-1]
        log_message(
            f"short fall loss: {short_fall_loss:>8.3f}, disposal loss: {disposal_loss:>8.3f}"
        )
        log_message("──────────────────────────────────────────────")

    def calculate_losses(
        self, offers: dict[str, tuple], world=None
    ) -> tuple[float, float]:
        if not offers:
            log_message("[calculate_losses] offers が空のため計算をスキップ")
            return (0.0, 0.0)
        sample_offer = next(iter(offers.values()))
        _, step, price = sample_offer
        qty_list, util_list = [], []
        for qty in range(0, 13):
            virtual_offer = (qty, step, price)
            u = self.ufun.from_offers(
                (virtual_offer,), (self.awi.is_first_level,), False, False
            )
            qty_list.append(qty)
            util_list.append(u)
        shortfall_loss = util_list[1] - util_list[0]
        disposal_loss = util_list[-2] - util_list[-1]
        log_message(
            f"Calculated shortfall loss: {shortfall_loss:8.3f}, disposal loss: {disposal_loss:8.3f}"
        )
        return shortfall_loss, disposal_loss


if __name__ == "__main__":
    import sys

    # run をインポート
    from helpers.runner import run

    for i in range(1):
        run([CostAverseAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
