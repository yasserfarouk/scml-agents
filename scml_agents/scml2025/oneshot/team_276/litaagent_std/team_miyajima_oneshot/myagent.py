#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.oneshot import *
from scml.std import *
from scml.oneshot.common import is_system_agent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

import numpy
import random

# def distribute(q: int, n: int) -> list[int]:
#     """
#     Distributes n values over m bins with at least one item per bin assuming q > n
    
#     INPUT:
#     - q: 必要な総数量
#     - n: 交渉相手の数
#     OUTPUT:
#     - 総和がqとなる0以上のn個の整数のランダムなリスト
#     e.g. distribute(10,4) = [2,3,1,4]
#     """
#     from numpy.random import choice
#     from collections import Counter
#     if q < n:
#         lst = [0] * (n - q) + [1] * q
#         random.shuffle(lst)
#         return lst

#     if q == n:
#         return [1] * n
#     r = Counter(choice(n, q - n))
#     return [r.get(_, 0) + 1 for _ in range(n)]

# def powerset(iterable):
#     """冪集合"""
#     from itertools import chain, combinations
#     s = list(iterable)
#     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# class UtilityBasedAgent(OneShotSyncAgent):
#     """
#     An agent that distributes its needs over its partners randomly.
#     改善できそうな点
#     - distributeの配分を直前の相手の提案量に依存させる
#       - 競合に対して要求している量がわからないため意味ないかも？
#     - 相手の提案の組合せの受諾の可否の決定方法
#       - 単価、生産コスト、廃棄コスト、欠損ペナルティなども含めて効用値ベースで判断する
#       - ラウンド序盤では、数量の不足は数量の過剰ほどリスクはない
#       - もとのcounter_allでは一発で受諾するoffersを決定してしまっているが、実際には一部ずつ決めていくてもありそう
#         - 例えば、自分の必要量が10で数量が[5,3,3,3]だったら、5,3だけ先に受け入れる(効用値が5,3,3の方が高かったとしても)
#     """

#     def before_step(self):
#         super().before_step()
#         self.ufun.find_limit(True)
#         self.ufun.find_limit(False)

#     def __init__(self, *args, threshold=0.95, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._threshold = threshold
#         self._threshold_max = 0.95
#         self._threshold_min = 0.3

#     def distribute_needs(self) -> dict[str, int]:
#         """Distributes my needs randomly over all my partners"""

#         dist = dict()
#         for needs, all_partners in [
#             (self.awi.needed_supplies, self.awi.my_suppliers),
#             (self.awi.needed_sales, self.awi.my_consumers),
#         ]:
#             # find suppliers and consumers still negotiating with me
#             partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
#             partners = len(partner_ids)

#             # if I need nothing, end all negotiations
#             if needs <= 0:
#                 dist.update(dict(zip(partner_ids, [0] * partners)))
#                 continue

#             # distribute my needs over my (remaining) partners.
#             dist.update(dict(zip(partner_ids, distribute(needs, partners))))
#         return dist

#     def first_proposals(self):
#         # just randomly distribute my needs over my partners (with best price for me).
#         s, p = self._step_and_price(best_price=True)
#         distribution = self.distribute_needs()
#         d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}
#         return d

#     def counter_all(self, offers, states):
#         response = dict()
#         # process for sales and supplies independently
#         for needs, all_partners, issues, is_selling in [
#             (
#                 self.awi.needed_supplies,
#                 self.awi.my_suppliers,
#                 self.awi.current_input_issues,
#                 False,
#             ),
#             (
#                 self.awi.needed_sales,
#                 self.awi.my_consumers,
#                 self.awi.current_output_issues,
#                 True,
#             ),
#         ]:
#             if (is_selling and all_partners == ['BUYER']) or (not is_selling and all_partners == ['SELLER']):
#                 continue

#             # get a random price
#             price = issues[UNIT_PRICE].rand()
#             # find active partners
#             partners = {_ for _ in all_partners if _ in offers.keys()}

#             # find the set of partners that gave me the best offer set
#             # (i.e. total quantity nearest to my needs)
#             plist = list(powerset(partners)) # 交渉相手の冪集合
#             #best_diff, best_indx = float("inf"), -1
#             u_max, best_indx = self.ufun.min_utility-1, -1
#             for i, partner_ids in enumerate(plist):
#                 """others = partners.difference(partner_ids)
#                 offered = sum(offers[p][QUANTITY] for p in partner_ids)
#                 diff = abs(offered - needs)
#                 if diff < best_diff:
#                     best_diff, best_indx = diff, i
#                 if diff == 0:
#                     break"""
#                 offers_i = [offers[p] for p in partner_ids]
#                 u = self.ufun.from_offers(
#                     offers = offers_i,
#                     outputs = [is_selling]*len(partner_ids),
#                     ignore_signed_contracts = False
#                 )
#                 if u > u_max:
#                     u_max, best_indx = u, i
            
#             neg_step_normalized = max([states[p].step/self.get_nmi(p).n_steps for p in (plist[best_indx] if len(plist)>0 else all_partners)])
            
            
#             threshold = self._threshold_max - (self._threshold_max - self._threshold_min) * neg_step_normalized

#             # If the best combination of offers is good enough, accept them and end all
#             # other negotiations
#             #if best_diff <= self._threshold:
#             if u_max >= threshold:
#                 partner_ids = plist[best_indx]
#                 others = list(partners.difference(partner_ids))
#                 response.update({
#                     k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k]) for k in partner_ids
#                 } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others})
#                 continue

#             # If I still do not have a good enough offer, distribute my current needs
#             # randomly over my partners.
#             distribution = self.distribute_needs()
#             response.update(
#                 {
#                     k: SAOResponse(ResponseType.END_NEGOTIATION, None)
#                     if q == 0
#                     else SAOResponse(
#                         ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
#                     )
#                     for k, q in distribution.items()
#                 }
#             )
#         return response

#     def _step_and_price(self, best_price=False):
#         """Returns current step and a random (or max) price"""
#         s = self.awi.current_step
#         seller = self.awi.is_first_level
#         issues = (
#             self.awi.current_output_issues if seller else self.awi.current_input_issues
#         )
#         pmin = issues[UNIT_PRICE].min_value
#         pmax = issues[UNIT_PRICE].max_value
#         if best_price:
#             return s, pmax if seller else pmin
#         return s, random.randint(pmin, pmax)

# class EnhancedSyncRandomAgent(OneShotSyncAgent):
#     """
#     An agent that distributes its needs over its partners randomly.
#     改善できそうな点
#     - distributeの配分を直前の相手の提案量に依存させる
#       - 競合に対して要求している量がわからないため意味ないかも？
#     - 相手の提案の組合せの受諾の可否の決定方法
#       - 単価、生産コスト、廃棄コスト、欠損ペナルティなども含めて効用値ベースで判断する
#       - ラウンド序盤では、数量の不足は数量の過剰ほどリスクはない
#     """

#     def __init__(self, *args, threshold=0.95, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._threshold = threshold

#     def distribute_needs(self) -> dict[str, int]:
#         """Distributes my needs randomly over all my partners"""

#         dist = dict()
#         for needs, all_partners in [
#             (self.awi.needed_supplies, self.awi.my_suppliers),
#             (self.awi.needed_sales, self.awi.my_consumers),
#         ]:
#             # find suppliers and consumers still negotiating with me
#             partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
#             partners = len(partner_ids)

#             # if I need nothing, end all negotiations
#             if needs <= 0:
#                 dist.update(dict(zip(partner_ids, [0] * partners)))
#                 continue

#             # distribute my needs over my (remaining) partners.
#             dist.update(dict(zip(partner_ids, distribute(needs, partners))))
#         return dist

#     def before_step(self):
#         #print(f"=== step {self.awi.current_step} ===")
#         return None

#     def first_proposals(self):
#         # just randomly distribute my needs over my partners (with best price for me).
#         s, p = self._step_and_price(best_price=True)
#         distribution = self.distribute_needs()
#         d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}
#         return d

#     def counter_all(self, offers, states):
#         response = dict()
#         # process for sales and supplies independently
#         for needs, all_partners, issues, is_selling in [
#             (
#                 self.awi.needed_supplies,
#                 self.awi.my_suppliers,
#                 self.awi.current_input_issues,
#                 False,
#             ),
#             (
#                 self.awi.needed_sales,
#                 self.awi.my_consumers,
#                 self.awi.current_output_issues,
#                 True,
#             ),
#         ]:
#             # get a random price
#             price = issues[UNIT_PRICE].rand()
#             # find active partners
#             partners = {_ for _ in all_partners if _ in offers.keys()}

#             # find the set of partners that gave me the best offer set
#             # (i.e. total quantity nearest to my needs)
#             plist = list(powerset(partners)) # 交渉相手の冪集合
#             best_diff, best_indx = float("inf"), -1
#             best_total_price = 0 if is_selling else float("inf")
#             for i, partner_ids in enumerate(plist):
#                 others = partners.difference(partner_ids)
#                 offered = sum([offers[p][QUANTITY] for p in partner_ids])
#                 diff = abs(offered - needs)
#                 total_price = sum([offers[p][UNIT_PRICE]*offers[p][QUANTITY] for p in partner_ids])
#                 if diff < best_diff:
#                     best_diff, best_total_price, best_indx = diff, total_price, i
#                     #print(self.awi.agent.id,is_selling,[(offers[p][UNIT_PRICE],offers[p][QUANTITY]) for p in partner_ids], needs, diff, total_price)
#                 elif diff == best_diff:
#                     #print(self.awi.agent.id,is_selling,[(offers[p][UNIT_PRICE],offers[p][QUANTITY]) for p in partner_ids], needs, diff, total_price)
#                     if (is_selling and total_price > best_total_price) or (not is_selling and total_price < best_total_price):
#                         best_diff, best_total_price, best_indx = diff, total_price, i
#                 # if diff == 0:
#                 #     break

#             # print(f"best: {self.awi.agent.id,is_selling,[(offers[p][UNIT_PRICE],offers[p][QUANTITY]) for p in plist[best_indx]], needs, best_diff, best_total_price}")
#             # If the best combination of offers is good enough, accept them and end all
#             # other negotiations
#             if best_diff <= self._threshold:
#                 partner_ids = plist[best_indx]
#                 others = list(partners.difference(partner_ids))
#                 response.update({
#                     k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k]) for k in partner_ids
#                 } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others})
#                 continue

#             # If I still do not have a good enough offer, distribute my current needs
#             # randomly over my partners.
#             distribution = self.distribute_needs()
#             response.update(
#                 {
#                     k: SAOResponse(ResponseType.END_NEGOTIATION, None)
#                     if q == 0
#                     else SAOResponse(
#                         ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
#                     )
#                     for k, q in distribution.items()
#                 }
#             )
#         return response

#     def _step_and_price(self, best_price=False):
#         """Returns current step and a random (or max) price"""
#         s = self.awi.current_step
#         seller = self.awi.is_first_level
#         issues = (
#             self.awi.current_output_issues if seller else self.awi.current_input_issues
#         )
#         pmin = issues[UNIT_PRICE].min_value
#         pmax = issues[UNIT_PRICE].max_value
#         if best_price:
#             return s, pmax if seller else pmin
#         return s, random.randint(pmin, pmax)

def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] = []
) -> list[int]:
    """Distributes q values over n bins.

    Args:
        q: Quantity to distribute
        n: number of bins to distribute q over
        mx: Maximum allowed per bin. `None` for no limit
        equal: Try to make the values in each bins as equal as possible
        concentrated: If true, will try to concentrate offers in few bins. `mx` must be passed in this case
        allow_zero: Allow some bins to be zero even if that is not necessary
    """
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        concentrated_lst = sorted(lst,reverse=True)[:len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i,x in zip(concentrated_idx,concentrated_lst):
            lst.insert(i,x)
        # print(lst,concentrated_idx)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]

def powerset(iterable):
    """冪集合"""
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class SyncRandomOneShotAgent(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, it the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation
                          step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        mismatch_max: float = 0.3,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max = overordering_max
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.mismatch_max = mismatch_max
        super().__init__(*args, **kwargs)

    def init(self):
        if 0 < self.mismatch_max < 1:
            self.mismatch_max *= self.awi.n_lines
        return super().init()

    def distribute_needs(self, t: float) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            int(needs * (1 + self._overordering_fraction(t))),
                            n_partners,
                            equal=self.equal_distribution,
                            allow_zero=self.awi.allow_zero_quantity,
                        ),
                    )
                )
            )
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs(t=0)
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
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
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break
            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            th = self._allowed_mismatch(min(state.relative_time for state in states.values()))
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            distribution = self.distribute_needs(t)
            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _allowed_mismatch(self, r: float):
        mn, mx = 0, self.mismatch_max
        return mn + (mx - mn) * (r**self.mismatch_exp)

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
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

class EnhancedSyncRandomOneShotAgentLegacy(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, if the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
        overmismatch_max: 相手から受けた提案の総量が自身の必要取引量を超える場合に許容する過剰量(n_linesに対する割合で付与)。
        undermismatching_min_selling: 自身が売り手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
        undermismatching_min_buying: 自身が買い手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        #mismatch_max: float = 0.3,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.2,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.3,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        #self.mismatch_max = mismatch_max
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        super().__init__(*args, **kwargs)

    def init(self):
        #if 0 < self.mismatch_max < 1:
            #self.mismatch_max *= self.awi.n_lines
        self.overordering_max = self.overordering_max_selling if self.awi.my_suppliers==["SELLER"] else self.overordering_max_buying
        self.overmismatch_max = self.awi.n_lines * (self.overmismatch_max_selling if self.awi.my_suppliers==["SELLER"] else self.overmismatch_max_buying)
        self.undermismatch_min = self.awi.n_lines * (self.undermismatch_min_selling if self.awi.my_suppliers==["SELLER"] else self.undermismatch_min_buying)
        #print("allow_zero_quantity:",self.awi.allow_zero_quantity)
        #print("storage_cost:",self.awi.current_storage_cost,"disposal_cost:",self.awi.current_disposal_cost,"shortfall_penalty",self.awi.current_shortfall_penalty)
        return super().init()

    def distribute_needs(self, t: float) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            int(needs * (1 + self._overordering_fraction(t))),
                            n_partners,
                            equal=self.equal_distribution,
                            allow_zero=self.awi.allow_zero_quantity,
                        ),
                    )
                )
            )
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs(t=0)
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
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
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            plus_best_diff, plus_best_indx = float("inf"), -1
            minus_best_diff, minus_best_indx = -float("inf"), -1
            best_diff, best_indx = float("inf"), -1
            
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff>=0: # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if is_selling: # 売り手の場合は高かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
                                plus_best_diff, plus_best_indx = diff, i
                        else: # 買い手の場合は安かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
                                plus_best_diff, plus_best_indx = diff, i
                if diff<=0: #必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if diff < 0 and len(partner_ids) < len(plist[minus_best_indx]): # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(plist[minus_best_indx]):
                            if is_selling: # 売り手の場合は高かったら更新
                                if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
                                    minus_best_diff, minus_best_indx = diff, i
                            else: # 買い手の場合は安かったら更新
                                if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
                                    minus_best_diff, minus_best_indx = diff, i      

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None) if not self.awi.allow_zero_quantity
                else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
            )

            th_min, th_max = self._allowed_mismatch(min(state.relative_time for state in states.values()))
            if th_min <= minus_best_diff or plus_best_diff <= th_max:
                if th_min <= minus_best_diff and plus_best_diff <= th_max:
                    if -minus_best_diff == plus_best_diff:
                        if is_selling:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else:
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif -minus_best_diff < plus_best_diff:
                        best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min and plus_best_diff <= th_max:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx
                    
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                
                if best_diff < 0 and len(others) > 0: # 必要量に足りないとき、CounterOfferで補う
                    s,p = self._step_and_price()
                    offer_quantities = dict(
                        zip(
                            others,
                            distribute(-best_diff,len(others),equal=self.equal_distribution,allow_zero=self.awi.allow_zero_quantity)
                        )
                    )
                    response.update({
                        k:(
                            unneeded_response if q==0
                            else SAOResponse(ResponseType.REJECT_OFFER, (q,s,p))
                        ) for k,q in offer_quantities.items()
                    })
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            distribution = self.distribute_needs(t)
            response.update({
                k: (
                    unneeded_response if q == 0
                    else SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                )for k, q in distribution.items()
            })
        return response

    def _allowed_mismatch(self, r: float):
        # mn, mx = 0, self.mismatch_max
        # return mn + (mx - mn) * (r**self.mismatch_exp)
        return self.undermismatch_min * ((1-r)**self.mismatch_exp), self.overmismatch_max * (r**self.mismatch_exp)

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = self.awi.current_output_issues if seller else self.awi.current_input_issues
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

########################################################################################################
########################################################################################################

class EnhancedSyncRandomOneShotAgent(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, if the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
        overmismatch_max: 相手から受けた提案の総量が自身の必要取引量を超える場合に許容する過剰量(n_linesに対する割合で付与)。
        undermismatching_min_selling: 自身が売り手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
        undermismatching_min_buying: 自身が買い手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.2,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.3,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        super().__init__(*args, **kwargs)

    def init(self):
        self.overordering_max = self.overordering_max_selling if self.awi.my_suppliers==["SELLER"] else self.overordering_max_buying
        # self.overmismatch_max = self.awi.n_lines * (self.overmismatch_max_selling if self.awi.my_suppliers==["SELLER"] else self.overmismatch_max_buying)
        # self.undermismatch_min = self.awi.n_lines * (self.undermismatch_min_selling if self.awi.my_suppliers==["SELLER"] else self.undermismatch_min_buying)
        self.overmismatch_max_selling *= self.awi.n_lines
        self.overmismatch_max_buying *= self.awi.n_lines
        self.undermismatch_min_selling *= self.awi.n_lines
        self.undermismatch_min_buying *= self.awi.n_lines

        # 各ラウンドでの相手一人あたりの(擬似的な)提案個数
        self.rounds_ave_offered = [self.awi.n_lines/len(self.awi.my_consumers)] + [self.awi.n_lines/2/len(self.awi.my_consumers)]*9 + [1]*10  if self.awi.my_suppliers==["SELLER"]\
            else [self.awi.n_lines/len(self.awi.my_suppliers)] + [self.awi.n_lines/2/len(self.awi.my_suppliers)]*9 + [1]*10  
        
        self.total_agreed_quantity = {k:0 for k in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers)}
        #print("allow_zero_quantity:",self.awi.allow_zero_quantity)
        #print("storage_cost:",self.awi.current_storage_cost,"disposal_cost:",self.awi.current_disposal_cost,"shortfall_penalty",self.awi.current_shortfall_penalty)
        return super().init()

    def distribute_needs(
        self, t: float, mx: int|None = None, equal: bool|None = None, allow_zero: bool|None = None,
        concentrated: bool = False, concentrated_ids: list[str] = []
    ) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        if equal is None: equal = self.equal_distribution
        if allow_zero is None: allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            # partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            # n_partners = len(partners)
            partners,n_partners = [],0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            int(needs * (1 + self._overordering_fraction(t))),
                            n_partners,
                            mx = mx,
                            equal = equal,
                            concentrated = concentrated,
                            allow_zero = allow_zero,
                            concentrated_idx = concentrated_idx 
                        ),
                    )
                )
            )
        return dist

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]
    
    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        my_negotiators = [p for p in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers) if p in self.negotiators.keys()]
        if self.awi.current_step > self.awi.n_steps * 0.5 and len(my_negotiators) > 0:
            concentrated_ids = sorted(my_negotiators, key=lambda x:self.total_agreed_quantity[x], reverse=True)[:1]
            distribution = self.distribute_needs(t=0,mx=self.awi.n_lines,allow_zero=False,concentrated=True,concentrated_ids=concentrated_ids)
        else:
            distribution = self.distribute_needs(t=0)
        
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {k for k, v in offers.items() if v[TIME] != self.awi.current_step}
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
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
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # ラウンドごとの相手一人あたりの平均提案個数を擬似的に算出
            if len(partners) > 0:
                neg_step = min(state.step for state in states.values())
                self.rounds_ave_offered[neg_step] = 0.7*self.rounds_ave_offered[neg_step] + 0.3*sum([offers[p][QUANTITY] for p in partners])/len(partners)
                
                # print("round",neg_step,self.rounds_ave_offered,sum([offers[p][QUANTITY] for p in partners]))

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None) if not self.awi.allow_zero_quantity
                else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
            )
            
            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            plus_best_diff, plus_best_expected_diff, plus_best_indx = float("inf"), float("inf"), -1
            minus_best_diff, minus_best_expected_diff, minus_best_indx = -float("inf"), -float("inf"), -1
            best_diff, best_indx = float("inf"), -1
            
            # for i, partner_ids in enumerate(plist):
            #     offered = sum(offers[p][QUANTITY] for p in partner_ids)
            #     neg_step = min(state.step for state in states.values())
            #     if neg_step < len(self.rounds_ave_offered) - 1:
            #         expected_next_offered = int(self.rounds_ave_offered[neg_step+1] * len(partners.difference(partner_ids).union(future_partners)))
            #     else:
            #         expected_next_offered = 0
            #     diff = offered - needs
            #     if diff >=0:
            #         expected_diff = diff
            #     elif -expected_next_offered <= diff < 0:
            #         expected_diff = 0
            #     else:
            #         expected_diff = diff + expected_next_offered

            #     if diff==0:
            #         # plus, minusともに更新
            #         if plus_best_diff > 0:
            #             plus_best_diff,plus_best_expected_diff,plus_best_indx = 0,0,i
            #         elif is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
            #             plus_best_diff,plus_best_expected_diff,plus_best_indx = 0,0,i
            #         elif not is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
            #             plus_best_diff,plus_best_expected_diff,plus_best_indx = 0,0,i

            #         if minus_best_diff < 0:
            #             minus_best_diff,minus_best_expected_diff,minus_best_indx = 0,0,i
            #         elif is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
            #             minus_best_diff,minus_best_expected_diff,minus_best_indx = 0,0,i
            #         elif not is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
            #             minus_best_diff,minus_best_expected_diff,minus_best_indx = 0,0,i
            #     elif diff > 0:
            #         # diff == expected_diff > 0
            #         if diff < plus_best_diff:
            #             plus_best_diff,plus_best_expected_diff,plus_best_indx =  diff, expected_diff, i
            #         elif diff == plus_best_diff:
            #             if is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
            #                 plus_best_diff,plus_best_expected_diff,plus_best_indx =  diff, expected_diff, i
            #             elif not is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
            #                 plus_best_diff,plus_best_expected_diff,plus_best_indx =  diff, expected_diff, i
            #     else:
            #         # diff < expected_diff == 0 の場合と diff < expected_diff < 0 の場合がある
            #         # expected_next_offer==0なら diff==expected_diff<0の場合もある
            #         if expected_diff == minus_best_expected_diff:
            #             if diff < minus_best_diff:
            #                 minus_best_diff,minus_best_expected_diff,minus_best_indx = diff,expected_diff,i
            #             elif diff == minus_best_diff:
            #                 if is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
            #                     minus_best_diff,minus_best_expected_diff,minus_best_indx = diff,expected_diff,i
            #                 elif not is_selling and sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
            #                     minus_best_diff,minus_best_expected_diff,minus_best_indx = diff,expected_diff,i
            #         elif expected_diff > minus_best_diff:
            #             minus_best_diff,minus_best_expected_diff,minus_best_indx = diff,expected_diff,i

                
            
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff>=0: # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if is_selling: # 売り手の場合は高かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
                                plus_best_diff, plus_best_indx = diff, i
                        else: # 買い手の場合は安かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
                                plus_best_diff, plus_best_indx = diff, i
                if diff<=0: #必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if diff < 0 and len(partner_ids) < len(plist[minus_best_indx]): # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(plist[minus_best_indx]):
                            if is_selling: # 売り手の場合は高かったら更新
                                if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
                                    minus_best_diff, minus_best_indx = diff, i
                            else: # 買い手の場合は安かったら更新
                                if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
                                    minus_best_diff, minus_best_indx = diff, i      

            # th_min_plus, th_max_plus = self._allowed_mismatch(
            #     min(state.relative_time for state in states.values()),
            #     len(partners.difference(plist[plus_best_indx]).union(future_partners)),
            #     is_selling)
            # th_min_minus, th_max_minus = self._allowed_mismatch(
            #     min(state.relative_time for state in states.values()),
            #     len(partners.difference(plist[minus_best_indx]).union(future_partners)),
            #     is_selling)
            # if th_min_minus <= minus_best_expected_diff or plus_best_expected_diff <= th_max_plus:
            #     if th_min_minus <= minus_best_expected_diff and plus_best_expected_diff <= th_max_plus:
            #         if -minus_best_diff == plus_best_diff:
            #             if is_selling: # 売り手のときは、best_diff>0だとshortfall penaltyが発生するのでminus優先
            #                 best_diff, best_indx = minus_best_diff, minus_best_indx
            #             else: # 買い手のときは、best_diff<0だとshortfall penaltyが発生するので、CounterOfferで補えるときは
            #                 best_diff, best_indx = plus_best_diff, plus_best_indx
            #         elif -minus_best_expected_diff < plus_best_expected_diff:
            #             best_diff, best_indx = minus_best_diff, minus_best_indx
            #         else:
            #             best_diff, best_indx = plus_best_diff, plus_best_indx
            #     elif minus_best_expected_diff < th_min_minus and plus_best_expected_diff <= th_max_plus:
            #         best_diff, best_indx = plus_best_diff, plus_best_indx
            #     else:
            #         best_diff, best_indx = minus_best_diff, minus_best_indx
            
            th_min_plus, th_max_plus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[plus_best_indx]).union(future_partners)),
                is_selling)
            th_min_minus, th_max_minus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[minus_best_indx]).union(future_partners)),
                is_selling)
            if th_min_minus <= minus_best_diff or plus_best_diff <= th_max_plus:
                if th_min_minus <= minus_best_diff and plus_best_diff <= th_max_plus:
                    if -minus_best_diff == plus_best_diff:
                        if is_selling: # 売り手のときは、best_diff>0だとshortfall penaltyが発生するのでminus優先
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else: # 買い手のときは、best_diff<0だとshortfall penaltyが発生するので、CounterOfferで補えるときは
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif -minus_best_diff < plus_best_diff:
                        best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min_minus and plus_best_diff <= th_max_plus:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx
                    
                    
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                
                if best_diff < 0 and len(others) > 0: # 必要量に足りないとき、CounterOfferで補う
                    s,p = self._step_and_price(best_price=True)
                    # distribution = dict(
                    #     zip(
                    #         others,
                    #         distribute(-best_diff,len(others),equal=self.equal_distribution,allow_zero=self.awi.allow_zero_quantity)
                    #     )
                    # )
                    t = min(states[p].relative_time for p in others)
                    if self.awi.current_step > self.awi.n_steps * 0.5:
                        concentrated_ids = sorted(others, key=lambda x:self.total_agreed_quantity[x], reverse=True)[:1]
                        concentrated_idx = [i for i,p in enumerate(others) if p in concentrated_ids]
                        distribution = dict(zip(
                            others,
                            distribute(
                                int(-best_diff * (1 + self._overordering_fraction(t))),
                                len(others),
                                mx = self.awi.n_lines,
                                concentrated = True,
                                concentrated_idx = concentrated_idx
                            )
                        ))
                    else:
                        distribution = dict(zip(
                            others,
                            distribute(int(-best_diff * (1 + self._overordering_fraction(t))), len(others), mx = self.awi.n_lines)
                        ))
                    response.update({
                        k:(
                            unneeded_response if q==0
                            else SAOResponse(ResponseType.REJECT_OFFER, (q,s,p))
                        ) for k,q in distribution.items()
                    })
                
                continue
  
            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            # distribution = self.distribute_needs(t)

            partners = partners.union(future_partners)
            partners = list(partners)
            if self.awi.current_step > self.awi.n_steps * 0.5 and len(partners) > 0:
                concentrated_ids = sorted(partners, key=lambda x:self.total_agreed_quantity[x], reverse=True)[:1]
                concentrated_idx = [i for i,p in enumerate(partners) if p in concentrated_ids]
                distribution = dict(zip(
                    partners,
                    distribute(
                        int(needs * (1 + self._overordering_fraction(t))),
                        len(partners),
                        mx = self.awi.n_lines,
                        concentrated = True,
                        concentrated_idx = concentrated_idx
                    )
                ))
            else:
                distribution = dict(zip(
                    partners,
                    distribute(int(needs * (1 + self._overordering_fraction(t))), len(partners), mx = self.awi.n_lines)
                ))

            response.update({
                k: (
                    unneeded_response if q == 0
                    else SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                )for k, q in distribution.items()
            })
        return response

    # def _allowed_mismatch(self, r:float, n_others:int, is_selling:bool):
    #     return self.undermismatch_min * ((1-r)**self.mismatch_exp), self.overmismatch_max * (r**self.mismatch_exp)

    def _allowed_mismatch(self, r:float, n_others:int, is_selling:bool):
    #     if is_selling:
    #         # 入荷量が一定、過剰な販売契約が良くない
    #         # 一人平均3個くらいの算段でOK?
    #         th_min = - 3 * n_others
    #         th_max = self.overmismatch_max_selling * (r**self.mismatch_exp)
    #     else:
    #         # 不足が良くない、n_othersが多いほどth_minが小さくても(不足が多くても)良い
    #         # 一人平均1.5個くらいの算段でOK?
    #         th_min = - 1.5 * n_others
    #         th_max = self.overmismatch_max_buying * (r**self.mismatch_exp)
    #     return th_min,th_max
        undermismatch_min = self.undermismatch_min_selling if is_selling else self.undermismatch_min_buying
        overmismatch_max = self.overmismatch_max_selling if is_selling else self.overmismatch_max_buying
        return undermismatch_min * ((1-r)**self.mismatch_exp), overmismatch_max * (r**self.mismatch_exp)

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
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

if __name__ == "__main__":
    import sys

    from helpers.runner import run
    from helpers.agents import QuantityOrientedAgent, KanbeAgent, AgentVSCforOneShot
    
    run([EnhancedSyncRandomOneShotAgent,EnhancedSyncRandomOneShotAgentLegacy,QuantityOrientedAgent,KanbeAgent,AgentVSCforOneShot], sys.argv[1] if len(sys.argv) > 1 else "oneshot",print_exceptions=True,n_steps=100)
