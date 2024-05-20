#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

from negmas import SAOResponse, ResponseType, Outcome, SAOState

from scml.oneshot.world import SCML2024OneShotWorld as W
from scml.oneshot import *


import random

# required for typing
from typing import Any

# required for development
from scml.oneshot import OneShotAWI, OneShotSyncAgent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState

def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at
    least one item per bin assuming q > n"""
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
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

class BetterSyncAgent(OneShotSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def distribute_needs(self) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers), # 自分がsecondレベルの時はこれだけ
            (self.awi.needed_sales, self.awi.my_consumers), # 自分がfirstレベルの時にはここだけ
        ]:
            # find suppliers and consumers still negotiating with me
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(dict(zip(partner_ids, distribute(needs, partners))))
        return dist

    def first_proposals(self):

        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}
        return d

    def counter_all(self, offers, states):
        response = dict()
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
            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                # best_diff_others を取得しなくていいのか？
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                # partner_idsにあるpartnerとの間のofferの数量の合計

                diff = abs(offered - needs)

                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            th = self._current_threshold(
                min([_.relative_time for _ in states.values()])
            )
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            distribution = self.distribute_needs()
            response.update(
                {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return mn + (mx - mn) * (r**4.0)

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

    run([BetterSyncAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")

#  不足違約金と，廃棄違約金のどちらが高いかを判断してどちらかを重視して数量を決める，ちょっと多くとるとかちょっと少なく取る


