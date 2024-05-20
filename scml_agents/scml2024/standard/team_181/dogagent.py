#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing
import random

# required for development
from scml.std import *

# required for typing
from negmas import *


from itertools import chain, combinations, repeat

__all__ = ["DogAgent"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
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


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class DogAgent(StdSyncAgent):
    """An agent that distributes today's needs randomly over 75% of its partners and
    samples future offers randomly."""

    def __init__(self, *args, threshold=None, ptoday=0.75, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = random.random() * 0.2 + 0.2
        self._threshold = threshold
        self._ptoday = ptoday
        self._productivity = productivity

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        # partners = self.negotiators.keys()
        s = self.awi.current_step

        distribution = self.distribute_todays_needs()
        return {
            k: (q, s, self.best_price(k))
            if q > 0
            else self.sample_future_offer(k).outcome
            for k, q in distribution.items()
        }

    def counter_all(self, offers, states):
        response = dict()
        # process for sales and supplies independently
        for edge_needs, all_partners, issues, judge in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
                0,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
                1,
            ),
        ]:
            # correct needs if I am in the middle
            if judge == 0:
                if self.awi.current_inventory_input:
                    needs = 0
                else:
                    needs = self.awi.n_lines - self.awi.current_inventory_input
            if judge == 1:
                needs = min(self.awi.n_lines, self.awi.current_inventory_input)

            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}
            totalsell = 0

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = needs - offered

                if judge == 0:
                    if diff <= 0:
                        if abs(diff) < abs(best_diff):
                            best_diff, best_indx = diff, i
                if judge == 1:
                    if diff >= 0:
                        if abs(diff) < abs(best_diff):
                            best_diff, best_indx = diff, i
                    if offered >= totalsell:
                        totalsell, total_indx = offered, i
                if diff == 0:
                    break

            if self.awi.is_first_level:
                if self.awi.current_inventory_input > 10:
                    if self.awi.n_lines * 0.7 <= totalsell <= self.awi.n_lines:
                        partner_ids = plist[best_indx]
                        # others = list(partners.difference(partner_ids))
                        others = list(partners.difference(partner_ids))
                        response.update(
                            {
                                # k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                                for k in partner_ids
                            }
                            | {
                                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                                for k in others
                            }
                        )
                        continue

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            if abs(best_diff) <= self._threshold:
                partner_ids = plist[best_indx]
                # others = list(partners.difference(partner_ids))
                others = list(partners.difference(partner_ids))
                response.update(
                    {
                        # k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    }
                    | {k: self.sample_future_offer(k) for k in others}
                )
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            distribution = self.distribute_todays_needs()
            response |= {
                k: (
                    self.sample_future_offer(k)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (q, self.awi.current_step, self.best_price(k)),
                    )
                )
                for k, q in distribution.items()
            }
        return response

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """Distributes my urgent (today's) needs randomly over some my partners"""
        if partners is None:
            partners = self.negotiators.keys()

        # initialize all quantities to zero
        response = dict(zip(partners, repeat(0)))
        # repeat for supplies and sales
        for is_partner, edge_needs, judge in (
            (self.is_supplier, self.awi.needed_supplies, 0),
            (self.is_consumer, self.awi.needed_sales, 1),
        ):
            # get my current needs
            if judge == 0:
                if self.awi.current_inventory_input:
                    needs = 0
                else:
                    needs = self.awi.n_lines - self.awi.current_inventory_input
            if judge == 1:
                needs = min(self.awi.n_lines, self.awi.current_inventory_input)

            #  Select a subset of my partners
            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            n_partners = len(active_partners)

            # if I need nothing or have no partnrs, just continue
            if needs <= 0 or n_partners <= 0:
                continue

            # If my needs are small, use a subset of negotiators
            if needs < n_partners:
                active_partners = random.sample(active_partners, needs)
                n_partners = len(active_partners)

            # distribute my needs over my (remaining) partners.
            response |= dict(zip(active_partners, distribute(needs, n_partners)))
        return response

    def sample_future_offer(self, partner):
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    # 最初の提案は、最低価格、最高価格でOK
    def best_price(self, partner):
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        t = self.awi.relative_time * 0.025
        if self.is_supplier(partner):
            return issue.min_value + ((issue.max_value - issue.min_value) * t)
        else:
            return issue.max_value - ((issue.max_value - issue.min_value) * t)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([DogAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
