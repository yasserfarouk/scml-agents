#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors*
Baris Arat, baris.arat@ozu.edu.tr
Ahmet Çağatay Savaşlı, cagatay.savasli@ozu.edu.tr
Alara Baysal, alara.baysal@ozu.edu.tr

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing

# required for development
from scml.oneshot import OneShotSyncAgent

# required for typing
from negmas import SAOResponse
import random

from negmas.sao import SAOResponse
from scml.common import distribute
from scml.oneshot.agent import OneShotSyncAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from negmas import SAOResponse, ResponseType
from scml.oneshot import *
from itertools import chain, combinations

__all__ = ["PeakPact"]


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
) -> list[int]:
    """Distributes q values over n bins.

    The same function used in the default EqualDistOneshotAgent
    We use it for the initial offers only.

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
        random.shuffle(lst)
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
    # used to store all possible combinations of partners
    # when finding the best set of partners
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class PeakPact(OneShotSyncAgent):
    def __init__(
        self,
        *args,
        equal: bool = True,
        # overordering_max: float = 0, #used for experimentation
        # overordering_min: float = 0, #used for experimentation
        # overordering_exp: float = 0, #used for experimentation
        # mismatch_exp: float =  0,  #used for experimentation
        # mismatch_max: float = 0,  #used for experimentation
        **kwargs,
    ):
        self.equal_distribution = equal
        # self.overordering_max = overordering_max  #used for experimentation
        # self.overordering_min = overordering_min  #used for experimentation
        # self.overordering_exp = overordering_exp  #used for experimentation
        # self.mismatch_exp = mismatch_exp  #used for experimentation
        # self.mismatch_max = mismatch_max  #used for experimentation
        super().__init__(*args, **kwargs)

    def init(self):
        # if 0 < self.mismatch_max < 1:
        #    self.mismatch_max *= self.awi.n_lines  #used for experimentation
        return super().init()

    def _distribute_needs_for_first_proposal(self, t: float) -> dict[str, int]:
        """
        Distribute needs over remaining partners equally.
        Uses 1.5x more needs than acutally needed.
        (Assumes all partners will not accept our initial offer.)
        Default uses the best price for us.
        Uses the same distribute function in the default EqualDistOneshotAgent.
        """

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
                            int(
                                needs * 1.5
                            ),  # offer slightly more than needed, assuming some will reject
                            n_partners,
                            equal=self.equal_distribution,
                            allow_zero=self.awi.allow_zero_quantity,
                        ),
                    )
                )
            )
        return dist

    def first_proposals(self):
        """
        Same as the default EqualDistOneshotAgent, but uses the best price for us.
        """
        s, p = self._step_and_price(best_price=True)  # start with best price
        distribution = self._distribute_needs_for_first_proposal(
            t=0
        )  # offers slightly more than needed
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        """
        Contains the main negotiation logic.

        """

        response = dict()
        future_partners = {
            k
            for k, v in offers.items()
            if v[TIME] != self.awi.current_step  # type: ignore
        }  # same as RandDistOneshotAgent
        offers = {
            k: v
            for k, v in offers.items()
            if v[TIME] == self.awi.current_step  # type: ignore
        }  # same as RandDistOneshotAgent

        # process needs and partners based on agent type (sales/consumer)
        # re-factored from RandDistOneshotAgent
        seller = self.awi.is_first_level
        if seller:
            needs, all_partners, issues = (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            )
        else:
            needs, all_partners, issues = (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            )

        # find active partners in some random order
        # same as RandDistOneshotAgent
        partners = [_ for _ in all_partners if _ in offers.keys()]
        random.shuffle(partners)
        partners = set(partners)

        # find the set of partners that gave me the best offer set
        # (i.e. total quantity nearest to my needs)
        plist = list(powerset(partners))[::-1]
        best_diff, best_indx = float("inf"), -1
        for i, partner_ids in enumerate(plist):
            offered = sum(offers[p][QUANTITY] for p in partner_ids)  # type: ignore
            diff = abs(offered - needs)
            if diff < best_diff:
                best_diff, best_indx = diff, i
            if diff == 0:
                break
        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )

        # th is used for: if the best combination of offers is good enough, accept them and end all
        # th = self._allowed_mismatch(min(_.relative_time for _ in states.values())) # RandDistOneshotAgent version
        th = 1  # we deactivated this feature, not allowing accepting all offers without matching the need completely
        if best_diff <= th:
            partner_ids = plist[best_indx]
            others = list(partners.difference(partner_ids).union(future_partners))
            response |= {
                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                for k in partner_ids
            } | {k: unneeded_response for k in others}
            return response

        # If I still do not have a combination of good enough offers,
        # first, I accept the top quantity offers, then
        # distribute my remaining current needs

        # we experimented time-senstive desicions but didn't benefit from it
        # t = min(_.relative_time for _ in states.values())

        distribution = dict()

        # find suppliers and consumers still negotiating with me
        # same as RandDistOneshotAgent
        partners = [_ for _ in all_partners if _ in self.negotiators.keys()]

        n_partners = len(partners)
        # if I need nothing, end all negotiations
        if n_partners == 0:
            return response  # empty dict

        need_per_partner = 0
        # adjusting allowed over ordering and acceptance threshold for the best offers based on market condition
        # consider partner count, as we get less partners, make less over ordering to prevent penalties
        # as the probabililty of getting acceptance from all increases because of lower number of partners
        if n_partners != 0:
            need_per_partner = int(needs / n_partners)
        if n_partners >= 15:
            need_per_partner = (
                need_per_partner * 1
            )  # adding a threshold, experimented with different values but didn'provide extra value
            over_order = 0.2
        elif n_partners >= 10:
            need_per_partner = (
                need_per_partner * 1
            )  # adding a threshold, experimented with different values but didn'provide extra value
            over_order = 0.15
        elif n_partners >= 5:
            need_per_partner = (
                need_per_partner * 1
            )  # adding a threshold, experimented with different values but didn'provide extra value
            over_order = 0.05
        else:
            need_per_partner = (
                need_per_partner * 1
            )  # adding a threshold, experimented with different values but didn'provide extra value
            over_order = 0

        # count the quantity to accept based on the expected quantity per partner
        # accept the offer if it's better than we accept from an agent on average
        # we tried setting higher and lower expected values but this approach provided the best
        # result since it has no bias and generalize better for variety of conditions
        quantity_will_be_accepted = 0
        total_quantity_offered_to_me = 0
        n_less_partners = 0
        total_quantity_will_be_rejected = 0
        for k in plist[best_indx]:  # partner ids
            if (
                offers[k][QUANTITY] > need_per_partner  # type: ignore
            ):  # by rule, these will be accepted
                quantity_will_be_accepted += offers[k][QUANTITY]  # type: ignore
                n_less_partners += (
                    1  # keep track of remaining partners after accepting some
                )
            else:
                total_quantity_will_be_rejected += offers[k][QUANTITY]  # type: ignore
            total_quantity_offered_to_me += offers[k][QUANTITY]  # type: ignore

        updated_needs = (
            needs - quantity_will_be_accepted
        )  # remove the amount will be accepted
        updated_shortage = (
            updated_needs - total_quantity_will_be_rejected
        )  # remaining shortage
        distributed_shortage_amount = updated_shortage / (
            n_partners - n_less_partners
        )  # equally distribute the shortage amount

        # update the response dict with the actual responses
        partner_ids = plist[best_indx]
        s, best_price = self._step_and_price(best_price=True)
        for k in partner_ids:
            # dist[k] gives quantity int directly
            if offers[k][QUANTITY] > (  # type: ignore
                need_per_partner
            ):  # we experimented with different cases like need_per_partner*0.8
                response[k] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
            else:
                a_bit_better_price = (
                    offers[k][UNIT_PRICE] + best_price  # type: ignore
                ) / 2  # better price for us
                response[k] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (
                        (int(offers[k][QUANTITY] + distributed_shortage_amount))  # type: ignore
                        * (1 + over_order),
                        self.awi.current_step,
                        a_bit_better_price,
                    ),
                )

        return response

    # this is a function used in EqualDistOneshotAgent
    # we don't use time based mismatch criteria but we experimented with this
    # def _allowed_mismatch(self, r: float):
    #    mn, mx = 0, self.mismatch_max
    #    return mn + (mx - mn) * (r**self.mismatch_exp)

    # this is a function used in EqualDistOneshotAgent
    # we don't use time based over_ordering criteria but we experimented with this
    # we both experiemented with increasing and decreasing overoverding fraction in time
    # our static approach worked better for our agent
    # def _overordering_fraction(self, t: float):
    #    mn, mx = self.overordering_min, self.overordering_max
    #    return mx - (mx - mn) * (t**self.overordering_exp)

    def _step_and_price(self, best_price=False):
        # this function is same as the EqualDistOneshotAgent
        # we use to get the best price
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

    from .helpers.runner import run

    run([PeakPact], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
