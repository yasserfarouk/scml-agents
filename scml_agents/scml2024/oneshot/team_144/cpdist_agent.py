# -*- coding: utf-8 -*-
import negmas
import scml

from collections import defaultdict
import random
from negmas import ResponseType
from scml.oneshot import *
from scml.utils import anac2024_oneshot
from negmas.sao.common import SAOResponse
from itertools import chain, combinations

__all__ = ["CPDistAgent"]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

class CPDistAgent(SyncRandomOneShotAgent):
    def __init__(self, *args, **kwargs):
        kwargs["equal"] = False
        super().__init__(*args, **kwargs)

    def before_step(self):
        super().before_step()
        self.q_opp_offer = defaultdict(lambda: float("inf"))

    def propose(self, negotiator_id: str, state) -> "Outcome":
        ami = self.get_nmi(negotiator_id)
        offer = super().propose(negotiator_id, state)
        seller = self.awi.is_first_level

        if offer is None:
            return None
        offer = list(offer)

        #compare q_opp_offer and q_need (cooperate)
        if seller:
            opponent = ami.annotation["buyer"]
            offer[QUANTITY] = min(self.q_opp_offer[opponent], offer[QUANTITY])
            offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
        else:
            opponent = ami.annotation["seller"]
            offer[QUANTITY] = min(self.q_opp_offer[opponent], offer[QUANTITY])
            offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
        
        return tuple(offer)

    def counter_all(self, offers, states):
        response = dict()
        seller = self.awi.is_first_level
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
            # price = issues[UNIT_PRICE].rand()
            pmin = issues[UNIT_PRICE].min_value
            pmax = issues[UNIT_PRICE].max_value
            price = pmax if seller else pmin

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
            th = self._allowed_mismatch(min(_.relative_time for _ in states.values()))
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

