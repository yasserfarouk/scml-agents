from .simple_sync_agent import SimpleSyncAgent
import random
from scml.oneshot.common import *
from negmas import SAOResponse, ResponseType


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


from itertools import chain, combinations, repeat


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class ProactiveAgent(SimpleSyncAgent):
    """An agent that distributes today's needs randomly over 75% of its partners and
    samples future offers randomly."""

    def __init__(self, *args, threshold=None, ptoday=0.75, productivity=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = random.random() * 0.2 + 0.2
        self._threshold = threshold
        self._ptoday = ptoday
        self._productivity = productivity

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        partners = self.negotiators.keys()
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
            # correct needs if I am in the middle
            needs = (
                max(edge_needs, int(self.awi.n_lines * self._productivity))
                if self.awi.is_middle_level
                else edge_needs
            )

            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            if best_diff <= self._threshold:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))
                response.update(
                    {
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
                k: self.sample_future_offer(k)
                if q == 0
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (q, self.awi.current_step, self.price(k))
                )
                for k, q in distribution.items()
            }
        return response

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """Distributes my urgent (today's) needs randomly over some my partners"""
        if partners is None:
            partners = self.negotiators.keys()

        # initialize all quantities to zero
        response = dict(zip(partners, repeat(0)))
        # repeat for supplies and sales
        for is_partner, edge_needs in (
            (self.is_supplier, self.awi.needed_supplies),
            (self.is_consumer, self.awi.needed_sales),
        ):
            # get my current needs
            needs = (
                max(edge_needs, int(self.awi.n_lines * self._productivity))
                if self.awi.is_middle_level
                else edge_needs
            )
            #  Select a subset of my partners
            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            random.shuffle(active_partners)
            active_partners = active_partners[
                : max(1, int(self._ptoday * len(active_partners)))
            ]
            n_partners = len(active_partners)

            # if I need nothing or have no partnrs, just continue
            if needs <= 0 or n_partners <= 0:
                continue

            # If my needs are small, use a subset of negotiators
            if needs < n_partners:
                active_partners = random.sample(
                    active_partners, random.randint(1, needs)
                )
                n_partners = len(active_partners)

            # distribute my needs over my (remaining) partners.
            response |= dict(zip(active_partners, distribute(needs, n_partners)))

        return response

    def sample_future_offer(self, partner):
        # get a random future offer. In reality an offer today may be returned
        nmi = self.get_nmi(partner)
        outcome = nmi.random_outcome()
        t = outcome[TIME]
        if t == self.awi.current_step:
            mn = max(nmi.issues[TIME].min_value, self.awi.current_step + 1)
            mx = max(nmi.issues[TIME].max_value, self.awi.current_step + 1)
            if mx <= mn:
                return SAOResponse(ResponseType.END_NEGOTIATION, None)
            t = random.randint(mn, mx)
        return SAOResponse(
            ResponseType.REJECT_OFFER, (outcome[QUANTITY], t, self.best_price(partner))
        )

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner):
        return self.get_nmi(partner).issues[UNIT_PRICE].rand()
