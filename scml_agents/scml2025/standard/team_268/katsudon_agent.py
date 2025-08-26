#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.std import StdAWI, StdSyncAgent
from scml.oneshot.common import *

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

from itertools import repeat, combinations, chain
import random
from collections import Counter
from numpy.random import choice

__all__ = ["KATSUDONAgent"]


class KATSUDONAgent(StdSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `StdUFun` in the docs for more details).
    """

    def __init__(self, *args, threshold=None, ptoday=0.75, productivity=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = random.random() * 0.2 + 0.2
        self._threshold = threshold
        self._ptoday = ptoday
        self._productivity = productivity
        # HACK
        self.production_level = 0.5
        self.future_concession = 0.1

    # =====================
    # Negotiation Callbacks
    # =====================

    def dynamic_price(self, partner, base_price, time_step):
        history = self.partner_history.get(partner, [])
        if not history:
            return base_price
        acceptance_rate = sum(1 for h in history if h["accepted"]) / len(history)
        adjustment = (1 - acceptance_rate) * 0.1  # 調整率
        if self.is_consumer(partner):
            return base_price * (1 - adjustment)
        else:
            return base_price * (1 + adjustment)

    def predict_future_needs(self, days_ahead=1):
        recent_needs = self.past_needs[-5:]  # 直近5日間の需要
        if not recent_needs:
            return self.awi.needed_supplies
        avg_need = sum(recent_needs) / len(recent_needs)
        return int(avg_need * days_ahead)

    def record_negotiation(self, partner, offer, accepted):
        if partner not in self.partner_history:
            self.partner_history[partner] = []
        self.partner_history[partner].append(
            {"offer": offer, "accepted": accepted, "time": self.awi.current_step}
        )

    def first_proposals(self) -> dict[str, Outcome | None]:
        partners = self.negotiators.keys()
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()
        return {
            k: (q, s, self.best_price(k))
            if q > 0
            else self.sample_future_offer(k).outcome
            for k, q in distribution.items()
        }

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response = dict()
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
            needs = (
                max(edge_needs, int(self.awi.n_lines * self._productivity))
                if self.awi.is_middle_level
                else edge_needs
            )

            partners = {_ for _ in all_partners if _ in offers.keys()}

            plist = list(self.powerset(partners))
            best_diff, best_index = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_index = diff, i
                if diff == 0:
                    break

            if best_diff <= self._threshold:
                partner_ids = plist[best_index]
                others = list(partners.difference(partner_ids))
                response.update(
                    {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    }
                    | {k: self.sample_future_offer(k) for k in others}
                )
                continue

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

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        "Distributes mu urgent (today's) needs randomly over some my partners"
        if partners is None:
            partners = self.negotiators.keys()

        response = dict(zip(partners, repeat(0)))
        for is_partner, edge_needs in (
            (self.is_supplier, self.awi.needed_supplies),
            (self.is_consumer, self.awi.needed_sales),
        ):
            needs = (
                max(edge_needs, int(self.awi.n_lines * self._productivity))
                if self.awi.is_middle_level
                else edge_needs
            )

            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            random.shuffle(active_partners)
            active_partners = active_partners[
                : max(1, int(self._ptoday * len(active_partners)))
            ]
            n_partners = len(active_partners)

            if needs <= 0 or n_partners <= 0:
                continue

            if needs < n_partners:
                active_partners = random.sample(
                    active_partners, random.randint(1, needs)
                )
                n_partners = len(active_partners)

            response |= dict(zip(active_partners, self.distribute(needs, n_partners)))

        return response

    def distribute(self, q: int, n: int) -> list[int]:
        if q < n:
            lst = [0] * (n - q) + [1] * q
            random.shuffle(lst)
            return lst

        if q == n:
            return [1] * n

        r = Counter(choice(n, q - n))
        return [r.get(_, 0) + 1 for _ in range(n)]

    def sample_future_offer(self, partner):
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

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        return (
            ResponseType.ACCEPT_OFFER
            if self.is_needed(negotiator_id, offer)
            and self.is_good_price(negotiator_id, offer, state)
            else ResponseType.REJECT_OFFER
        )

    def is_needed(self, partner, offer):
        if offer is None:
            return False
        return offer[QUANTITY] <= self._needs(partner, offer[TIME])

    def is_good_price(self, partner, offer, state):
        if offer is None:
            return False
        nmi = self.get_nmi(partner)
        if not nmi:
            return False
        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value
        r = state.relative_time
        if offer[TIME] > self.awi.current_step:
            r *= self.future_concession
        if self.is_consumer(partner):
            return offer[UNIT_PRICE] >= minp + (1 - r) * (maxp - minp)
        return -offer[UNIT_PRICE] >= -minp + (1 - r) * (minp - maxp)

    def propose(self, negotiator_id: str, state):
        return self.good_offer(negotiator_id, state)

    def good_offer(self, partner, state):
        nmi = self.get_nmi(partner)
        if not nmi:
            return None
        issues = nmi.issues
        qissue = issues[QUANTITY]
        pissue = issues[UNIT_PRICE]
        for t in sorted(list(issues[TIME].all)):
            needed = self._needs(partner, t)
            if needed <= 0:
                continue
            offer = [-1] * 3
            offer[QUANTITY] = max(min(needed, qissue.max_value), qissue.min_value)
            offer[TIME] = t
            r = state.relative_time
            if t > self.awi.current_step:
                r *= self.future_concession
            minp, maxp = pissue.min_value, pissue.max_value

            if self.is_consumer(partner):
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * (1 - r) + 0.5)
            else:
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * r + 0.5)
            return tuple(offer)
        return None

    def secure_future_contracts(self):
        """
        Proactively initiates proposals for future needs based on predicted demand
        to avoid last-minute shortages and ensure stable production.
        """
        future_demand = self.predict_future_needs(days_ahead=2)
        future_step = self.awi.current_step + 1
        proactive_offers = {}

        for partner in self.negotiators:
            nmi = self.get_nmi(partner)
            if not nmi or not self.is_supplier(partner):
                continue

            qissue = nmi.issues[QUANTITY]
            t_min, t_max = nmi.issues[TIME].min_value, nmi.issues[TIME].max_value

            if not (t_min <= future_step <= t_max):
                continue

            offer = (
                min(future_demand, qissue.max_value),
                future_step,
                self.best_price(partner),
            )

            proactive_offers[partner] = offer

        for partner, offer in proactive_offers.items():
            self.awi.request_negotiation(partner, offer)

    def before_step(self):
        self.secure_future_contracts()

    def _needs(self, partner, t):
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines

        if self.is_consumer(partner):
            total_needs += (
                self.production_level * self.awi.n_lines * (t - self.awi.current_step)
            )
            total_needs -= self.awi.total_sales_until(t)
        else:
            total_needs -= (
                self.production_level * self.awi.n_lines * (self.awi.n_steps - t - 1)
            )
            total_needs -= self.awi.total_supplies_between(t, self.awi.n_steps - 1)
        return int(total_needs)

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner):
        return self.get_nmi(partner).issues[UNIT_PRICE].rand()

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
