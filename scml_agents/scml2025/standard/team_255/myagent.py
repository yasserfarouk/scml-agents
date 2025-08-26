from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.oneshot.common import *

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

from itertools import repeat, combinations, chain
import random
from collections import Counter
from numpy.random import choice

from .proactive_agent import ProactiveAgent

__all__ = ["PonponAgent"]


class PonponAgent(ProactiveAgent):
    def __init__(self, *args, threshold=None, ptoday=0.75, productivity=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = random.random() * 0.2 + 0.2
        self._threshold = threshold
        self._ptoday = ptoday
        self._productivity = productivity
        self.production_level = 0.5
        self.future_concession = 0.1
        self._printed_level = False
        self.future_contracts = []
        self.max_future_accept = 2

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
        inventory_forecast = self.forecast_inventory(n_steps=3)

        self.ptoday = 1.0

        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))
        forecast_horizon = 1
        safety_margin = 1
        needs_now = self.awi.needed_supplies
        inventory_now = self.awi.inventory.get(self.awi.my_input_product, 0)
        needed_to_order = max(0, needs_now + safety_margin - inventory_now)
        suppliers = [p for p in partners if self.is_supplier(p)]
        if suppliers and needed_to_order > 0:
            per_supplier = max(1, needed_to_order // len(suppliers))
            for s in suppliers:
                response[s] = per_supplier
        return response

        consumers = [p for p in partners if self.is_consumer(p)]
        for c in consumers:
            response[c] = 0

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
        if self.is_consumer(partner):
            if hasattr(self.awi, "inventory"):
                inventory = self.awi.inventory.get(self.awi.my_input_product, 0)
            else:
                inventory = 0
            up_to = offer[TIME] if offer and TIME in offer else self.awi.current_step
            future_in = self.total_future_incoming(up_to_step=up_to)
            deliverable = inventory + future_in

            threshold = deliverable
            if inventory > self.awi.needed_supplies * 2:
                threshold += 5
            elif inventory > self.awi.needed_supplies * 1.2:
                threshold += 2
            return offer[QUANTITY] <= threshold

        now = self.awi.current_step
        if offer[TIME] == now:
            return True
        future_step = offer[TIME] - now
        if 2 <= future_step <= 4:
            if self.count_future_contracts(future_step) < self.max_future_accept:
                return True
        return False

    def count_future_contracts(self, offset):
        return sum(
            1 for c in self.future_contracts if c[1] == self.awi.current_step + offset
        )

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
        cost = self.get_minimum_profitable_price(partner)

        inventory = (
            self.awi.inventory.get(self.awi.my_input_product, 0)
            if hasattr(self.awi, "inventory")
            else 0
        )
        needed = getattr(self.awi, "needed_supplies", 0)

        if self.is_supplier(partner):
            shortfall_risk = inventory < needed
            if shortfall_risk:
                threshold = cost * 2.0
            else:
                threshold = cost * 1.10
            return offer[UNIT_PRICE] <= threshold
        else:
            return offer[UNIT_PRICE] >= cost * 1.02

    def get_minimum_profitable_price(self, partner):
        profile = self.awi.profile
        if hasattr(profile, "costs"):
            return profile.costs[0][self.awi.my_input_product]
        elif hasattr(profile, "cost"):
            return profile.cost
        else:
            return 0

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

    def _needs(self, partner, t):
        base_needs = (
            self.awi.needed_supplies
            if self.is_supplier(partner)
            else self.awi.needed_sales
        )
        buffer = self.calc_adaptive_buffer() if self.is_supplier(partner) else 0
        return base_needs + buffer

    def calc_adaptive_buffer(self):
        target_inventory = self.awi.needed_supplies * 1.5
        if hasattr(self.awi, "inventory"):
            inventory_now = self.awi.inventory.get(self.awi.my_input_product, 0)
        else:
            inventory_now = 0
        if inventory_now > target_inventory:
            return 0
        elif inventory_now < target_inventory * 0.5:
            return 3
        elif inventory_now < target_inventory * 0.8:
            return 2
        else:
            return 1

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

    def before_step(self):
        if not self._printed_level:
            pass  # print(f"MyAgent 担当階層:{self.awi.my_input_product}")
        self._printed_level = True
        now = self.awi.current_step
        self.future_contracts = [c for c in self.future_contracts if c[1] > now]

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism,
        state,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        if hasattr(contract, "outcome"):
            time = contract.outcome[TIME]
            if time > self.awi.current_step:
                self.future_contracts.append((contract, time))

    def forecast_inventory(self, n_steps=5):
        inventory = (
            self.awi.inventory.get(self.awi.my_input_product, 0)
            if hasattr(self.awi, "inventory")
            else 0
        )
        forecast = [inventory]
        for i in range(1, n_steps + 1):
            incoming = self.scheduled_incoming(i)
            outgoing = self.scheduled_outgoing(i)
            forecast.append(forecast[-1] + incoming - outgoing)
        return forecast

    def total_future_incoming(self, up_to_step=None):
        """現ステップ以降、up_to_stepまでに入る予定のfuture納品量を合計"""
        if up_to_step is None:
            up_to_step = self.awi.n_steps
        total = 0
        for contract, t in self.future_contracts:
            if self.awi.current_step < t <= up_to_step:
                total += contract.outcome[QUANTITY]
        return total
