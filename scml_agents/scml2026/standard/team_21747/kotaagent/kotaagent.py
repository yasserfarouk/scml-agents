#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* Hibino Kota <hibino.kota@otsukalab.nitech.ac.jp>

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations

from typing import Any

from scml.std import UNIT_PRICE, StdAWI, StdSyncAgent
from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState


class KotaAgent(StdSyncAgent):
    """
    KotaAgent implements adaptive negotiation strategies for the SCML Standard track.

    Key strategies:
    - Profit margin enforcement: only buy below min-sell price, sell above max-buy price
    - Delivery window constraint: reject contracts more than 4 steps ahead
    - Separate adaptive fractions for buying and selling, learned via EMA from outcomes
    - Failure learning: relaxes price thresholds when negotiations repeatedly fail
    - Fallback buying: accepts minimum-price deals when no profitable margin exists,
      to avoid shortfall penalties from zero production
    """

    def _future_supply_needed(self, t: int) -> int:
        steps_ahead = t - self.awi.current_step
        target = int(self.awi.n_lines * 0.5 * steps_ahead)
        secured = self.awi.total_supplies_until(t)
        return max(0, target - secured)

    def _future_sales_needed(self, t: int) -> int:
        steps_from_t = self.awi.n_steps - t
        target = int(self.awi.n_lines * 0.5 * steps_from_t)
        secured = self.awi.total_sales_from(t)
        return max(0, target - secured)

    def first_proposals(self) -> dict[str, Outcome | None]:
        offers = {}
        suppliers = [p for p in self.negotiators if p in self.awi.my_suppliers]
        consumers = [p for p in self.negotiators if p in self.awi.my_consumers]
        today = self.awi.current_step

        max_buy = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        min_sell = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )

        needed_buy = self.awi.needed_supplies
        needed_sell = self.awi.needed_sales

        if suppliers and needed_buy > 0:
            n = len(suppliers)
            qty_each = min((needed_buy + n - 1) // n, self.awi.current_input_issues[0].max_value)
            for p in suppliers:
                nmi = self.get_nmi(p)
                if nmi is None:
                    continue
                mn = nmi.issues[2].min_value
                price = int(mn + (max_buy - mn) * self._buy_fraction) if max_buy > mn else mn
                offers[p] = (qty_each, today, price)

        if consumers and needed_sell > 0:
            n = len(consumers)
            qty_each = min((needed_sell + n - 1) // n, self.awi.current_output_issues[0].max_value)
            for p in consumers:
                nmi = self.get_nmi(p)
                if nmi is None:
                    continue
                mx = nmi.issues[2].max_value
                price = int(mx - (mx - min_sell) * self._sell_fraction) if mx > min_sell else mx
                offers[p] = (qty_each, today, price)

        return offers

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses = {}
        today = self.awi.current_step
        remaining_buy = self.awi.needed_supplies
        remaining_sell = self.awi.needed_sales
        future_buy_budget: dict[int, int] = {}
        future_sell_budget: dict[int, int] = {}

        max_buy = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        min_sell = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )

        for partner, offer in offers.items():
            if offer is None:
                continue
            nmi = self.get_nmi(partner)
            if nmi is None:
                continue

            q, t, p = offer
            if t - today > self._max_delivery_ahead:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, None)
                continue
            relative_time = states[partner].relative_time
            is_future = t > today

            if partner in self.awi.my_suppliers:
                mn = nmi.issues[2].min_value
                mx = max_buy
                if mn > mx:
                    mx = mn  # no profitable margin: accept at minimum price to avoid shortfall
                if is_future:
                    if t not in future_buy_budget:
                        future_buy_budget[t] = self._future_supply_needed(t)
                    budget = future_buy_budget[t]
                else:
                    budget = remaining_buy
                threshold = int(mn + (mx - mn) * (self._buy_fraction + (1 - self._buy_fraction) * relative_time))
                if p <= threshold and q <= budget and budget > 0:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    if is_future:
                        future_buy_budget[t] -= q
                    else:
                        remaining_buy -= q
                elif budget > 0:
                    counter_q = min(q, budget)
                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER, (counter_q, t, threshold)
                    )
                else:
                    responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, None)
            else:
                mn = min_sell
                mx = nmi.issues[2].max_value
                if mn > mx:
                    continue
                if is_future:
                    if t not in future_sell_budget:
                        future_sell_budget[t] = self._future_sales_needed(t)
                    budget = future_sell_budget[t]
                else:
                    budget = remaining_sell
                threshold = int(mx - (mx - mn) * (self._sell_fraction + (1 - self._sell_fraction) * relative_time))
                if p >= threshold and q <= budget and budget > 0:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    if is_future:
                        future_sell_budget[t] -= q
                    else:
                        remaining_sell -= q
                elif budget > 0:
                    counter_q = min(q, budget)
                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER, (counter_q, t, threshold)
                    )
                else:
                    responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, None)

        return responses

    def init(self):
        self._buy_fraction = 0.5
        self._sell_fraction = 0.5
        self._max_delivery_ahead = 4

    def before_step(self):
        pass

    def step(self):
        pass

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        is_seller = annotation.get("seller") == self.id
        if not is_seller:
            self._buy_fraction = min(0.9, self._buy_fraction + 0.05)
        else:
            self._sell_fraction = min(0.9, self._sell_fraction + 0.05)

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        price = contract.agreement["unit_price"]
        is_seller = contract.annotation.get("seller") == self.id

        max_buy = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        min_sell = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )

        if not is_seller:
            mn = self.awi.current_input_issues[UNIT_PRICE].min_value
            if max_buy > mn:
                fraction = (price - mn) / (max_buy - mn)
                self._buy_fraction = max(0.1, min(0.9, 0.7 * self._buy_fraction + 0.3 * fraction))
        else:
            mx = self.awi.current_output_issues[UNIT_PRICE].max_value
            if mx > min_sell:
                fraction = (mx - price) / (mx - min_sell)
                self._sell_fraction = max(0.1, min(0.9, 0.7 * self._sell_fraction + 0.3 * fraction))


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([KotaAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
