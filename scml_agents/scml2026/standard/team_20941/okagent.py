#!/usr/bin/env python
from __future__ import annotations

from typing import Any

from scml.std import StdAWI, StdSyncAgent
from negmas import (
    Contract,
    Outcome,
    SAOResponse,
    SAOState,
    ResponseType,
)


class OkAgent(StdSyncAgent):

    # =====================
    # Utility Functions
    # =====================

    def needed_quantity(self):

        production = self.awi.n_lines

        try:
            inventory = self.awi.current_inventory_input
        except Exception:
            inventory = 0

        return max(0, production - inventory)

    def estimated_trading_price(self, partner):

        nmi = self.negotiators[partner][0].nmi

        min_price = nmi.issues[2].min_value
        max_price = nmi.issues[2].max_value

        return (min_price + max_price) / 2

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self):
        return {
            partner: None
            for partner in self.negotiators.keys()
        }

    def counter_all(
        self,
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
    ) -> dict[str, SAOResponse]:

        responses = {}

        need = self.needed_quantity()

        for partner, offer in offers.items():

            quantity, delivery_time, price = offer

            tp = self.estimated_trading_price(partner)

            is_supplier = partner in self.awi.my_suppliers

            # =====================
            # BUY STRATEGY
            # =====================

            if is_supplier:

                # とても安い
                if price <= tp * 0.80:

                    responses[partner] = SAOResponse(
                        ResponseType.ACCEPT_OFFER,
                        None,
                    )

                # 安い
                elif price <= tp * 0.90 and quantity <= need * 2:

                    responses[partner] = SAOResponse(
                        ResponseType.ACCEPT_OFFER,
                        None,
                    )

                # 通常価格
                elif price <= tp and quantity <= need:

                    responses[partner] = SAOResponse(
                        ResponseType.ACCEPT_OFFER,
                        None,
                    )

                # 高すぎる
                else:

                    counter_offer = (
                        max(1, need),
                        self.awi.current_step,
                        int(tp * 0.90),
                    )

                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        counter_offer,
                    )

            # =====================
            # SELL STRATEGY
            # =====================

            else:

                if price >= tp:

                    responses[partner] = SAOResponse(
                        ResponseType.ACCEPT_OFFER,
                        None,
                    )

                else:

                    counter_offer = (
                        quantity,
                        self.awi.current_step,
                        int(tp * 1.10),
                    )

                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        counter_offer,
                    )

        return responses

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        self.max_inventory_days = 3

    def before_step(self):
        pass

    def step(self):
        pass

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        pass

    def on_negotiation_success(
        self,
        contract: Contract,
        mechanism: StdAWI,
    ) -> None:
        pass


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run(
        [OkAgent],
        sys.argv[1] if len(sys.argv) > 1 else "std",
    )

