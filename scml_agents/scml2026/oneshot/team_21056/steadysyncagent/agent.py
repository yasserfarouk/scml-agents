#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* Sora Kawase <sora.17.1126@gmail.com>

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

from negmas import Outcome, SAOResponse, SAOState
from negmas.gb.common import ResponseType
from scml.common import distribute
from scml.oneshot import OneShotSyncAgent
from scml.oneshot.common import QUANTITY, UNIT_PRICE


class SteadySyncAgent(OneShotSyncAgent):
    """
    SCML OneShot Agent that combines Boulware pricing with Sync-style
    quantity distribution.

    Upgraded with 'Profit-Guardrail' and 'Quantity-First' logic for long-term dominance.
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    def distribute_needs(self, needs: int, partners: list[str]) -> dict[str, int]:
        """必要量をパートナーに分配する"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}

        # 過剰注文を避け、正確な必要量のみを分配する
        target = needs
        qs = distribute(target, len(partners))
        return dict(zip(partners, qs))

    def first_proposals(self) -> dict[str, Outcome | None]:
        responses = {}
        s = self.awi.current_step

        for needs, all_partners, issues, is_buyer in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
                True,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
                False,
            ),
        ]:
            partners = [p for p in all_partners if p in self.negotiators]
            if not partners:
                continue

            dist = self.distribute_needs(needs, partners)
            price = (
                issues[UNIT_PRICE].min_value
                if is_buyer
                else issues[UNIT_PRICE].max_value
            )

            for nid, q in dist.items():
                responses[nid] = (q, s, price) if q > 0 else None

        return responses

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses = {}
        if not states:
            return {}

        for needs, all_partners, issues, is_buyer in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
                True,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
                False,
            ),
        ]:
            if not issues:
                continue

            partners = [p for p in all_partners if p in offers]
            if not partners:
                continue

            in_min, in_max = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value
            time = max(states[p].relative_time for p in partners)

            # 確実な利益を確保するための安全マージン設定
            if self.awi.current_input_issues and self.awi.current_output_issues:
                input_max = self.awi.current_input_issues[UNIT_PRICE].max_value
                output_min = self.awi.current_output_issues[UNIT_PRICE].min_value

                if is_buyer:
                    # 売値の最低ラインを少し下回るくらいまでを限度とする
                    limit = (
                        min(in_max, output_min + (input_max - output_min) * 0.7)
                        if output_min < input_max
                        else in_max
                    )
                    target_price = self.get_boulware_price(
                        in_min, limit, time, is_buyer, exponent=2.0
                    )
                else:
                    # 買値の最高ラインを少し上回るくらいまでを限度とする
                    limit = (
                        max(in_min, input_max - (input_max - output_min) * 0.7)
                        if output_min < input_max
                        else in_min
                    )
                    target_price = self.get_boulware_price(
                        limit, in_max, time, is_buyer, exponent=2.0
                    )
            else:
                target_price = self.get_boulware_price(
                    in_min, in_max, time, is_buyer, exponent=2.0
                )

            target_price = round(target_price)

            # 価格でソート (買い手なら安い順、売り手なら高い順)
            if is_buyer:
                sorted_partners = sorted(partners, key=lambda p: offers[p][UNIT_PRICE])
            else:
                sorted_partners = sorted(
                    partners, key=lambda p: offers[p][UNIT_PRICE], reverse=True
                )

            accepted_q = 0
            for p in sorted_partners:
                offer = offers[p]
                price = offer[UNIT_PRICE]
                q = offer[QUANTITY]

                is_acceptable_price = (is_buyer and price <= target_price) or (
                    not is_buyer and price >= target_price
                )

                # 時間切れ直前は少し妥協してでも契約を取りに行く
                if time > 0.95:
                    if is_buyer and price <= in_max * 0.9:
                        is_acceptable_price = True
                    elif not is_buyer and price >= in_min * 1.1:
                        is_acceptable_price = True

                if is_acceptable_price and accepted_q + q <= needs:
                    responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    accepted_q += q
                else:
                    rem_q = needs - accepted_q
                    if rem_q > 0:
                        responses[p] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (rem_q, self.awi.current_step, target_price),
                        )
                    else:
                        responses[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)

        return responses

    def get_boulware_price(
        self, min_p, max_p, relative_time, is_buyer=True, exponent=2.5
    ):
        """Boulware戦略に基づいた価格を計算"""
        if is_buyer:
            return min_p + (max_p - min_p) * (relative_time**exponent)
        else:
            return max_p - (max_p - min_p) * (relative_time**exponent)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([SteadySyncAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
