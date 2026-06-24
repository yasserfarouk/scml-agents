#!/usr/bin/env python
"""
**Submitted to ANAC 2026 SCML (OneShot track)**
*Authors* Rinon Asanuma <asanuma@katfuji.lab.tuat.ac.jp>

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2026 SCML.
"""
from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.oneshot import QUANTITY, UNIT_PRICE, OneShotAWI, OneShotSyncAgent

# required for typing
from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState


class AssariAsari(OneShotSyncAgent):
    """SCML One-shot agent: あっさりあさり (AssariAsari)."""

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
        - You can access your ufun using `self.ufun` (See `OneShotUFun` in the docs for more details).
    """

    def __init__(
        self,
        *args,
        concession_exponent: float = 0.50,
        late_quantity_slack: float = 0.55,
        price_weight: float = 0.135,
        reliability_weight: float = 0.22,
        overfill_penalty: float = 1.30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concession_exponent = concession_exponent
        self.late_quantity_slack = late_quantity_slack
        self.price_weight = price_weight
        self.reliability_weight = reliability_weight
        self.overfill_penalty = overfill_penalty
        self._partner_successes: dict[str, int] = {}
        self._partner_failures: dict[str, int] = {}
        self._attempted_partners: set[str] = set()

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        return {
            partner: self._make_offer(partner, quantity)
            for partner, quantity in self._distribute_needs().items()
        }

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        responses = {
            partner: self._reject_or_end(partner, quantity)
            for partner, quantity in self._distribute_needs().items()
            if partner in self.active_negotiators
        }

        for needs, partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            active = [p for p in partners if p in offers]
            if not active:
                continue

            if needs <= 0:
                responses.update(
                    {p: SAOResponse(ResponseType.END_NEGOTIATION, None) for p in active}
                )
                continue

            chosen, diff = self._best_offer_subset(active, offers, needs)
            slack = self._quantity_slack(states, active)
            if chosen and diff <= slack:
                accepted = set(chosen)
                responses.update(
                    {
                        p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                        for p in accepted
                    }
                )
                remaining = max(0, needs - sum(offers[p][QUANTITY] for p in accepted))
                others = [p for p in active if p not in accepted]
                for partner, quantity in self._split_quantity(remaining, others).items():
                    responses[partner] = self._reject_or_end(partner, quantity)

        return responses

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        self._attempted_partners.clear()

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""
        my_id = getattr(self, "id", None)
        for partner in partners:
            if partner != my_id and partner in self._attempted_partners:
                self._partner_failures[partner] = (
                    self._partner_failures.get(partner, 0) + 1
                )

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
        my_id = getattr(self, "id", None)
        for partner in getattr(contract, "partners", []):
            if partner != my_id:
                self._partner_successes[partner] = (
                    self._partner_successes.get(partner, 0) + 1
                )

    # =================
    # Strategy Helpers
    # =================

    def _distribute_needs(self) -> dict[str, int]:
        """Split each side's current need across still-active partners."""
        distribution: dict[str, int] = {}
        for needs, partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            active = [p for p in partners if p in self.active_negotiators]
            distribution.update(self._split_quantity(max(0, needs), active))
        return distribution

    def _split_quantity(self, quantity: int, partners: list[str]) -> dict[str, int]:
        if not partners:
            return {}
        if quantity <= 0:
            return {p: 0 for p in partners}

        weights = {p: self._partner_weight(p, partners) for p in partners}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            ranked = sorted(partners, key=self._partner_rank)
            base, extra = divmod(quantity, len(ranked))
            return {p: base + (1 if i < extra else 0) for i, p in enumerate(ranked)}

        raw = {p: quantity * weights[p] / total_weight for p in partners}
        distribution = {p: int(raw[p]) for p in partners}
        remaining = quantity - sum(distribution.values())
        ranked = sorted(
            partners,
            key=lambda p: (-(raw[p] - distribution[p]), self._partner_rank(p)),
        )
        for partner in ranked[:remaining]:
            distribution[partner] += 1
        return distribution

    def _partner_rank(self, partner: str) -> tuple[int, str]:
        nmi = self.get_nmi(partner)
        if not nmi:
            return (0, partner)
        quantity_issue = nmi.issues[QUANTITY]
        return (-int(quantity_issue.max_value), partner)

    def _make_offer(self, partner: str, quantity: int) -> Outcome | None:
        if quantity <= 0:
            return None

        nmi = self.get_nmi(partner)
        if not nmi:
            return None

        qissue = nmi.issues[QUANTITY]
        q = min(max(quantity, qissue.min_value), qissue.max_value)
        self._attempted_partners.add(partner)
        return (q, self.awi.current_step, self._target_price(partner))

    def _reject_or_end(self, partner: str, quantity: int) -> SAOResponse:
        offer = self._make_offer(partner, quantity)
        if offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, offer)

    def _best_offer_subset(
        self, partners: list[str], offers: dict[str, Outcome], needs: int
    ) -> tuple[tuple[str, ...], int]:
        best: tuple[tuple[str, ...], float, float] = (
            tuple(),
            self._quantity_mismatch_cost(0, needs),
            float("-inf"),
        )

        # Partner counts are usually small. For larger cases, keep the search bounded.
        if len(partners) <= 10:
            masks = range(1, 1 << len(partners))
            subsets = (
                tuple(p for i, p in enumerate(partners) if mask & (1 << i))
                for mask in masks
            )
        else:
            ordered = sorted(
                partners,
                key=lambda p: (
                    abs(offers[p][QUANTITY] - needs),
                    -self._price_score(p, offers[p]),
                ),
            )
            subsets = (tuple(ordered[:i]) for i in range(1, len(ordered) + 1))

        for subset in subsets:
            total = sum(offers[p][QUANTITY] for p in subset)
            mismatch_cost = self._quantity_mismatch_cost(total, needs)
            price_score = sum(self._price_score(p, offers[p]) for p in subset)
            reliability_score = sum(self._partner_reliability(p) for p in subset)
            average_price_score = price_score / len(subset)
            score = (
                average_price_score
                + self.reliability_weight * reliability_score
                - self.price_weight * mismatch_cost
            )
            if mismatch_cost < best[1] or (
                mismatch_cost == best[1] and score > best[2]
            ):
                best = (subset, mismatch_cost, score)
        return best[0], best[1]

    def _price_score(self, partner: str, offer: Outcome) -> float:
        nmi = self.get_nmi(partner)
        if not nmi:
            return 0.0

        pissue = nmi.issues[UNIT_PRICE]
        price_range = max(1, pissue.max_value - pissue.min_value)
        normalized = (offer[UNIT_PRICE] - pissue.min_value) / price_range
        return normalized if self._is_selling(partner) else 1.0 - normalized

    def _target_price(self, partner: str) -> int:
        nmi = self.get_nmi(partner)
        if not nmi:
            return 0

        pissue = nmi.issues[UNIT_PRICE]
        mn, mx = pissue.min_value, pissue.max_value
        concession = self._concession(getattr(nmi.state, "relative_time", 0.0))
        if self._is_selling(partner):
            return int(mx - concession * (mx - mn) + 0.5)
        return int(mn + concession * (mx - mn) + 0.5)

    def _quantity_slack(self, states: dict[str, SAOState], partners: list[str]) -> float:
        known_partners = [p for p in partners if p in states]
        if not known_partners:
            return 0.0
        relative_time = min(states[p].relative_time for p in known_partners)
        return self.late_quantity_slack * self.awi.n_lines * (relative_time**3.5)

    def _quantity_mismatch_cost(self, total: int, needs: int) -> float:
        shortage = max(0, needs - total)
        overfill = max(0, total - needs)
        return shortage + self.overfill_penalty * overfill

    def _concession(self, relative_time: float) -> float:
        return relative_time**self.concession_exponent

    def _is_selling(self, partner: str) -> bool:
        return partner in self.awi.my_consumers

    def _partner_reliability(self, partner: str) -> float:
        successes = self._partner_successes.get(partner, 0)
        failures = self._partner_failures.get(partner, 0)
        return (successes + 1.0) / (successes + failures + 2.0)

    def _partner_weight(self, partner: str, partners: list[str]) -> float:
        nmi = self.get_nmi(partner)
        if not nmi:
            return 1.0

        capacities = [
            max(1, int(self.get_nmi(p).issues[QUANTITY].max_value))
            for p in partners
            if self.get_nmi(p)
        ]
        average_capacity = sum(capacities) / len(capacities) if capacities else 1.0
        capacity = max(1, int(nmi.issues[QUANTITY].max_value))
        return 0.80 + 0.20 * capacity / average_capacity
