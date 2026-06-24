from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, isfinite
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.std import StdSyncAgent
from scml.std.common import QUANTITY, TIME, UNIT_PRICE


@dataclass
class PartnerStats:
    successes: int = 0
    failures: int = 0
    offers_seen: int = 0
    last_price: float | None = None

    @property
    def reliability(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


class Agent01StdAgent(StdSyncAgent):
    """Portfolio-style synchronous agent for the SCML Standard league."""

    max_exact_offers = 14
    beam_width = 96
    accept_slack = 0.006
    future_weight = 0.0
    future_productivity = 0.72
    concession_exp = 0.55
    early_overorder = 0.11
    late_overorder = 0.015
    risk_gate_start_fraction = 0.60
    risk_gate_imbalance_lines = 0.35
    risk_gate_target_factor = 0.75
    risk_gate_overorder_factor = 0.70

    def init(self):
        self.partner_stats: dict[str, PartnerStats] = {}

    def before_step(self):
        try:
            self.ufun.find_limit(True, ignore_signed_contracts=False)
            self.ufun.find_limit(False, ignore_signed_contracts=False)
        except Exception:
            pass

    def first_proposals(self) -> dict[str, Outcome | None]:
        return self._make_proposals(self._partners(), t=0.0)

    def counter_all(
        self, offers: dict[str, Outcome | None], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        valid_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and offer[QUANTITY] > 0
            and offer[TIME] >= self.awi.current_step
        }
        for partner, offer in valid_offers.items():
            stat = self._stats(partner)
            stat.offers_seen += 1
            stat.last_price = float(offer[UNIT_PRICE])

        t = min((s.relative_time for s in states.values()), default=1.0)
        accepted = self._accepted_portfolio(valid_offers, t)
        responses: dict[str, SAOResponse] = {
            partner: SAOResponse(ResponseType.ACCEPT_OFFER, valid_offers[partner])
            for partner in accepted
            if partner in valid_offers
        }

        accepted_offers = {p: valid_offers[p] for p in accepted if p in valid_offers}
        remaining = [p for p in offers if p not in responses]
        counters = self._make_proposals(
            remaining, t=t, already_accepted=accepted_offers
        )
        for partner in remaining:
            counter = counters.get(partner)
            if counter is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return responses

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism,
        state: SAOState,
    ) -> None:
        for partner in partners:
            if partner != self.id:
                self._stats(partner).failures += 1

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:  # type: ignore
        unit_price = contract.agreement.get("unit_price", None)
        for partner in contract.partners:
            if partner == self.id:
                continue
            stat = self._stats(partner)
            stat.successes += 1
            if unit_price is not None:
                stat.last_price = float(unit_price)

    # Portfolio acceptance

    def _accepted_portfolio(self, offers: dict[str, Outcome], t: float) -> set[str]:
        if not offers:
            return set()
        base_score = self._portfolio_score({})
        best_partners, best_score = self._best_subset(offers)
        if not best_partners:
            return set()
        margin = self.accept_slack * (1.0 - min(1.0, max(0.0, t)) ** 0.7)
        if best_score <= base_score + margin:
            return set()

        if t < 0.6 and len(best_partners) > 1:
            robust: set[str] = set()
            for partner in best_partners:
                without = {
                    p: offer
                    for p, offer in offers.items()
                    if p in best_partners and p != partner
                }
                if best_score - self._portfolio_score(without) > margin:
                    robust.add(partner)
            if robust:
                best_partners = robust
        return set(best_partners)

    def _best_subset(self, offers: dict[str, Outcome]) -> tuple[set[str], float]:
        items = list(offers.items())
        if len(items) <= self.max_exact_offers:
            best_partners: set[str] = set()
            best_score = self._portfolio_score({})
            for r in range(1, len(items) + 1):
                for combo in combinations(items, r):
                    subset = {p: o for p, o in combo}
                    score = self._portfolio_score(subset)
                    if score > best_score:
                        best_score = score
                        best_partners = set(subset)
            return best_partners, best_score

        beam: list[tuple[float, dict[str, Outcome]]] = [(self._portfolio_score({}), {})]
        for partner, offer in items:
            expanded = beam + [
                (
                    self._portfolio_score({**subset, partner: offer}),
                    {**subset, partner: offer},
                )
                for _, subset in beam
            ]
            expanded.sort(key=lambda x: x[0], reverse=True)
            beam = expanded[: self.beam_width]
        score, subset = beam[0]
        return set(subset), score

    def _portfolio_score(self, offers: dict[str, Outcome]) -> float:
        current = {
            partner: offer
            for partner, offer in offers.items()
            if offer[TIME] == self.awi.current_step
        }
        future = {
            partner: offer
            for partner, offer in offers.items()
            if offer[TIME] > self.awi.current_step
        }
        utility = self._current_utility(current)
        if not isfinite(utility):
            return utility
        return utility + self._future_score(future)

    def _current_utility(self, offers: dict[str, Outcome]) -> float:
        try:
            value = self.ufun.from_offers(offers, ignore_signed_contracts=False)
        except TypeError:
            value = self.ufun.from_offers(offers)
        if value is None:
            return float("-inf")
        return float(value)

    def _future_score(self, offers: dict[str, Outcome]) -> float:
        if not offers or self.future_weight <= 0.0:
            return 0.0
        by_time: dict[int, dict[str, float]] = {}
        for partner, offer in offers.items():
            delivery = int(offer[TIME])
            bucket = by_time.setdefault(
                delivery, {"sq": 0.0, "sc": 0.0, "dq": 0.0, "dr": 0.0}
            )
            quantity = float(offer[QUANTITY])
            value = quantity * float(offer[UNIT_PRICE])
            if self._is_selling_partner(partner):
                bucket["dq"] += quantity
                bucket["dr"] += value
            else:
                bucket["sq"] += quantity
                bucket["sc"] += value

        score = 0.0
        scale = max(1.0, self._price_scale() * max(1.0, float(self.awi.n_lines)))
        for delivery, extra in by_time.items():
            baseline = self._future_day_value(delivery, 0.0, 0.0, 0.0, 0.0)
            with_extra = self._future_day_value(
                delivery, extra["sq"], extra["sc"], extra["dq"], extra["dr"]
            )
            distance = max(1, delivery - self.awi.current_step)
            discount = 1.0 / (1.0 + 0.18 * (distance - 1))
            score += discount * (with_extra - baseline) / scale
        return self.future_weight * score

    def _future_day_value(
        self,
        delivery: int,
        extra_supply_q: float,
        extra_supply_cost: float,
        extra_sale_q: float,
        extra_sale_revenue: float,
    ) -> float:
        supply_q = float(self._total_supplies_at(delivery)) + extra_supply_q
        sale_q = float(self._total_sales_at(delivery)) + extra_sale_q
        supply_cost = self._total_supply_cost_at(delivery) + extra_supply_cost
        sale_revenue = self._total_sale_revenue_at(delivery) + extra_sale_revenue

        input_ref = self._reference_price(selling=False)
        output_ref = self._reference_price(selling=True)
        if getattr(self.awi, "is_first_level", False):
            virtual_q = float(self.awi.n_lines)
            supply_q += virtual_q
            supply_cost += virtual_q * input_ref
        if getattr(self.awi, "is_last_level", False):
            virtual_q = float(self.awi.n_lines)
            sale_q += virtual_q
            sale_revenue += virtual_q * output_ref

        capacity = float(max(1, self.awi.n_lines))
        producible = min(supply_q, sale_q, capacity)
        avg_supply = supply_cost / supply_q if supply_q > 0 else input_ref
        avg_sale = sale_revenue / sale_q if sale_q > 0 else output_ref
        production_cost = float(getattr(self.awi.profile, "cost", 0.0) or 0.0)

        unmatched_supply = max(0.0, supply_q - producible)
        unmatched_sales = max(0.0, sale_q - producible)
        disposal = self._penalty(shortfall=False)
        shortfall = self._penalty(shortfall=True)
        return (
            producible * (avg_sale - avg_supply - production_cost)
            - unmatched_supply * disposal
            - unmatched_sales * shortfall
        )

    # Counter-offer policy

    def _make_proposals(
        self,
        partners: list[str],
        t: float,
        already_accepted: dict[str, Outcome] | None = None,
    ) -> dict[str, Outcome | None]:
        partners = [p for p in partners if self.get_nmi(p) is not None]
        already_accepted = already_accepted or {}
        proposals: dict[str, Outcome | None] = {p: None for p in partners}
        for selling in (False, True):
            side_partners = [p for p in partners if self._is_selling_partner(p) == selling]
            current = self._current_side_proposals(
                side_partners,
                selling=selling,
                t=t,
                already_covered=self._accepted_quantity(
                    already_accepted, selling=selling, current_only=True
                ),
            )
            proposals.update(current)
            unused = [p for p in side_partners if current.get(p) is None]
            proposals.update(
                self._future_side_proposals(
                    unused,
                    selling=selling,
                    t=t,
                    already_accepted=already_accepted,
                )
            )
        return proposals

    def _current_side_proposals(
        self,
        partners: list[str],
        selling: bool,
        t: float,
        already_covered: int = 0,
    ) -> dict[str, Outcome | None]:
        if not partners:
            return {}
        need = self._side_need(selling) - already_covered
        if need <= 0:
            return {p: None for p in partners}
        quantity = max(1, int(ceil(need * (1.0 + self._overorder_fraction(t)))))
        quantities = self._allocate_quantities(quantity, partners)
        proposals: dict[str, Outcome | None] = {}
        for partner in partners:
            q = self._clip_quantity(partner, quantities.get(partner, 0))
            if q <= 0:
                proposals[partner] = None
                continue
            price = self._counter_price(partner, selling, q, self.awi.current_step, t)
            proposals[partner] = (q, self.awi.current_step, price)
        return proposals

    def _future_side_proposals(
        self,
        partners: list[str],
        selling: bool,
        t: float,
        already_accepted: dict[str, Outcome],
    ) -> dict[str, Outcome | None]:
        # Local stress tests showed that active future contracting creates
        # unstable over-commitment before a stronger cross-time model is added.
        return {p: None for p in partners}

    def _future_needs(
        self, selling: bool, already_accepted: dict[str, Outcome]
    ) -> dict[int, int]:
        current = self.awi.current_step
        end = min(self.awi.n_steps - 1, current + max(1, int(getattr(self.awi, "horizon", 1) or 1)))
        target = max(1, int(round(self.future_productivity * self.awi.n_lines)))
        needs: dict[int, int] = {}
        for delivery in range(current + 1, end + 1):
            if selling and getattr(self.awi, "is_last_level", False):
                continue
            if not selling and getattr(self.awi, "is_first_level", False):
                continue
            signed_supply = self._total_supplies_at(delivery)
            signed_sale = self._total_sales_at(delivery)
            accepted_supply = self._accepted_quantity_at(
                already_accepted, selling=False, delivery=delivery
            )
            accepted_sale = self._accepted_quantity_at(
                already_accepted, selling=True, delivery=delivery
            )
            if selling:
                desired = max(target, signed_supply + accepted_supply)
                needs[delivery] = max(0, min(self.awi.n_lines, desired) - signed_sale - accepted_sale)
            else:
                desired = max(target, signed_sale + accepted_sale)
                needs[delivery] = max(0, min(self.awi.n_lines, desired) - signed_supply - accepted_supply)
        return {k: v for k, v in needs.items() if v > 0}

    def _counter_price(
        self, partner: str, selling: bool, quantity: int, delivery: int, t: float
    ) -> int:
        nmi = self.get_nmi(partner)
        assert nmi is not None
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = int(p_issue.min_value), int(p_issue.max_value)
        if pmin >= pmax:
            return pmin

        concession = min(1.0, max(0.0, t) ** self.concession_exp)
        if delivery == self.awi.current_step:
            shadow = self._shadow_price(partner, selling, quantity, pmin, pmax)
            if selling:
                curve = pmax - concession * (pmax - pmin)
                price = max(shadow, curve)
            else:
                curve = pmin + concession * (pmax - pmin)
                price = min(shadow, curve)
        else:
            input_ref = self._reference_price(selling=False)
            output_ref = self._reference_price(selling=True)
            production_cost = float(getattr(self.awi.profile, "cost", 0.0) or 0.0)
            if selling:
                floor = max(pmin, int(round(input_ref + production_cost)))
                price = pmax - 0.65 * concession * (pmax - floor)
            else:
                ceiling = min(pmax, int(round(output_ref - production_cost)))
                price = pmin + 0.65 * concession * (ceiling - pmin)
        return int(min(max(round(price), pmin), pmax))

    def _shadow_price(
        self, partner: str, selling: bool, quantity: int, pmin: int, pmax: int
    ) -> float:
        base = self._current_utility({})
        if selling:
            acceptable = pmin
            for price in range(pmin, pmax + 1):
                if self._current_utility(
                    {partner: (quantity, self.awi.current_step, price)}
                ) >= base:
                    acceptable = price
                    break
            return float(acceptable)
        acceptable = pmax
        for price in range(pmax, pmin - 1, -1):
            if self._current_utility(
                {partner: (quantity, self.awi.current_step, price)}
            ) >= base:
                acceptable = price
                break
        return float(acceptable)

    # State helpers

    def _partners(self) -> list[str]:
        active = getattr(self, "active_negotiators", None)
        if active:
            return list(active.keys())
        return list(self.negotiators.keys())

    def _progress(self) -> float:
        n_steps = max(2, int(getattr(self.awi, "n_steps", 2) or 2))
        current = min(max(0, int(getattr(self.awi, "current_step", 0) or 0)), n_steps - 1)
        return current / max(1, n_steps - 1)

    def _current_needs(self) -> tuple[int, int]:
        return int(self.awi.needed_supplies), int(self.awi.needed_sales)

    def _need_imbalance(self) -> float:
        supplies, sales = self._current_needs()
        return abs(supplies - sales) / max(1.0, float(self.awi.n_lines))

    def _risk_gate_active(self) -> bool:
        if not getattr(self.awi, "is_middle_level", False):
            return False
        if self._progress() >= self.risk_gate_start_fraction:
            return True
        return self._need_imbalance() >= self.risk_gate_imbalance_lines

    def _side_need(self, selling: bool) -> int:
        need = int(self.awi.needed_sales if selling else self.awi.needed_supplies)
        if getattr(self.awi, "is_middle_level", False):
            target = int(round(self.future_productivity * self.awi.n_lines))
            if self._risk_gate_active():
                supply_need, sale_need = self._current_needs()
                opposite_need = supply_need if selling else sale_need
                if need <= 0 and opposite_need > 0:
                    return 0
                target = int(round(target * self.risk_gate_target_factor))
            need = max(need, target)
        return need

    def _is_selling_partner(self, partner: str) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is not None:
            return nmi.annotation["product"] == self.awi.my_output_product
        return partner in getattr(self.awi, "my_consumers", [])

    def _allocate_quantities(
        self, total_quantity: int, partners: list[str]
    ) -> dict[str, int]:
        ranked = sorted(
            partners,
            key=lambda p: (self._stats(p).reliability, -self._stats(p).offers_seen),
            reverse=True,
        )
        weights = [max(0.05, self._stats(p).reliability) for p in ranked]
        wsum = sum(weights) or 1.0
        allocation = {p: 0 for p in partners}
        remaining = int(max(0, total_quantity))
        for i, partner in enumerate(ranked):
            if remaining <= 0:
                break
            q = remaining if i == len(ranked) - 1 else max(
                1, int(round(total_quantity * weights[i] / wsum))
            )
            allocation[partner] = min(q, remaining)
            remaining -= allocation[partner]
        return allocation

    def _clip_quantity(self, partner: str, quantity: int) -> int:
        nmi = self.get_nmi(partner)
        if nmi is None or quantity <= 0:
            return 0
        q_issue = nmi.issues[QUANTITY]
        return int(min(max(quantity, q_issue.min_value), q_issue.max_value))

    def _time_allowed(self, partner: str, delivery: int) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        issue = nmi.issues[TIME]
        return int(issue.min_value) <= delivery <= int(issue.max_value)

    def _overorder_fraction(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        value = self.early_overorder - (
            self.early_overorder - self.late_overorder
        ) * (t**0.7)
        if self._risk_gate_active():
            value *= self.risk_gate_overorder_factor
        return value

    def _accepted_quantity(
        self,
        offers: dict[str, Outcome],
        selling: bool,
        current_only: bool = False,
    ) -> int:
        return sum(
            int(offer[QUANTITY])
            for partner, offer in offers.items()
            if self._is_selling_partner(partner) == selling
            and (not current_only or offer[TIME] == self.awi.current_step)
        )

    def _accepted_quantity_at(
        self, offers: dict[str, Outcome], selling: bool, delivery: int
    ) -> int:
        return sum(
            int(offer[QUANTITY])
            for partner, offer in offers.items()
            if self._is_selling_partner(partner) == selling
            and int(offer[TIME]) == delivery
        )

    def _total_sales_at(self, delivery: int) -> int:
        method = getattr(self.awi, "total_sales_at", None)
        return int(method(delivery) if method else 0)

    def _total_supplies_at(self, delivery: int) -> int:
        method = getattr(self.awi, "total_supplies_at", None)
        return int(method(delivery) if method else 0)

    def _total_sale_revenue_at(self, delivery: int) -> float:
        costs = getattr(self.awi, "future_sales_cost", {}) or {}
        return float(sum((costs.get(delivery, {}) or {}).values()))

    def _total_supply_cost_at(self, delivery: int) -> float:
        costs = getattr(self.awi, "future_supplies_cost", {}) or {}
        return float(sum((costs.get(delivery, {}) or {}).values()))

    def _reference_price(self, selling: bool) -> float:
        product = self.awi.my_output_product if selling else self.awi.my_input_product
        for attr in ("trading_prices", "catalog_prices"):
            prices = getattr(self.awi, attr, None)
            if prices is None:
                continue
            try:
                price = float(prices[product])
            except Exception:
                continue
            if price > 0:
                return price
        return self._price_scale()

    def _price_scale(self) -> float:
        prices = getattr(self.awi, "catalog_prices", None)
        if prices is not None:
            try:
                return max(1.0, float(max(prices)))
            except Exception:
                pass
        return 10.0

    def _penalty(self, shortfall: bool) -> float:
        value = getattr(
            self.awi,
            "current_shortfall_penalty" if shortfall else "current_disposal_cost",
            0.0,
        )
        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def _stats(self, partner: str) -> PartnerStats:
        if not hasattr(self, "partner_stats"):
            self.partner_stats = {}
        if partner not in self.partner_stats:
            self.partner_stats[partner] = PartnerStats()
        return self.partner_stats[partner]


__all__ = ["Agent01StdAgent"]
