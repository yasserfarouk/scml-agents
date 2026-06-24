#!/usr/bin/env python
"""
CoReOneShotAgent for ANAC 2026 SCML OneShot.

The agent treats simultaneous one-shot negotiations as a portfolio decision.
It accepts offer sets using global utility, then prices counter-offers using a
counterfactual regret / shadow-price rule instead of a fixed concession curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, isfinite
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.oneshot import OneShotAWI, OneShotSyncAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE


@dataclass
class PartnerStats:
    successes: int = 0
    failures: int = 0
    offers_seen: int = 0
    last_price: float | None = None

    @property
    def reliability(self) -> float:
        # Smoothed empirical success rate. New partners start neutral.
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


class CoReOneShotAgent(OneShotSyncAgent):
    """
    Counterfactual-regret one-shot supply-chain negotiator.

    Main design:
    - Portfolio acceptance: evaluate subsets of concurrent offers by the true
      SCML utility function, including already signed contracts.
    - Regret gate: accept only if the best current portfolio clears a dynamic
      aspiration level, which falls as negotiation time runs out.
    - Shadow-price counters: generate prices around marginal utility rather
      than relying only on a hard-coded time concession schedule.
    """

    max_exact_offers = 16
    beam_width = 128
    use_regret_guard = True
    use_shadow_price = True
    use_partner_reliability = True

    def __init__(
        self,
        *args,
        concession_exp: float = 0.55,
        early_overorder: float = 0.18,
        late_overorder: float = 0.03,
        accept_slack: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concession_exp = concession_exp
        self.early_overorder = early_overorder
        self.late_overorder = late_overorder
        self.accept_slack = accept_slack
        self.partner_stats: dict[str, PartnerStats] = {}

    # =====================
    # Negotiation callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        partners = list(self.active_negotiators.keys())
        supply_partners = [p for p in partners if not self._is_selling_partner(p)]
        sale_partners = [p for p in partners if self._is_selling_partner(p)]
        proposals: dict[str, Outcome | None] = {}
        proposals.update(self._proposals_for_side(supply_partners, selling=False, t=0.0))
        proposals.update(self._proposals_for_side(sale_partners, selling=True, t=0.0))
        return proposals

    def counter_all(
        self, offers: dict[str, Outcome | None], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        valid_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }
        for partner, offer in valid_offers.items():
            stat = self._stats(partner)
            stat.offers_seen += 1
            stat.last_price = float(offer[UNIT_PRICE])

        t = min((s.relative_time for s in states.values()), default=1.0)
        accepted = self._accepted_portfolio(valid_offers, t)
        accepted_supply = sum(
            valid_offers[p][QUANTITY]
            for p in accepted
            if p in valid_offers and not self._is_selling_partner(p)
        )
        accepted_sales = sum(
            valid_offers[p][QUANTITY]
            for p in accepted
            if p in valid_offers and self._is_selling_partner(p)
        )

        responses: dict[str, SAOResponse] = {
            partner: SAOResponse(ResponseType.ACCEPT_OFFER, valid_offers[partner])
            for partner in accepted
            if partner in valid_offers
        }

        remaining_partners = [p for p in offers if p not in responses]
        supply_partners = [
            p for p in remaining_partners if not self._is_selling_partner(p)
        ]
        sale_partners = [p for p in remaining_partners if self._is_selling_partner(p)]
        counters: dict[str, Outcome | None] = {}
        counters.update(
            self._proposals_for_side(
                supply_partners,
                selling=False,
                t=t,
                already_covered=accepted_supply,
            )
        )
        counters.update(
            self._proposals_for_side(
                sale_partners,
                selling=True,
                t=t,
                already_covered=accepted_sales,
            )
        )

        for partner in remaining_partners:
            counter = counters.get(partner)
            if counter is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return responses

    # =====================
    # Time-driven callbacks
    # =====================

    def init(self):
        self.partner_stats = {}

    def before_step(self):
        # Limits are used only for aspiration scaling. If limit search fails in
        # unusual domains, the agent falls back to local portfolio utilities.
        try:
            self.ufun.find_limit(True)
            self.ufun.find_limit(False)
        except Exception:
            pass

    def step(self):
        pass

    # ================================
    # Negotiation feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        for partner in partners:
            if partner != self.id:
                self._stats(partner).failures += 1

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:  # type: ignore
        partners = [p for p in contract.partners if p != self.id]
        unit_price = contract.agreement.get("unit_price", None)
        for partner in partners:
            stat = self._stats(partner)
            stat.successes += 1
            if unit_price is not None:
                stat.last_price = float(unit_price)

    # =====================
    # Portfolio acceptance
    # =====================

    def _accepted_portfolio(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if not offers:
            return set()
        base_utility = self._utility({})
        best_partners, best_utility = self._best_subset(offers)
        if not best_partners:
            return set()

        aspiration = self._aspiration(base_utility, best_utility, t)
        if best_utility + self.accept_slack < aspiration:
            return set()

        # Do not lock in a partner whose removal barely changes utility early in
        # the negotiation. This is the counterfactual regret guard.
        if self.use_regret_guard and t < 0.65 and len(best_partners) > 1:
            robust: set[str] = set()
            for partner in best_partners:
                without = {
                    p: offer
                    for p, offer in offers.items()
                    if p in best_partners and p != partner
                }
                regret = best_utility - self._utility(without)
                if regret > self.accept_slack * max(1.0, abs(best_utility)):
                    robust.add(partner)
            if robust:
                best_partners = robust

        return set(best_partners)

    def _best_subset(self, offers: dict[str, Outcome]) -> tuple[set[str], float]:
        items = list(offers.items())
        if len(items) <= self.max_exact_offers:
            best_partners: set[str] = set()
            best_utility = self._utility({})
            for r in range(1, len(items) + 1):
                for combo in combinations(items, r):
                    subset = {p: o for p, o in combo}
                    value = self._utility(subset)
                    if value > best_utility:
                        best_utility = value
                        best_partners = set(subset)
            return best_partners, best_utility

        # Beam fallback for unusually large partner counts.
        beam: list[tuple[float, dict[str, Outcome]]] = [(self._utility({}), {})]
        for partner, offer in items:
            expanded = beam + [
                (self._utility({**subset, partner: offer}), {**subset, partner: offer})
                for _, subset in beam
            ]
            expanded.sort(key=lambda x: x[0], reverse=True)
            beam = expanded[: self.beam_width]
        value, subset = beam[0]
        return set(subset), value

    def _aspiration(self, base_utility: float, best_utility: float, t: float) -> float:
        max_utility = getattr(self.ufun, "max_utility", None)
        if max_utility is None or not isfinite(float(max_utility)):
            max_utility = max(best_utility, base_utility)
        max_utility = float(max_utility)
        if max_utility <= base_utility:
            return base_utility + self.accept_slack
        concession = min(1.0, max(0.0, t) ** self.concession_exp)
        return max_utility - (max_utility - base_utility) * concession

    def _utility(self, offers: dict[str, Outcome]) -> float:
        try:
            value = self.ufun.from_offers(offers, ignore_signed_contracts=False)
        except TypeError:
            value = self.ufun.from_offers(offers)
        if value is None:
            return float("-inf")
        return float(value)

    # =====================
    # Counter-offer policy
    # =====================

    def _proposals_for_side(
        self,
        partners: list[str],
        selling: bool,
        t: float,
        already_covered: int = 0,
    ) -> dict[str, Outcome | None]:
        partners = [p for p in partners if self.get_nmi(p) is not None]
        if not partners:
            return {}

        need = self._side_need(selling) - already_covered
        if need <= 0:
            return {p: None for p in partners}

        over = self._overorder_fraction(t)
        target_quantity = max(1, int(ceil(need * (1.0 + over))))
        quantities = self._allocate_quantities(target_quantity, partners)
        proposals: dict[str, Outcome | None] = {}
        for partner in partners:
            quantity = quantities.get(partner, 0)
            if quantity <= 0:
                proposals[partner] = None
                continue
            nmi = self.get_nmi(partner)
            if nmi is None:
                proposals[partner] = None
                continue
            q_issue = nmi.issues[QUANTITY]
            quantity = min(max(quantity, q_issue.min_value), q_issue.max_value)
            price = self._counter_price(partner, selling, quantity, t)
            proposals[partner] = (quantity, self.awi.current_step, price)
        return proposals

    def _counter_price(
        self, partner: str, selling: bool, quantity: int, t: float
    ) -> int:
        nmi = self.get_nmi(partner)
        assert nmi is not None
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = int(p_issue.min_value), int(p_issue.max_value)
        if pmin >= pmax:
            return pmin

        concession = min(1.0, max(0.0, t) ** self.concession_exp)
        risk = self._risk_pressure(selling)
        concession = min(1.0, concession * (0.75 + 0.75 * risk))

        if selling:
            curve_price = pmax - concession * (pmax - pmin)
            shadow = (
                self._shadow_price(partner, selling, quantity, pmin, pmax)
                if self.use_shadow_price
                else pmin
            )
            price = max(shadow, curve_price)
        else:
            curve_price = pmin + concession * (pmax - pmin)
            shadow = (
                self._shadow_price(partner, selling, quantity, pmin, pmax)
                if self.use_shadow_price
                else pmax
            )
            price = min(shadow, curve_price)
        return int(min(max(round(price), pmin), pmax))

    def _shadow_price(
        self, partner: str, selling: bool, quantity: int, pmin: int, pmax: int
    ) -> float:
        base = self._utility({})
        if selling:
            acceptable = pmin
            for price in range(pmin, pmax + 1):
                value = self._utility(
                    {partner: (quantity, self.awi.current_step, price)}
                )
                if value >= base:
                    acceptable = price
                    break
            return float(acceptable)

        acceptable = pmax
        for price in range(pmax, pmin - 1, -1):
            value = self._utility({partner: (quantity, self.awi.current_step, price)})
            if value >= base:
                acceptable = price
                break
        return float(acceptable)

    def _allocate_quantities(
        self, total_quantity: int, partners: list[str]
    ) -> dict[str, int]:
        if not self.use_partner_reliability:
            allocation = {p: 0 for p in partners}
            remaining = total_quantity
            for i, partner in enumerate(partners):
                if remaining <= 0:
                    break
                slots_left = len(partners) - i
                q = max(1, int(round(remaining / slots_left)))
                allocation[partner] = min(q, remaining)
                remaining -= allocation[partner]
            return allocation

        ranked = sorted(
            partners,
            key=lambda p: (self._stats(p).reliability, -self._stats(p).offers_seen),
            reverse=True,
        )
        weights = [max(0.05, self._stats(p).reliability) for p in ranked]
        wsum = sum(weights)
        allocation = {p: 0 for p in partners}
        remaining = total_quantity
        for i, partner in enumerate(ranked):
            if remaining <= 0:
                break
            if i == len(ranked) - 1:
                q = remaining
            else:
                q = max(1, int(round(total_quantity * weights[i] / wsum)))
                q = min(q, remaining)
            allocation[partner] = q
            remaining -= q
        return allocation

    # =====================
    # State helpers
    # =====================

    def _side_need(self, selling: bool) -> int:
        return int(self.awi.needed_sales if selling else self.awi.needed_supplies)

    def _risk_pressure(self, selling: bool) -> float:
        need = max(0, self._side_need(selling))
        capacity = max(1, int(getattr(self.awi, "n_lines", 1)))
        need_pressure = min(1.0, need / capacity)
        if selling:
            penalty = float(getattr(self.awi, "current_disposal_cost", 0.0) or 0.0)
        else:
            penalty = float(getattr(self.awi, "current_shortfall_penalty", 0.0) or 0.0)
        penalty_pressure = min(1.0, penalty)
        return 0.65 * need_pressure + 0.35 * penalty_pressure

    def _overorder_fraction(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        return self.early_overorder - (
            self.early_overorder - self.late_overorder
        ) * (t**0.7)

    def _is_selling_partner(self, partner: str) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return partner in getattr(self.awi, "my_consumers", [])
        return nmi.annotation["product"] == self.awi.my_output_product

    def _stats(self, partner: str) -> PartnerStats:
        if partner not in self.partner_stats:
            self.partner_stats[partner] = PartnerStats()
        return self.partner_stats[partner]


class CoReNoRegretAgent(CoReOneShotAgent):
    """Ablation: portfolio utility + shadow pricing + reliability, no regret gate."""

    use_regret_guard = False


class CoReNoShadowAgent(CoReOneShotAgent):
    """Ablation: portfolio utility + regret gate + reliability, no shadow price."""

    use_shadow_price = False


class CoReNoReliabilityAgent(CoReOneShotAgent):
    """Ablation: portfolio utility + regret gate + shadow pricing, equal allocation."""

    use_partner_reliability = False


class CoRePortfolioOnlyAgent(CoReOneShotAgent):
    """Ablation: only portfolio utility selection plus time-based counters."""

    use_regret_guard = False
    use_shadow_price = False
    use_partner_reliability = False


class CoReCapacityAwareAgent(CoRePortfolioOnlyAgent):
    """Portfolio agent with quantity allocation constrained by issue capacities."""

    def _allocate_quantities(
        self, total_quantity: int, partners: list[str]
    ) -> dict[str, int]:
        allocation = {p: 0 for p in partners}
        caps: dict[str, int] = {}
        for partner in partners:
            nmi = self.get_nmi(partner)
            if nmi is None:
                caps[partner] = 0
                continue
            caps[partner] = max(0, int(nmi.issues[QUANTITY].max_value))

        remaining = min(max(0, total_quantity), sum(caps.values()))
        active = [p for p in partners if caps[p] > 0]
        while remaining > 0 and active:
            slots_left = len(active)
            q = max(1, int(ceil(remaining / slots_left)))
            next_active: list[str] = []
            for partner in active:
                if remaining <= 0:
                    break
                room = caps[partner] - allocation[partner]
                if room <= 0:
                    continue
                add = min(q, room, remaining)
                allocation[partner] += add
                remaining -= add
                if allocation[partner] < caps[partner]:
                    next_active.append(partner)
            if len(next_active) == len(active) and all(
                caps[p] <= allocation[p] for p in next_active
            ):
                break
            active = next_active
        return allocation


class CoReMarketAnchorAgent(CoRePortfolioOnlyAgent):
    """Portfolio agent with market-price anchored counter-offers."""

    def _counter_price(
        self, partner: str, selling: bool, quantity: int, t: float
    ) -> int:
        nmi = self.get_nmi(partner)
        assert nmi is not None
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = int(p_issue.min_value), int(p_issue.max_value)
        if pmin >= pmax:
            return pmin

        concession = min(1.0, max(0.0, t) ** self.concession_exp)
        risk = self._risk_pressure(selling)
        concession = min(1.0, concession * (0.75 + 0.75 * risk))
        reference = self._market_reference_price(selling, pmin, pmax)

        if selling:
            floor = (1.0 - risk) * reference + risk * pmin
            price = pmax - concession * (pmax - floor)
        else:
            ceiling = (1.0 - risk) * reference + risk * pmax
            price = pmin + concession * (ceiling - pmin)
        return int(min(max(round(price), pmin), pmax))

    def _market_reference_price(self, selling: bool, pmin: int, pmax: int) -> float:
        product = self.awi.my_output_product if selling else self.awi.my_input_product
        for attr in ("trading_prices", "catalog_prices"):
            prices = getattr(self.awi, attr, None)
            if prices is None:
                continue
            try:
                value = float(prices[product])
            except Exception:
                continue
            if isfinite(value):
                return min(max(value, float(pmin)), float(pmax))
        return 0.5 * (pmin + pmax)


class CoReResponsiveAllocatorAgent(CoRePortfolioOnlyAgent):
    """Portfolio agent that allocates counter quantities to favorable partners."""

    favor_weight_base = 0.35
    favor_weight_scale = 1.0

    def _allocate_quantities(
        self, total_quantity: int, partners: list[str]
    ) -> dict[str, int]:
        allocation = {p: 0 for p in partners}
        caps: dict[str, int] = {}
        weights: dict[str, float] = {}
        for partner in partners:
            nmi = self.get_nmi(partner)
            if nmi is None:
                caps[partner] = 0
                weights[partner] = 0.0
                continue
            q_issue = nmi.issues[QUANTITY]
            p_issue = nmi.issues[UNIT_PRICE]
            caps[partner] = max(0, int(q_issue.max_value))
            pmin, pmax = float(p_issue.min_value), float(p_issue.max_value)
            last_price = self._stats(partner).last_price
            if last_price is None or pmax <= pmin:
                favorability = 0.5
            elif self._is_selling_partner(partner):
                favorability = (last_price - pmin) / (pmax - pmin)
            else:
                favorability = (pmax - last_price) / (pmax - pmin)
            favorability = min(1.0, max(0.0, favorability))
            weights[partner] = self.favor_weight_base + (
                self.favor_weight_scale * favorability
            )

        remaining = min(max(0, total_quantity), sum(caps.values()))
        active = [p for p in partners if caps[p] > 0 and weights[p] > 0]
        while remaining > 0 and active:
            wsum = sum(weights[p] for p in active)
            if wsum <= 0:
                break
            next_active: list[str] = []
            progressed = False
            for partner in sorted(active, key=lambda p: weights[p], reverse=True):
                if remaining <= 0:
                    break
                room = caps[partner] - allocation[partner]
                if room <= 0:
                    continue
                quota = max(1, int(round(remaining * weights[partner] / wsum)))
                add = min(quota, room, remaining)
                if add > 0:
                    allocation[partner] += add
                    remaining -= add
                    progressed = True
                if allocation[partner] < caps[partner]:
                    next_active.append(partner)
            if not progressed:
                break
            active = next_active
        return allocation


class CoReResponsiveMildAgent(CoReResponsiveAllocatorAgent):
    """Responsive allocation with a weak current-offer price signal."""

    favor_weight_base = 0.80
    favor_weight_scale = 0.40


class CoReResponsiveTinyAgent(CoReResponsiveAllocatorAgent):
    """Responsive allocation with a very small current-offer price signal."""

    favor_weight_base = 1.00
    favor_weight_scale = 0.15


class CoReSelectiveResponsiveAgent(CoRePortfolioOnlyAgent):
    """Use current price signal only when choosing among too many partners."""

    def _allocate_quantities(
        self, total_quantity: int, partners: list[str]
    ) -> dict[str, int]:
        active = [p for p in partners if self.get_nmi(p) is not None]
        if total_quantity >= len(active):
            return super()._allocate_quantities(total_quantity, partners)

        allocation = {p: 0 for p in partners}
        ranked = sorted(active, key=self._partner_favorability, reverse=True)
        remaining = max(0, total_quantity)
        for partner in ranked:
            if remaining <= 0:
                break
            nmi = self.get_nmi(partner)
            if nmi is None:
                continue
            cap = max(0, int(nmi.issues[QUANTITY].max_value))
            if cap <= 0:
                continue
            allocation[partner] = 1
            remaining -= 1
        return allocation

    def _partner_favorability(self, partner: str) -> float:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return 0.0
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = float(p_issue.min_value), float(p_issue.max_value)
        last_price = self._stats(partner).last_price
        if last_price is None or pmax <= pmin:
            return 0.5
        if self._is_selling_partner(partner):
            return min(1.0, max(0.0, (last_price - pmin) / (pmax - pmin)))
        return min(1.0, max(0.0, (pmax - last_price) / (pmax - pmin)))


class CoReRiskAdaptivePriceAgent(CoRePortfolioOnlyAgent):
    """Portfolio agent with concession speed adapted to current risk."""

    def _counter_price(
        self, partner: str, selling: bool, quantity: int, t: float
    ) -> int:
        nmi = self.get_nmi(partner)
        assert nmi is not None
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = int(p_issue.min_value), int(p_issue.max_value)
        if pmin >= pmax:
            return pmin

        risk = self._risk_pressure(selling)
        effective_exp = min(1.2, max(0.35, self.concession_exp + 0.40 - 0.65 * risk))
        concession = min(1.0, max(0.0, t) ** effective_exp)
        concession = min(1.0, concession * (0.70 + 0.85 * risk))
        if selling:
            price = pmax - concession * (pmax - pmin)
        else:
            price = pmin + concession * (pmax - pmin)
        return int(min(max(round(price), pmin), pmax))


class CoReLatePositiveAgent(CoRePortfolioOnlyAgent):
    """Portfolio agent that accepts clearly positive portfolios near deadline."""

    late_accept_time = 0.88
    late_accept_slack = 0.0

    def _accepted_portfolio(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if not offers:
            return set()
        base_utility = self._utility({})
        best_partners, best_utility = self._best_subset(offers)
        if (
            best_partners
            and t >= self.late_accept_time
            and best_utility > base_utility + self.late_accept_slack
        ):
            return set(best_partners)
        return super()._accepted_portfolio(offers, t)


class CoReLatePositiveEarlyAgent(CoReLatePositiveAgent):
    """More aggressive late-positive acceptance candidate for ablation."""

    late_accept_time = 0.80


class CoReLatePositiveSafeAgent(CoReLatePositiveAgent):
    """Conservative late-positive acceptance candidate for ablation."""

    late_accept_time = 0.94


class CoReLevelAdaptiveAgent(CoRePortfolioOnlyAgent):
    """Use the portfolio baseline except on last-level buyer roles."""

    def init(self):
        super().init()
        use_buyer_refinement = bool(getattr(self.awi, "is_last_level", False))
        self.use_shadow_price = use_buyer_refinement
        self.use_partner_reliability = use_buyer_refinement
        self.use_regret_guard = False


class CoReLevelConcessionAgent(CoRePortfolioOnlyAgent):
    """Use a slower concession curve for first-level seller roles only."""

    def init(self):
        super().init()
        if bool(getattr(self.awi, "is_first_level", False)):
            self.concession_exp = 0.90


class CoReLevelComboAgent(CoRePortfolioOnlyAgent):
    """Combine first-level slow concession with last-level buyer refinement."""

    def init(self):
        super().init()
        if bool(getattr(self.awi, "is_first_level", False)):
            self.concession_exp = 0.90
        if bool(getattr(self.awi, "is_last_level", False)):
            self.use_shadow_price = True
            self.use_partner_reliability = True
        self.use_regret_guard = False


class CoRePenaltyAwarePortfolioAgent(CoRePortfolioOnlyAgent):
    """Portfolio selector with a lightweight probabilistic mismatch penalty."""

    risk_aversion = 0.035
    utility_drawdown_limit = 0.035
    future_fill_floor = 0.08
    future_fill_ceiling = 0.80

    def _accepted_portfolio(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if not offers:
            return set()
        base_utility = self._utility({})
        empty_score = self._risk_adjusted_score({}, offers, t, base_utility)
        best_partners, best_utility, best_score = self._best_risk_subset(offers, t)
        if not best_partners:
            return set()
        if best_score + self.accept_slack < empty_score:
            return set()
        if best_utility < base_utility - self.utility_drawdown_limit:
            return set()
        return set(best_partners)

    def _best_risk_subset(
        self, offers: dict[str, Outcome], t: float
    ) -> tuple[set[str], float, float]:
        items = list(offers.items())
        if len(items) > self.max_exact_offers:
            partners, utility = super()._best_subset(offers)
            subset = {p: offers[p] for p in partners}
            return partners, utility, self._risk_adjusted_score(
                subset, offers, t, utility
            )

        best_partners: set[str] = set()
        best_utility = self._utility({})
        best_score = self._risk_adjusted_score({}, offers, t, best_utility)
        for r in range(1, len(items) + 1):
            for combo in combinations(items, r):
                subset = {p: o for p, o in combo}
                utility = self._utility(subset)
                score = self._risk_adjusted_score(subset, offers, t, utility)
                if score > best_score:
                    best_score = score
                    best_utility = utility
                    best_partners = set(subset)
        return best_partners, best_utility, best_score

    def _risk_adjusted_score(
        self,
        subset: dict[str, Outcome],
        all_offers: dict[str, Outcome],
        t: float,
        utility: float,
    ) -> float:
        if not isfinite(utility):
            return utility

        accepted_supply = sum(
            offer[QUANTITY]
            for partner, offer in subset.items()
            if not self._is_selling_partner(partner)
        )
        accepted_sales = sum(
            offer[QUANTITY]
            for partner, offer in subset.items()
            if self._is_selling_partner(partner)
        )
        need_supply = max(0.0, float(self._side_need(False)))
        need_sales = max(0.0, float(self._side_need(True)))

        remaining_partners = [p for p in all_offers if p not in subset]
        expected_supply = self._expected_future_quantity(
            remaining_partners, all_offers, selling=False, t=t
        )
        expected_sales = self._expected_future_quantity(
            remaining_partners, all_offers, selling=True, t=t
        )

        supply_shortage = max(0.0, need_supply - accepted_supply - expected_supply)
        sales_shortage = max(0.0, need_sales - accepted_sales - expected_sales)
        oversold = max(0.0, accepted_sales - need_sales - expected_supply)
        oversupplied = max(0.0, accepted_supply - need_supply - expected_sales)

        capacity = max(1.0, float(getattr(self.awi, "n_lines", 1) or 1))
        shortfall_weight = self._penalty_weight(shortfall=True)
        disposal_weight = self._penalty_weight(shortfall=False)
        risk = (
            shortfall_weight * (supply_shortage + oversold)
            + disposal_weight * (sales_shortage + oversupplied)
        ) / capacity
        return utility - self.risk_aversion * risk

    def _expected_future_quantity(
        self,
        partners: list[str],
        offers: dict[str, Outcome],
        selling: bool,
        t: float,
    ) -> float:
        time_left = 1.0 - min(1.0, max(0.0, t))
        base_prob = self.future_fill_floor + (
            self.future_fill_ceiling - self.future_fill_floor
        ) * (time_left**0.7)
        expected = 0.0
        for partner in partners:
            if self._is_selling_partner(partner) != selling:
                continue
            nmi = self.get_nmi(partner)
            if nmi is None:
                continue
            q_issue = nmi.issues[QUANTITY]
            q_cap = max(0.0, float(q_issue.max_value))
            if q_cap <= 0:
                continue
            favorability = self._offer_favorability(partner, offers.get(partner))
            prob = min(0.95, max(0.02, base_prob * (0.65 + 0.70 * favorability)))
            expected += q_cap * prob
        return expected

    def _offer_favorability(self, partner: str, offer: Outcome | None) -> float:
        nmi = self.get_nmi(partner)
        if nmi is None or offer is None:
            return 0.5
        p_issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = float(p_issue.min_value), float(p_issue.max_value)
        if pmax <= pmin:
            return 0.5
        price = float(offer[UNIT_PRICE])
        if self._is_selling_partner(partner):
            value = (price - pmin) / (pmax - pmin)
        else:
            value = (pmax - price) / (pmax - pmin)
        return min(1.0, max(0.0, value))

    def _penalty_weight(self, shortfall: bool) -> float:
        attr = "current_shortfall_penalty" if shortfall else "current_disposal_cost"
        penalty = float(getattr(self.awi, attr, 0.0) or 0.0)
        return 0.75 + min(2.0, max(0.0, penalty))


class CoRePenaltyAwareMildAgent(CoRePenaltyAwarePortfolioAgent):
    """Lower-risk penalty-aware portfolio candidate."""

    risk_aversion = 0.015
    utility_drawdown_limit = 0.020


class CoRePenaltyAwareStrictAgent(CoRePenaltyAwarePortfolioAgent):
    """Higher-risk-aversion penalty-aware portfolio candidate."""

    risk_aversion = 0.070
    utility_drawdown_limit = 0.050


class CoRePenaltyTiebreakAgent(CoRePenaltyAwarePortfolioAgent):
    """Use penalty risk only to choose among near-equal utility portfolios."""

    utility_band = 0.006
    risk_aversion = 0.040

    def _accepted_portfolio(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if not offers:
            return set()
        base_utility = self._utility({})
        best_partners, best_utility = CoRePortfolioOnlyAgent._best_subset(
            self, offers
        )
        if not best_partners:
            return set()
        aspiration = self._aspiration(base_utility, best_utility, t)
        if best_utility + self.accept_slack < aspiration:
            return set()

        chosen_partners = best_partners
        chosen_utility = best_utility
        chosen_subset = {p: offers[p] for p in chosen_partners}
        chosen_score = self._risk_adjusted_score(
            chosen_subset, offers, t, chosen_utility
        )

        items = list(offers.items())
        if len(items) <= self.max_exact_offers:
            cutoff = best_utility - self.utility_band
            for r in range(1, len(items) + 1):
                for combo in combinations(items, r):
                    subset = {p: o for p, o in combo}
                    utility = self._utility(subset)
                    if utility < cutoff:
                        continue
                    score = self._risk_adjusted_score(subset, offers, t, utility)
                    if score > chosen_score:
                        chosen_score = score
                        chosen_utility = utility
                        chosen_partners = set(subset)

        return set(chosen_partners)


class CoRePenaltyTiebreakTinyAgent(CoRePenaltyTiebreakAgent):
    """Very conservative penalty tie-breaker."""

    utility_band = 0.002
    risk_aversion = 0.020


class CoReLearnedPenaltyAgent(CoRePenaltyTiebreakTinyAgent):
    """Use the learned tiny penalty tie-breaker only on first-level roles."""

    def _accepted_portfolio(
        self, offers: dict[str, Outcome], t: float
    ) -> set[str]:
        if bool(getattr(self.awi, "is_first_level", False)):
            return super()._accepted_portfolio(offers, t)
        return CoRePortfolioOnlyAgent._accepted_portfolio(self, offers, t)


class Agent03Agent(CoReLearnedPenaltyAgent):
    """Competition submission class for the registered agent03 entry."""

    def init(self):
        super().init()
        if bool(getattr(self.awi, "is_last_level", False)):
            self.use_shadow_price = True


class CoreOSAgent(Agent03Agent):
    """Backward-compatible class name used by the first live submission."""


# Keep the official skeleton entry point stable. The live site has seen both
# names during retries, so keep the aliases harmless and import-compatible.
Agent01Agent = Agent03Agent
Agent02Agent = Agent03Agent
MyAgent = Agent03Agent
MyAgentFork = Agent03Agent


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([MyAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
