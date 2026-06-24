from __future__ import annotations

from math import ceil, sqrt
import random

from scml.oneshot.common import QUANTITY, UNIT_PRICE

from .core_agent import AdaptivePortfolioOneShotAgent


def _equal_split(total: int, n: int) -> list[int]:
    if n <= 0:
        return []
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


class QuantityOrientedAdaptiveAgent(AdaptivePortfolioOneShotAgent):
    """Inspired by quantity-oriented winners: prioritize fulfilling need reliably."""

    def _bundle_rank(self, info, chosen, need):
        quantity = sum(offer[0] for offer in chosen.values())
        fill = min(1.0, quantity / max(1, need))
        return (
            -float(info.shortfall_quantity),
            fill,
            info.utility,
            -float(info.remaining_quantity),
            -abs(quantity - need),
            -float(len(chosen)),
        )


class PatientPortfolioAgent(AdaptivePortfolioOneShotAgent):
    """Inspired by patient winners: slow concession, wait for stronger prices."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            concession_power=1.8,
            start_aspiration=0.92,
            end_aspiration=0.24,
            partner_markup=0.0,
            **kwargs,
        )


class CautiousCostAgent(AdaptivePortfolioOneShotAgent):
    """Inspired by cautious/cost-averse winners: avoid costly over-commitment."""

    def _bundle_rank(self, info, chosen, need):
        quantity = sum(offer[0] for offer in chosen.values())
        overfill = max(0, quantity - need)
        fill = min(1.0, quantity / max(1, need))
        return (
            info.utility,
            -float(info.shortfall_penalty + info.disposal_cost + info.storage_cost),
            -float(info.shortfall_quantity),
            -float(overfill),
            fill,
            -float(info.remaining_quantity),
        )

    def _estimate_market_pressure(self) -> float:
        return min(1.2, max(0.8, super()._estimate_market_pressure()))


class AlmostEqualAspirationAgent(AdaptivePortfolioOneShotAgent):
    """Inspired by AlmostEqualAgent: distribute quantities evenly and settle quickly."""

    def __init__(
        self,
        *args,
        concession_power=0.75,
        start_aspiration=0.82,
        end_aspiration=0.36,
        partner_markup=0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            concession_power=concession_power,
            start_aspiration=start_aspiration,
            end_aspiration=end_aspiration,
            partner_markup=partner_markup,
            **kwargs,
        )

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        active = []
        for partner in partners:
            nmi = self.get_nmi(partner)
            if nmi:
                active.append((partner, int(nmi.issues[QUANTITY].min_value), int(nmi.issues[QUANTITY].max_value)))
        if not active:
            return {partner: 0 for partner in partners}
        raw = _equal_split(need, len(active))
        plan = {partner: 0 for partner in partners}
        for (partner, floor, cap), qty in zip(active, raw, strict=False):
            qty = min(cap, max(floor if qty > 0 else 0, qty))
            plan[partner] = qty
        return plan


class DynamicMarketPortfolioAgent(AdaptivePortfolioOneShotAgent):
    """Inspired by Rchan: stronger market estimation and dynamic aspiration."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            concession_power=0.95,
            start_aspiration=0.88,
            end_aspiration=0.28,
            partner_markup=0.015,
            **kwargs,
        )

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        volatility = abs(self._trading_price - float(self.awi.catalog_prices[self.awi.my_output_product if self._selling else self.awi.my_input_product]))
        scale = min(0.15, volatility / max(1.0, self._trading_price + 1.0))
        if self._selling:
            return min(1.5, base + scale)
        return max(0.55, base - scale)


class SuccessWeightedPortfolioAgent(AdaptivePortfolioOneShotAgent):
    """Use success frequency more aggressively in allocation and pricing."""

    def _partner_weight(self, partner: str) -> float:
        stats = self._partner_stats[partner]
        nmi = self.get_nmi(partner)
        if not nmi:
            return 1.0
        mn = float(nmi.issues[UNIT_PRICE].min_value)
        mx = float(nmi.issues[UNIT_PRICE].max_value)
        span = max(1.0, mx - mn)
        anchor = stats.average_success_price
        if anchor is None:
            anchor = stats.best_price if stats.best_price is not None else stats.last_price
        if anchor is None:
            return 1.0
        attractiveness = 1.0 + ((anchor - mn) / span if self._selling else (mx - anchor) / span)
        total = stats.successes + stats.failures
        success_rate = stats.successes / total if total else 0.5
        momentum = 1.0 + max(-0.2, min(0.2, stats.concession_score / span))
        return max(0.2, attractiveness * (0.7 + success_rate) * momentum)


class OverorderingEqualAgent(AlmostEqualAspirationAgent):
    """Equal distribution with early over-ordering, similar to top distribution agents."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        relative_time = min(
            1.0,
            max(
                0.0,
                getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0),
            ),
        )
        over = 1.18 - 0.18 * relative_time
        adjusted_need = max(need, ceil(need * over))
        return super()._allocate_quantities(partners, adjusted_need, reference_offers)

    def counter_all(self, offers, states):
        self._current_state_for_alloc = next(iter(states.values()), None)
        return super().counter_all(offers, states)


class BalancedBundleAgent(AdaptivePortfolioOneShotAgent):
    """Prioritize balanced bundles with low shortfall and low excess."""

    def _bundle_rank(self, info, chosen, need):
        quantity = sum(offer[0] for offer in chosen.values())
        mismatch = abs(quantity - need)
        return (
            info.utility,
            -mismatch,
            -float(info.shortfall_quantity),
            -float(info.remaining_quantity),
            sum(self._partner_weight(p) for p in chosen),
        )


class MarginalGreedyPortfolioAgent(AdaptivePortfolioOneShotAgent):
    """Greedy marginal-utility subset construction for larger offer sets."""

    def _choose_offer_subset_beam(self, offers, partners, base_info):
        need = max(1, self._needed_quantity())
        chosen = {}
        current = base_info
        improved = True
        remaining = set(partners)
        while improved and remaining:
            improved = False
            best_partner = None
            best_info = current
            best_rank = self._bundle_rank(current, chosen, need)
            for partner in list(remaining):
                extended = dict(chosen)
                extended[partner] = offers[partner]
                info = self.ufun.from_offers(
                    extended,
                    return_info=True,
                    ignore_signed_contracts=False,
                )
                rank = self._bundle_rank(info, extended, need)
                if rank > best_rank:
                    best_rank = rank
                    best_info = info
                    best_partner = partner
                    improved = True
            if improved and best_partner is not None:
                chosen[best_partner] = offers[best_partner]
                current = best_info
                remaining.remove(best_partner)
        return chosen, current


class AdaptiveMismatchPortfolioAgent(AdaptivePortfolioOneShotAgent):
    """Accept quantity mismatch early if price is attractive; close mismatch late."""

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        accepted_qty = sum(offer[0] for offer in accepted.values())
        mismatch = abs(accepted_qty - need)
        tolerance = max(1, ceil((0.5 - 0.4 * relative_time) * need))
        return info.utility > self._base_utility and mismatch <= tolerance


class OmegaHistoricalHybridAgent(AdaptivePortfolioOneShotAgent):
    """Hybrid of cautious, quantity-oriented, and equal-distribution winner ideas."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            concession_power=1.0,
            start_aspiration=0.84,
            end_aspiration=0.32,
            partner_markup=0.01,
            **kwargs,
        )

    def _bundle_rank(self, info, chosen, need):
        quantity = sum(offer[0] for offer in chosen.values())
        mismatch = abs(quantity - need)
        reliability = sum(self._partner_weight(p) for p in chosen)
        risk_cost = info.shortfall_penalty + info.disposal_cost + info.storage_cost
        return (
            info.utility,
            -risk_cost,
            -float(info.shortfall_quantity),
            -mismatch,
            -float(info.remaining_quantity),
            reliability,
        )

    def _allocate_quantities(self, partners, need, reference_offers=None):
        partner_list = list(partners)
        if need <= 0 or not partner_list:
            return {partner: 0 for partner in partner_list}
        if len(partner_list) <= 3:
            return super()._allocate_quantities(tuple(partner_list), need, reference_offers)
        weights = [(partner, self._partner_weight(partner)) for partner in partner_list if self.get_nmi(partner)]
        weights.sort(key=lambda x: x[1], reverse=True)
        chosen = [p for p, _ in weights[: max(2, min(len(weights), ceil(len(weights) * 0.6)))]]
        raw = _equal_split(need, len(chosen))
        plan = {partner: 0 for partner in partner_list}
        for partner, qty in zip(chosen, raw, strict=False):
            nmi = self.get_nmi(partner)
            if not nmi:
                continue
            q_issue = nmi.issues[QUANTITY]
            plan[partner] = min(int(q_issue.max_value), max(int(q_issue.min_value) if qty > 0 else 0, qty))
        return plan


class OmegaHistoricalHybridEqualAgent(OmegaHistoricalHybridAgent):
    """Hybrid plus equal distribution and mild over-ordering early."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        partner_list = list(partners)
        if need <= 0 or not partner_list:
            return {partner: 0 for partner in partner_list}
        rt = min(
            1.0,
            max(
                0.0,
                getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0),
            ),
        )
        adjusted_need = max(need, ceil(need * (1.12 - 0.12 * rt)))
        return AlmostEqualAspirationAgent._allocate_quantities(
            self, tuple(partner_list), adjusted_need, reference_offers
        )

    def counter_all(self, offers, states):
        self._current_state_for_alloc = next(iter(states.values()), None)
        return super().counter_all(offers, states)


class OmegaHistoricalHybridMarginalAgent(OmegaHistoricalHybridAgent):
    """Hybrid strategy with greedy marginal search for larger offer sets."""

    def _choose_offer_subset_beam(self, offers, partners, base_info):
        return MarginalGreedyPortfolioAgent._choose_offer_subset_beam(
            self, offers, partners, base_info
        )


class OmegaHistoricalHybridMarketAgent(OmegaHistoricalHybridAgent):
    """Hybrid strategy with stronger market response."""

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        if self._selling:
            return min(1.5, base + 0.05)
        return max(0.55, base - 0.05)


class AlmostEqualMarketAgent(AlmostEqualAspirationAgent):
    """Equal distribution plus stronger market response."""

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        if self._selling:
            return min(1.5, base + 0.06)
        return max(0.55, base - 0.04)


class AlmostEqualOverorderAgent(AlmostEqualAspirationAgent):
    """Equal distribution plus mild early over-ordering."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        rt = min(
            1.0,
            max(0.0, getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0)),
        )
        adjusted_need = max(need, ceil(need * (1.14 - 0.14 * rt)))
        return super()._allocate_quantities(partners, adjusted_need, reference_offers)

    def counter_all(self, offers, states):
        self._current_state_for_alloc = next(iter(states.values()), None)
        return super().counter_all(offers, states)


class AlmostEqualRiskAwareAgent(AlmostEqualAspirationAgent):
    """Equal distribution with cost-aware bundle selection."""

    def _bundle_rank(self, info, chosen, need):
        quantity = sum(offer[0] for offer in chosen.values())
        mismatch = abs(quantity - need)
        return (
            info.utility,
            -(info.shortfall_penalty + info.disposal_cost + info.storage_cost),
            -float(info.shortfall_quantity),
            -mismatch,
            -float(info.remaining_quantity),
        )


class AlmostEqualSelectiveAgent(AlmostEqualAspirationAgent):
    """Equal split over a selected subset of better partners."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        weights = [(p, self._partner_weight(p)) for p in partners if self.get_nmi(p)]
        weights.sort(key=lambda x: x[1], reverse=True)
        selected = tuple(p for p, _ in weights[: max(2, ceil(len(weights) * 0.65))])
        return super()._allocate_quantities(selected, need, reference_offers) | {
            p: 0 for p in partners if p not in selected
        }


class AlmostEqualSuccessSelectiveAgent(AlmostEqualAspirationAgent):
    """Select partners by success history before equal split."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        scored = []
        for p in partners:
            stats = self._partner_stats[p]
            score = stats.successes - 0.5 * stats.failures + 0.1 * stats.offers_seen
            scored.append((score + 0.2 * self._partner_weight(p), p))
        scored.sort(reverse=True)
        selected = tuple(p for _, p in scored[: max(2, ceil(len(scored) * 0.6))])
        return super()._allocate_quantities(selected, need, reference_offers) | {
            p: 0 for p in partners if p not in selected
        }


class AlmostEqualMismatchAgent(AlmostEqualAspirationAgent):
    """More tolerant to partial fulfillment early if utility improves."""

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        qty = sum(offer[0] for offer in accepted.values())
        mismatch = abs(qty - need)
        return info.utility > self._base_utility and mismatch <= max(1, ceil(0.35 * need * (1.0 - relative_time)))


class AlmostEqualMarketRiskAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal distribution plus stronger market response."""

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        if self._selling:
            return min(1.45, base + 0.05)
        return max(0.55, base - 0.04)


class AlmostEqualFocusedOverorderAgent(AlmostEqualOverorderAgent):
    """Early over-ordering but only over a focused partner subset."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        weights = [(p, self._partner_weight(p)) for p in partners if self.get_nmi(p)]
        weights.sort(key=lambda x: x[1], reverse=True)
        selected = tuple(p for p, _ in weights[: max(2, ceil(len(weights) * 0.6))])
        return super()._allocate_quantities(selected, need, reference_offers) | {
            p: 0 for p in partners if p not in selected
        }


class AlmostEqualSoftEndAgent(AlmostEqualAspirationAgent):
    """Delay ending negotiations if counterpart is close to our target."""

    def _should_end_negotiation(self, partner, qty, price, relative_time):
        if relative_time < 0.9:
            return False
        return super()._should_end_negotiation(partner, qty, price, relative_time)


class AlmostEqualMarketAdaptiveAgent(AlmostEqualAspirationAgent):
    """Market-aware equal agent with adaptive concession defaults."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("concession_power", None)
        kwargs.pop("start_aspiration", None)
        kwargs.pop("end_aspiration", None)
        kwargs.pop("partner_markup", None)
        super().__init__(
            *args,
            concession_power=0.9,
            start_aspiration=0.80,
            end_aspiration=0.38,
            partner_markup=0.0,
            **kwargs,
        )

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        return min(1.45, max(0.6, base))


class AlmostEqualRiskOverorderAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal strategy with mild early over-ordering."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        rt = min(
            1.0,
            max(0.0, getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0)),
        )
        adjusted_need = max(need, ceil(need * (1.10 - 0.10 * rt)))
        return super()._allocate_quantities(partners, adjusted_need, reference_offers)

    def counter_all(self, offers, states):
        self._current_state_for_alloc = next(iter(states.values()), None)
        return super().counter_all(offers, states)


class AlmostEqualRiskSoftEndAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal strategy with softer late ending."""

    def _should_end_negotiation(self, partner, qty, price, relative_time):
        if relative_time < 0.9:
            return False
        return super()._should_end_negotiation(partner, qty, price, relative_time)


class AlmostEqualRiskMismatchAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal strategy with controlled mismatch acceptance."""

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        qty = sum(offer[0] for offer in accepted.values())
        mismatch = abs(qty - need)
        return (
            info.utility > self._base_utility
            and mismatch <= max(1, ceil(0.25 * need * (1.0 - 0.5 * relative_time)))
            and info.shortfall_quantity <= max(1, ceil(0.2 * need))
        )


class AlmostEqualRiskLightMarketAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal strategy with a small market bias only."""

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        if self._selling:
            return min(1.42, base + 0.025)
        return max(0.58, base - 0.02)


class AlmostEqualRiskTunedAgent(AlmostEqualRiskAwareAgent):
    """Risk-aware equal strategy with tuned concession parameters."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("concession_power", None)
        kwargs.pop("start_aspiration", None)
        kwargs.pop("end_aspiration", None)
        kwargs.pop("partner_markup", None)
        super().__init__(
            *args,
            concession_power=0.82,
            start_aspiration=0.80,
            end_aspiration=0.38,
            partner_markup=0.0,
            **kwargs,
        )


class AlmostEqualTimeMismatchAgent(AlmostEqualRiskOverorderAgent):
    """Allow a small time-weighted mismatch when utility is clearly positive."""

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        qty = sum(offer[0] for offer in accepted.values())
        mismatch = abs(qty - need)
        allowed = max(
            1,
            ceil(max(0.12 * self.awi.n_lines, 0.18 * need) * (relative_time**3.0)),
        )
        return (
            info.utility > self._base_utility
            and mismatch <= allowed
            and info.shortfall_quantity <= max(1, ceil(0.22 * need))
        )


class AlmostEqualNeedAdaptiveAgent(AlmostEqualRiskOverorderAgent):
    """Lower aspiration faster when the remaining need is still large."""

    def _aspiration(self, relative_time: float) -> float:
        base = super()._aspiration(relative_time)
        need_pressure = min(1.0, self._needed_quantity() / max(1, self.awi.n_lines))
        relief = 0.065 * need_pressure * (0.25 + 0.75 * relative_time)
        return max(self._end_aspiration, min(self._start_aspiration, base - relief))


class AlmostEqualConcessionPriceAgent(AlmostEqualRiskOverorderAgent):
    """Blend target prices toward the partner's latest concession path."""

    def _target_price(self, partner: str, nmi, relative_time: float) -> int:
        base = super()._target_price(partner, nmi, relative_time)
        stats = self._partner_stats[partner]
        if stats.last_price is None:
            return base
        mn = int(nmi.issues[UNIT_PRICE].min_value)
        mx = int(nmi.issues[UNIT_PRICE].max_value)
        span = max(1.0, float(mx - mn))
        momentum = max(-1.0, min(1.0, stats.concession_score / (0.15 * span)))
        blend = 0.08 + 0.27 * relative_time
        blended = round((1.0 - blend) * base + blend * float(stats.last_price))
        if self._selling:
            price = max(base, blended) if momentum > 0 else blended
        else:
            price = min(base, blended) if momentum > 0 else blended
        return max(mn, min(mx, price))


class AlmostEqualBayesWeightAgent(AlmostEqualRiskOverorderAgent):
    """Use a simple optimistic posterior on partner reliability."""

    def _partner_weight(self, partner: str) -> float:
        base = super()._partner_weight(partner)
        stats = self._partner_stats[partner]
        alpha = 1.0 + stats.successes
        beta = 1.0 + stats.failures
        posterior_mean = alpha / (alpha + beta)
        optimism = 0.9 / sqrt(alpha + beta + 1.0)
        activity = min(0.12, 0.02 * stats.offers_seen)
        return max(0.2, base * (0.82 + 0.32 * posterior_mean + 0.18 * optimism + activity))


class AlmostEqualCapacityBalancedAgent(AlmostEqualRiskOverorderAgent):
    """Stay close to equal split but lean toward large-capacity partners."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        rt = min(
            1.0,
            max(0.0, getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0)),
        )
        adjusted_need = max(need, ceil(need * (1.10 - 0.10 * rt)))
        active = []
        for partner in partners:
            nmi = self.get_nmi(partner)
            if not nmi:
                continue
            floor = int(nmi.issues[QUANTITY].min_value)
            cap = int(nmi.issues[QUANTITY].max_value)
            if cap <= 0:
                continue
            active.append((partner, floor, cap, self._partner_weight(partner)))
        if not active:
            return {partner: 0 for partner in partners}

        max_cap = max(cap for _, _, cap, _ in active)
        scored = []
        total_score = 0.0
        for partner, floor, cap, weight in active:
            score = 0.55 + 0.45 * (cap / max(1, max_cap)) * weight
            scored.append((partner, floor, cap, score))
            total_score += score

        plan = {partner: 0 for partner in partners}
        remainders: list[tuple[float, str, int]] = []
        assigned = 0
        for partner, floor, cap, score in scored:
            raw = adjusted_need * score / max(total_score, 1e-9)
            qty = min(cap, int(raw))
            if qty > 0:
                qty = max(floor, qty)
            if reference_offers and partner in reference_offers:
                qty = int(round((qty + reference_offers[partner][QUANTITY]) / 2))
                qty = min(cap, max(floor if qty > 0 else 0, qty))
            plan[partner] = qty
            assigned += qty
            remainders.append((raw - int(raw), partner, cap))

        if assigned < adjusted_need:
            for _, partner, cap in sorted(remainders, reverse=True):
                if assigned >= adjusted_need:
                    break
                if plan[partner] >= cap:
                    continue
                plan[partner] += 1
                assigned += 1
        return plan


class AlmostEqualNeedBufferAgent(AlmostEqualRiskOverorderAgent):
    """Accept bundles slightly below target when coverage is already strong."""

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        qty = sum(offer[0] for offer in accepted.values())
        coverage = qty / max(1, need)
        gap = max(0.0, self._ideal_utility - self._base_utility)
        target = self._base_utility + self._aspiration(relative_time) * gap
        buffer = gap * (0.045 - 0.015 * relative_time)
        return (
            info.utility >= target - buffer
            and coverage >= 0.72
            and info.shortfall_quantity <= max(1, ceil(0.18 * need))
        )


class AlmostEqualPressureAdaptiveAgent(AlmostEqualRiskOverorderAgent):
    """Adapt market pressure using partner scarcity and remaining-need pressure."""

    def _estimate_market_pressure(self) -> float:
        base = super()._estimate_market_pressure()
        partner_ratio = len(self.negotiators) / max(1, self.awi.n_lines)
        need_ratio = min(1.4, self._needed_quantity() / max(1, self.awi.n_lines))
        if self._selling:
            shift = max(0.0, partner_ratio - 1.0) * 0.06 + need_ratio * 0.015
            return min(1.5, base + shift)
        shift = max(0.0, 1.0 - partner_ratio) * 0.05 + need_ratio * 0.01
        return max(0.55, base - shift)


class AlmostEqualLateSalvageAgent(AlmostEqualRiskOverorderAgent):
    """Keep close negotiations alive a little longer when need remains high."""

    def _should_end_negotiation(self, partner, qty, price, relative_time):
        if relative_time < 0.94:
            return False
        need_pressure = self._needed_quantity() / max(1, self.awi.n_lines)
        stats = self._partner_stats[partner]
        nmi = self.get_nmi(partner)
        if (
            need_pressure > 0.35
            and nmi
            and stats.last_price is not None
        ):
            span = max(
                1.0,
                float(nmi.issues[UNIT_PRICE].max_value) - float(nmi.issues[UNIT_PRICE].min_value),
            )
            if abs(float(stats.last_price) - price) <= 0.16 * span:
                return False
        return super()._should_end_negotiation(partner, qty, price, relative_time)


class AlmostEqualSuccessAnchorAgent(AlmostEqualRiskOverorderAgent):
    """Use successful deal prices as the main partner anchor when available."""

    def _partner_anchor_price(self, partner: str, mn: int, mx: int) -> float:
        base = super()._partner_anchor_price(partner, mn, mx)
        stats = self._partner_stats[partner]
        if stats.average_success_price is None:
            return base
        success_weight = min(0.72, 0.28 + 0.1 * stats.successes)
        best = float(stats.best_price if stats.best_price is not None else base)
        blended = success_weight * float(stats.average_success_price) + (1.0 - success_weight) * best
        if stats.last_price is not None:
            blended = 0.82 * blended + 0.18 * float(stats.last_price)
        return max(float(mn), min(float(mx), blended))


class AlmostEqualCompositeAgent(AlmostEqualRiskOverorderAgent):
    """Combine optimistic partner weighting, buffered acceptance, and smarter pricing."""

    def _partner_weight(self, partner: str) -> float:
        base = super()._partner_weight(partner)
        stats = self._partner_stats[partner]
        alpha = 1.0 + stats.successes
        beta = 1.0 + stats.failures
        posterior_mean = alpha / (alpha + beta)
        optimism = 0.9 / sqrt(alpha + beta + 1.0)
        activity = min(0.12, 0.02 * stats.offers_seen)
        return max(0.2, base * (0.82 + 0.32 * posterior_mean + 0.18 * optimism + activity))

    def _partner_anchor_price(self, partner: str, mn: int, mx: int) -> float:
        base = super()._partner_anchor_price(partner, mn, mx)
        stats = self._partner_stats[partner]
        if stats.average_success_price is None:
            return base
        success_weight = min(0.72, 0.28 + 0.1 * stats.successes)
        best = float(stats.best_price if stats.best_price is not None else base)
        blended = success_weight * float(stats.average_success_price) + (1.0 - success_weight) * best
        if stats.last_price is not None:
            blended = 0.82 * blended + 0.18 * float(stats.last_price)
        return max(float(mn), min(float(mx), blended))

    def _target_price(self, partner: str, nmi, relative_time: float) -> int:
        base = super()._target_price(partner, nmi, relative_time)
        stats = self._partner_stats[partner]
        if stats.last_price is None:
            return base
        mn = int(nmi.issues[UNIT_PRICE].min_value)
        mx = int(nmi.issues[UNIT_PRICE].max_value)
        span = max(1.0, float(mx - mn))
        momentum = max(-1.0, min(1.0, stats.concession_score / (0.15 * span)))
        blend = 0.08 + 0.27 * relative_time
        blended = round((1.0 - blend) * base + blend * float(stats.last_price))
        if self._selling:
            price = max(base, blended) if momentum > 0 else blended
        else:
            price = min(base, blended) if momentum > 0 else blended
        return max(mn, min(mx, price))

    def _should_accept(self, info, relative_time, accepted, need):
        if super()._should_accept(info, relative_time, accepted, need):
            return True
        qty = sum(offer[0] for offer in accepted.values())
        coverage = qty / max(1, need)
        mismatch = abs(qty - need)
        gap = max(0.0, self._ideal_utility - self._base_utility)
        target = self._base_utility + self._aspiration(relative_time) * gap
        buffer = gap * (0.045 - 0.015 * relative_time)
        allowed = max(
            1,
            ceil(max(0.10 * self.awi.n_lines, 0.15 * need) * (relative_time**2.8)),
        )
        return (
            info.utility >= target - buffer
            and coverage >= 0.7
            and mismatch <= allowed
            and info.shortfall_quantity <= max(1, ceil(0.2 * need))
        )


class AlmostEqualPressureLateAgent(
    AlmostEqualPressureAdaptiveAgent,
    AlmostEqualLateSalvageAgent,
):
    """Combine stronger market-pressure adaptation with late salvage."""


class AlmostEqualPressureSuccessAgent(
    AlmostEqualPressureAdaptiveAgent,
    AlmostEqualSuccessAnchorAgent,
):
    """Combine stronger market-pressure adaptation with success anchors."""


class AlmostEqualPressureBayesAgent(
    AlmostEqualPressureAdaptiveAgent,
    AlmostEqualBayesWeightAgent,
):
    """Combine stronger market-pressure adaptation with Bayesian partner weighting."""


class AlmostEqualPressureLateSuccessAgent(
    AlmostEqualPressureAdaptiveAgent,
    AlmostEqualLateSalvageAgent,
    AlmostEqualSuccessAnchorAgent,
):
    """Combine pressure adaptation, late salvage, and success anchors."""


class AlmostEqualBayesSuccessAgent(
    AlmostEqualBayesWeightAgent,
    AlmostEqualSuccessAnchorAgent,
):
    """Combine Bayesian partner weighting with success-based anchor prices."""


class AlmostEqualBayesSuccessLateAgent(
    AlmostEqualBayesWeightAgent,
    AlmostEqualSuccessAnchorAgent,
    AlmostEqualLateSalvageAgent,
):
    """Combine Bayesian weighting, success anchors, and late salvage."""


class ScalableAlmostEqualAgent(AlmostEqualBayesWeightAgent):
    """A bounded-search adaptation of the quantity-price strategy."""

    overbuying = 0.20
    quantity_price_balance = 0.85
    score_threshold = 0.60
    quantity_cap = 3
    random_counter_price = True

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        active = []
        for partner in partners:
            nmi = self.get_nmi(partner)
            if not nmi:
                continue
            floor = int(nmi.issues[QUANTITY].min_value)
            cap = min(self.quantity_cap, int(nmi.issues[QUANTITY].max_value))
            active.append((partner, floor, cap))
        if not active:
            return {partner: 0 for partner in partners}
        adjusted_need = int(need * (1.0 + self.overbuying)) if not self._selling else need
        raw = _equal_split(adjusted_need, len(active))
        plan = {partner: 0 for partner in partners}
        for (partner, floor, cap), quantity in zip(active, raw, strict=False):
            if quantity > 0:
                quantity = max(floor, quantity)
            plan[partner] = min(cap, quantity)
        return plan

    def _quantity_price_score(self, chosen, need):
        quantity = sum(offer[QUANTITY] for offer in chosen.values())
        quantity_diff = quantity - need
        total_price = sum(
            offer[UNIT_PRICE] * offer[QUANTITY] for offer in chosen.values()
        )
        if self._selling:
            penalty = (
                max(0, quantity - need) * self.awi.current_shortfall_penalty
                + max(0, need - quantity) * self.awi.current_disposal_cost
            )
            price_score = total_price - penalty
        else:
            penalty = (
                max(0, need - quantity) * self.awi.current_shortfall_penalty
                + max(0, quantity - need) * self.awi.current_disposal_cost
            )
            price_score = -(total_price - penalty)
        normalized_price = max(0.0, min(1.0, 0.5 + price_score / 1000.0))
        normalized_quantity = 1.0 - abs(quantity_diff) / max(1, need)
        score = (
            self.quantity_price_balance * normalized_quantity
            + (1.0 - self.quantity_price_balance) * normalized_price
        )
        return score, quantity_diff

    def _bundle_rank(self, info, chosen, need):
        score, diff = self._quantity_price_score(chosen, need)
        return (
            score,
            info.utility,
            -abs(diff),
            -float(info.shortfall_quantity),
            -float(len(chosen)),
        )

    def _should_accept(self, info, relative_time, accepted, need):
        if not accepted:
            return False
        score, _ = self._quantity_price_score(accepted, need)
        threshold = self.score_threshold * (1.0 - 0.5 * relative_time)
        return score >= threshold

    def _target_price(self, partner, nmi, relative_time):
        if not self.random_counter_price:
            return super()._target_price(partner, nmi, relative_time)
        mn = int(nmi.issues[UNIT_PRICE].min_value)
        mx = int(nmi.issues[UNIT_PRICE].max_value)
        return random.randint(mn, mx)

    def _should_end_negotiation(self, partner, qty, price, relative_time):
        return False


class ScalableAlmostEqualTenAgent(ScalableAlmostEqualAgent):
    overbuying = 0.10


class ScalableAlmostEqualDeterministicAgent(ScalableAlmostEqualAgent):
    random_counter_price = False


class ScalableAlmostEqualTenDeterministicAgent(ScalableAlmostEqualTenAgent):
    random_counter_price = False


class FinalAdaptiveCapAgent(ScalableAlmostEqualTenDeterministicAgent):
    """Raise the per-partner cap only when scarce partners cannot cover need."""

    def _allocate_quantities(self, partners, need, reference_offers=None):
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        adjusted_need = int(need * (1.0 + self.overbuying)) if not self._selling else need
        active = []
        for partner in partners:
            nmi = self.get_nmi(partner)
            if not nmi:
                continue
            floor = int(nmi.issues[QUANTITY].min_value)
            issue_cap = int(nmi.issues[QUANTITY].max_value)
            active.append((partner, floor, issue_cap))
        if not active:
            return {partner: 0 for partner in partners}
        cap = max(self.quantity_cap, ceil(adjusted_need / len(active)))
        raw = _equal_split(adjusted_need, len(active))
        plan = {partner: 0 for partner in partners}
        for (partner, floor, issue_cap), quantity in zip(active, raw, strict=False):
            if quantity > 0:
                quantity = max(floor, quantity)
            plan[partner] = min(cap, issue_cap, quantity)
        return plan
