from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import ceil

from negmas import Outcome, ResponseType, SAOResponse
from scml.oneshot import OneShotSyncAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.ufun import UtilityInfo


@dataclass
class PartnerStats:
    offers_seen: int = 0
    first_price: float | None = None
    best_price: float | None = None
    last_price: float | None = None
    average_success_price: float | None = None
    successes: int = 0
    failures: int = 0
    concession_score: float = 0.0

    def register_offer(self, price: float, selling: bool) -> None:
        self.offers_seen += 1
        if self.first_price is None:
            self.first_price = price
        if self.best_price is None:
            self.best_price = price
        elif selling:
            self.best_price = max(self.best_price, price)
        else:
            self.best_price = min(self.best_price, price)
        if self.last_price is not None:
            delta = price - self.last_price
            signed_delta = delta if selling else -delta
            self.concession_score = 0.7 * self.concession_score + 0.3 * signed_delta
        self.last_price = price

    def register_success(self, price: float) -> None:
        self.successes += 1
        if self.average_success_price is None:
            self.average_success_price = price
            return
        n = self.successes
        self.average_success_price = ((n - 1) * self.average_success_price + price) / n


class AdaptivePortfolioOneShotAgent(OneShotSyncAgent):
    """
    A synchronous one-shot negotiation agent for SCML OneShot.

    Strategy summary:
    - Treat the day's negotiations as a portfolio selection problem and use the
      provided utility function to choose the best subset of current offers.
    - Keep an aspiration threshold that concedes over negotiation time.
    - Allocate remaining quantity conservatively across open negotiations so
      simultaneous acceptances do not create avoidable over-commitment.
    - Learn simple partner-specific price anchors during a simulation.
    """

    def __init__(
        self,
        *args,
        concession_power: float = 1.0,
        start_aspiration: float = 0.84,
        end_aspiration: float = 0.32,
        partner_markup: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._concession_power = concession_power
        self._start_aspiration = start_aspiration
        self._end_aspiration = end_aspiration
        self._partner_markup = partner_markup
        self._partner_stats: defaultdict[str, PartnerStats] = defaultdict(PartnerStats)
        self._ideal_utility = 0.0
        self._base_utility = 0.0
        self._selling = False
        self._reservation_price = 0
        self._extreme_price = 0
        self._market_pressure = 1.0
        self._trading_price = 0.0
        self._exact_subset_limit = 10
        self._beam_width = 24
        self._max_ranked_offers = 14

    def init(self) -> None:
        self._partner_stats = defaultdict(PartnerStats)

    def before_step(self) -> None:
        self._selling = self.awi.is_first_level
        self._base_utility = self.ufun.from_offers({}, ignore_signed_contracts=False)
        ideal_utility = self.ufun.max_utility
        if ideal_utility is None:
            ideal_utility = self._base_utility
        self._ideal_utility = float(ideal_utility)
        self._reservation_price, self._extreme_price = self._estimate_price_bounds()
        self._trading_price = float(self.awi.trading_prices[self.awi.my_output_product if self._selling else self.awi.my_input_product])
        self._market_pressure = self._estimate_market_pressure()

    def on_negotiation_success(self, contract, mechanism) -> None:
        price = contract.agreement["unit_price"]
        partner = contract.annotation["buyer"] if self._selling else contract.annotation["seller"]
        self._partner_stats[partner].register_success(price)

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        partner = annotation["buyer"] if self._selling else annotation["seller"]
        self._partner_stats[partner].failures += 1

    def first_proposals(self) -> dict[str, Outcome | None]:
        need = self._needed_quantity()
        if need <= 0:
            return {partner: None for partner in self.negotiators}
        plan = self._allocate_quantities(tuple(self.negotiators.keys()), need)
        proposals: dict[str, Outcome | None] = {}
        soft_cap = max(1, ceil(need / max(1, len(self.negotiators))))
        for partner, qty in plan.items():
            if qty <= 0:
                proposals[partner] = None
                continue
            nmi = self.get_nmi(partner)
            if not nmi:
                proposals[partner] = None
                continue
            qty = min(qty, soft_cap + 1)
            qty = max(int(nmi.issues[QUANTITY].min_value), min(int(nmi.issues[QUANTITY].max_value), qty))
            proposals[partner] = (
                qty,
                self.awi.current_step,
                self._target_price(partner, nmi, relative_time=0.0),
            )
        return proposals

    def counter_all(self, offers: dict[str, Outcome | None], states) -> dict[str, SAOResponse]:
        responses = {partner: SAOResponse(ResponseType.REJECT_OFFER, None) for partner in offers}
        if not offers:
            return responses
        self._current_state_for_alloc = next(iter(states.values()), None)

        relative_time = max(getattr(state, "relative_time", 0.0) for state in states.values())
        need = self._needed_quantity()
        if need <= 0:
            return {
                partner: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for partner in offers
            }

        valid_offers: dict[str, tuple[int, int, int]] = {}
        for partner, offer in offers.items():
            if offer is None:
                continue
            offer = tuple(offer)
            if offer[TIME] != self.awi.current_step or offer[QUANTITY] <= 0:
                continue
            valid_offers[partner] = offer  # type: ignore[assignment]
            self._partner_stats[partner].register_offer(offer[UNIT_PRICE], self._selling)

        accepted, best_info = self._choose_offer_subset(valid_offers)
        should_accept = self._should_accept(best_info, relative_time, accepted, need)

        accepted_qty = 0
        if should_accept:
            for partner in accepted:
                responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                accepted_qty += accepted[partner][QUANTITY]

        remaining_need = max(0, need - accepted_qty)
        open_partners = tuple(
            p for p in offers if (p not in accepted or not should_accept)
        )
        if remaining_need <= 0:
            for partner in open_partners:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            return responses

        plan = self._allocate_quantities(open_partners, remaining_need, valid_offers)
        for partner in open_partners:
            qty = plan.get(partner, 0)
            if qty <= 0:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            nmi = self.get_nmi(partner)
            if not nmi:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            target_price = self._target_price(partner, nmi, relative_time)
            if self._should_end_negotiation(partner, qty, target_price, relative_time):
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            responses[partner] = SAOResponse(
                ResponseType.REJECT_OFFER,
                (
                    qty,
                    self.awi.current_step,
                    target_price,
                ),
            )
        return responses

    def _should_accept(
        self,
        info: UtilityInfo,
        relative_time: float,
        accepted: dict[str, tuple[int, int, int]],
        need: int,
    ) -> bool:
        if not accepted:
            return False
        utility = info.utility
        aspiration = self._aspiration(relative_time)
        target = self._base_utility + aspiration * (self._ideal_utility - self._base_utility)
        accepted_qty = sum(offer[QUANTITY] for offer in accepted.values())
        covers_need = accepted_qty >= need
        improves = utility > self._base_utility + 1e-9
        low_shortfall = info.shortfall_quantity <= max(1, ceil(0.15 * need))
        meaningful_cover = accepted_qty >= max(1, ceil(0.6 * need))
        if utility >= target:
            return True
        if improves and low_shortfall and meaningful_cover:
            return True
        if relative_time >= 0.97 and utility > self._base_utility:
            return True
        if relative_time >= 0.9 and improves:
            return True
        if relative_time >= 0.65 and improves and low_shortfall:
            return True
        if relative_time >= 0.75 and covers_need and improves:
            return True
        return False

    def _aspiration(self, relative_time: float) -> float:
        relative_time = min(1.0, max(0.0, relative_time))
        concede = relative_time ** self._concession_power
        need = max(1, self._needed_quantity())
        partner_count = max(1, len(self.negotiators))
        urgency = min(1.0, need / max(1, self.awi.n_lines))
        penalty = self.awi.current_shortfall_penalty if not self._selling else self.awi.current_disposal_cost
        penalty_scale = min(0.15, penalty / max(1.0, self._trading_price + 1.0))
        partner_relief = min(0.1, 0.02 * partner_count)
        start = max(0.55, self._start_aspiration - 0.1 * urgency - penalty_scale + partner_relief)
        end = min(0.45, self._end_aspiration + 0.12 * urgency + penalty_scale)
        aspiration = start + (end - start) * concede
        return min(self._start_aspiration, max(self._end_aspiration, aspiration))

    def _needed_quantity(self) -> int:
        need = self.awi.needed_sales if self._selling else self.awi.needed_supplies
        return max(0, int(need))

    def _choose_offer_subset(
        self, offers: dict[str, tuple[int, int, int]]
    ) -> tuple[dict[str, tuple[int, int, int]], UtilityInfo]:
        base_info = self.ufun.from_offers(
            {},
            return_info=True,
            ignore_signed_contracts=False,
        )
        if not offers:
            return {}, base_info

        partners = list(offers.keys())
        if len(partners) <= self._exact_subset_limit:
            return self._choose_offer_subset_exact(offers, partners, base_info)
        return self._choose_offer_subset_beam(offers, partners, base_info)

    def _choose_offer_subset_exact(
        self,
        offers: dict[str, tuple[int, int, int]],
        partners: list[str],
        base_info: UtilityInfo,
    ) -> tuple[dict[str, tuple[int, int, int]], UtilityInfo]:
        if len(partners) > self._exact_subset_limit:
            return self._choose_offer_subset_beam(offers, partners, base_info)
        best_selection: dict[str, tuple[int, int, int]] = {}
        best_info = base_info
        need = max(1, self._needed_quantity())
        best_rank = self._bundle_rank(base_info, best_selection, need)
        for r in range(1, len(partners) + 1):
            for subset in combinations(partners, r):
                chosen = {partner: offers[partner] for partner in subset}
                info = self.ufun.from_offers(
                    chosen,
                    return_info=True,
                    ignore_signed_contracts=False,
                )
                rank = self._bundle_rank(info, chosen, need)
                if rank > best_rank:
                    best_info = info
                    best_selection = chosen
                    best_rank = rank
        return best_selection, best_info

    def _choose_offer_subset_beam(
        self,
        offers: dict[str, tuple[int, int, int]],
        partners: list[str],
        base_info: UtilityInfo,
    ) -> tuple[dict[str, tuple[int, int, int]], UtilityInfo]:
        need = max(1, self._needed_quantity())
        ordered_partners = self._rank_offer_candidates(offers, partners)
        ordered_partners = ordered_partners[: self._max_ranked_offers]
        best_selection: dict[str, tuple[int, int, int]] = {}
        best_info = base_info
        best_rank = self._bundle_rank(base_info, best_selection, need)
        beam: list[tuple[dict[str, tuple[int, int, int]], UtilityInfo, tuple[float, ...]]] = [
            ({}, base_info, best_rank)
        ]
        beam_width = min(40, max(self._beam_width, 2 * len(ordered_partners)))

        for partner in ordered_partners:
            offer = offers[partner]
            candidates = list(beam)
            for chosen, _, _ in beam:
                extended = dict(chosen)
                extended[partner] = offer
                info = self.ufun.from_offers(
                    extended,
                    return_info=True,
                    ignore_signed_contracts=False,
                )
                rank = self._bundle_rank(info, extended, need)
                candidates.append((extended, info, rank))
                if rank > best_rank:
                    best_selection = extended
                    best_info = info
                    best_rank = rank
            beam = sorted(
                candidates,
                key=lambda item: item[2],
                reverse=True,
            )[: beam_width]

        return best_selection, best_info

    def _bundle_rank(
        self,
        info: UtilityInfo,
        chosen: dict[str, tuple[int, int, int]],
        need: int,
    ) -> tuple[float, ...]:
        quantity = sum(offer[QUANTITY] for offer in chosen.values())
        mismatch = abs(quantity - need)
        reliability = sum(self._partner_weight(partner) for partner in chosen)
        risk_cost = info.shortfall_penalty + info.disposal_cost + info.storage_cost
        return (
            info.utility,
            -risk_cost,
            -float(info.shortfall_quantity),
            -mismatch,
            -float(info.remaining_quantity),
            reliability,
        )

    def _rank_offer_candidates(
        self,
        offers: dict[str, tuple[int, int, int]],
        partners: list[str],
    ) -> list[str]:
        scored: list[tuple[tuple[float, ...], str]] = []
        need = max(1, self._needed_quantity())
        for partner in partners:
            offer = offers[partner]
            info = self.ufun.from_offers(
                {partner: offer},
                return_info=True,
                ignore_signed_contracts=False,
            )
            rank = (
                info.utility,
                -float(info.shortfall_quantity),
                -float(info.remaining_quantity),
                min(1.0, offer[QUANTITY] / need),
                self._partner_weight(partner),
            )
            scored.append((rank, partner))
        scored.sort(reverse=True)
        return [partner for _, partner in scored]

    def _allocate_quantities(
        self,
        partners: tuple[str, ...],
        need: int,
        reference_offers: dict[str, tuple[int, int, int]] | None = None,
    ) -> dict[str, int]:
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}

        scored = []
        for partner in partners:
            nmi = self.get_nmi(partner)
            if not nmi:
                continue
            q_issue = nmi.issues[QUANTITY]
            cap = int(q_issue.max_value)
            floor = int(q_issue.min_value)
            if cap <= 0:
                continue
            weight = self._partner_weight(partner)
            scored.append((partner, weight, floor, cap))

        if not scored:
            return {partner: 0 for partner in partners}

        scored.sort(key=lambda item: item[1], reverse=True)
        scored = self._focus_partners(scored, need)
        relative_time = min(
            1.0,
            max(
                0.0,
                getattr(getattr(self, "_current_state_for_alloc", None), "relative_time", 0.0),
            ),
        )
        adjusted_need = max(need, ceil(need * (1.12 - 0.12 * relative_time)))
        plan = {partner: 0 for partner in partners}
        chosen = scored[: max(2, min(len(scored), ceil(len(scored) * 0.6)))]
        equal_base = adjusted_need // len(chosen)
        equal_rem = adjusted_need % len(chosen)
        for i, (partner, _, floor, cap) in enumerate(chosen):
            qty = equal_base + (1 if i < equal_rem else 0)
            if reference_offers and partner in reference_offers:
                qty = int(round((qty + reference_offers[partner][QUANTITY]) / 2))
            if qty > 0:
                qty = max(floor, qty)
            plan[partner] = min(cap, max(0, qty))
        return plan

    def _focus_partners(
        self, scored: list[tuple[str, float, int, int]], need: int
    ) -> list[tuple[str, float, int, int]]:
        if len(scored) <= 3:
            return scored
        focused: list[tuple[str, float, int, int]] = []
        covered_capacity = 0
        target_capacity = max(need, ceil(1.25 * need))
        for item in scored:
            focused.append(item)
            covered_capacity += item[3]
            if covered_capacity >= target_capacity and len(focused) >= 2:
                break
        return focused

    def _partner_weight(self, partner: str) -> float:
        stats = self._partner_stats[partner]
        if stats.best_price is None:
            return 1.0
        nmi = self.get_nmi(partner)
        if not nmi:
            return 1.0
        mn = float(nmi.issues[UNIT_PRICE].min_value)
        mx = float(nmi.issues[UNIT_PRICE].max_value)
        span = max(1.0, mx - mn)
        reference_price = stats.average_success_price
        if reference_price is None:
            reference_price = stats.best_price
        if reference_price is None:
            reference_price = stats.last_price
        if reference_price is None:
            return 1.0
        if self._selling:
            attractiveness = 1.0 + (reference_price - mn) / span
        else:
            attractiveness = 1.0 + (mx - reference_price) / span
        total_outcomes = stats.successes + stats.failures
        reliability_rate = stats.successes / total_outcomes if total_outcomes else 0.5
        reliability = 0.85 + 0.5 * reliability_rate + 0.05 * min(4, stats.successes)
        concession = 1.0 + max(-0.25, min(0.25, stats.concession_score / span))
        return max(0.2, attractiveness * reliability * concession)

    def _estimate_price_bounds(self) -> tuple[int, int]:
        issues = self.awi.current_output_issues if self._selling else self.awi.current_input_issues
        if not issues:
            return 0, 0
        mn = int(issues[UNIT_PRICE].min_value)
        mx = int(issues[UNIT_PRICE].max_value)
        typical_qty = max(
            1,
            min(
                int(issues[QUANTITY].max_value),
                ceil(self._needed_quantity() / max(1, len(self.negotiators) or 1)),
            ),
        )
        neutral = mx if self._selling else mn
        prices = range(mn, mx + 1)
        for price in prices:
            if self._selling:
                candidate = price
            else:
                candidate = mx - (price - mn)
            offer = {"probe": (typical_qty, self.awi.current_step, candidate)}
            utility = self.ufun.from_offers(offer, ignore_signed_contracts=False)
            if utility >= self._base_utility:
                neutral = candidate
                if self._selling:
                    break
            elif not self._selling:
                break
        return neutral, mx if self._selling else mn

    def _estimate_market_pressure(self) -> float:
        summary = self.awi.exogenous_contract_summary
        qty_ratio = 1.0
        if summary:
            raw_qty = max(1, int(summary[0][0]))
            final_qty = max(1, int(summary[-1][0]))
            qty_ratio = final_qty / raw_qty
        product = self.awi.my_output_product if self._selling else self.awi.my_input_product
        catalog = float(self.awi.catalog_prices[product])
        trading_ratio = self._trading_price / max(1.0, catalog)
        blended = 0.55 * qty_ratio + 0.45 * trading_ratio
        bias = 0.05 if self._selling else -0.05
        return min(1.5, max(0.55, blended + bias))

    def _should_end_negotiation(
        self, partner: str, qty: int, price: int, relative_time: float
    ) -> bool:
        if relative_time < 0.85:
            return False
        stats = self._partner_stats[partner]
        nmi = self.get_nmi(partner)
        if nmi and stats.last_price is not None:
            span = max(
                1.0,
                float(nmi.issues[UNIT_PRICE].max_value) - float(nmi.issues[UNIT_PRICE].min_value),
            )
            if abs(float(stats.last_price) - price) <= 0.08 * span:
                return False
            if stats.concession_score > 0:
                return False
        counter_utility = self.ufun.from_offers(
            {partner: (qty, self.awi.current_step, price)},
            ignore_signed_contracts=False,
        )
        if relative_time < 0.95 and counter_utility > self._base_utility - 0.02 * abs(self._base_utility + 1.0):
            return False
        return counter_utility <= self._base_utility

    def _target_price(self, partner: str, nmi, relative_time: float) -> int:
        mn = int(nmi.issues[UNIT_PRICE].min_value)
        mx = int(nmi.issues[UNIT_PRICE].max_value)
        stats = self._partner_stats[partner]
        concede = relative_time ** self._concession_power
        anchor = self._partner_anchor_price(partner, mn, mx)

        if self._selling:
            aspiration_price = round(self._extreme_price - concede * (self._extreme_price - self._reservation_price))
            market_anchor = int(round((self._trading_price * self._market_pressure + self._reservation_price) / 2))
            price = max(
                aspiration_price,
                market_anchor,
                int(round(anchor * (1.0 - self._partner_markup * concede))),
            )
            return max(mn, min(mx, price))

        aspiration_price = round(self._extreme_price + concede * (self._reservation_price - self._extreme_price))
        market_anchor = int(round((self._trading_price / self._market_pressure + self._reservation_price) / 2))
        price = min(
            aspiration_price,
            market_anchor,
            int(round(anchor * (1.0 + self._partner_markup * concede))),
        )
        return max(mn, min(mx, price))

    def _partner_anchor_price(self, partner: str, mn: int, mx: int) -> float:
        stats = self._partner_stats[partner]
        anchors = [
            stats.best_price,
            stats.last_price,
            stats.average_success_price,
            stats.first_price,
        ]
        values = [float(x) for x in anchors if x is not None]
        if not values:
            return float(mx if self._selling else mn)
        if self._selling:
            weights = [0.45, 0.2, 0.25, 0.1]
        else:
            weights = [0.45, 0.25, 0.2, 0.1]
        total = 0.0
        used = 0.0
        for value, weight in zip(anchors, weights):
            if value is None:
                continue
            total += float(value) * weight
            used += weight
        anchor = total / max(used, 1e-9)
        return max(float(mn), min(float(mx), anchor))
