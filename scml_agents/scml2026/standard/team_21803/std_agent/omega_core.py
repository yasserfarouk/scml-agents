from __future__ import annotations

import math
import random
from collections import defaultdict

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.std import StdSyncAgent
from scml.std.common import QUANTITY, TIME, UNIT_PRICE


class OmegaSubmissionCore(StdSyncAgent):
    """Quantity-first standard agent with bounded bundle optimization."""

    def __init__(
        self,
        *args,
        productivity: float = 0.75,
        today_partner_fraction: float = 0.75,
        future_horizon: int = 3,
        future_accept_ratio: float = 0.95,
        acceptance_concession: float = 1.0,
        counter_start_concession: float = 0.25,
        counter_concession_exponent: float = 1.7,
        partner_exploration: int = 1,
        partner_memory_weight: float = 0.25,
        overfill_threshold: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.productivity = productivity
        self.today_partner_fraction = today_partner_fraction
        self.future_horizon = future_horizon
        self.future_accept_ratio = future_accept_ratio
        self.acceptance_concession = acceptance_concession
        self.counter_start_concession = counter_start_concession
        self.counter_concession_exponent = counter_concession_exponent
        self.partner_exploration = partner_exploration
        self.partner_memory_weight = partner_memory_weight
        self.overfill_threshold = overfill_threshold

        self._partner_successes: dict[str, int] = defaultdict(int)
        self._partner_attempts: dict[str, int] = defaultdict(int)
        self._balance_history: list[float] = []
        self._aggressive = False
        self._conservative = False
        self._threshold = 1

    def before_step(self) -> None:
        super().before_step()
        balance = float(getattr(self.awi, "current_balance", 0.0))
        self._balance_history.append(balance)
        self._balance_history = self._balance_history[-5:]
        base = max(1, int(round(self.awi.n_lines * 0.1)))
        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if inventory_ratio < 0.3:
            base = max(1, int(math.ceil(base * 1.5)))
        elif inventory_ratio > 0.8:
            base = max(1, int(round(base * 0.7)))
        progress = self.awi.current_step / max(1, self.awi.n_steps - 1)
        self._aggressive = progress > 0.7
        if self._aggressive:
            base = max(1, int(math.ceil(base * 1.8)))
        elif progress > 0.4:
            base = max(1, int(math.ceil(base * 1.3)))
        self._threshold = self.overfill_threshold if self.overfill_threshold is not None else base
        self._conservative = self._profit_is_falling()

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:  # type: ignore[override]
        super().on_negotiation_success(contract, mechanism)
        partner = self._contract_partner(contract)
        if partner:
            self._partner_attempts[partner] += 1
            self._partner_successes[partner] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:  # type: ignore[override]
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        for partner in partners:
            if partner != self.id:
                self._partner_attempts[partner] += 1

    def first_proposals(self):  # type: ignore[override]
        partners = list(self.negotiators)
        quantities = self._distribute_today(partners)
        future_partners = [partner for partner in partners if quantities.get(partner, 0) <= 0]
        future = self._future_offers(future_partners)
        unneeded = None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        return {
            partner: (
                (
                    quantity,
                    self.awi.current_step,
                    self._price(partner, relative_time=0.0, first=True),
                )
                if quantity > 0
                else future.get(partner, unneeded)
            )
            for partner, quantity in quantities.items()
        }

    def counter_all(
        self,
        offers: dict[str, Outcome | None],
        states: dict[str, SAOState],
    ) -> dict[str, SAOResponse]:
        responses: dict[str, SAOResponse] = {}
        future_accepted = self._accept_future(offers, states)
        for partner in future_accepted:
            offer = offers[partner]
            if offer is not None:
                responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        accepted_today: dict[str, Outcome] = {}
        for selling in (False, True):
            need = self._need(self.awi.current_step, selling)
            candidates = {
                partner: offer
                for partner, offer in offers.items()
                if partner not in responses
                and offer is not None
                and offer[TIME] == self.awi.current_step
                and self._is_selling(partner) == selling
                and self._acceptable_price(
                    partner,
                    offer[UNIT_PRICE],
                    states[partner].relative_time,
                )
            }
            selected = self._select_quantity_subset(candidates, need, selling)
            for partner in selected:
                offer = candidates[partner]
                accepted_today[partner] = offer
                responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        remaining = [partner for partner in offers if partner not in responses]
        accepted_buy = sum(
            offer[QUANTITY]
            for partner, offer in accepted_today.items()
            if not self._is_selling(partner)
        )
        accepted_sell = sum(
            offer[QUANTITY]
            for partner, offer in accepted_today.items()
            if self._is_selling(partner)
        )
        quantities = self._distribute_today(
            remaining,
            buy_override=max(0, self._need(self.awi.current_step, False) - accepted_buy),
            sell_override=max(0, self._need(self.awi.current_step, True) - accepted_sell),
        )
        future = self._future_offers(
            [partner for partner in remaining if quantities.get(partner, 0) <= 0]
        )
        unneeded = None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        for partner in remaining:
            quantity = quantities.get(partner, 0)
            if quantity > 0:
                offer = (
                    quantity,
                    self.awi.current_step,
                    self._price(
                        partner,
                        relative_time=states[partner].relative_time,
                        first=False,
                    ),
                )
            else:
                offer = future.get(partner, unneeded)
            responses[partner] = SAOResponse(
                ResponseType.END_NEGOTIATION if offer is None else ResponseType.REJECT_OFFER,
                offer,
            )
        return responses

    def _accept_future(
        self,
        offers: dict[str, Outcome | None],
        states: dict[str, SAOState],
    ) -> set[str]:
        current = self.awi.current_step
        accepted: set[str] = set()
        booked: dict[tuple[bool, int], int] = defaultdict(int)
        candidates = []
        for partner, offer in offers.items():
            if offer is None or offer[TIME] <= current:
                continue
            if self.future_horizon > 0 and offer[TIME] - current > self.future_horizon:
                continue
            if not self._acceptable_price(
                partner,
                offer[UNIT_PRICE],
                states[partner].relative_time,
            ):
                continue
            selling = self._is_selling(partner)
            price_quality = offer[UNIT_PRICE] if selling else -offer[UNIT_PRICE]
            candidates.append((offer[TIME], -price_quality, partner, offer, selling))
        for _, _, partner, offer, selling in sorted(candidates):
            day = offer[TIME]
            key = (selling, day)
            limit = int(self._need(day, selling) * self.future_accept_ratio)
            if offer[QUANTITY] + booked[key] <= max(0, limit):
                accepted.add(partner)
                booked[key] += offer[QUANTITY]
        return accepted

    def _select_quantity_subset(
        self,
        offers: dict[str, Outcome],
        need: int,
        selling: bool,
    ) -> set[str]:
        if need <= 0 or not offers:
            return set()
        max_offer = max(offer[QUANTITY] for offer in offers.values())
        quantity_cap = need + max(self._threshold, max_offer)
        states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, tuple())}
        for partner, offer in sorted(
            offers.items(),
            key=lambda item: self._offer_quality(item[0], item[1], selling),
            reverse=True,
        ):
            quality = self._offer_quality(partner, offer, selling)
            updated = dict(states)
            for quantity, (score, selected) in states.items():
                total = quantity + offer[QUANTITY]
                if total > quantity_cap:
                    continue
                candidate = (score + quality, selected + (partner,))
                if total not in updated or candidate[0] > updated[total][0]:
                    updated[total] = candidate
            states = updated

        overshoot = [
            (quantity, value)
            for quantity, value in states.items()
            if quantity >= need and quantity - need <= self._threshold
        ]
        if overshoot:
            _, (_, selected) = min(
                overshoot,
                key=lambda item: (item[0] - need, -item[1][0]),
            )
            return set(selected)
        underfill = [(quantity, value) for quantity, value in states.items() if 0 < quantity < need]
        if not underfill:
            return set()
        _, (_, selected) = min(
            underfill,
            key=lambda item: (need - item[0], -item[1][0]),
        )
        return set(selected)

    def _distribute_today(
        self,
        partners: list[str],
        buy_override: int | None = None,
        sell_override: int | None = None,
    ) -> dict[str, int]:
        result = {partner: 0 for partner in partners}
        for selling, override in ((False, buy_override), (True, sell_override)):
            side = [partner for partner in partners if self._is_selling(partner) == selling]
            need = self._need(self.awi.current_step, selling) if override is None else override
            if need <= 0 or not side:
                continue
            max_quantity = int(
                (
                    self.awi.current_output_issues
                    if selling
                    else self.awi.current_input_issues
                )[QUANTITY].max_value
            )
            selected = self._select_partners(side, need, max_quantity)
            allocation = self._allocate(need, selected, max_quantity)
            result.update(allocation)
        return result

    def _future_offers(self, partners: list[str]) -> dict[str, Outcome]:
        current = self.awi.current_step
        result: dict[str, Outcome] = {}
        for selling in (False, True):
            side = self._rank_partners(
                [partner for partner in partners if self._is_selling(partner) == selling]
            )
            if not side:
                continue
            cuts = (math.ceil(len(side) * 0.5), math.ceil(len(side) * 0.8))
            groups = (side[: cuts[0]], side[cuts[0] : cuts[1]], side[cuts[1] :])
            for offset, group in enumerate(groups, start=1):
                day = current + offset
                if offset > self.future_horizon or day >= self.awi.n_steps or not group:
                    continue
                need = max(0, self._need(day, selling) // 3)
                issue = (
                    self.awi.current_output_issues
                    if selling
                    else self.awi.current_input_issues
                )[QUANTITY]
                for partner, quantity in self._allocate(
                    need,
                    group,
                    int(issue.max_value),
                ).items():
                    if quantity > 0:
                        result[partner] = (
                            quantity,
                            day,
                            self._price(partner, relative_time=0.0, first=True),
                        )
        return result

    def _select_partners(self, partners: list[str], need: int, max_quantity: int) -> list[str]:
        ranked = self._rank_partners(partners)
        fraction = self.today_partner_fraction
        if self._aggressive:
            fraction = min(0.95, fraction + 0.15)
        elif self._conservative:
            fraction = max(0.45, fraction - 0.10)
        required = max(1, math.ceil(need / max(1, max_quantity)))
        count = max(required, int(math.ceil(len(ranked) * fraction)))
        selected = ranked[: min(len(ranked), count)]
        if self.partner_exploration > 0 and len(selected) < len(ranked):
            tail = ranked[len(selected) :]
            selected.extend(random.sample(tail, min(self.partner_exploration, len(tail))))
        return selected

    @staticmethod
    def _allocate(need: int, partners: list[str], max_quantity: int) -> dict[str, int]:
        if need <= 0 or not partners:
            return {partner: 0 for partner in partners}
        remaining = min(need, len(partners) * max_quantity)
        allocation = {partner: 0 for partner in partners}
        while remaining > 0:
            active = [partner for partner in partners if allocation[partner] < max_quantity]
            if not active:
                break
            share = max(1, remaining // len(active))
            for partner in active:
                quantity = min(share, max_quantity - allocation[partner], remaining)
                allocation[partner] += quantity
                remaining -= quantity
                if remaining <= 0:
                    break
        return allocation

    def _need(self, day: int, selling: bool) -> int:
        production = int(round(self.awi.n_lines * self.productivity))
        if selling:
            target = min(self.awi.n_lines, production + self.awi.current_inventory_input)
            return max(0, target - self.awi.total_sales_at(day))
        return max(
            0,
            production - self.awi.current_inventory_input - self.awi.total_supplies_at(day),
        )

    def _price(self, partner: str, relative_time: float, first: bool) -> int:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return 0
        issue = nmi.issues[UNIT_PRICE]
        minimum, maximum = issue.min_value, issue.max_value
        if first:
            return int(maximum if self._is_selling(partner) else minimum)
        concession = self.counter_start_concession + (
            1.0 - self.counter_start_concession
        ) * relative_time**self.counter_concession_exponent
        if self._aggressive:
            concession = min(1.0, concession + 0.10)
        if self._is_selling(partner):
            return int(round(maximum - concession * (maximum - minimum)))
        return int(round(minimum + concession * (maximum - minimum)))

    def _acceptable_price(self, partner: str, price: float, relative_time: float) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        issue = nmi.issues[UNIT_PRICE]
        concession = self.acceptance_concession + (
            1.0 - self.acceptance_concession
        ) * relative_time**self.counter_concession_exponent
        if self._is_selling(partner):
            threshold = issue.max_value - concession * (issue.max_value - issue.min_value)
            return price >= threshold
        threshold = issue.min_value + concession * (issue.max_value - issue.min_value)
        return price <= threshold

    def _offer_quality(self, partner: str, offer: Outcome, selling: bool) -> float:
        price = float(offer[UNIT_PRICE]) * offer[QUANTITY]
        price_quality = price if selling else -price
        return price_quality + self.partner_memory_weight * self._partner_rate(partner)

    def _rank_partners(self, partners: list[str]) -> list[str]:
        return sorted(partners, key=self._partner_rate, reverse=True)

    def _partner_rate(self, partner: str) -> float:
        attempts = self._partner_attempts.get(partner, 0)
        if attempts <= 0:
            return 0.5
        return self._partner_successes.get(partner, 0) / attempts

    def _is_selling(self, partner: str) -> bool:
        return partner in self.awi.my_consumers

    def _contract_partner(self, contract: Contract) -> str | None:
        if contract.annotation.get("seller") == self.id:
            return contract.annotation.get("buyer")
        return contract.annotation.get("seller")

    def _profit_is_falling(self) -> bool:
        if len(self._balance_history) < 4:
            return False
        midpoint = len(self._balance_history) // 2
        early = self._balance_history[:midpoint]
        late = self._balance_history[midpoint:]
        return sum(late) / len(late) < sum(early) / len(early) - 50.0
