#!/usr/bin/env python
"""A value-led SCML Standard agent.

The agent keeps a small internal ledger of signed contracts and treats every
offer as a marginal change to that ledger.  Decisions are driven by shadow
prices rather than by distributing today's need over partners.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.std import StdAWI, StdSyncAgent


@dataclass
class PartnerBook:
    seen: int = 0
    deals: int = 0
    offered_value: float = 0.0

    @property
    def trust(self) -> float:
        return (1.0 + self.deals) / (2.0 + self.seen)


class SupmerkosV3(StdSyncAgent):
    """Ledger based planner for the Standard track."""

    def init(self):
        self._partners: dict[str, PartnerBook] = defaultdict(PartnerBook)
        self._buy_cost = float(self.awi.catalog_prices[self.awi.my_input_product])
        self._sell_price = float(self.awi.catalog_prices[self.awi.my_output_product])
        self._signed = defaultdict(int)
        self._future_buys = defaultdict(int)
        self._future_sells = defaultdict(int)

    def before_step(self):
        self._signed.clear()
        current = int(self.awi.current_step)
        for ledger in (self._future_buys, self._future_sells):
            for time in list(ledger):
                if time < current:
                    del ledger[time]

    def step(self):
        return None

    def first_proposals(self) -> dict[str, Outcome | None]:
        proposals: dict[str, Outcome | None] = {}
        for partner in self.active_negotiators:
            proposals[partner] = self._proposal_for(partner, opening=True)
        return proposals

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses: dict[str, SAOResponse] = {}
        candidates: list[tuple[float, str, Outcome]] = []
        current_candidates: list[tuple[str, Outcome]] = []

        for partner, offer in offers.items():
            if offer is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            self._partners[partner].seen += 1
            if int(offer[TIME]) == self.awi.current_step:
                current_candidates.append((partner, offer))
                continue
            score = self._offer_score(partner, offer, states.get(partner))
            candidates.append((score, partner, offer))

        chosen = self._best_current_subset(current_candidates)
        for partner, offer in chosen.items():
            responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        reserved: dict[tuple[bool, int], int] = defaultdict(int)
        accepted: dict[str, Outcome] = dict(chosen)
        baseline = self._utility(accepted)
        for score, partner, offer in sorted(candidates, reverse=True):
            is_buy = self._is_supplier(partner)
            time = int(offer[TIME])
            quantity = int(offer[QUANTITY])
            key = (is_buy, time)
            room = self._room(is_buy, time) - reserved[key]
            min_score = self._future_acceptance_cutoff(partner, offer, states.get(partner))

            if (
                quantity > 0
                and room > 0
                and quantity <= room
                and score >= min_score
                and self._future_is_exceptional(partner, offer)
            ):
                responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                reserved[key] += quantity
                accepted[partner] = offer
                baseline += score
                continue

            counter = self._proposal_for(partner, opening=False)
            if counter is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        for partner, _offer in current_candidates:
            if partner in responses:
                continue
            counter = self._proposal_for(partner, opening=False)
            if counter is None:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return responses

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        for partner in partners:
            if partner != self.id:
                self._partners[partner].seen += 1

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        agreement = contract.agreement
        quantity = int(agreement["quantity"])
        price = float(agreement["unit_price"])
        time = int(agreement["time"])
        is_seller = contract.annotation["seller"] == self.id
        partner = contract.annotation["buyer"] if is_seller else contract.annotation["seller"]
        book = self._partners[partner]
        book.seen += 1
        book.deals += 1
        book.offered_value += price
        self._signed[(not is_seller, time)] += quantity
        if time > self.awi.current_step:
            if is_seller:
                self._future_sells[time] += quantity
            else:
                self._future_buys[time] += quantity
        if is_seller:
            self._sell_price = 0.8 * self._sell_price + 0.2 * price
        else:
            self._buy_cost = 0.8 * self._buy_cost + 0.2 * price

    def _proposal_for(self, partner: str, opening: bool) -> Outcome | None:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        is_buy = self._is_supplier(partner)
        best = None
        times = self._candidate_times(nmi, is_buy)
        for time in times:
            time = int(time)
            if time < self.awi.current_step or time >= self.awi.n_steps:
                continue
            room = self._room(is_buy, time)
            if room <= 0:
                continue
            qissue = nmi.issues[QUANTITY]
            if time == self.awi.current_step:
                offer = self._single_current_quote(partner, room, opening)
                if offer is None:
                    continue
                score = self._offer_score(partner, offer, None, proposed_by_me=True) + 3.0
            else:
                quantity = max(int(qissue.min_value), min(int(qissue.max_value), room))
                if quantity <= 0:
                    continue
                price = self._target_price(partner, time, opening)
                price = self._clip_price(price, nmi.issues[UNIT_PRICE])
                offer = (quantity, time, price)
                score = self._offer_score(partner, offer, None, proposed_by_me=True)
                score -= self._future_proposal_penalty(is_buy, time, quantity)
            if best is None or score > best[0]:
                best = (score, offer)
        return None if best is None or best[0] < -0.25 else best[1]

    def _single_current_quote(
        self, partner: str, room: int, opening: bool
    ) -> Outcome | None:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        qissue = nmi.issues[QUANTITY]
        pissue = nmi.issues[UNIT_PRICE]
        minq = max(1, int(qissue.min_value))
        maxq = min(int(qissue.max_value), int(room))
        if maxq < minq:
            return None
        quantities = sorted({minq, max(minq, maxq // 2), maxq})
        mn, mx = float(pissue.min_value), float(pissue.max_value)
        span = max(1.0, mx - mn)
        is_buy = self._is_supplier(partner)
        if is_buy:
            shares = (0.14, 0.22, 0.31, 0.40) if opening else (0.30, 0.40, 0.52, 0.62)
        else:
            shares = (0.86, 0.78, 0.69, 0.60) if opening else (0.70, 0.60, 0.48, 0.38)
        best: tuple[float, Outcome] | None = None
        for quantity in quantities:
            for share in shares:
                price = self._clip_price(mn + share * span, pissue)
                offer = (int(quantity), int(self.awi.current_step), price)
                util = self._utility({partner: offer})
                if util <= 0.0:
                    continue
                acceptability = share if is_buy else 1.0 - share
                weight = 0.03 if opening else 0.055
                score = util + weight * quantity * acceptability
                if best is None or score > best[0]:
                    best = (score, offer)
        return None if best is None else best[1]

    def _offer_score(
        self,
        partner: str,
        offer: Outcome,
        state: SAOState | None,
        proposed_by_me: bool = False,
    ) -> float:
        quantity = int(offer[QUANTITY])
        time = int(offer[TIME])
        price = float(offer[UNIT_PRICE])
        if quantity <= 0 or time < self.awi.current_step or time >= self.awi.n_steps:
            return -10_000.0

        is_buy = self._is_supplier(partner)
        room = self._room(is_buy, time)
        overflow = max(0, quantity - room)
        trust = self._partners[partner].trust
        lateness = max(0, time - self.awi.current_step)
        urgency = 1.0 / (1.0 + lateness)

        if is_buy:
            unit = self._input_value(time) - price
            if self.awi.is_last_level:
                unit += 0.05 * self._output_value(time)
        else:
            unit = price - self._output_cost(time)
            if self.awi.is_first_level:
                unit += 0.03 * self._output_value(time)

        concession = 0.0
        if state is not None:
            concession = 0.10 * float(getattr(state, "relative_time", 0.0))
        if proposed_by_me:
            concession -= 0.04

        risk = overflow * (abs(unit) + 2.0)
        if lateness > 0:
            risk += self._future_risk(is_buy, time, quantity, unit)
        reliability = 0.15 * trust * quantity
        timing = 0.05 * urgency * quantity
        return unit * quantity + reliability + timing + concession - risk

    def _acceptance_cutoff(self, partner: str, state: SAOState | None) -> float:
        rt = float(getattr(state, "relative_time", 0.0)) if state is not None else 0.0
        time_left = 1.0 - self.awi.current_step / max(1, self.awi.n_steps - 1)
        strictness = 0.04 * time_left * (1.0 - rt)
        if self._partners[partner].trust > 0.65:
            strictness *= 0.75
        return strictness

    def _future_acceptance_cutoff(
        self, partner: str, offer: Outcome, state: SAOState | None
    ) -> float:
        is_buy = self._is_supplier(partner)
        time = int(offer[TIME])
        quantity = int(offer[QUANTITY])
        price = float(offer[UNIT_PRICE])
        lateness = max(1, time - self.awi.current_step)
        rt = float(getattr(state, "relative_time", 0.0)) if state is not None else 0.0
        if is_buy:
            unit_margin = self._input_value(time) - price
        else:
            unit_margin = price - self._output_cost(time)
        base = 0.55 + 0.18 * lateness + 0.05 * quantity
        base += 0.12 * (1.0 - rt)
        if unit_margin < 0:
            base += abs(unit_margin) * quantity
        if self._partners[partner].trust > 0.7:
            base *= 0.82
        return base

    def _future_is_exceptional(self, partner: str, offer: Outcome) -> bool:
        time = int(offer[TIME])
        quantity = int(offer[QUANTITY])
        price = float(offer[UNIT_PRICE])
        if time != self.awi.current_step + 1:
            return False
        if self.awi.current_step >= int(0.45 * self.awi.n_steps):
            return False
        is_buy = self._is_supplier(partner)
        if not self._future_is_balanced(is_buy, time, quantity):
            return False
        if is_buy:
            margin = self._input_value(time) - price
        else:
            margin = price - self._output_cost(time)
        return margin * quantity > 2.0 + 0.15 * quantity

    def _utility(self, offers: dict[str, Outcome]) -> float:
        try:
            return float(self.ufun.from_offers(offers, ignore_signed_contracts=False))
        except Exception:
            return sum(self._offer_score(p, o, None) for p, o in offers.items() if o)

    def _best_current_subset(self, candidates: list[tuple[str, Outcome]]) -> dict[str, Outcome]:
        if not candidates:
            return {}
        if len(candidates) > 12:
            candidates = sorted(
                candidates,
                key=lambda x: self._offer_score(x[0], x[1], None),
                reverse=True,
            )[:12]
        best: dict[str, Outcome] = {}
        best_value = self._utility({})
        indexed = list(range(len(candidates)))
        for r in range(1, len(candidates) + 1):
            for combo in combinations(indexed, r):
                offers = {candidates[i][0]: candidates[i][1] for i in combo}
                value = self._utility(offers)
                value += 0.25 * self._need_fit_bonus(offers)
                if value > best_value + 1e-9:
                    best_value = value
                    best = offers
        return best

    def _need_fit_bonus(self, offers: dict[str, Outcome]) -> float:
        if not offers:
            return 0.0
        buy_q = sum(
            int(offer[QUANTITY])
            for partner, offer in offers.items()
            if self._is_supplier(partner)
        )
        sell_q = sum(
            int(offer[QUANTITY])
            for partner, offer in offers.items()
            if not self._is_supplier(partner)
        )
        buy_need = max(0, int(self.awi.needed_supplies))
        sell_need = max(0, int(self.awi.needed_sales))
        lines = max(1, int(self.awi.n_lines))
        bonus = 0.0
        if buy_need > 0:
            bonus += self._side_fit_bonus(buy_q, buy_need, lines)
        elif buy_q > 0:
            bonus -= 0.18 * buy_q
        if sell_need > 0:
            bonus += self._side_fit_bonus(sell_q, sell_need, lines)
        elif sell_q > 0:
            bonus -= 0.18 * sell_q
        return bonus

    @staticmethod
    def _side_fit_bonus(quantity: int, need: int, lines: int) -> float:
        if quantity <= 0:
            return 0.0
        shortfall = max(0, need - quantity)
        surplus = max(0, quantity - need)
        covered = min(quantity, need)
        return 0.22 * covered - 0.34 * shortfall - 0.28 * surplus - 0.04 * max(0, quantity - lines)

    def _room(self, is_buy: bool, time: int) -> int:
        if is_buy:
            planned = (
                self.awi.total_supplies_at(time)
                + self._signed[(True, time)]
                + self._future_buys[time]
            )
            target = self._buy_target(time)
        else:
            planned = (
                self.awi.total_sales_at(time)
                + self._signed[(False, time)]
                + self._future_sells[time]
            )
            target = self._sell_target(time)
        return max(0, int(target - planned))

    def _buy_target(self, time: int) -> int:
        lines = max(1, int(self.awi.n_lines))
        if time == self.awi.current_step:
            base = max(self.awi.needed_supplies, lines)
        else:
            horizon = time - self.awi.current_step
            decay = 0.65 if horizon <= 2 else 0.35
            base = int(lines * decay)
        if self.awi.current_step > 0.75 * self.awi.n_steps:
            base = min(base, lines // 2 + 1)
        return max(0, base)

    def _sell_target(self, time: int) -> int:
        lines = max(1, int(self.awi.n_lines))
        inventory = int(self.awi.current_inventory_input + self.awi.current_inventory_output)
        if time == self.awi.current_step:
            base = max(self.awi.needed_sales, lines)
        else:
            horizon_capacity = max(0, time - self.awi.current_step) * lines
            base = min(int(0.65 * lines), inventory + horizon_capacity)
        return max(0, base)

    def _candidate_times(self, nmi, is_buy: bool) -> list[int]:
        values = set(self._issue_values(nmi.issues[TIME]))
        current = int(self.awi.current_step)
        times = [current]
        return [time for time in times if time in values]

    def _future_proposal_penalty(self, is_buy: bool, time: int, quantity: int) -> float:
        lateness = max(1, time - self.awi.current_step)
        penalty = 0.30 * lateness + 0.04 * quantity
        if not self._future_is_balanced(is_buy, time, quantity):
            penalty += 2.5 + 0.2 * quantity
        return penalty

    def _future_risk(self, is_buy: bool, time: int, quantity: int, unit: float) -> float:
        lateness = max(1, time - self.awi.current_step)
        risk = 0.16 * lateness * quantity
        if not self._future_is_balanced(is_buy, time, quantity):
            risk += (abs(unit) + 1.0) * max(1, quantity) * 0.65
        if time > self.awi.n_steps - 3:
            risk += 0.35 * quantity
        return risk

    def _future_is_balanced(self, is_buy: bool, time: int, quantity: int) -> bool:
        lines = max(1, int(self.awi.n_lines))
        if is_buy:
            committed_sales = self.awi.total_sales_at(time) + self._future_sells[time]
            planned_buys = self.awi.total_supplies_at(time) + self._future_buys[time]
            output_stock = int(self.awi.current_inventory_output)
            future_capacity = max(0, time - self.awi.current_step) * lines
            coverable_sales = output_stock + future_capacity
            expected_need = max(0, min(lines, committed_sales + lines // 2) - planned_buys)
            return quantity <= max(lines, expected_need + lines // 2) and coverable_sales > 0
        committed_buys = self.awi.total_supplies_at(time) + self._future_buys[time]
        committed_sales = self.awi.total_sales_at(time) + self._future_sells[time]
        output_stock = int(self.awi.current_inventory_output)
        future_capacity = max(0, time - self.awi.current_step) * lines
        available_output = output_stock + min(future_capacity, committed_buys + self.awi.current_inventory_input)
        return committed_sales + quantity <= available_output + lines

    def _input_value(self, time: int) -> float:
        expected_sale = self._output_value(min(self.awi.n_steps - 1, time + 1))
        cost = float(self.awi.profile.cost)
        inventory_pressure = max(0.0, 1.0 - self.awi.current_inventory_input / max(1, self.awi.n_lines))
        end_discount = 0.18 * time / max(1, self.awi.n_steps - 1)
        return expected_sale - cost + inventory_pressure * 2.0 - end_discount * expected_sale

    def _output_value(self, time: int) -> float:
        product = self.awi.my_output_product
        catalog = float(self.awi.catalog_prices[product])
        trading = float(self.awi.trading_prices[product])
        learned = self._sell_price
        pressure = 1.0 - 0.12 * time / max(1, self.awi.n_steps - 1)
        return max(0.0, (0.45 * catalog + 0.35 * trading + 0.20 * learned) * pressure)

    def _output_cost(self, time: int) -> float:
        product = self.awi.my_input_product
        catalog = float(self.awi.catalog_prices[product])
        trading = float(self.awi.trading_prices[product])
        input_cost = 0.40 * catalog + 0.35 * trading + 0.25 * self._buy_cost
        carrying = 0.10 * max(0, time - self.awi.current_step)
        return input_cost + float(self.awi.profile.cost) + carrying

    def _target_price(self, partner: str, time: int, opening: bool) -> int:
        nmi = self.get_nmi(partner)
        if nmi is not None:
            issue = nmi.issues[UNIT_PRICE]
            mn, mx = float(issue.min_value), float(issue.max_value)
            span = mx - mn
            current = time == self.awi.current_step
            phase = self.awi.current_step / max(1, self.awi.n_steps - 1)
            if self._is_supplier(partner):
                share = 0.32 + 0.06 * phase if opening and current else 0.44 + 0.04 * phase
                if not current:
                    share = 0.35
                return int(round(mn + share * span))
            share = 0.68 - 0.06 * phase if opening and current else 0.56 - 0.04 * phase
            if not current:
                share = 0.68
            return int(round(mn + share * span))
        trust = self._partners[partner].trust
        if self._is_supplier(partner):
            fair = self._input_value(time)
            discount = 0.18 if opening else 0.08
            return int(round(fair * (1.0 - discount + 0.06 * trust)))
        fair = self._output_cost(time)
        premium = 0.22 if opening else 0.10
        return int(round(fair * (1.0 + premium - 0.04 * trust)))

    @staticmethod
    def _clip_price(price: float, issue) -> int:
        return int(max(issue.min_value, min(issue.max_value, round(price))))

    @staticmethod
    def _issue_values(issue) -> list[int]:
        values = getattr(issue, "all", None)
        if values is not None:
            return list(values)
        return list(range(int(issue.min_value), int(issue.max_value) + 1))

    def _is_supplier(self, partner: str) -> bool:
        return partner in self.awi.my_suppliers
