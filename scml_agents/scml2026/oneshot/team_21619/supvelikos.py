#!/usr/bin/env python
from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAWI, OneShotSyncAgent


class Supvelikos(OneShotSyncAgent):
    """A synchronous OneShot agent driven by portfolio utility.

    Instead of accepting offers side-by-side using hand-written quantity scores,
    this agent evaluates the whole candidate contract set with its actual utility
    function. Counter offers then keep a broad market-making stance so the policy
    stays useful against unknown opponents instead of fitting local agent names.
    """

    overorder_max = 0.30
    overorder_min = 0.01
    overorder_exp = 0.38
    mismatch_max = 0.26
    mismatch_exp = 4.2
    price_exp = 1.7
    price_guard_slack = 0.18
    equal_distribution = False
    utility_margin = 0.002

    def init(self):
        self._best_selling: dict[str, int] = {}
        self._best_buying: dict[str, int] = {}
        self._deal_prices: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._last_prices: dict[str, int] = {}
        self._use_guarded_prices = False
        self._use_equal_split = False

    def before_step(self):
        self._secured_sales = 0
        self._secured_supplies = 0

    def first_proposals(self) -> dict[str, Outcome | None]:
        step, price = self._step_and_price(best=True, t=0.0)
        return {
            partner: (quantity, step, price) if quantity > 0 else None
            for partner, quantity in self._distribute_needs(0.0).items()
        }

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response: dict[str, SAOResponse] = {}
        t = min((state.relative_time for state in states.values()), default=1.0)
        current_offers = {
            p: o
            for p, o in offers.items()
            if o is not None and o[TIME] == self.awi.current_step
        }
        future_partners = set(offers) - set(current_offers)

        for partner, offer in current_offers.items():
            self._remember_price(partner, offer[UNIT_PRICE])

        for needs, partners, selling in (
            (self.awi.needed_supplies, self.awi.my_suppliers, False),
            (self.awi.needed_sales, self.awi.my_consumers, True),
        ):
            active = [p for p in partners if p in current_offers]
            if not active:
                continue
            selected = self._select_resilient_subset(
                active, current_offers, int(needs), selling, t
            )
            for partner in selected:
                response[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, current_offers[partner]
                )

            remaining = [p for p in active if p not in selected]
            if selling:
                remaining += [p for p in future_partners if p in self.awi.my_consumers]
            else:
                remaining += [p for p in future_partners if p in self.awi.my_suppliers]

            selected_qty = sum(
                current_offers[p][QUANTITY]
                for p in selected
                if p in partners and p in current_offers
            )
            remaining_need = max(0, int(needs) - selected_qty)
            quantities = self._partner_split(remaining_need, remaining, t)
            prices = self._price_ladder(
                [p for p, q in quantities.items() if q > 0], selling, t
            )
            for partner, quantity in quantities.items():
                if quantity <= 0:
                    response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                else:
                    response[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (quantity, self.awi.current_step, prices[partner]),
                    )

        for partner in set(offers) - set(response):
            response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return response

    def _select_utility_portfolio(self, offers: dict[str, Outcome], t: float) -> set[str]:
        partners = list(offers)
        if not partners:
            return set()

        baseline = float(self.ufun.from_offers({}))
        best_score = baseline
        best: tuple[str, ...] = tuple()
        max_size = min(len(partners), 10)

        for n in range(1, max_size + 1):
            for subset in combinations(partners, n):
                candidate = {p: offers[p] for p in subset}
                utility = float(self.ufun.from_offers(candidate))
                if utility > best_score:
                    best_score = utility
                    best = subset

        margin = self.utility_margin * (1.0 - min(1.0, max(0.0, t)))
        if best_score <= baseline + margin:
            return set()
        return set(best)

    def _select_resilient_subset(
        self,
        partners: list[str],
        offers: dict[str, Outcome],
        needs: int,
        selling: bool,
        t: float,
    ) -> set[str]:
        if needs <= 0:
            return set()

        baseline = float(self.ufun.from_offers({}))
        best: tuple[str, ...] = tuple()
        best_score = float("-inf")
        allowed = self._allowed_mismatch(t)
        guard = self._price_guard(selling, t) if self._use_guarded_prices else None

        for n in range(1, len(partners) + 1):
            for subset in combinations(partners, n):
                quantity = sum(offers[p][QUANTITY] for p in subset)
                diff = abs(quantity - needs)
                if diff > allowed:
                    continue
                total_value = sum(
                    offers[p][QUANTITY] * offers[p][UNIT_PRICE] for p in subset
                )
                average_price = total_value / max(1, quantity)
                if guard is not None and selling and average_price < guard:
                    continue
                if guard is not None and not selling and average_price > guard:
                    continue
                price_score = total_value if selling else -total_value
                utility_gain = float(
                    self.ufun.from_offers({p: offers[p] for p in subset})
                ) - baseline
                score = -diff * 1000.0 + price_score + 0.01 * utility_gain - 0.02 * n
                if score > best_score:
                    best_score = score
                    best = subset

        return set(best)

    def _select_best_subset(
        self,
        partners: list[str],
        offers: dict[str, Outcome],
        needs: int,
        selling: bool,
        t: float,
    ) -> set[str]:
        if needs <= 0:
            return set()

        best: tuple[str, ...] = tuple()
        best_score = float("-inf")
        allowed = self._allowed_mismatch(t)

        for n in range(1, len(partners) + 1):
            for subset in combinations(partners, n):
                quantity = sum(offers[p][QUANTITY] for p in subset)
                diff = abs(quantity - needs)
                if diff > allowed:
                    continue
                prices = [offers[p][UNIT_PRICE] for p in subset]
                avg_price = sum(prices) / len(prices)
                price_score = avg_price if selling else -avg_price
                compact_bonus = -0.01 * len(subset)
                score = -diff * 1000.0 + price_score + compact_bonus
                if score > best_score:
                    best_score = score
                    best = subset

        return set(best)

    def _distribute_needs(self, t: float) -> dict[str, int]:
        result: dict[str, int] = {}
        for needs, partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            active = [p for p in partners if p in self.negotiators]
            quantity = int(round(needs * (1 + self._overorder(t))))
            result.update(self._partner_split(quantity, active, t))
        return result

    def _partner_split(self, quantity: int, partners: list[str], t: float) -> dict[str, int]:
        if not partners:
            return {}
        quantity = max(0, int(quantity))
        if quantity <= 0:
            return {p: 0 for p in partners}
        if len(partners) == 1:
            return {partners[0]: quantity}

        if self._use_equal_split:
            values = self._split(quantity, len(partners), t)
            return dict(zip(partners, values, strict=False))

        learned = [self._partner_weight(p, t) for p in partners]
        mean_weight = sum(learned) / len(learned)
        # Keep most of the allocation diversified. Partner history is noisy in
        # OneShot worlds, so it is useful as a nudge rather than a commitment.
        weights = [
            0.78 + 0.22 * weight / max(1e-9, mean_weight) for weight in learned
        ]
        total = sum(weights)
        if total <= 0:
            values = self._split(quantity, len(partners), t)
            return dict(zip(partners, values, strict=False))

        raw = [quantity * w / total for w in weights]
        values = [int(v) for v in raw]
        remainder = quantity - sum(values)
        order = sorted(range(len(partners)), key=lambda i: raw[i] - values[i], reverse=True)
        for i in order[:remainder]:
            values[i] += 1

        if not self.awi.allow_zero_quantity and quantity >= len(partners):
            for i, v in enumerate(values):
                if v == 0:
                    donor = max(range(len(values)), key=values.__getitem__)
                    if values[donor] > 1:
                        values[donor] -= 1
                        values[i] = 1
        return dict(zip(partners, values, strict=False))

    def _partner_weight(self, partner: str, t: float) -> float:
        successes = self._successes.get(partner, 0)
        failures = self._failures.get(partner, 0)
        reliability = (successes + 1.5) / (successes + failures + 3.0)
        price_memory = 1.0
        if partner in self.awi.my_consumers and partner in self._best_selling:
            price_memory = 1.0 + 0.12 * min(1.0, successes + 1)
        elif partner in self.awi.my_suppliers and partner in self._best_buying:
            price_memory = 1.0 + 0.12 * min(1.0, successes + 1)
        exploration = 0.35 * (1.0 - min(1.0, max(0.0, t)))
        return max(0.05, reliability * price_memory + exploration * random.random())

    def _split(self, quantity: int, n: int, t: float) -> list[int]:
        if n <= 0:
            return []
        quantity = max(0, int(quantity))
        if quantity <= 0:
            return [0] * n
        if self._use_equal_split or self.equal_distribution or t > 0.45:
            base, rem = divmod(quantity, n)
            values = [base + (1 if i < rem else 0) for i in range(n)]
            random.shuffle(values)
            return values
        values = [0] * n
        remaining = quantity
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if remaining <= 0:
                break
            q = max(1, round(remaining / max(1, len(order))))
            values[i] = q
            remaining -= q
        if remaining > 0:
            values[order[0]] += remaining
        return values

    def _step_and_price(
        self, best: bool = False, t: float = 0.0, selling: bool | None = None
    ) -> tuple[int, int]:
        if selling is None:
            selling = self.awi.is_first_level
        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        pmin = int(issues[UNIT_PRICE].min_value)
        pmax = int(issues[UNIT_PRICE].max_value)
        if best:
            return self.awi.current_step, pmax if selling else pmin
        reservation = max(pmin, min(pmax, self._reservation_price(selling)))
        concession = min(1.0, max(0.0, t)) ** self.price_exp
        if selling:
            price = pmax - (pmax - reservation) * concession
        else:
            price = pmin + (reservation - pmin) * concession
        return self.awi.current_step, max(pmin, min(pmax, int(round(price))))

    def _price_guard(self, selling: bool, t: float) -> float:
        _, target = self._step_and_price(best=False, t=min(1.0, t + self.price_guard_slack), selling=selling)
        return float(target)

    def _random_price(self, selling: bool) -> int:
        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        issue = issues[UNIT_PRICE]
        return random.randint(int(issue.min_value), int(issue.max_value))

    def _target_price(self, partner: str, selling: bool, t: float) -> int:
        if not self._use_guarded_prices:
            return self._learned_price(partner, selling, t)
        _, price = self._step_and_price(best=False, t=t, selling=selling)
        return price

    def _price_ladder(
        self, partners: list[str], selling: bool, t: float
    ) -> dict[str, int]:
        if not partners:
            return {}
        if len(partners) == 1:
            return {partners[0]: self._target_price(partners[0], selling, t)}

        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        issue = issues[UNIT_PRICE]
        pmin, pmax = int(issue.min_value), int(issue.max_value)
        width = pmax - pmin + 1
        shuffled = list(partners)
        random.shuffle(shuffled)
        prices: dict[str, int] = {}
        for i, partner in enumerate(shuffled):
            low = pmin + width * i // len(shuffled)
            high = pmin + width * (i + 1) // len(shuffled) - 1
            prices[partner] = random.randint(low, max(low, min(pmax, high)))
        return prices

    def _learned_price(self, partner: str, selling: bool, t: float) -> int:
        return self._random_price(selling)

    def _adaptive_price(self, partner: str, selling: bool, t: float) -> int:
        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        issue = issues[UNIT_PRICE]
        pmin, pmax = int(issue.min_value), int(issue.max_value)
        _, target = self._step_and_price(best=False, t=t, selling=selling)
        noise = random.randint(pmin, pmax)
        trust = min(0.65, 0.15 + 0.10 * self._successes.get(partner, 0))
        if selling and partner in self._best_selling:
            memory = self._best_selling[partner]
            price = (1.0 - trust) * noise + trust * max(target, memory)
        elif not selling and partner in self._best_buying:
            memory = self._best_buying[partner]
            price = (1.0 - trust) * noise + trust * min(target, memory)
        else:
            price = 0.55 * noise + 0.45 * target
        return max(pmin, min(pmax, int(round(price))))

    def _market_price(self, partner: str, selling: bool, t: float) -> int:
        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        issue = issues[UNIT_PRICE]
        pmin, pmax = int(issue.min_value), int(issue.max_value)
        random_price = random.randint(pmin, pmax)

        if t < 0.18:
            return random_price

        _, target = self._step_and_price(best=False, t=t, selling=selling)
        memory = None
        if selling:
            memory = self._best_selling.get(partner)
        else:
            memory = self._best_buying.get(partner)

        if memory is None:
            weight = 0.25 + 0.25 * t
            price = (1.0 - weight) * random_price + weight * target
        elif selling:
            anchor = max(target, memory)
            weight = min(0.55, 0.20 + 0.10 * self._successes.get(partner, 0) + 0.25 * t)
            price = (1.0 - weight) * random_price + weight * anchor
        else:
            anchor = min(target, memory)
            weight = min(0.55, 0.20 + 0.10 * self._successes.get(partner, 0) + 0.25 * t)
            price = (1.0 - weight) * random_price + weight * anchor

        return max(pmin, min(pmax, int(round(price))))

    def _reservation_price(self, selling: bool) -> int:
        cost = int(round(getattr(self.awi.profile, "cost", 0) or 0))
        penalty = 0.08 * max(self.awi.current_disposal_cost, self.awi.current_shortfall_penalty)
        if selling:
            q = self.awi.current_exogenous_input_quantity
            unit_input = (
                self.awi.current_exogenous_input_price / q
                if q
                else self.awi.trading_prices[self.awi.my_input_product]
            )
            return int(round(unit_input + cost + penalty))
        q = self.awi.current_exogenous_output_quantity
        unit_output = (
            self.awi.current_exogenous_output_price / q
            if q
            else self.awi.trading_prices[self.awi.my_output_product]
        )
        return int(round(unit_output - cost - penalty))

    def _overorder(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        return self.overorder_max - (self.overorder_max - self.overorder_min) * (t**self.overorder_exp)

    def _allowed_mismatch(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        shortage = max(1e-9, float(self.awi.current_shortfall_penalty))
        disposal = max(1e-9, float(self.awi.current_disposal_cost))
        cheaper_error = min(shortage, disposal) / max(shortage, disposal)
        risk_budget = 0.04 + 0.08 * cheaper_error + 0.24 * (t**3.2)
        return risk_budget * self.awi.n_lines

    def _remember_price(self, partner: str, price: int) -> None:
        self._last_prices[partner] = int(price)
        if partner in self.awi.my_consumers:
            self._best_selling[partner] = max(price, self._best_selling.get(partner, price))
        else:
            self._best_buying[partner] = min(price, self._best_buying.get(partner, price))

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        for partner in partners:
            self._failures[partner] = self._failures.get(partner, 0) + 1

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:
        price = int(contract.agreement["unit_price"])
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            self._secured_sales += contract.agreement["quantity"]
        else:
            partner = contract.annotation["seller"]
            self._secured_supplies += contract.agreement["quantity"]
        previous = self._deal_prices.get(partner)
        if previous is None:
            self._deal_prices[partner] = price
        elif partner in self.awi.my_consumers:
            self._deal_prices[partner] = max(previous, price)
        else:
            self._deal_prices[partner] = min(previous, price)
        self._successes[partner] = self._successes.get(partner, 0) + 1

