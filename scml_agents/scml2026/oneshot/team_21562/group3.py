from negmas import ResponseType
from scml.oneshot import OneShotAgent


# SCML offer tuple indices
QUANTITY = 0
TIME = 1
UNIT_PRICE = 2


class Group3(OneShotAgent):
    name = "Group3"

    def init(self):
        # Daily portfolio state
        self.daily_target_quantity = 0
        self.secured_quantity = 0
        self.remaining_quantity = 0

        # Negotiation tracking
        self.successful_contracts = 0
        self.failed_negotiations = 0
        self.offer_history = {}
        self.sent_offer_history = {}

        # Opponent behavior model
        self.opponent_model = {}

        # Bidding parameters
        self.base_e = 0.3
        self.min_offer_quantity = 1

        # Behavior parameters
        self.silent_threshold = 1e-6
        self.nice_threshold = 0.25

        # Sensitivity parameters
        self.sensitivity_eta = 0.7
        self.sensitivity_epsilon = 1e-3
        self.default_sensitivity = 0.5

        # Stability parameters
        self.completion_strictness_threshold = 0.90
        self.max_late_buffer_fraction = 0.10
        self.max_early_buffer_fraction = 0.20
        self.selfish_quantity_penalty = 0.95
        self.low_sensitivity_quantity_penalty = 0.97

        # Debug history
        self.day_logs = []

    def before_step(self):

        # At the beginning of every simulation day.

        # Opponent model is not reset because SCML allows learning
        # within a simulation through partner identities.

        self.secured_quantity = 0
        self.remaining_quantity = 0
        self.offer_history = {}
        self.sent_offer_history = {}

        self.daily_target_quantity = self._estimate_daily_target_quantity()
        self.remaining_quantity = self.daily_target_quantity

    def propose(self, negotiator_id, state):

        # portfolio-safe quantity
        # stability-aware quantity adjustment
        # Boulware price concession
        # sensitivity-adjusted concession exponent

        quantity = self._stability_aware_offer_quantity(negotiator_id, state)

        if quantity <= 0:
            return None

        price = self._adaptive_price(state, negotiator_id)
        time = self.awi.current_step

        offer = (quantity, time, price)
        self._record_sent_offer(negotiator_id, offer)

        return offer

    def respond(self, negotiator_id, state, source=None):

        # Respond to incoming offer.

        # record opponent offer
        # update behavior and sensitivity model
        # apply stability-aware portfolio feasibility
        # apply utility-style acceptance

        offer = state.current_offer

        if offer is None:
            return ResponseType.REJECT_OFFER

        self._record_offer(negotiator_id, offer)
        self._update_opponent_model(negotiator_id, offer)

        if not self._portfolio_feasible(offer, negotiator_id, state):
            return ResponseType.REJECT_OFFER

        if not self._utility_acceptable(offer, state, negotiator_id):
            return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def on_negotiation_success(self, contract, mechanism):

        # Update portfolio after successful negotiation.

        self.successful_contracts += 1

        agreement = contract.agreement
        if agreement is None:
            return

        quantity = agreement.get("quantity", 0)
        self.secured_quantity += quantity
        self.remaining_quantity = max(
            0,
            self.daily_target_quantity - self.secured_quantity
        )

    def on_negotiation_failure(self, partners, annotation, mechanism, state):

        # Track failed negotiations.

        self.failed_negotiations += 1

    def step(self):

        # End-of-day logging.

        self.day_logs.append(
            {
                "step": self.awi.current_step,
                "target": self.daily_target_quantity,
                "secured": self.secured_quantity,
                "remaining": self.remaining_quantity,
                "completion_ratio": self._completion_ratio(),
                "successes": self.successful_contracts,
                "failures": self.failed_negotiations,
                "opponents": self._opponent_summary(),
            }
        )


    # Portfolio Manager + Stability Guard

    def _estimate_daily_target_quantity(self):
        input_q = getattr(self.awi, "current_exogenous_input_quantity", 0)
        output_q = getattr(self.awi, "current_exogenous_output_quantity", 0)

        target = max(input_q, output_q)

        if target <= 0:
            target = input_q + output_q

        return int(max(0, target))

    def _portfolio_safe_quantity(self):
        remaining = self.daily_target_quantity - self.secured_quantity
        return int(max(0, remaining))

    def _completion_ratio(self):

        # How much of today's target has already been secured.

        if self.daily_target_quantity <= 0:
            return 1.0

        return max(
            0.0,
            min(1.0, self.secured_quantity / self.daily_target_quantity)
        )

    def _relative_time(self, state):
        relative_time = getattr(state, "relative_time", 0.0)
        return max(0.0, min(1.0, float(relative_time)))

    def _quantity_buffer(self, state):

        # Controlled buffer that shrinks over time and with portfolio completion.

        # Early and incomplete portfolio: allow slightly more flexibility.
        # Late or near-complete portfolio: become stricter to avoid overcommitment.

        if self.daily_target_quantity <= 0:
            return 0

        relative_time = self._relative_time(state)
        completion = self._completion_ratio()

        # Early buffer is larger than late buffer.
        buffer_fraction = (
            self.max_early_buffer_fraction * (1.0 - relative_time)
            + self.max_late_buffer_fraction * relative_time
        )

        # Shrink buffer as we approach completion.
        buffer_fraction *= (1.0 - completion)

        buffer = int(round(self.daily_target_quantity * buffer_fraction))

        return max(0, buffer)

    def _safe_acceptance_limit(self, negotiator_id, state):

        # Maximum quantity we are willing to accept from an offer.

        # This protects against accepting contracts that may create excess or shortfall exposure.

        remaining = self._portfolio_safe_quantity()
        buffer = self._quantity_buffer(state)

        limit = remaining + buffer

        # Become stricter when nearly complete.
        if self._completion_ratio() >= self.completion_strictness_threshold:
            limit = remaining

        # Penalize risky opponent behavior.
        behavior = self.get_opponent_behavior(negotiator_id)
        sensitivity = self.get_opponent_sensitivity(negotiator_id)

        if behavior == "selfish":
            limit *= self.selfish_quantity_penalty

        if sensitivity < 0.5:
            limit *= self.low_sensitivity_quantity_penalty

        return int(max(0, round(limit)))

    def _portfolio_feasible(self, offer, negotiator_id=None, state=None):

        # Stability-aware feasibility check.

        # Reject offers that exceed the safe acceptance limit.

        quantity = offer[QUANTITY]

        if quantity <= 0:
            return False

        remaining = self._portfolio_safe_quantity()

        if remaining <= 0:
            return False

        if state is None:
            limit = remaining
        else:
            limit = self._safe_acceptance_limit(negotiator_id, state)

        if quantity > limit:
            return False

        return True

    def _stability_aware_offer_quantity(self, negotiator_id, state):

        # Quantity proposed by the agent.

        # Keeps quantity close to remaining need, but becomes
        # more conservative near portfolio completion or against risky opponents.

        remaining = self._portfolio_safe_quantity()

        if remaining <= 0:
            return 0

        quantity = remaining

        behavior = self.get_opponent_behavior(negotiator_id)
        sensitivity = self.get_opponent_sensitivity(negotiator_id)

        # If nearly complete, avoid large commitments.
        if self._completion_ratio() >= self.completion_strictness_threshold:
            quantity = min(quantity, remaining)

        # If opponent appears selfish or non-reciprocal, reduce exposure.
        if behavior == "selfish":
            quantity = int(max(1, round(quantity * self.selfish_quantity_penalty)))

        if sensitivity < 0.5:
            quantity = int(max(1, round(quantity * self.low_sensitivity_quantity_penalty)))

        return int(max(1, quantity))


    # Adaptive Bidding Engine

    def _adaptive_price(self, state, negotiator_id=None):

        # Compute Boulware-style adaptive price.

        min_price, max_price = self._price_range()
        alpha = self._concession_factor(state, negotiator_id)

        if self._is_selling():
            price = min_price + alpha * (max_price - min_price)
        else:
            price = max_price - alpha * (max_price - min_price)

        return int(round(self._clip_price(price, min_price, max_price)))

    def _concession_factor(self, state, negotiator_id=None):
        relative_time = self._relative_time(state)

        e = self._adaptive_e(negotiator_id)
        e = max(0.05, e)

        alpha = 1.0 - (relative_time ** (1.0 / e))

        return max(0.0, min(1.0, alpha))

    def _adaptive_e(self, negotiator_id):

        # Adapt concession exponent using opponent behavior and sensitivity.

        e = self.base_e

        if negotiator_id is None or negotiator_id not in self.opponent_model:
            return e

        model = self.opponent_model[negotiator_id]
        behavior = model.get("current_behavior", "unknown")
        sensitivity = model.get("sensitivity", self.default_sensitivity)

        if behavior == "selfish":
            e *= 0.85
        elif behavior in ("concession", "nice"):
            e *= 1.10

        if sensitivity >= 1.0:
            e *= 1.10
        elif sensitivity < 0.5:
            e *= 0.90

        return max(0.15, min(0.75, e))

    def _clip_price(self, price, min_price, max_price):
        return max(min_price, min(max_price, price))


    # Utility Engine / Acceptance

    def _utility_acceptable(self, offer, state, negotiator_id=None):

        # Utility-style acceptance approximation.

        # Adds stability strictness near portfolio completion.

        quantity, time, price = offer

        min_price, max_price = self._price_range()

        relative_time = self._relative_time(state)
        threshold_adjustment = self._acceptance_adjustment(negotiator_id)

        # Stronger selectivity once portfolio is mostly satisfied.
        if self._completion_ratio() >= self.completion_strictness_threshold:
            threshold_adjustment += 0.05 * (max_price - min_price)

        if self._is_selling():
            threshold = max_price - relative_time * (max_price - min_price)
            threshold += threshold_adjustment
            return price >= threshold

        threshold = min_price + relative_time * (max_price - min_price)
        threshold -= threshold_adjustment
        return price <= threshold

    def _acceptance_adjustment(self, negotiator_id):

        # Small threshold adjustment based on opponent behavior.

        if negotiator_id is None or negotiator_id not in self.opponent_model:
            return 0.0

        min_price, max_price = self._price_range()
        price_span = max_price - min_price

        if price_span <= 0:
            return 0.0

        model = self.opponent_model[negotiator_id]
        behavior = model.get("current_behavior", "unknown")
        sensitivity = model.get("sensitivity", self.default_sensitivity)

        adjustment = 0.0

        if behavior == "selfish":
            adjustment += 0.05 * price_span
        elif behavior in ("concession", "nice"):
            adjustment -= 0.03 * price_span

        if sensitivity < 0.5:
            adjustment += 0.03 * price_span
        elif sensitivity >= 1.0:
            adjustment -= 0.02 * price_span

        return adjustment

    def _offer_value_for_us(self, offer):

        # Lightweight local score used for behavior and sensitivity tracking.

        quantity, time, price = offer
        min_price, max_price = self._price_range()

        if max_price == min_price:
            price_score = 1.0
        elif self._is_selling():
            price_score = (price - min_price) / (max_price - min_price)
        else:
            price_score = (max_price - price) / (max_price - min_price)

        useful_quantity = min(quantity, max(1, self._portfolio_safe_quantity()))

        return price_score * useful_quantity

    def _offer_value_for_opponent_proxy(self, offer):

        # Approximate how good our offer is for the opponent.

        quantity, time, price = offer
        min_price, max_price = self._price_range()

        if max_price == min_price:
            price_score = 1.0
        elif self._is_selling():
            # We sell to them. Lower price is better for opponent.
            price_score = (max_price - price) / (max_price - min_price)
        else:
            # We buy from them. Higher price is better for opponent.
            price_score = (price - min_price) / (max_price - min_price)

        return price_score * max(1, quantity)


    # Opponent Modeling

    def _ensure_opponent(self, negotiator_id):
        if negotiator_id not in self.opponent_model:
            self.opponent_model[negotiator_id] = {
                "offers": [],
                "values": [],
                "our_offers": [],
                "our_offer_values_for_opp": [],
                "moves": [],
                "behavior_counts": {
                    "concession": 0,
                    "selfish": 0,
                    "silent": 0,
                    "nice": 0,
                },
                "current_behavior": "unknown",
                "sensitivity": self.default_sensitivity,
                "sensitivity_observations": 0,
            }

    def _update_opponent_model(self, negotiator_id, offer):

        # Update opponent behavior and sensitivity using new opponent offer.

        self._ensure_opponent(negotiator_id)

        model = self.opponent_model[negotiator_id]

        value = self._offer_value_for_us(offer)

        model["offers"].append(offer)
        model["values"].append(value)

        if len(model["values"]) >= 2:
            previous_value = model["values"][-2]
            current_value = model["values"][-1]

            delta_opp = current_value - previous_value

            behavior = self._classify_behavior(delta_opp, previous_value)

            model["moves"].append(behavior)
            model["behavior_counts"][behavior] += 1
            model["current_behavior"] = self._dominant_behavior(model)

            self._update_sensitivity(negotiator_id, delta_opp)

    def _update_sensitivity(self, negotiator_id, delta_opp):

        # Estimate reciprocal concession sensitivity.

        model = self.opponent_model[negotiator_id]

        our_values = model["our_offer_values_for_opp"]

        if len(our_values) < 2:
            return

        delta_us = our_values[-1] - our_values[-2]

        if delta_us <= 0:
            return

        reciprocal_response = max(0.0, delta_opp) / max(
            self.sensitivity_epsilon,
            delta_us
        )

        old_sensitivity = model.get("sensitivity", self.default_sensitivity)

        new_sensitivity = (
            self.sensitivity_eta * old_sensitivity
            + (1.0 - self.sensitivity_eta) * reciprocal_response
        )

        model["sensitivity"] = max(0.0, min(2.0, new_sensitivity))
        model["sensitivity_observations"] += 1

    def _classify_behavior(self, delta, previous_value):
        if abs(delta) <= self.silent_threshold:
            return "silent"

        base = max(abs(previous_value), 1e-6)
        relative_delta = delta / base

        if relative_delta >= self.nice_threshold:
            return "nice"

        if delta > 0:
            return "concession"

        return "selfish"

    def _dominant_behavior(self, model):
        counts = model["behavior_counts"]
        return max(counts, key=counts.get)

    def _opponent_summary(self):
        summary = {}

        for opponent, model in self.opponent_model.items():
            summary[opponent] = {
                "current_behavior": model["current_behavior"],
                "behavior_counts": dict(model["behavior_counts"]),
                "sensitivity": round(model.get("sensitivity", 0.0), 3),
                "sensitivity_observations": model.get(
                    "sensitivity_observations",
                    0
                ),
                "n_offers": len(model["offers"]),
                "n_our_offers": len(model["our_offers"]),
            }

        return summary

    def get_opponent_behavior(self, negotiator_id):
        if negotiator_id not in self.opponent_model:
            return "unknown"

        return self.opponent_model[negotiator_id]["current_behavior"]

    def get_opponent_sensitivity(self, negotiator_id):
        if negotiator_id not in self.opponent_model:
            return self.default_sensitivity

        return self.opponent_model[negotiator_id].get(
            "sensitivity",
            self.default_sensitivity
        )


    # Price Helpers

    def _best_price(self):
        min_price, max_price = self._price_range()

        if self._is_selling():
            return max_price

        return min_price

    def _price_range(self):
        if self._is_selling():
            issues = self.awi.current_output_issues
        else:
            issues = self.awi.current_input_issues

        price_issue = issues[UNIT_PRICE]
        return price_issue.min_value, price_issue.max_value

    def _is_selling(self):
        input_q = getattr(self.awi, "current_exogenous_input_quantity", 0)
        return input_q > 0


    # Offer Tracking

    def _record_offer(self, negotiator_id, offer):
        if negotiator_id not in self.offer_history:
            self.offer_history[negotiator_id] = []

        self.offer_history[negotiator_id].append(offer)

    def _record_sent_offer(self, negotiator_id, offer):
        self._ensure_opponent(negotiator_id)

        if negotiator_id not in self.sent_offer_history:
            self.sent_offer_history[negotiator_id] = []

        self.sent_offer_history[negotiator_id].append(offer)

        model = self.opponent_model[negotiator_id]
        model["our_offers"].append(offer)
        model["our_offer_values_for_opp"].append(
            self._offer_value_for_opponent_proxy(offer)
        )