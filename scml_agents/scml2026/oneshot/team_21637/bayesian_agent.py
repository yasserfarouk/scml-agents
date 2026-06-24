from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from negmas import Contract, Outcome, SAOResponse, ResponseType
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent, powerset


class BayesianAgent(SyncRandomOneShotAgent):
    
    OPPONENT_TYPES = ("GreedyOneShotAgent", "NonGreedy")
    _PARENT_GOOD_ACCEPT_WEIGHT = 1.20

    def _bayes__init__(
        self,
        *args,
        classification_threshold: float = 0.60,
        min_observations: int = 1,
        greedy_time_concession: float = 0.15,
        softmax_temperature: float = 1.0,
        counter_good_price_time: float = 0.65,
        counter_good_price_shortage_ratio: float = 0.40,
        counter_bad_price_penalty: float = 0.30,
        max_counter_partners: int = 4,
        counter_beam_width: int = 24,
        max_accept_subsets: int = 16,
        seller_quantity_bias: float = 1.0,
        buyer_quantity_bias: float = 1.0,
        seller_greedy_fill_threshold: float = 0.55,
        nongreedy_seller_firm: bool = False,
        single_partner_utility_floor: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classification_threshold = classification_threshold
        self.min_observations = min_observations
        self.greedy_time_concession = greedy_time_concession
        self.softmax_temperature = max(0.01, float(softmax_temperature))
        self.exploration_days = 6
        self.max_exploration_days = 10
        self.unknown_exploration_ratio = 0.5
        self.min_strategy_classifications = 3
        self.sync_equal_classification_threshold = 0.50
        self.strategy_random_threshold = 0.55
        self.exploration_quantity_multiplier = 1.3
        self.buyer_favorable_market_ratio = 1.2
        self.buyer_favorable_scaled_target_multiplier = 1.5
        self.small_dist_early_quantity_multiplier = 1.3
        self.small_dist_midpoint = 0.5
        self.non_greedy_success_default = 0.5
        self.classification_log_path = os.environ.get("BAYES_CLASSIFICATION_LOG")
        self.counter_good_price_time = float(counter_good_price_time)
        self.counter_good_price_shortage_ratio = float(counter_good_price_shortage_ratio)
        self.counter_bad_price_penalty = float(counter_bad_price_penalty)
        self.max_counter_partners = int(max_counter_partners)
        self.counter_beam_width = int(counter_beam_width)
        self.max_accept_subsets = int(max_accept_subsets)
        self.seller_quantity_bias = float(seller_quantity_bias)
        self.buyer_quantity_bias = float(buyer_quantity_bias)
        self.seller_greedy_fill_threshold = float(seller_greedy_fill_threshold)
        # Optional experiment: stay firm on sell-side NonGreedy counters.
        # Disabled by default because it can lose deals when concession is needed.
        self.nongreedy_seller_firm = bool(nongreedy_seller_firm)
        # Improvement: in a 1-on-1 (single remaining partner) endgame, only move
        # toward the opponent's offer up to the quantity at which our utility is
        # still no worse than the no-agreement (disagreement) utility.
        self.single_partner_utility_floor = bool(single_partner_utility_floor)

    def _bayes_init(self):
        super().init()
        self._opponent_logp: dict[str, dict[str, float]] = {}
        self._opponent_logits = {}
        self._opponent_observations = defaultdict(int)
        self._non_greedy_veto = {}
        self._opponent_offer_history = defaultdict(list)
        self._sent_offer_history = defaultdict(list)
        self._own_offer_result_history = defaultdict(list)
        self._own_offer_end_history = defaultdict(list)
        self._non_greedy_initial_offer_results = []
        self._partner_non_greedy_initial_offer_results = defaultdict(list)
        self._initial_good_rejection_streak_observed = defaultdict(int)
        self._received_offer_counts = defaultdict(int)
        self._received_first_offer_history = defaultdict(list)
        self._logit_history = defaultdict(list)
        self._evidence_counts = defaultdict(lambda: defaultdict(int))
        self._history_pattern_observed = defaultdict(set)
        self._first_offer_trials_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_accepts_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_reoffers_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_counter_quantities_by_price = defaultdict(
            lambda: defaultdict(list)
        )
        self._good_first_offer_rejection_streak = defaultdict(int)
        self._good_first_offer_cumulative_rejection_observed = defaultdict(bool)
        for partner in self._all_partners():
            self._ensure_partner(partner)

    def before_step(self):
        super().before_step()
        self._received_offer_counts.clear()
        for partner in self._all_partners():
            self._ensure_partner(partner)
        self._dump_classification_log("before_step")

    def _all_partners(self):
        return list(dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers)))

    def _ensure_partner(self, partner):
        if partner is None:
            return
        if partner not in self._opponent_logp:
            prior = -math.log(len(self.OPPONENT_TYPES))
            self._opponent_logp[partner] = {name: prior for name in self.OPPONENT_TYPES}
        if partner not in self._opponent_logits:
            self._opponent_logits[partner] = {
                "GreedyOneShotAgent": 0.0,
                "NonGreedy": 0.0,
            }
        self._sent_offer_history.setdefault(partner, [])
        self._own_offer_result_history.setdefault(partner, [])
        self._own_offer_end_history.setdefault(partner, [])
        self._partner_non_greedy_initial_offer_results.setdefault(partner, [])
        self._received_first_offer_history.setdefault(partner, [])
        self._logit_history.setdefault(partner, [])
        self._evidence_counts.setdefault(partner, defaultdict(int))
        self._history_pattern_observed.setdefault(partner, set())
        self._first_offer_trials_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_accepts_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_reoffers_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_counter_quantities_by_price.setdefault(
            partner,
            defaultdict(list),
        )

    def _is_process_one_agent(self) -> bool:
        return str(getattr(self, "id", "")).endswith("@1")

    def _is_process_zero_agent(self) -> bool:
        return str(getattr(self, "id", "")).endswith("@0")

    def _is_seller_to(self, partner) -> bool:
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        return self.awi.current_output_issues if self._is_seller_to(partner) else self.awi.current_input_issues

    def _best_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(issues[UNIT_PRICE].max_value if self._is_seller_to(partner) else issues[UNIT_PRICE].min_value)

    def _worst_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(issues[UNIT_PRICE].min_value if self._is_seller_to(partner) else issues[UNIT_PRICE].max_value)

    def _clamp_quantity(self, partner, quantity: int) -> int:
        issues = self._issues_for(partner)
        qmin = int(issues[QUANTITY].min_value)
        qmax = int(issues[QUANTITY].max_value)
        if quantity <= 0 and self.awi.allow_zero_quantity:
            return 0
        return max(qmin, min(qmax, int(quantity)))

    def _market_exogenous_quantity(self, values, agent_id) -> int:
        try:
            return int(values[agent_id])
        except Exception:
            return 0

    def _input_market_sell_buy_targets(self) -> tuple[int, int]:
        world = getattr(self.awi, "_world", None)
        if world is None:
            return 0, 0
        input_product = int(getattr(self.awi, "my_input_product", -1))
        if input_product < 0:
            return 0, 0
        all_suppliers = getattr(self.awi, "all_suppliers", [])
        all_consumers = getattr(self.awi, "all_consumers", [])
        if input_product >= len(all_suppliers) or input_product >= len(all_consumers):
            return 0, 0
        exogenous_qin = getattr(world, "exogenous_qin", {})
        exogenous_qout = getattr(world, "exogenous_qout", {})
        sell_target = sum(
            self._market_exogenous_quantity(exogenous_qin, partner)
            for partner in all_suppliers[input_product]
        )
        buy_target = sum(
            self._market_exogenous_quantity(exogenous_qout, partner)
            for partner in all_consumers[input_product]
        )
        return int(sell_target), int(buy_target)

    def _input_market_sell_buy_ratio(self) -> float | None:
        sell_target, buy_target = self._input_market_sell_buy_targets()
        if buy_target <= 0:
            return None
        return sell_target / buy_target

    def _is_process_one_buy_side(self, partners: list[str]) -> bool:
        return (
            self._is_process_one_agent()
            and bool(partners)
            and all(partner in self.awi.my_suppliers for partner in partners)
        )

    def _exploration_enabled(self) -> bool:
        if self.awi.current_step < self.exploration_days:
            return True
        if self.awi.current_step >= self.max_exploration_days:
            return False
        return self._too_many_unknown_opponents()

    def _active_partners(self):
        return [
            partner
            for partner in self._all_partners()
            if partner in getattr(self, "negotiators", {})
        ]

    def _too_many_unknown_opponents(self) -> bool:
        partners = self._active_partners()
        if not partners:
            return False
        types = [self.opponent_type(partner) for partner in partners]
        unknown_count = sum(opponent_type == "Unknown" for opponent_type in types)
        classified_count = sum(
            opponent_type not in {"Unknown", "Other"}
            for opponent_type in types
        )
        unknown_ratio = unknown_count / len(partners)
        return (
            unknown_ratio >= self.unknown_exploration_ratio
            or classified_count < min(self.min_strategy_classifications, len(partners))
        )

    def _day_progress(self) -> float:
        total_steps = getattr(self.awi, "n_steps", None)
        if total_steps is None:
            total_steps = getattr(self.awi, "n_days", None)
        if not total_steps:
            return 0.0
        return max(0.0, min(1.0, self.awi.current_step / max(1, int(total_steps) - 1)))

    def _small_dist_quantity_multiplier(self) -> float:
        progress = self._day_progress()
        if progress >= self.small_dist_midpoint:
            return 1.0
        extra = self.small_dist_early_quantity_multiplier - 1.0
        remaining = 1.0 - progress / max(0.01, self.small_dist_midpoint)
        return 1.0 + extra * remaining

    def _world_name(self) -> str:
        world = getattr(self.awi, "_world", None)
        if world is None:
            world = getattr(self.awi, "world", None)
        for name in ("name", "id"):
            value = getattr(world, name, None)
            if value is not None:
                return str(value)
        return ""

    def _dump_classification_log(self, event: str):
        if not self.classification_log_path:
            return
        partners = [
            partner
            for partner in self._all_partners()
            if partner in getattr(self, "negotiators", {})
        ]
        if not partners:
            return
        strategy_types = self._strategy_opponent_types(partners)
        row = {
            "event": event,
            "world": self._world_name(),
            "step": int(self.awi.current_step),
            "agent": self.id,
            "predictions": {
                partner: {
                    "strict": self.opponent_type(partner),
                    "strategy": strategy_types[partner],
                    "observations": int(self._opponent_observations[partner]),
                    "posteriors": self.opponent_posteriors(partner),
                }
                for partner in partners
            },
        }
        try:
            path = Path(self.classification_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            pass

    def _price_for_me_score(self, partner, price: float) -> float:
        """0.0 is worst for me, 1.0 is best for me."""
        issues = self._issues_for(partner)
        pmin = float(issues[UNIT_PRICE].min_value)
        pmax = float(issues[UNIT_PRICE].max_value)
        span = max(1.0, pmax - pmin)
        if self._is_seller_to(partner):
            return max(0.0, min(1.0, (float(price) - pmin) / span))
        return max(0.0, min(1.0, (pmax - float(price)) / span))

    def _price_for_opponent_score(self, partner, price: float) -> float:
        return 1.0 - self._price_for_me_score(partner, price)

    def _price_good_for_opponent(self, partner, price: float) -> bool:
        return self._price_for_opponent_score(partner, price) >= 0.5

    def _opponent_price_label(self, partner, price: float) -> str:
        score = self._price_for_opponent_score(partner, price)
        if score >= 0.80:
            return "good"
        if score <= 0.20:
            return "bad"
        return "neutral"

    def _max_lines(self) -> int:
        return max(
            1,
            int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 1)) or 1),
        )

    def _relative_time(self, states) -> float:
        values = [getattr(state, "relative_time", None) for state in states.values()]
        values = [float(value) for value in values if value is not None]
        if values:
            return max(0.0, min(1.0, min(values)))
        return 0.0

    def _round_index(self, states) -> int:
        values = []
        for state in states.values():
            for name in ("step", "current_offer_index"):
                value = getattr(state, name, None)
                if isinstance(value, int):
                    values.append(value)
                    break
        return min(values, default=0)

    def opponent_posteriors(self, partner) -> dict[str, float]:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return {"GreedyOneShotAgent": 0.0, "NonGreedy": 1.0}

        logits = self._opponent_logits[partner]
        scaled = {
            name: value / self.softmax_temperature
            for name, value in logits.items()
        }
        base = max(scaled.values())
        weights = {name: math.exp(value - base) for name, value in scaled.items()}
        total = sum(weights.values())
        if total <= 0:
            return {"GreedyOneShotAgent": 0.5, "NonGreedy": 0.5}
        return {name: value / total for name, value in weights.items()}

    def opponent_type(self, partner) -> str:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return "NonGreedy"

        if self._opponent_observations[partner] < self.min_observations:
            return "Unknown"

        posteriors = self.opponent_posteriors(partner)
        best = max(posteriors, key=posteriors.get)
        threshold = 0.51 if best == "GreedyOneShotAgent" else self.classification_threshold

        if posteriors[best] >= threshold:
            return best

        return "Unknown"

    def _best_non_greedy_type(self, partner, posteriors) -> str:
        if self._opponent_observations[partner] < self.min_observations:
            return "Unknown"
        candidates = {
            name: value
            for name, value in posteriors.items()
            if name != "GreedyOneShotAgent"
        }
        best = max(candidates, key=candidates.get)
        if best in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}:
            if candidates[best] >= 0.30:
                return best
        elif candidates[best] >= 0.40:
            return best
        return "Unknown"

    def _greedy_binary_decision(self, partner, posteriors) -> bool | None:
        if self._opponent_observations[partner] < self.min_observations:
            return None

        greedy_p = posteriors.get("GreedyOneShotAgent", 0.0)
        non_greedy_best = max(
            posteriors.get("RandomOneShotAgent", 0.0),
            posteriors.get("SyncRandomDistOneShotAgent", 0.0),
            posteriors.get("EqualDistOneShotAgent", 0.0),
            posteriors.get("Other", 0.0),
        )

        own_history = self._own_offer_result_history.get(partner, [])[-12:]
        batch_decision = self._greedy_probe_batch_decision(partner)
        if batch_decision is not None:
            return batch_decision

        good_results = [
            item for item in own_history if bool(item.get("price_good"))
        ]
        bad_results = [
            item for item in own_history if not bool(item.get("price_good"))
        ]
        good_accepts = sum(item["accepted"] for item in good_results)
        bad_accepts = sum(item["accepted"] for item in bad_results)
        good_trials = len(good_results)
        bad_trials = len(bad_results)
        early_good_results = [
            item for item in good_results if float(item.get("relative_time", 1.0)) < 0.5
        ]
        early_bad_results = [
            item for item in bad_results if float(item.get("relative_time", 1.0)) < 0.5
        ]
        early_bad_rejections = sum(not item["accepted"] for item in early_bad_results)
        early_bad_accepts = sum(item["accepted"] for item in early_bad_results)
        equal_like_history = self._equal_like_offer_history(partner)
        if (
            early_bad_accepts >= 2
            and good_accepts == 0
            and non_greedy_best >= greedy_p * 1.20
        ):
            return False
        if (
            bad_accepts >= 2
            and good_accepts == 0
            and non_greedy_best >= greedy_p * 1.20
        ):
            return False
        if len(good_results) >= 1 and len(bad_results) >= 1:
            good_accept_rate = sum(item["accepted"] for item in good_results) / len(good_results)
            bad_accept_rate = sum(item["accepted"] for item in bad_results) / len(bad_results)
            if (
                bad_accept_rate >= 0.50
                and good_accept_rate < 0.50
                and non_greedy_best >= greedy_p * 1.20
            ):
                return False
        offer_history = self._opponent_offer_history.get(partner, [])[-8:]
        early_offers = [
            item
            for item in offer_history
            if float(item.get("time", 1.0)) < 0.5
        ]
        if early_offers:
            early_good_ratio = sum(bool(item["price_good"]) for item in early_offers) / len(early_offers)
            early_quantities = [max(0, int(item["quantity"])) for item in early_offers]
            early_small_ratio = sum(quantity <= 3 for quantity in early_quantities) / len(early_quantities)
            early_equal_like = (
                early_small_ratio >= 0.70
                and sum(early_quantities) / len(early_quantities) <= 3.5
            )
            if early_good_ratio <= 0.50 and non_greedy_best >= greedy_p * 0.90:
                return False
            if len(early_offers) >= 2 and not early_equal_like:
                if (
                    early_bad_rejections >= 1
                    and early_bad_accepts == 0
                    and greedy_p >= 0.18
                    and greedy_p >= non_greedy_best * 0.50
                ):
                    return True
                if (
                    len(early_offers) >= 3
                    and greedy_p >= 0.25
                    and greedy_p >= non_greedy_best * 0.85
                ):
                    return True
        if len(offer_history) >= 4:
            price_goods = [bool(item["price_good"]) for item in offer_history]
            quantities = [max(0, int(item["quantity"])) for item in offer_history]
            good_ratio = sum(price_goods) / len(price_goods)
            price_flip_ratio = sum(
                previous != current
                for previous, current in zip(price_goods, price_goods[1:], strict=False)
            ) / max(1, len(price_goods) - 1)
            mean_quantity = sum(quantities) / len(quantities)
            quantity_range = max(quantities) - min(quantities)
            small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
            equal_like_quantity = (
                small_ratio >= 0.70
                and mean_quantity <= 3.5
            )
            early = price_goods[: max(1, len(price_goods) // 2)]
            late = price_goods[max(1, len(price_goods) // 2) :]
            early_good_ratio = sum(early) / len(early)
            late_good_ratio = sum(late) / len(late)

            if (
                good_ratio >= 0.75
                and price_flip_ratio <= 0.25
                and not equal_like_quantity
                and greedy_p >= 0.35
                and greedy_p >= non_greedy_best * 0.85
            ):
                return True
            if (
                early_good_ratio >= 0.75
                and late_good_ratio <= 0.50
                and not equal_like_quantity
                and greedy_p >= 0.30
                and greedy_p >= non_greedy_best * 0.80
            ):
                return True

            if (
                price_flip_ratio >= 0.45
                and 0.25 <= good_ratio <= 0.75
                and non_greedy_best >= greedy_p * 0.90
            ):
                return False
            if good_ratio <= 0.45 and non_greedy_best >= greedy_p * 0.80:
                return False

        if greedy_p >= 0.82 and greedy_p >= non_greedy_best * 1.20:
            return True
        if non_greedy_best >= 0.60 and greedy_p <= non_greedy_best * 0.45:
            return False
        return None

    def _greedy_probe_batch_decision(self, partner) -> bool | None:
        probes = [
            item
            for item in self._own_offer_result_history.get(partner, [])
            if float(item.get("relative_time", 1.0)) < 0.5
            and bool(item.get("initial", False))
        ][-6:]
        if len(probes) < 6:
            return None

        good = [item for item in probes if bool(item.get("price_good"))]
        bad = [item for item in probes if not bool(item.get("price_good"))]
        if len(good) < 2 or len(bad) < 2:
            return None

        good_accept_rate = sum(item["accepted"] for item in good) / len(good)
        bad_accept_rate = sum(item["accepted"] for item in bad) / len(bad)

        if bad_accept_rate == 0.0 and good_accept_rate >= 0.67:
            return True
        if bad_accept_rate >= 0.34:
            return False
        if 0.25 < good_accept_rate < 0.75 and 0.0 < bad_accept_rate < 0.75:
            return False
        return None

    def _equal_like_offer_history(self, partner) -> bool:
        history = self._opponent_offer_history.get(partner, [])[-8:]
        if len(history) < 4:
            return False
        quantities = [max(0, int(item["quantity"])) for item in history]
        mean_quantity = sum(quantities) / len(quantities)
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
        return small_ratio >= 0.70 and mean_quantity <= 3.5

    def _greedy_override_type(self, partner, posteriors) -> str | None:
        if self._opponent_observations[partner] < self.min_observations:
            return None
        greedy_p = posteriors.get("GreedyOneShotAgent", 0.0)
        sync_p = posteriors.get("SyncRandomDistOneShotAgent", 0.0)
        equal_p = posteriors.get("EqualDistOneShotAgent", 0.0)

        offer_history = self._opponent_offer_history.get(partner, [])[-8:]
        own_history = self._own_offer_result_history.get(partner, [])[-12:]
        offer_score = 0

        if offer_history:
            price_goods = [bool(item["price_good"]) for item in offer_history]
            good_ratio = sum(price_goods) / len(price_goods)
            price_flip_ratio = sum(
                previous != current
                for previous, current in zip(price_goods, price_goods[1:], strict=False)
            ) / max(1, len(price_goods) - 1)
            if good_ratio >= 0.75 and price_flip_ratio <= 0.25:
                offer_score += 1
            if (
                len(price_goods) >= 4
                and sum(price_goods[: len(price_goods) // 2]) / max(1, len(price_goods) // 2)
                >= 0.75
                and sum(price_goods[len(price_goods) // 2 :])
                / max(1, len(price_goods) - len(price_goods) // 2)
                <= 0.50
            ):
                offer_score += 1

        good_results = [
            item for item in own_history if bool(item.get("price_good"))
        ]
        bad_results = [
            item for item in own_history if not bool(item.get("price_good"))
        ]
        accept_score = 0
        if len(good_results) >= 2 and len(bad_results) >= 1:
            good_accept_rate = sum(item["accepted"] for item in good_results) / len(good_results)
            bad_accept_rate = sum(item["accepted"] for item in bad_results) / len(bad_results)
            if good_accept_rate >= 0.65 and bad_accept_rate <= 0.35:
                accept_score += 2

        if (
            accept_score >= 2
            and greedy_p >= 0.30
            and greedy_p >= sync_p * 0.75
            and greedy_p >= equal_p * 0.80
        ):
            return "GreedyOneShotAgent"
        if (
            accept_score >= 1
            and greedy_p >= 0.60
            and greedy_p >= sync_p * 0.90
            and greedy_p >= equal_p * 0.90
            and offer_score >= 1
        ):
            return "GreedyOneShotAgent"
        return None

    def _best_behavior_type(self, partner) -> tuple[str, float]:
        posteriors = self.opponent_posteriors(partner)
        candidates = dict(posteriors)
        best = max(candidates, key=candidates.get)
        return best, candidates[best]

    def _strategy_opponent_types(self, partners: list[str]) -> dict[str, str]:
        return {partner: self.opponent_type(partner) for partner in partners}

    def _ensure_min_dist_strategy_types(self, partners, types):
        return dict(types)

    def _add_evidence(self, partner, weights: dict[str, float], strength: float = 1.0):
        del weights, strength
        self._ensure_partner(partner)

    def _add_logit_evidence(
        self,
        partner,
        *,
        greedy: float = 0.0,
        non_greedy: float = 0.0,
        reason: str,
    ):
        self._ensure_partner(partner)
        self._opponent_logits[partner]["GreedyOneShotAgent"] += float(greedy)
        self._opponent_logits[partner]["NonGreedy"] += float(non_greedy)
        self._opponent_observations[partner] += 1
        self._logit_history[partner].append(
            {
                "step": int(self.awi.current_step),
                "greedy_delta": float(greedy),
                "non_greedy_delta": float(non_greedy),
                "logits": dict(self._opponent_logits[partner]),
                "reason": reason,
            }
        )
        if len(self._logit_history[partner]) > 50:
            del self._logit_history[partner][:-50]

    def _add_evidence_count(self, partner, name: str):
        self._ensure_partner(partner)
        self._evidence_counts[partner][name] += 1

    def _veto_non_greedy(self, partner, reason: str):
        self._ensure_partner(partner)
        if partner not in self._non_greedy_veto:
            self._non_greedy_veto[partner] = reason
        self._add_logit_evidence(
            partner,
            non_greedy=8.0,
            reason=reason,
        )

    def _record_sent_offers(
        self,
        proposals,
        initial: bool = True,
        relative_time: float = 0.0,
    ):
        for partner, offer in proposals.items():
            self._record_sent_offer(
                partner,
                offer,
                initial=initial,
                relative_time=relative_time,
            )

    def _record_response_offers(self, responses, relative_time: float = 0.0):
        for partner, response in responses.items():
            if response is None or response.response != ResponseType.REJECT_OFFER:
                continue
            self._record_sent_offer(
                partner,
                response.outcome,
                initial=False,
                relative_time=relative_time,
            )

    def _record_sent_offer(
        self,
        partner,
        offer,
        initial: bool = True,
        relative_time: float = 0.0,
    ):
        if partner is None or offer is None or len(offer) <= UNIT_PRICE:
            return
        self._ensure_partner(partner)
        opponent_type = self.opponent_type(partner)
        price_label = self._opponent_price_label(partner, offer[UNIT_PRICE])
        self._sent_offer_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": max(0.0, min(1.0, float(relative_time))),
                "initial": bool(initial),
                "partner": partner,
                "end_response_observed": False,
                "first_result_observed": False,
                "offer": tuple(offer),
                "price_label": price_label,
                "price_good": self._price_good_for_opponent(partner, offer[UNIT_PRICE]),
                "non_greedy_initial_probe": initial
                and opponent_type
                in {
                    "RandomOneShotAgent",
                    "SyncRandomDistOneShotAgent",
                    "EqualDistOneShotAgent",
                    "Other",
                    "Unknown",
                    "NonGreedy",
                },
            }
        )
        if initial:
            self._first_offer_trials_by_price[partner][price_label] += 1
        if len(self._sent_offer_history[partner]) > 20:
            del self._sent_offer_history[partner][:-20]

    def _contract_partner(self, contract: Contract):
        for partner in getattr(contract, "partners", ()):
            if partner != self.id:
                return partner
        return None

    def _contract_outcome(self, contract: Contract) -> Outcome | None:
        agreement = getattr(contract, "agreement", None)
        if agreement is None:
            return None
        if isinstance(agreement, dict):
            quantity = agreement.get(QUANTITY, agreement.get("quantity"))
            delivery_step = agreement.get(TIME, agreement.get("time"))
            unit_price = agreement.get(UNIT_PRICE, agreement.get("unit_price"))
            if quantity is None or delivery_step is None or unit_price is None:
                return None
            return (quantity, delivery_step, unit_price)
        if len(agreement) <= UNIT_PRICE:
            return None
        return tuple(agreement)

    def _same_offer(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return (
            int(left[QUANTITY]) == int(right[QUANTITY])
            and int(left[TIME]) == int(right[TIME])
            and int(left[UNIT_PRICE]) == int(right[UNIT_PRICE])
        )

    def _same_quantity_offer(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return int(left[QUANTITY]) == int(right[QUANTITY])

    def _latest_sent_offer(self, partner):
        history = self._sent_offer_history.get(partner, [])
        current = [
            item
            for item in history
            if item["step"] == self.awi.current_step
        ]
        return current[-1] if current else None

    def _matching_sent_offer(self, partner, outcome):
        history = self._sent_offer_history.get(partner, [])
        for item in reversed(history):
            if item["step"] != self.awi.current_step:
                continue
            if self._same_offer(item["offer"], outcome):
                return item
        return None

    def _latest_unobserved_first_offer(self, partner):
        for item in reversed(self._sent_offer_history.get(partner, [])):
            if item["step"] != self.awi.current_step:
                continue
            if not item.get("initial", False):
                continue
            if item.get("first_result_observed", False):
                continue
            return item
        return None

    def _bayes_observe_first_offer_classification_result(
        self,
        partner,
        sent_offer,
        accepted: bool,
    ):
        if sent_offer is None or sent_offer.get("first_result_observed", False):
            return

        sent_offer["first_result_observed"] = True
        price_label = sent_offer.get("price_label", "neutral")
        accepted = bool(accepted)

        if accepted:
            self._first_offer_accepts_by_price[partner][price_label] += 1

        if price_label == "bad" and accepted:
            self._good_first_offer_rejection_streak[partner] = 0
            self._add_evidence_count(partner, "bad_first_offer_accepted")
            self._veto_non_greedy(partner, "bad_first_offer_accepted")
        elif price_label == "bad" and not accepted:
            self._good_first_offer_rejection_streak[partner] = 0
            self._add_evidence_count(partner, "bad_first_offer_rejected")
            self._add_logit_evidence(
                partner,
                greedy=0.08,
                reason="bad_first_offer_rejected",
            )
        elif price_label == "good" and accepted:
            self._good_first_offer_rejection_streak[partner] = 0
            self._add_evidence_count(partner, "good_first_offer_accepted")
            self._add_logit_evidence(
                partner,
                greedy=1.20,
                reason="good_first_offer_accepted",
            )
        elif price_label == "good":
            self._good_first_offer_rejection_streak[partner] += 1
            self._add_evidence_count(partner, "good_first_offer_rejected")
            if (
                self._evidence_counts[partner]["good_first_offer_rejected"] >= 3
                and not self._good_first_offer_cumulative_rejection_observed[partner]
            ):
                self._good_first_offer_cumulative_rejection_observed[partner] = True
                self._add_logit_evidence(
                    partner,
                    greedy=-2.50,
                    non_greedy=1.00,
                    reason="good_first_offer_rejected_cumulative_3",
                )
            if self._good_first_offer_rejection_streak[partner] >= 2:
                self._add_logit_evidence(
                    partner,
                    greedy=-2.00,
                    non_greedy=0.75,
                    reason="good_first_offer_rejected_streak",
                )
            else:
                self._add_logit_evidence(
                    partner,
                    non_greedy=0.15,
                    reason="good_first_offer_rejected",
                )
        elif accepted:
            self._good_first_offer_rejection_streak[partner] = 0
            self._add_logit_evidence(
                partner,
                greedy=0.05,
                reason="neutral_first_offer_accepted",
            )
        else:
            self._good_first_offer_rejection_streak[partner] = 0
            self._add_logit_evidence(
                partner,
                non_greedy=0.05,
                reason="neutral_first_offer_rejected",
            )

    def _observe_own_first_offer_counter(self, partner, sent_offer, counter_offer):
        if sent_offer is None or sent_offer.get("first_result_observed", False):
            return

        price_label = sent_offer.get("price_label", "neutral")
        self._first_offer_reoffers_by_price[partner][price_label] += 1
        self._first_offer_counter_quantities_by_price[partner][price_label].append(
            max(0, int(counter_offer[QUANTITY]))
        )
        quantities = self._first_offer_counter_quantities_by_price[partner][price_label]
        if len(quantities) > 50:
            del quantities[:-50]

        self._observe_first_offer_classification_result(
            partner,
            sent_offer,
            accepted=False,
        )

    def _observe_received_first_offer(self, partner, offer):
        if offer is None or len(offer) <= UNIT_PRICE:
            return

        self._ensure_partner(partner)
        if self._received_offer_counts[partner] > 0:
            self._received_offer_counts[partner] += 1
            return

        self._received_offer_counts[partner] += 1
        price_label = self._opponent_price_label(partner, offer[UNIT_PRICE])

        if price_label == "good":
            self._add_evidence_count(partner, "opponent_first_offer_good_price")
            self._add_logit_evidence(
                partner,
                greedy=0.03,
                reason="opponent_first_offer_good_price",
            )
        elif price_label == "bad":
            self._add_evidence_count(partner, "opponent_first_offer_bad_price")
            self._veto_non_greedy(partner, "opponent_first_offer_bad_price")

        self._received_first_offer_history[partner].append(
            {
                "step": int(self.awi.current_step),
                "quantity": int(offer[QUANTITY]),
                "price_label": price_label,
            }
        )

        if len(self._received_first_offer_history[partner]) > 30:
            del self._received_first_offer_history[partner][:-30]

    def _partner_ended_after_sent_offer(self, sent_offer, state) -> bool:
        if sent_offer is None or state is None:
            return False
        if not bool(getattr(state, "broken", False)):
            return False
        if bool(getattr(state, "timedout", False)) or bool(getattr(state, "has_error", False)):
            return False
        if getattr(state, "agreement", None) is not None:
            return False
        return self._same_offer(getattr(state, "current_offer", None), sent_offer["offer"])

    def _observe_partner_end_response(self, partner, sent_offer, ended: bool):
        if sent_offer is None or sent_offer.get("end_response_observed", False):
            return
        sent_offer["end_response_observed"] = True

        history = self._own_offer_end_history[partner]
        history.append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "ended": bool(ended),
            }
        )
        if len(history) > 20:
            del history[:-20]
        self._observe_partner_end_pattern(partner)

    def _observe_partner_end_pattern(self, partner):
        return
        history = self._own_offer_end_history[partner][-10:]
        if len(history) < 4:
            return
        ended_count = sum(item["ended"] for item in history)
        end_rate = sum(item["ended"] for item in history) / len(history)
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if ended_count >= 3 and end_rate >= 0.30:
            evidence["RandomOneShotAgent"] -= 0.90
            evidence["SyncRandomDistOneShotAgent"] += 0.08
            evidence["EqualDistOneShotAgent"] += 0.05
        elif ended_count >= 2 and end_rate >= 0.20:
            evidence["RandomOneShotAgent"] -= 0.45
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.4)

    def _observe_own_offer_result(self, partner, sent_offer, accepted: bool):
        if sent_offer is None:
            return
        if sent_offer.get("initial", False):
            self._observe_first_offer_classification_result(
                partner,
                sent_offer,
                accepted=accepted,
            )
        price_good = bool(sent_offer["price_good"])
        self._own_offer_result_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "initial": bool(sent_offer.get("initial", False)),
                "price_good": price_good,
                "accepted": bool(accepted),
            }
        )
        if len(self._own_offer_result_history[partner]) > 30:
            del self._own_offer_result_history[partner][:-30]
        return

        early = float(sent_offer.get("relative_time", 1.0)) < 0.5
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if accepted and price_good:
            evidence["GreedyOneShotAgent"] += 0.45 if early else 1.00
            evidence["RandomOneShotAgent"] -= 0.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.10
            evidence["EqualDistOneShotAgent"] -= 0.05
        elif not accepted and not price_good:
            pass
        elif accepted and not price_good:
            evidence["RandomOneShotAgent"] += 0.35 if early else 0.25
            evidence["SyncRandomDistOneShotAgent"] += 0.20 if early else 0.12
            evidence["EqualDistOneShotAgent"] += 0.15 if early else 0.10
        elif not accepted and price_good:
            # Greedy can reject good-price offers when the quantity is unsuitable.
            evidence["GreedyOneShotAgent"] -= 0.10 if early else 0.45
            evidence["Other"] += 0.05
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.6)
        self._own_offer_result_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "initial": bool(sent_offer.get("initial", False)),
                "price_good": price_good,
                "accepted": bool(accepted),
            }
        )
        if len(self._own_offer_result_history[partner]) > 30:
            del self._own_offer_result_history[partner][:-30]
        self._observe_own_offer_result_pattern(partner)
        self._observe_initial_good_price_rejection_streak(partner)

    def _observe_own_offer_result_pattern(self, partner):
        return
        history = self._own_offer_result_history[partner][-12:]
        good = [item for item in history if item["price_good"]]
        if len(good) < 3:
            return
        good_accept_rate = sum(item["accepted"] for item in good) / len(good)
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if good_accept_rate >= 0.67:
            evidence["GreedyOneShotAgent"] += 1.40
            evidence["RandomOneShotAgent"] -= 0.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.25
            evidence["EqualDistOneShotAgent"] -= 0.10
        elif good_accept_rate <= 0.25:
            evidence["GreedyOneShotAgent"] -= 0.90
            evidence["RandomOneShotAgent"] += 0.20
            evidence["SyncRandomDistOneShotAgent"] += 0.25
            evidence["EqualDistOneShotAgent"] += 0.15
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.8)

    def _observe_initial_good_price_rejection_streak(self, partner):
        return
        history = self._own_offer_result_history[partner]
        streak = 0
        for item in reversed(history):
            if not item.get("initial", False):
                break
            if not item.get("price_good", False) or item.get("accepted", False):
                break
            streak += 1

        if streak < 2:
            return
        if streak <= self._initial_good_rejection_streak_observed[partner]:
            return
        self._initial_good_rejection_streak_observed[partner] = streak

        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        evidence["GreedyOneShotAgent"] -= 0.65 if streak == 2 else 1.00
        evidence["RandomOneShotAgent"] += 0.12
        evidence["SyncRandomDistOneShotAgent"] += 0.15
        evidence["EqualDistOneShotAgent"] += 0.10
        self._add_evidence(partner, evidence, strength=0.7)

    def _renormalize_logp(self, partner):
        logp = self._opponent_logp[partner]
        center = max(logp.values())
        for name in logp:
            logp[name] -= center

    def _observe_offer(self, partner, offer: Outcome, states):
        if offer is None:
            return
        self._ensure_partner(partner)
        if len(offer) <= UNIT_PRICE:
            return

        quantity = int(offer[QUANTITY])
        delivery_step = offer[TIME]
        price = float(offer[UNIT_PRICE])
        t = self._relative_time(states)
        round_index = self._round_index(states)
        price_good = self._price_good_for_opponent(partner, price)
        history = self._opponent_offer_history[partner]
        history.append(
            {
                "step": self.awi.current_step,
                "round": round_index,
                "time": t,
                "quantity": quantity,
                "price": price,
                "price_good": price_good,
                "price_label": "good" if price_good else "bad",
            }
        )
        if len(history) > 30:
            del history[:-30]
        self._observe_binary_offer_history(partner)
        return

    def _add_history_pattern_logit(
        self,
        partner,
        *,
        greedy: float = 0.0,
        non_greedy: float = 0.0,
        reason: str,
    ):
        key = (int(self.awi.current_step), reason)
        if key in self._history_pattern_observed[partner]:
            return
        self._history_pattern_observed[partner].add(key)
        self._add_evidence_count(partner, reason)
        self._add_logit_evidence(
            partner,
            greedy=greedy,
            non_greedy=non_greedy,
            reason=reason,
        )

    def _observe_binary_offer_history(self, partner):
        history = self._opponent_offer_history[partner][-8:]
        if len(history) < 4:
            return

        quantities = [max(0, int(item["quantity"])) for item in history]
        price_goods = [bool(item["price_good"]) for item in history]
        if not quantities:
            return

        issues = self._issues_for(partner)
        qmax = max(1, int(issues[QUANTITY].max_value))
        mean = sum(quantities) / len(quantities)
        variance = sum((quantity - mean) ** 2 for quantity in quantities) / len(quantities)
        coefficient = math.sqrt(variance) / max(1.0, mean)
        mean_ratio = mean / qmax
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= max(2, math.ceil(0.35 * qmax)) for quantity in quantities) / len(quantities)
        good_ratio = sum(price_goods) / len(price_goods)
        price_flip_ratio = sum(
            previous != current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        ) / max(1, len(price_goods) - 1)

        early = history[: max(1, len(history) // 2)]
        late = history[max(1, len(history) // 2) :]
        early_good_ratio = sum(bool(item["price_good"]) for item in early) / len(early)
        late_good_ratio = sum(bool(item["price_good"]) for item in late) / len(late)

        stable_quantity = coefficient <= 0.30 or quantity_range <= 1
        if small_ratio >= 0.70 and stable_quantity:
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.70,
                reason="history_small_stable_quantity",
            )
        elif (
            small_ratio >= 0.60
            and coefficient >= 0.45
            and (good_ratio <= 0.65 or price_flip_ratio >= 0.30)
        ):
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.40,
                reason="history_small_variable_quantity",
            )

        if price_flip_ratio >= 0.40 and coefficient >= 0.35 and good_ratio < 0.90:
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.35,
                reason="history_random_like_flips",
            )

        if good_ratio >= 0.75 and mean_ratio >= 0.55 and stable_quantity:
            self._add_history_pattern_logit(
                partner,
                greedy=0.35,
                reason="history_large_selfish_stable",
            )

        if early_good_ratio >= 0.75 and late_good_ratio <= early_good_ratio - 0.35:
            self._add_history_pattern_logit(
                partner,
                greedy=0.85,
                reason="history_selfish_then_concedes",
            )

    def _observe_history_pattern(self, partner):
        return
        history = self._opponent_offer_history[partner]
        if len(history) < 4:
            return

        recent = history[-8:]
        quantities = [max(0, int(item["quantity"])) for item in recent]
        price_goods = [bool(item["price_good"]) for item in recent]
        if not quantities or sum(quantities) <= 0:
            return

        issues = self._issues_for(partner)
        qmax = max(1, int(issues[QUANTITY].max_value))
        mean = sum(quantities) / len(quantities)
        variance = sum((quantity - mean) ** 2 for quantity in quantities) / len(quantities)
        coefficient = math.sqrt(variance) / max(1.0, mean)
        mean_ratio = mean / qmax
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
        extreme_ratio = sum(quantity >= 0.8 * qmax for quantity in quantities) / len(quantities)
        good_ratio = sum(price_goods) / len(price_goods)
        good_to_bad = sum(
            previous and not current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        )
        bad_to_good = sum(
            (not previous) and current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        )
        price_flip_ratio = sum(
            previous != current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        ) / max(1, len(price_goods) - 1)
        early = recent[: max(1, len(recent) // 2)]
        late = recent[max(1, len(recent) // 2) :]
        early_good_ratio = sum(bool(item["price_good"]) for item in early) / len(early)
        late_good_ratio = sum(bool(item["price_good"]) for item in late) / len(late)
        last_two_bad = len(price_goods) >= 2 and not price_goods[-1] and not price_goods[-2]
        previous_good_streak = len(price_goods) >= 4 and all(price_goods[-4:-2])

        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        stable_quantity = coefficient <= 0.25 or quantity_range <= 1
        if small_ratio >= 0.70 and mean <= 3.5 and stable_quantity:
            if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
                evidence["EqualDistOneShotAgent"] += 0.75
                evidence["RandomOneShotAgent"] -= 0.20
                evidence["GreedyOneShotAgent"] -= 0.65
            else:
                evidence["EqualDistOneShotAgent"] += 1.25
                evidence["SyncRandomDistOneShotAgent"] -= 0.20
                evidence["GreedyOneShotAgent"] -= 0.10
                evidence["RandomOneShotAgent"] -= 0.20
        elif small_ratio >= 0.70 and (coefficient >= 0.40 or quantity_range >= 3):
            evidence["SyncRandomDistOneShotAgent"] += 0.45 if good_ratio >= 0.65 else 1.00
            evidence["EqualDistOneShotAgent"] -= 0.45
        elif extreme_ratio >= 0.25:
            if good_ratio >= 0.65 and price_flip_ratio <= 0.35:
                evidence["GreedyOneShotAgent"] += 0.65
                evidence["RandomOneShotAgent"] -= 0.10
            else:
                evidence["RandomOneShotAgent"] += 0.75
                evidence["GreedyOneShotAgent"] += 0.05
            evidence["EqualDistOneShotAgent"] -= 0.50
            evidence["SyncRandomDistOneShotAgent"] -= 0.55
        elif stable_quantity and mean_ratio <= 0.65:
            evidence["EqualDistOneShotAgent"] += 0.80
            evidence["SyncRandomDistOneShotAgent"] -= 0.05
            evidence["GreedyOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.10
        elif mean_ratio <= 0.65:
            evidence["SyncRandomDistOneShotAgent"] += 0.25

        if coefficient >= 0.55:
            evidence["SyncRandomDistOneShotAgent"] += 0.30 if good_ratio < 0.75 else -0.10
            evidence["GreedyOneShotAgent"] += 0.10
            evidence["EqualDistOneShotAgent"] -= 0.40
            if good_ratio >= 0.75 and price_flip_ratio <= 0.25:
                evidence["GreedyOneShotAgent"] += 0.45
                evidence["SyncRandomDistOneShotAgent"] -= 0.10
            if extreme_ratio >= 0.20 and 0.20 <= good_ratio <= 0.80:
                evidence["RandomOneShotAgent"] += 0.20 if good_ratio >= 0.65 else 0.75
                evidence["SyncRandomDistOneShotAgent"] -= 0.45

        quantity_unstable = coefficient >= 0.45 or quantity_range >= 3
        if (
            good_to_bad > 0
            and bad_to_good > 0
            and 0.20 <= good_ratio <= 0.80
            and quantity_unstable
        ):
            if extreme_ratio >= 0.20:
                evidence["RandomOneShotAgent"] += 0.65
                evidence["SyncRandomDistOneShotAgent"] -= 0.20
            else:
                evidence["RandomOneShotAgent"] += 0.25
                evidence["SyncRandomDistOneShotAgent"] += 0.65
            evidence["EqualDistOneShotAgent"] += 0.10
            evidence["GreedyOneShotAgent"] -= 0.45
        elif good_to_bad > 0 and bad_to_good > 0 and 0.20 <= good_ratio <= 0.80:
            evidence["RandomOneShotAgent"] += 0.10
            evidence["SyncRandomDistOneShotAgent"] += 0.40
            evidence["EqualDistOneShotAgent"] += 0.10
        elif price_flip_ratio >= 0.35 and good_ratio < 0.90:
            evidence["SyncRandomDistOneShotAgent"] += 0.25 if good_ratio < 0.75 else -0.05
            evidence["EqualDistOneShotAgent"] += 0.25
            evidence["RandomOneShotAgent"] += 0.08
            evidence["GreedyOneShotAgent"] -= 0.20
        if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
            evidence["GreedyOneShotAgent"] += 0.85
            evidence["EqualDistOneShotAgent"] -= 0.35
            evidence["RandomOneShotAgent"] -= 0.20
            evidence["SyncRandomDistOneShotAgent"] -= 0.20
        if early_good_ratio >= 0.75 and late_good_ratio <= early_good_ratio - 0.35:
            evidence["GreedyOneShotAgent"] += 1.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.15
            evidence["EqualDistOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.25
        if previous_good_streak and last_two_bad and bad_to_good == 0:
            evidence["GreedyOneShotAgent"] += 0.75
            evidence["SyncRandomDistOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.20

        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.6)

    def _exploration_first_proposals(self, base_proposals):
        proposals = {}
        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue
            proposals.update(self._equal_dist_exploration_proposals(int(needs), partners))

        if proposals:
            return proposals
        return base_proposals

    def _equal_dist_exploration_proposals(self, needs: int, partners: list[str]):
        proposals = {partner: None for partner in partners}
        n = len(partners)
        multiplier = self._exploration_quantity_multiplier_for(partners)
        target_quantity = max(0, math.ceil(int(needs) * multiplier))
        if target_quantity <= 0 or n <= 0:
            return proposals

        if target_quantity < n:
            for index, partner in enumerate(partners[:target_quantity]):
                proposals[partner] = self._offer(
                    partner,
                    1,
                    self._exploration_probe_price(partner, index),
                )
            return proposals

        base = max(1, target_quantity // n)
        if target_quantity >= 2 * n:
            base = max(base, 2)
        base = min(3, base)
        remainder = max(0, target_quantity - base * n)

        for index, partner in enumerate(partners):
            quantity = base
            if remainder > 0:
                extra = min(remainder, max(0, 3 - quantity))
                quantity += extra
                remainder -= extra
            proposals[partner] = self._offer(
                partner,
                self._clamp_quantity(partner, quantity),
                self._exploration_probe_price(partner, index),
            )
        return proposals

    def _exploration_quantity_multiplier_for(self, partners: list[str]) -> float:
        count = len(partners)
        if count >= 6:
            return 1.5
        if count == 5:
            return 1.4
        return self.exploration_quantity_multiplier

    def _exploration_probe_price(self, partner, index: int) -> int:
        # Alternate between prices that are good and bad for the opponent.  The
        # accept/reject pattern is especially informative for Greedy agents.
        opponent_good = (self.awi.current_step + index) % 2 == 0
        return self._worst_price_for_me(partner) if opponent_good else self._best_price_for_me(partner)

    def counter_all(self, offers, states):
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
        }
        for partner, offer in current_offers.items():
            if len(offer) <= UNIT_PRICE or offer[TIME] != self.awi.current_step:
                continue
            sent_offer = self._latest_unobserved_first_offer(partner)
            if sent_offer is not None and not self._same_offer(sent_offer["offer"], offer):
                self._observe_own_first_offer_counter(
                    partner,
                    sent_offer,
                    offer,
                )
            self._observe_partner_end_response(
                partner,
                self._latest_sent_offer(partner),
                ended=False,
            )
            self._observe_received_first_offer(partner, offer)
            self._observe_offer(partner, offer, states)

        return self._current_offer_responses(offers, states)

    def _greedy_only_first_proposals(self, needs: int, partners: list[str]):
        proposals = {partner: None for partner in partners}
        if needs <= 0:
            return proposals

        opponent_types = self._ensure_min_dist_strategy_types(
            partners,
            {partner: self.opponent_type(partner) for partner in partners},
        )
        greedy_partners = [
            partner
            for partner in partners
            if opponent_types[partner] == "GreedyOneShotAgent"
        ]
        greedy_partners.sort(
            key=lambda partner: self.opponent_posteriors(partner).get(
                "GreedyOneShotAgent",
                0.0,
            ),
            reverse=True,
        )
        success_scaled_partners = self._success_scaled_partners(
            partners,
            opponent_types,
        )

        success_rate = self._non_greedy_initial_offer_success_rate()
        selected_greedy_partners = []
        strong_greedy_partners = [
            partner for partner in greedy_partners if self._strong_initial_greedy_partner(partner)
        ]
        if len(greedy_partners) >= 2 and len(strong_greedy_partners) >= 2:
            selected_greedy_partners = strong_greedy_partners[:2]
            greedy_quantities = [
                math.ceil(needs / 2),
                math.floor(needs / 2),
            ]
            for greedy_partner, greedy_quantity in zip(
                selected_greedy_partners,
                greedy_quantities,
                strict=False,
            ):
                proposals[greedy_partner] = self._offer(
                    greedy_partner,
                    greedy_quantity,
                    self._worst_price_for_me(greedy_partner),
                )
            return proposals

        if greedy_partners:
            if len(greedy_partners) >= 2:
                selected_greedy_partners = greedy_partners[:2]
                greedy_quantities = self._split_greedy_eighty_quantities(needs)
                scaled_target = self._success_adjusted_quantity(
                    needs * 0.2,
                    success_rate,
                )
            else:
                selected_greedy_partners = greedy_partners[:1]
                greedy_quantity = min(7, int(needs))
                greedy_quantities = [greedy_quantity]
                scaled_target = self._success_adjusted_quantity(
                    max(0, int(needs) - greedy_quantity),
                    success_rate,
                )

            for greedy_partner, greedy_quantity in zip(
                selected_greedy_partners,
                greedy_quantities,
                strict=False,
            ):
                proposals[greedy_partner] = self._offer(
                    greedy_partner,
                    greedy_quantity,
                    self._worst_price_for_me(greedy_partner),
                )
        else:
            scaled_target = self._success_adjusted_quantity(needs, success_rate)

        if len(greedy_partners) >= 3:
            for greedy_partner in greedy_partners[2:]:
                if greedy_partner not in success_scaled_partners:
                    success_scaled_partners.append(greedy_partner)

        scaled_target = self._buyer_favorable_scaled_target_cap(
            scaled_target,
            needs,
            partners,
        )

        if success_scaled_partners:
            scaled_target = self._minimum_two_dist_quantity(
                scaled_target,
                success_scaled_partners,
                opponent_types,
            )
            if self._is_process_one_agent():
                self._assign_success_weighted_quantities(
                    proposals,
                    success_scaled_partners,
                    scaled_target,
                    price_getter=self._best_price_for_me,
                    quantity_caps=self._half_quantity_caps(
                        int(needs),
                        len(success_scaled_partners),
                    ),
                )
            else:
                self._assign_equal_quantities(
                    proposals,
                    success_scaled_partners,
                    scaled_target,
                    price_getter=self._best_price_for_me,
                )
        elif greedy_partners and scaled_target > 0:
            scaled_target = max(
                int(scaled_target),
                min(2, len(selected_greedy_partners)),
            )
            self._add_equal_quantities(
                proposals,
                selected_greedy_partners,
                scaled_target,
                price_getter=self._worst_price_for_me,
            )
        return proposals

    def _success_scaled_partners(self, partners, opponent_types):
        sync_equal_types = {
            "SyncRandomDistOneShotAgent",
            "EqualDistOneShotAgent",
        }
        fallback_types = {"RandomOneShotAgent", "Other", "Unknown", "NonGreedy"}

        sync_equal_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in sync_equal_types
        ]
        sync_equal_partners.sort(
            key=lambda partner: max(
                self.opponent_posteriors(partner).get(
                    "SyncRandomDistOneShotAgent",
                    0.0,
                ),
                self.opponent_posteriors(partner).get(
                    "EqualDistOneShotAgent",
                    0.0,
                ),
            ),
            reverse=True,
        )

        fallback_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in fallback_types
        ]
        return sync_equal_partners + fallback_partners

    def _minimum_two_dist_quantity(self, target_quantity, partners, opponent_types):
        if not partners:
            return 0
        if int(target_quantity) <= 0:
            return 0
        return max(int(target_quantity), min(2, len(partners)))

    def _buyer_favorable_scaled_target_cap(
        self,
        scaled_target: int,
        needs: int,
        partners: list[str],
    ) -> int:
        scaled_target = int(scaled_target)
        if not self._is_process_one_buy_side(partners):
            return scaled_target
        ratio = self._input_market_sell_buy_ratio()
        if ratio is None or ratio < self.buyer_favorable_market_ratio:
            return scaled_target
        cap = math.ceil(int(needs) * self.buyer_favorable_scaled_target_multiplier)
        return min(scaled_target, max(0, cap))

    def _success_adjusted_quantity(self, target_quantity: float, success_rate: float) -> int:
        success_rate = max(0.05, min(1.0, float(success_rate)))
        return math.ceil(target_quantity / success_rate)

    def _split_greedy_eighty_quantities(self, needs: int) -> list[int]:
        total = max(0, math.ceil(int(needs) * 0.8))
        if total <= 0:
            return [0, 0]

        difference = 2 if total % 2 == 0 else 3
        if total < difference:
            return [total, 0]

        high = (total + difference) // 2
        low = total - high
        return [high, low]

    def _half_quantity_caps(self, needs: int, count: int):
        if needs <= 0 or count <= 0:
            return []

        low = needs // 2
        high = needs - low
        if count == 1:
            return [high]
        return [low if index % 2 == 0 else high for index in range(count)]

    def _assign_equal_quantities(self, proposals, partners, target_quantity, price_getter):
        partners = list(partners)
        if not partners or target_quantity <= 0:
            return

        base = target_quantity // len(partners)
        remainder = target_quantity - base * len(partners)
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            if quantity <= 0:
                continue
            proposals[partner] = self._offer(
                partner,
                quantity,
                price_getter(partner),
            )

    def _assign_success_weighted_quantities(
        self,
        proposals,
        partners,
        target_quantity,
        price_getter,
        quantity_caps=None,
    ):
        partners = list(partners)
        if not partners or target_quantity <= 0:
            return

        target_quantity = int(target_quantity)
        if quantity_caps is not None:
            quantity_caps = [
                max(0, int(cap))
                for cap in list(quantity_caps)[: len(partners)]
            ]
            if len(quantity_caps) < len(partners):
                quantity_caps.extend([0] * (len(partners) - len(quantity_caps)))

        weights = [
            0.75 + 0.5 * self._partner_non_greedy_initial_offer_success_rate(partner)
            for partner in partners
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            self._assign_equal_quantities(proposals, partners, target_quantity, price_getter)
            return

        raw_quantities = [
            target_quantity * weight / total_weight
            for weight in weights
        ]
        quantities = []
        for index, quantity in enumerate(raw_quantities):
            assigned = math.floor(quantity)
            if quantity_caps is not None:
                assigned = min(assigned, quantity_caps[index])
            quantities.append(assigned)
        remainder = target_quantity - sum(quantities)
        order = sorted(
            range(len(partners)),
            key=lambda index: (
                raw_quantities[index] - quantities[index],
                weights[index],
            ),
            reverse=True,
        )
        while remainder > 0:
            changed = False
            for index in order:
                if (
                    quantity_caps is not None
                    and quantities[index] >= quantity_caps[index]
                ):
                    continue
                quantities[index] += 1
                remainder -= 1
                changed = True
                if remainder <= 0:
                    break
            if not changed:
                break

        for partner, quantity in zip(partners, quantities, strict=False):
            if quantity <= 0:
                continue
            proposals[partner] = self._offer(
                partner,
                quantity,
                price_getter(partner),
            )

    def _add_equal_quantities(self, proposals, partners, target_quantity, price_getter):
        partners = list(partners)
        if not partners or target_quantity <= 0:
            return

        additions = {partner: 0 for partner in partners}
        base = target_quantity // len(partners)
        remainder = target_quantity - base * len(partners)
        for index, partner in enumerate(partners):
            additions[partner] = base + (1 if index < remainder else 0)

        for partner, addition in additions.items():
            if addition <= 0:
                continue
            current = proposals.get(partner)
            current_quantity = int(current[QUANTITY]) if current is not None else 0
            proposals[partner] = self._offer(
                partner,
                current_quantity + addition,
                price_getter(partner),
            )

    def _non_greedy_initial_offer_success_rate(self) -> float:
        if not self._non_greedy_initial_offer_results:
            return self.non_greedy_success_default
        return (
            sum(self._non_greedy_initial_offer_results)
            / len(self._non_greedy_initial_offer_results)
        )

    def _partner_non_greedy_initial_offer_success_rate(self, partner) -> float:
        results = self._partner_non_greedy_initial_offer_results.get(partner, [])
        if not results:
            return self._non_greedy_initial_offer_success_rate()
        return sum(results) / len(results)

    def _strong_initial_greedy_partner(self, partner) -> bool:
        greedy_probability = self.opponent_posteriors(partner).get(
            "GreedyOneShotAgent",
            0.0,
        )
        if greedy_probability < 0.80:
            return False

        streak = 0
        for item in reversed(self._own_offer_result_history.get(partner, [])):
            if not item.get("initial", False):
                continue
            if not item.get("accepted", False):
                break
            streak += 1
            if streak >= 3:
                return True
        return False

    def _record_non_greedy_initial_offer_result(self, sent_offer, accepted: bool):
        if not sent_offer or not sent_offer.get("non_greedy_initial_probe"):
            return
        partner = sent_offer.get("partner")
        self._non_greedy_initial_offer_results.append(bool(accepted))
        if len(self._non_greedy_initial_offer_results) > 100:
            del self._non_greedy_initial_offer_results[:-100]
        if partner is None:
            return
        partner_results = self._partner_non_greedy_initial_offer_results[partner]
        partner_results.append(bool(accepted))
        if len(partner_results) > 20:
            del partner_results[:-20]

    def _oneshot_counter_all(self, offers, states):
        responses = {}
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and len(offer) > UNIT_PRICE
            and offer[TIME] == self.awi.current_step
        }

        t = self._relative_time(states)

        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            side_partners = [
                partner
                for partner in all_partners
                if partner in current_offers
            ]
            if not side_partners:
                continue

            if needs <= 0:
                for partner in side_partners:
                    responses[partner] = self._unneeded_response()
                continue

            responses.update(
                self._probabilistic_counter_side(
                    int(needs),
                    side_partners,
                    current_offers,
                    t,
                )
            )

        self._record_response_offers(
            responses,
            relative_time=t,
        )
        return responses

    def _probabilistic_counter_side(
        self,
        needs: int,
        partners: list[str],
        current_offers,
        t: float,
    ):
        best_score = None
        best_accept_set = set()
        best_counters = {}
        partners = list(partners)
        accept_subsets = self._candidate_accept_subsets(
            partners,
            current_offers,
            needs,
        )

        for accept_subset in accept_subsets:
            accept_subset = set(accept_subset)
            accepted_quantity = sum(
                int(current_offers[partner][QUANTITY])
                for partner in accept_subset
            )
            remaining_needs = max(0, int(needs) - accepted_quantity)
            remaining_partners = [
                partner
                for partner in partners
                if partner not in accept_subset
            ]

            accept_value = 0.0
            accept_bad_price_count = 0
            if accept_subset:
                accept_quantities = [
                    int(current_offers[partner][QUANTITY])
                    for partner in accept_subset
                ]
                accept_expected_units = [float(quantity) for quantity in accept_quantities]
                accept_prices = {
                    partner: float(current_offers[partner][UNIT_PRICE])
                    for partner in accept_subset
                }
                accept_value = self._rdvo_value(
                    needs,
                    list(accept_subset),
                    accept_quantities,
                    accept_expected_units,
                    t,
                    prices=accept_prices,
                    offered_total=accepted_quantity,
                )
                accept_bad_price_count = sum(
                    1
                    for partner in accept_subset
                    if self._price_for_me_score(
                        partner,
                        current_offers[partner][UNIT_PRICE],
                    )
                    <= 0.20
                )

            if remaining_needs > 0 and remaining_partners:
                (
                    counters,
                    counter_expected_total,
                    counter_value,
                    counter_bad_price_count,
                    counter_offered_total,
                ) = self._best_oneshot_counter_proposals(
                    remaining_needs,
                    remaining_partners,
                    t,
                )
            else:
                counters = {}
                counter_expected_total = 0.0
                counter_value = 0.0
                counter_bad_price_count = 0
                counter_offered_total = 0

            total_expected = float(accepted_quantity) + float(counter_expected_total)
            expected_gap = abs(float(needs) - total_expected)
            expected_over = max(0.0, total_expected - float(needs))
            total_value = accept_value + counter_value
            bad_price_count = accept_bad_price_count + counter_bad_price_count
            offered_gap = abs(
                float(needs) - float(accepted_quantity + counter_offered_total)
            )
            score = (
                expected_gap,
                expected_over,
                bad_price_count * self.counter_bad_price_penalty,
                -total_value,
                offered_gap,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_accept_set = accept_subset
                best_counters = counters

        responses = {}
        for partner in partners:
            if partner in best_accept_set:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    current_offers[partner],
                )
            else:
                offer = best_counters.get(partner)
                if offer is None:
                    responses[partner] = self._unneeded_response()
                else:
                    responses[partner] = self._counter_or_accept_response(
                        partner,
                        current_offers[partner],
                        offer,
                    )
        return responses

    def _candidate_accept_subsets(self, partners, current_offers, needs: int):
        partners = list(partners)
        needs = max(0, int(needs))
        candidates = [set()]

        sorted_single = sorted(
            partners,
            key=lambda partner: (
                abs(int(current_offers[partner][QUANTITY]) - needs),
                -self._price_for_me_score(
                    partner,
                    current_offers[partner][UNIT_PRICE],
                ),
            ),
        )
        for partner in sorted_single[: min(6, len(sorted_single))]:
            candidates.append({partner})

        by_good_price = sorted(
            partners,
            key=lambda partner: (
                -self._price_for_me_score(
                    partner,
                    current_offers[partner][UNIT_PRICE],
                ),
                abs(int(current_offers[partner][QUANTITY]) - needs),
            ),
        )
        total = 0
        subset = set()
        for partner in by_good_price:
            quantity = int(current_offers[partner][QUANTITY])
            if total + quantity <= needs:
                subset.add(partner)
                total += quantity
        if subset:
            candidates.append(subset)

        by_quantity = sorted(
            partners,
            key=lambda partner: int(current_offers[partner][QUANTITY]),
            reverse=True,
        )
        total = 0
        subset = set()
        for partner in by_quantity:
            quantity = int(current_offers[partner][QUANTITY])
            if abs(needs - (total + quantity)) <= abs(needs - total):
                subset.add(partner)
                total += quantity
        if subset:
            candidates.append(subset)

        good_price_partners = [
            partner
            for partner in partners
            if self._price_for_me_score(
                partner,
                current_offers[partner][UNIT_PRICE],
            )
            >= 0.80
        ]
        total = 0
        subset = set()
        for partner in sorted(
            good_price_partners,
            key=lambda partner: int(current_offers[partner][QUANTITY]),
        ):
            quantity = int(current_offers[partner][QUANTITY])
            if total + quantity <= needs:
                subset.add(partner)
                total += quantity
        if subset:
            candidates.append(subset)

        unique = []
        seen = set()
        for subset in candidates:
            key = tuple(sorted(subset))
            if key in seen:
                continue
            seen.add(key)
            unique.append(set(subset))
        return unique[: self.max_accept_subsets]

    def _rank_counter_partners(self, partners):
        return sorted(
            list(partners),
            key=lambda partner: (
                self._price_conditioned_accept_probability(
                    partner,
                    1,
                    self._best_price_for_me(partner),
                ),
                self._partner_non_greedy_initial_offer_success_rate(partner),
                self._mean_reoffer_quantity(partner),
            ),
            reverse=True,
        )

    def _best_oneshot_counter_proposals(
        self,
        needs: int,
        partners: list[str],
        t: float,
    ):
        if not partners or needs <= 0:
            return {}, 0.0, 0.0, 0, 0

        ranked_partners = self._rank_counter_partners(partners)
        active_partners = ranked_partners[: self.max_counter_partners]
        inactive_partners = ranked_partners[self.max_counter_partners :]
        quantity_allocations = self._beam_quantity_allocations_for_counter(
            needs,
            active_partners,
            t,
            beam_width=self.counter_beam_width,
        )

        best_score = None
        best_proposals = {}
        best_expected_total = 0.0
        best_value = 0.0
        best_bad_price_count = 0
        best_offered_total = 0

        for quantities, _, _, offered_total in quantity_allocations:
            price_patterns = self._counter_price_patterns(
                active_partners,
                quantities,
                t,
                needs,
            )
            for prices in price_patterns:
                expected_units = []
                price_dict = {}
                bad_price_count = 0
                for partner, quantity, price in zip(
                    active_partners,
                    quantities,
                    prices,
                    strict=False,
                ):
                    quantity = int(quantity)
                    price = int(price)
                    price_dict[partner] = float(price)
                    if quantity <= 0:
                        expected_units.append(0.0)
                        continue
                    if price == self._worst_price_for_me(partner):
                        bad_price_count += 1
                    expected_units.append(
                        self._oneshot_counter_expected_units(
                            partner,
                            quantity,
                            price,
                        )
                    )

                expected_total = max(0.0, sum(expected_units))
                expected_gap = abs(float(needs) - expected_total)
                expected_over = max(0.0, expected_total - float(needs))
                value = self._rdvo_value(
                    needs,
                    active_partners,
                    quantities,
                    expected_units,
                    t,
                    prices=price_dict,
                    offered_total=offered_total,
                )
                offered_gap = abs(float(needs) - float(offered_total))
                score = (
                    expected_gap,
                    expected_over,
                    bad_price_count * self.counter_bad_price_penalty,
                    -value,
                    offered_gap,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_expected_total = expected_total
                    best_value = value
                    best_bad_price_count = bad_price_count
                    best_offered_total = offered_total
                    proposals = {}
                    for partner, quantity, price in zip(
                        active_partners,
                        quantities,
                        prices,
                        strict=False,
                    ):
                        quantity = int(quantity)
                        if quantity <= 0:
                            proposals[partner] = None
                        else:
                            proposals[partner] = self._offer(
                                partner,
                                quantity,
                                int(price),
                            )
                    for partner in inactive_partners:
                        proposals[partner] = None
                    best_proposals = proposals

        return (
            best_proposals,
            best_expected_total,
            best_value,
            best_bad_price_count,
            best_offered_total,
        )

    def _beam_quantity_allocations_for_counter(
        self,
        needs: int,
        partners: list[str],
        t: float,
        *,
        beam_width: int | None = None,
    ):
        needs = max(0, int(needs))
        partners = list(partners)
        if beam_width is None:
            beam_width = self.counter_beam_width

        beam = [([], [], 0.0, 0)]
        for partner in partners:
            candidates = self._quantity_candidates_for_oneshot_counter(
                needs,
                partner,
            )
            next_beam = []
            for (
                quantities_so_far,
                expected_units_so_far,
                expected_total,
                offered_total,
            ) in beam:
                for quantity in candidates:
                    quantity = int(quantity)
                    price = self._best_price_for_me(partner)
                    expected = self._oneshot_counter_expected_units(
                        partner,
                        quantity,
                        price,
                    )
                    new_quantities = quantities_so_far + [quantity]
                    new_expected_units = expected_units_so_far + [expected]
                    new_expected_total = expected_total + expected
                    new_offered_total = offered_total + max(0, quantity)
                    if new_expected_total > needs * 1.5 + 2:
                        continue
                    next_beam.append(
                        (
                            new_quantities,
                            new_expected_units,
                            new_expected_total,
                            new_offered_total,
                        )
                    )

            def partial_key(item):
                _, _, expected_total, offered_total = item
                expected_over = max(0.0, expected_total - float(needs))
                offered_over = max(0, offered_total - needs)
                return (
                    expected_over,
                    offered_over,
                    abs(float(needs) - expected_total) * 0.2,
                    offered_total,
                )

            next_beam.sort(key=partial_key)
            beam = next_beam[:beam_width]
            if not beam:
                break

        def final_key(item):
            _, _, expected_total, offered_total = item
            expected_gap = abs(float(needs) - expected_total)
            expected_over = max(0.0, expected_total - float(needs))
            offered_gap = abs(float(needs) - float(offered_total))
            return (
                expected_gap,
                expected_over,
                offered_gap,
            )

        beam.sort(key=final_key)
        return beam

    def _counter_price_patterns(
        self,
        partners: list[str],
        quantities: list[int],
        t: float,
        remaining_needs: int,
    ):
        best_prices = [self._best_price_for_me(partner) for partner in partners]
        patterns = [best_prices]
        t = max(0.0, min(1.0, float(t)))
        shortage_ratio = float(remaining_needs) / max(1.0, float(self._max_lines()))
        allow_worst = (
            t >= self.counter_good_price_time
            or shortage_ratio >= self.counter_good_price_shortage_ratio
        )
        if not allow_worst:
            return patterns

        all_worst = [
            self._worst_price_for_me(partner)
            if int(quantity) > 0
            else self._best_price_for_me(partner)
            for partner, quantity in zip(partners, quantities, strict=False)
        ]
        patterns.append(all_worst)

        active_indices = [
            index
            for index, quantity in enumerate(quantities)
            if int(quantity) > 0
        ]
        if active_indices:
            weakest = min(
                active_indices,
                key=lambda index: self._price_conditioned_accept_probability(
                    partners[index],
                    quantities[index],
                    self._best_price_for_me(partners[index]),
                ),
            )
            one_worst = list(best_prices)
            one_worst[weakest] = self._worst_price_for_me(partners[weakest])
            patterns.append(one_worst)

        greedy_indices = [
            index
            for index, partner in enumerate(partners)
            if int(quantities[index]) > 0
            and self.opponent_type(partner) == "GreedyOneShotAgent"
        ]
        if greedy_indices:
            greedy_worst = list(best_prices)
            for index in greedy_indices:
                greedy_worst[index] = self._worst_price_for_me(partners[index])
            patterns.append(greedy_worst)

        unique = []
        seen = set()
        for pattern in patterns:
            key = tuple(pattern)
            if key in seen:
                continue
            seen.add(key)
            unique.append(pattern)
        return unique

    def _quantity_candidates_for_oneshot_counter(
        self,
        needs: int,
        partner,
    ) -> list[int]:
        qmax = max(
            1,
            int(self._issues_for(partner)[QUANTITY].max_value),
        )
        upper = min(max(0, int(needs)), qmax)
        return list(range(0, upper + 1))

    def _price_conditioned_accept_probability(
        self,
        partner,
        quantity: int,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)
        trials = self._first_offer_trials_by_price[partner][label]
        accepts = self._first_offer_accepts_by_price[partner][label]
        if trials > 0:
            base = (accepts + 1.0) / (trials + 2.0)
        else:
            base = self._accept_probability(partner, quantity, price)
        return max(0.02, min(0.98, base))

    def _price_conditioned_reoffer_probability(
        self,
        partner,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)
        trials = self._first_offer_trials_by_price[partner][label]
        accepts = self._first_offer_accepts_by_price[partner][label]
        rejected = max(0, trials - accepts)
        reoffers = self._first_offer_reoffers_by_price[partner][label]
        if trials > 0:
            return (reoffers + 1.0) / (rejected + 2.0)
        return self._reoffer_probability(partner)

    def _price_conditioned_mean_reoffer_quantity(
        self,
        partner,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)
        quantities = self._first_offer_counter_quantities_by_price[partner][label]
        if quantities:
            recent = quantities[-20:]
            return sum(recent) / len(recent)
        return self._mean_reoffer_quantity(partner)

    def _oneshot_counter_expected_units(
        self,
        partner,
        quantity: int,
        price: float,
    ) -> float:
        if quantity <= 0:
            return 0.0
        p = self._price_conditioned_accept_probability(
            partner,
            quantity,
            price,
        )
        r = self._price_conditioned_reoffer_probability(
            partner,
            price,
        )
        mu = self._price_conditioned_mean_reoffer_quantity(
            partner,
            price,
        )
        return p * quantity + (1.0 - p) * r * mu

    def _accept_probability(self, partner, quantity: int, price: float) -> float:
        del quantity, price
        history = self._own_offer_result_history.get(partner, [])[-20:]
        if history:
            accepts = sum(bool(item.get("accepted", False)) for item in history)
            return (accepts + 1.0) / (len(history) + 2.0)
        return 0.5

    def _reoffer_probability(self, partner) -> float:
        total_reoffers = sum(
            len(quantities)
            for quantities in self._first_offer_counter_quantities_by_price[partner].values()
        )
        trials = sum(self._first_offer_trials_by_price[partner].values())
        accepts = sum(self._first_offer_accepts_by_price[partner].values())
        rejected = max(0, trials - accepts)
        if trials > 0:
            return (total_reoffers + 1.0) / (rejected + 2.0)
        return 0.5

    def _mean_reoffer_quantity(self, partner) -> float:
        quantities = []
        for values in self._first_offer_counter_quantities_by_price[partner].values():
            quantities.extend(values[-20:])
        if quantities:
            recent = quantities[-20:]
            return sum(recent) / len(recent)
        return 2.0

    def _rdvo_value(
        self,
        needs: int,
        partners: list[str],
        quantities,
        expected_units,
        t: float,
        prices: dict[str, float] | None = None,
        offered_total: float | None = None,
    ) -> float:
        del quantities, t
        if not partners:
            return 0.0
        value = 0.0
        for partner, expected in zip(partners, expected_units, strict=False):
            if expected <= 0:
                continue
            price = (
                float(prices[partner])
                if prices and partner in prices
                else float(self._best_price_for_me(partner))
            )
            value += float(expected) * self._price_for_me_score(partner, price)
        expected_total = max(0.0, sum(float(unit) for unit in expected_units))
        expected_gap = abs(float(needs) - expected_total)
        offered_gap = (
            abs(float(needs) - float(offered_total))
            if offered_total is not None
            else 0.0
        )
        return value - 0.05 * expected_gap - 0.02 * offered_gap

    def _conceded_counter_quantity(
        self, partner, desired_quantity: int, offer, t: float, accepted_offers=None
    ) -> int:
        desired_quantity = int(desired_quantity)
        if offer is None or t <= 0.5:
            quantity = desired_quantity
        else:
            opponent_quantity = int(offer[QUANTITY])
            concession = max(0.0, min(1.0, (float(t) - 0.5) / 0.45))
            quantity = self._clamp_quantity(
                partner,
                round(
                    desired_quantity
                    + (opponent_quantity - desired_quantity) * concession
                ),
            )
        # Improvement: when conceding toward the opponent in a 1-on-1 endgame, do
        # not move past the quantity at which our utility drops below the
        # no-agreement (disagreement) utility.
        if self.single_partner_utility_floor and accepted_offers is not None:
            quantity = self._max_floor_quantity(
                partner,
                desired_quantity,
                quantity,
                self._conceded_price_for_me(partner, t),
                accepted_offers,
            )
        return quantity

    def _max_floor_quantity(
        self,
        partner,
        desired_q: int,
        target_q: int,
        price,
        accepted_offers,
    ) -> int:
        """Move from desired_q toward target_q only while utility stays above
        the disagreement utility.  As a seller, never move above the remaining
        sales need represented by desired_q."""
        desired_q = int(desired_q)
        target_q = int(target_q)
        if target_q <= 0:
            return target_q
        desired_q = self._clamp_quantity(partner, desired_q)
        target_q = self._clamp_quantity(partner, target_q)
        if self._is_seller_to(partner):
            target_q = min(target_q, desired_q)
        step = self.awi.current_step
        try:
            base = dict(accepted_offers) if accepted_offers else {}
            u_no = self.ufun.from_offers(base) if base else self.ufun.from_offers({})
            best = 0
            direction = 1 if target_q >= desired_q else -1
            for q in range(desired_q, target_q + direction, direction):
                if q <= 0:
                    continue
                trial = dict(base)
                trial[partner] = (int(q), step, int(price))
                if self.ufun.from_offers(trial) >= u_no:
                    best = q
                else:
                    break
            return best
        except Exception:
            return target_q

    def _final_single_partner_response(
        self,
        partner,
        offer,
        accepted_offers,
        desired_quantity: int | None = None,
    ):
        accept_offers = dict(accepted_offers)
        accept_offers[partner] = offer
        reject_offers = dict(accepted_offers)
        try:
            accept_utility = self.ufun.from_offers(accept_offers)
            reject_utility = self.ufun.from_offers(reject_offers)
        except Exception:
            accept_utility = self._single_offer_profit_heuristic(partner, offer)
            reject_utility = 0

        if (
            self.single_partner_utility_floor
            and desired_quantity is not None
            and offer is not None
        ):
            limited_quantity = self._max_floor_quantity(
                partner,
                int(desired_quantity),
                int(offer[QUANTITY]),
                int(offer[UNIT_PRICE]),
                accepted_offers,
            )
            if limited_quantity <= 0:
                return self._unneeded_response()
            if limited_quantity != int(offer[QUANTITY]):
                counter_offer = self._raw_offer(
                    partner,
                    limited_quantity,
                    int(offer[UNIT_PRICE]),
                )
                if counter_offer is None:
                    return self._unneeded_response()
                return SAOResponse(ResponseType.REJECT_OFFER, counter_offer)

        if accept_utility >= reject_utility:
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return self._unneeded_response()

    def _single_remaining_near_need_buy_response(
        self,
        partner,
        offer,
        remaining_needs: int,
    ):
        if not self._is_process_one_agent():
            return None
        if partner not in self.awi.my_suppliers:
            return None
        remaining_needs = int(remaining_needs)
        if remaining_needs < 4 or offer is None:
            return None
        offer_quantity = int(offer[QUANTITY])
        if abs(offer_quantity - remaining_needs) > 1:
            return None
        return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

    def _single_offer_profit_heuristic(self, partner, offer) -> float:
        if offer is None:
            return 0.0
        quantity = int(offer[QUANTITY])
        price = float(offer[UNIT_PRICE])
        if self._is_seller_to(partner):
            return quantity * price
        return -quantity * price

    def _has_exact_offer_subset(self, partners, offers, needs: int) -> bool:
        if needs <= 0:
            return False
        reachable = {0}
        for partner in partners:
            quantity = int(offers[partner][QUANTITY])
            if quantity <= 0 or quantity > needs:
                continue
            next_reachable = set(reachable)
            for total in reachable:
                offered = total + quantity
                if offered == needs:
                    return True
                if offered < needs:
                    next_reachable.add(offered)
            reachable = next_reachable
        return False

    def _eighty_percent_acceptance_subset(self, partners, offers, needs: int):
        n_partners = len(partners)
        max_accept_partners = n_partners - 1
        if needs <= 0 or max_accept_partners <= 0:
            return None

        target = min(needs, max(1, math.ceil(needs * 0.8)))
        best_for_total = {0: tuple()}
        for partner in partners:
            quantity = int(offers[partner][QUANTITY])
            if quantity <= 0 or quantity >= needs:
                continue
            updates = {}
            for total, partner_ids in best_for_total.items():
                offered = total + quantity
                if offered >= needs:
                    continue
                candidate = partner_ids + (partner,)
                current = updates.get(offered)
                if current is None:
                    current = best_for_total.get(offered)
                if current is None or (len(candidate), candidate) < (len(current), current):
                    updates[offered] = candidate
            for offered, candidate in updates.items():
                current = best_for_total.get(offered)
                if current is None or (len(candidate), candidate) < (len(current), current):
                    best_for_total[offered] = candidate

        best = None
        for offered in range(target, needs):
            partner_ids = best_for_total.get(offered)
            if not partner_ids or len(partner_ids) > max_accept_partners:
                continue
            candidate = (offered, len(partner_ids), partner_ids)
            if (
                best is None
                or offered > best[0]
                or (offered == best[0] and candidate[1:] < best[1:])
            ):
                best = candidate
        return None if best is None else best[-1]

    def _seller_greedy_fill_plan(self, partners, offers, needs: int):
        if needs <= 0:
            return None

        greedy_partners = [
            partner
            for partner in partners
            if self._greedy_fill_probability(partner)
            >= self.seller_greedy_fill_threshold
        ]
        if not greedy_partners:
            return None

        greedy_partners.sort(
            key=self._greedy_fill_probability,
            reverse=True,
        )
        greedy_partner = greedy_partners[0]
        non_greedy_partners = [
            partner
            for partner in partners
            if partner != greedy_partner
        ]

        accepted_partners = self._max_under_needs_subset_for_seller(
            non_greedy_partners,
            offers,
            needs,
        )
        accepted_quantity = sum(
            int(offers[partner][QUANTITY])
            for partner in accepted_partners
        )
        remaining_needs = max(0, int(needs) - accepted_quantity)
        if remaining_needs <= 0:
            return None
        return set(accepted_partners), greedy_partner, remaining_needs

    def _greedy_fill_probability(self, partner) -> float:
        if partner in self._non_greedy_veto:
            return 0.0
        return self.opponent_posteriors(partner).get("GreedyOneShotAgent", 0.0)

    def _max_under_needs_subset_for_seller(self, partners, offers, needs: int):
        best_for_total = {0: (0.0, 0, tuple())}
        for partner in partners:
            quantity = int(offers[partner][QUANTITY])
            if quantity <= 0 or quantity > needs:
                continue
            price_value = quantity * float(offers[partner][UNIT_PRICE])
            updates = {}
            for total, (total_price, neg_size, partner_ids) in best_for_total.items():
                offered = total + quantity
                if offered > needs:
                    continue
                candidate = (
                    total_price + price_value,
                    neg_size - 1,
                    partner_ids + (partner,),
                )
                current = updates.get(offered)
                if current is None:
                    current = best_for_total.get(offered)
                if current is None or candidate > current:
                    updates[offered] = candidate
            for offered, candidate in updates.items():
                current = best_for_total.get(offered)
                if current is None or candidate > current:
                    best_for_total[offered] = candidate

        best = None
        for total, (price_value, neg_size, partner_ids) in best_for_total.items():
            candidate = (total, price_value, neg_size, partner_ids)
            if best is None or candidate > best:
                best = candidate
        return tuple() if best is None else best[-1]

    def _counter_remainder_partners(self, partners, opponent_types, count: int = 2):
        if not partners or count <= 0:
            return []
        preferred = self._success_scaled_partners(partners, opponent_types)
        selected = []
        for partner in preferred + list(partners):
            if partner in selected:
                continue
            selected.append(partner)
            if len(selected) >= count:
                break
        return selected

    def _equal_counter_quantities(self, needs: int, partners: list[str]) -> dict[str, int]:
        if not partners or needs <= 0:
            return {}
        quantities = {}
        remaining = int(needs)
        for index, partner in enumerate(partners):
            if remaining <= 0:
                quantities[partner] = 0
                continue
            slots_left = len(partners) - index
            quantity = math.ceil(remaining / slots_left)
            quantity = self._clamp_quantity(partner, quantity)
            quantity = min(quantity, remaining)
            quantities[partner] = quantity
            remaining = max(0, remaining - quantity)
        return quantities

    def _unneeded_response(self):
        if self.awi.allow_zero_quantity:
            return SAOResponse(
                ResponseType.REJECT_OFFER,
                (0, self.awi.current_step, 0),
            )
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _type_based_first_proposals(self, needs: int, partners: list[str], base_proposals):
        opponent_types = self._strategy_opponent_types(partners)
        greedy_partners = self._ranked_partners_of_type(partners, "GreedyOneShotAgent", opponent_types)
        random_partners = self._ranked_partners_of_type(partners, "RandomOneShotAgent", opponent_types)
        small_dist_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}
        ]
        strategic_partners = set(greedy_partners[:2]) | set(random_partners) | set(small_dist_partners)
        if strategic_partners:
            proposals = {partner: None for partner in partners}
        else:
            proposals = {partner: base_proposals.get(partner) for partner in partners}
        remaining = max(0, int(needs))

        if greedy_partners:
            selected_greedy = greedy_partners[:2]
            if len(selected_greedy) == 1:
                quantities = [min(7, int(needs))]
            else:
                quantities = self._split_greedy_eighty_quantities(needs)
            for partner, quantity in zip(selected_greedy, quantities, strict=False):
                quantity = self._clamp_quantity(partner, quantity)
                proposals[partner] = self._offer(partner, quantity, self._worst_price_for_me(partner))
            remaining = max(0, remaining - sum(int(proposals[p][QUANTITY]) for p in selected_greedy if proposals[p] is not None))

        main_partners = [partner for partner in small_dist_partners if partner not in greedy_partners[:2]]
        if not greedy_partners and main_partners:
            self._assign_small_equal_quantities(
                proposals,
                main_partners,
                self._scaled_small_dist_quantity(remaining),
            )
            remaining = 0
        elif remaining > 0 and main_partners:
            self._assign_small_equal_quantities(
                proposals,
                main_partners,
                self._scaled_small_dist_quantity(remaining),
            )
            remaining = 0

        for partner in random_partners:
            if partner in greedy_partners[:2] or partner in main_partners:
                continue
            proposals[partner] = self._offer(partner, self._clamp_quantity(partner, 1), self._best_price_for_me(partner))

        for partner in partners:
            if proposals.get(partner) is None:
                continue
            quantity = int(proposals[partner][QUANTITY])
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                proposals[partner] = None

        return proposals

    def _ranked_partners_of_type(self, partners, type_name: str, opponent_types=None) -> list[str]:
        if opponent_types is None:
            opponent_types = {partner: self.opponent_type(partner) for partner in partners}
        return sorted(
            [partner for partner in partners if opponent_types[partner] == type_name],
            key=lambda partner: self.opponent_posteriors(partner).get(type_name, 0.0),
            reverse=True,
        )

    def _assign_small_equal_quantities(self, proposals, partners: list[str], target_quantity: int):
        if not partners:
            return
        n = len(partners)
        if target_quantity <= 0:
            for partner in partners:
                proposals[partner] = self._offer(partner, 0, self._best_price_for_me(partner))
            return
        if target_quantity <= 3 * n:
            base = min(3, target_quantity // n)
            remainder = max(0, target_quantity - base * n)
        else:
            base = target_quantity // n
            remainder = target_quantity - base * n
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            proposals[partner] = self._offer(partner, self._clamp_quantity(partner, quantity), self._best_price_for_me(partner))

    def _scaled_small_dist_quantity(self, quantity: int) -> int:
        if quantity <= 0:
            return 0
        return max(1, math.ceil(quantity * self._small_dist_quantity_multiplier()))

    def _offer(self, partner, quantity: int, price: int) -> Outcome | None:
        quantity = self._role_biased_quantity(partner, quantity)
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

    def _raw_offer(self, partner, quantity: int, price: int) -> Outcome | None:
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

    def _role_biased_quantity(self, partner, quantity: int) -> int:
        # Sellers are penalized for over-commitment (shortfall) so bias
        # quantities down; buyers should avoid under-procurement so bias up.
        # Greedy partners keep their exact quantities; exploration probes are
        # left untouched to avoid distorting classification evidence.
        quantity = int(quantity)
        if quantity <= 0:
            return quantity
        if self._exploration_enabled():
            return quantity
        if self.opponent_type(partner) == "GreedyOneShotAgent":
            return quantity
        if self._is_seller_to(partner):
            return max(1, math.floor(quantity * self.seller_quantity_bias))
        return int(quantity * self.buyer_quantity_bias)

    def _counter_or_accept_response(self, partner, current_offer, counter_offer):
        if counter_offer is None:
            return self._unneeded_response()
        if self._same_offer(counter_offer, current_offer) or self._same_quantity_offer(
            counter_offer,
            current_offer,
        ):
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                current_offer,
            )
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            counter_offer,
        )

    def _adapt_proposals(self, proposals: dict[str, Outcome | None], t: float):
        adapted = dict(proposals)
        for partner, offer in proposals.items():
            if offer is None:
                continue
            opponent_type = self.opponent_type(partner)
            quantity = int(offer[QUANTITY])
            price = int(offer[UNIT_PRICE])

            if opponent_type == "RandomOneShotAgent":
                # Random agents sometimes accept extreme prices, so stay firm.
                price = self._best_price_for_me(partner)
            elif opponent_type == "GreedyOneShotAgent":
                price = self._worst_price_for_me(partner)
            elif opponent_type in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}:
                price = self._best_price_for_me(partner)

            adapted[partner] = (quantity, self.awi.current_step, price)
        return adapted

    def _adapt_responses(self, responses, offers, states, t: float):
        adapted = dict(responses)
        for partner, response in list(adapted.items()):
            if response is None:
                continue
            offer = offers.get(partner)
            opponent_type = self.opponent_type(partner)

            if response.response == ResponseType.REJECT_OFFER and response.outcome is not None:
                quantity = int(response.outcome[QUANTITY])
                price = int(response.outcome[UNIT_PRICE])
                if opponent_type == "RandomOneShotAgent":
                    price = self._best_price_for_me(partner)
                elif opponent_type == "GreedyOneShotAgent":
                    price = self._worst_price_for_me(partner)
                adapted[partner] = SAOResponse(ResponseType.REJECT_OFFER, (quantity, self.awi.current_step, price))
                continue

        return adapted

    def _conceded_price_for_me(self, partner, t: float) -> int:
        best = self._best_price_for_me(partner)
        if (
            self.nongreedy_seller_firm
            and self._is_seller_to(partner)
            and self.opponent_type(partner) != "GreedyOneShotAgent"
        ):
            return best  # no price concession vs NonGreedy buyers
        worst = self._worst_price_for_me(partner)
        concession = max(0.0, min(1.0, t + self.greedy_time_concession))
        if self._is_seller_to(partner):
            return int(round(best - (best - worst) * concession * 0.35))
        return int(round(best + (worst - best) * concession * 0.35))

    def on_negotiation_success(self, contract: Contract, mechanism: Any):
        super().on_negotiation_success(contract, mechanism)
        partner = self._contract_partner(contract)
        outcome = self._contract_outcome(contract)
        if partner is None or outcome is None:
            return
        sent_offer = self._matching_sent_offer(partner, outcome)
        if sent_offer is not None:
            self._observe_partner_end_response(partner, sent_offer, ended=False)
            self._record_non_greedy_initial_offer_result(sent_offer, accepted=True)
            self._observe_own_offer_result(partner, sent_offer, accepted=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except StopIteration:
            return
        sent_offer = self._latest_sent_offer(partner)
        if sent_offer is not None:
            if self._partner_ended_after_sent_offer(sent_offer, state):
                self._observe_partner_end_response(partner, sent_offer, ended=True)
            self._record_non_greedy_initial_offer_result(sent_offer, accepted=False)
            self._observe_own_offer_result(partner, sent_offer, accepted=False)

    def __init__(
        self,
        *args,
        accept_base_tolerance: float = 0.20,
        accept_favorable_factor: float = 0.85,
        accept_unfavorable_factor: float = 1.25,
        single_partner_acceptance_factor: float = 2.50,
        neutral_market_ratio: float = 1.06,
        market_band: float = 0.13,
        forced_accept_time: float = 0.925,
        # カウンター数量の時間譲歩を開始する t (これ以降、相手量へ寄せる)。
        quantity_concession_start: float = 0.5,
        # 不足方向 (売り手の過剰契約) の受諾を禁止する。
        forbid_shortfall_accept: bool = True,
        eighty_success_threshold: float = 0.75,
        eighty_main_ratio: float = 0.65,
        greedy_single_offer_cap: int = 7,
        greedy_first_offer_cap: int = 6,
        greedy_second_offer_cap: int = 4,
        min_success_rate: float = 0.05,
        # NonGreedy 買い手の過剰調達抑制。
        #   target 算出に使う成功率の下限 (target ブランチ)。
        target_success_floor: float = 0.45,
        #   提案総量キャップ = ceil(needs / max(success_cap_floor, 平均成功率))。
        #   成功率の下限 (これ未満だとキャップが過大になり過剰調達を許すため)。
        success_cap_floor: float = 0.45,
        # 主軸のヒステリシス: 昇格は成功率 >= eighty_success_threshold(0.75)、
        # 一度主軸になった相手は降格閾値を下回るまで維持 (1サンプルの揺れで降格しない)。
        main_demote_threshold: float = 0.75,
        # target数変更ガード: 成功率由来の生 target が active と異なっても即変更せず、
        # 同方向のズレが target_confirm_days 日連続したときだけ active を
        # ±target_change_step 動かす (1日のノイズで下がらないよう様子見する)。
        target_confirm_days: int = 2,
        target_change_step: int = 1,
        **kwargs,
    ):
        self._bayes__init__(*args, **kwargs)
        # 受諾閾値 (誤差÷残り必要数) の基準値と需給による補正係数。
        self.accept_base_tolerance = float(accept_base_tolerance)
        self.accept_favorable_factor = float(accept_favorable_factor)
        self.accept_unfavorable_factor = float(accept_unfavorable_factor)
        self.single_partner_acceptance_factor = float(single_partner_acceptance_factor)
        # この市場は構造的に供給>需要 (実測の供給÷需要 ≈ 1.06、供給超過が約68%)。
        # そのため需給の「中立」を 1.0 ではなく実測平均 ratio に置き、その上下に
        # band 幅の帯を作って有利/不利を判定する。
        self.neutral_market_ratio = float(neutral_market_ratio)
        self.market_band = float(market_band)
        self.favorable_market_ratio = self.neutral_market_ratio * (1.0 + self.market_band)
        self.unfavorable_market_ratio = self.neutral_market_ratio * (1.0 - self.market_band)
        # t がこの値を超えたら利益最大化の組み合わせを強制受諾する。
        self.forced_accept_time = float(forced_accept_time)
        self.quantity_concession_start = float(quantity_concession_start)
        self.forbid_shortfall_accept = bool(forbid_shortfall_accept)
        # 初手オファー契約成功率がこの値以上の相手を「主軸」とみなす。
        self.eighty_success_threshold = float(eighty_success_threshold)
        self.eighty_main_ratio = float(eighty_main_ratio)
        # Greedy 環境でのオファー数量上限。
        self.greedy_single_offer_cap = int(greedy_single_offer_cap)
        self.greedy_first_offer_cap = int(greedy_first_offer_cap)
        self.greedy_second_offer_cap = int(greedy_second_offer_cap)
        self.min_success_rate = float(min_success_rate)
        # target ブランチの過剰調達抑制 (案A: 成功率下限 / 案B: 総量キャップ)。
        self.target_success_floor = float(target_success_floor)
        self.success_cap_floor = float(success_cap_floor)
        # 主軸ヒステリシス / target変更ガード。
        self.main_demote_threshold = float(main_demote_threshold)
        self.target_confirm_days = int(target_confirm_days)
        self.target_change_step = int(target_change_step)

    def init(self):
        self._bayes_init()
        # is_buy -> 現在の主軸相手 (降格閾値を下回るまで維持)。
        self._current_main: dict = {}
        # target変更の「2日確認」ガード用の状態。
        self._active_target: dict = {}   # is_buy -> 現在採用中の target
        self._pending_days: dict = {}    # is_buy -> 同方向にズレた連続日数
        self._pending_dir: dict = {}     # is_buy -> ズレの方向 (+1 / -1)

    def _is_greedy(self, partner) -> bool:
        return self.opponent_type(partner) == "GreedyOneShotAgent"

    def _greedy_partners_sorted(self, partners) -> list[str]:
        greedy = [partner for partner in partners if self._is_greedy(partner)]
        greedy.sort(
            key=lambda partner: self.opponent_posteriors(partner).get(
                "GreedyOneShotAgent",
                0.0,
            ),
            reverse=True,
        )
        return greedy

    def _success_rate(self, partner) -> float:
        return self._partner_non_greedy_initial_offer_success_rate(partner)

    def _average_success_rate(self, partners) -> float:
        rates = [self._success_rate(partner) for partner in partners]
        if not rates:
            return self._non_greedy_initial_offer_success_rate()
        return sum(rates) / len(rates)

    def _good_price_for(self, partner, greedy_partners) -> int:
        # Greedy には常に good price (= 自分の worst price)、それ以外は best price。
        if partner in greedy_partners:
            return self._worst_price_for_me(partner)
        return self._best_price_for_me(partner)

    def _good_accept_greedy_weight(self, quantity: int) -> float:
        """good価格オファーの受諾に対する greedy 加点を数量で重み付け。

        実測 (good価格・数量別の初手受諾率) の対数尤度比に基づく:
          - 小口(<=2): Greedy/NonGreedy 共に受けやすく非識別的 → 弱
          - 中数量(4〜7): NonGreedy はほぼ受けず Greedy のみ受ける → 強
          - 大数量(>=8): Greedy 自身も needs 超過で断り始める → 弱
        """
        q = int(quantity)
        if q <= 2:
            return 0.5
        if q == 3:
            return 0.9
        if q <= 7:
            return 1.3
        return 0.5

    def _observe_first_offer_classification_result(self, partner, sent_offer, accepted):
        # 親の分類ロジックをそのまま走らせた上で、good価格受諾の加点だけ数量補正する。
        if sent_offer is None:
            return self._bayes_observe_first_offer_classification_result(
                partner, sent_offer, accepted
            )
        already = sent_offer.get("first_result_observed", False)
        price_label = sent_offer.get("price_label", "neutral")
        offer = sent_offer.get("offer")
        quantity = int(offer[QUANTITY]) if offer is not None and len(offer) > QUANTITY else 0

        self._bayes_observe_first_offer_classification_result(partner, sent_offer, accepted)

        if (not already) and bool(accepted) and price_label == "good":
            adjusted = self._good_accept_greedy_weight(quantity)
            # 親は +_PARENT_GOOD_ACCEPT_WEIGHT を加点済み → 差分で目標重みに補正。
            self._ensure_partner(partner)
            self._opponent_logits[partner]["GreedyOneShotAgent"] += (
                adjusted - self._PARENT_GOOD_ACCEPT_WEIGHT
            )

    def first_proposals(self):
        # 分類のための探索フェーズは親の挙動を踏襲する。
        if self._exploration_enabled():
            base_proposals = SyncRandomOneShotAgent.first_proposals(self)
            proposals = self._exploration_first_proposals(base_proposals)
            self._record_sent_offers(proposals)
            return proposals

        proposals = {}
        for needs, all_partners, is_buy in (
            (self.awi.needed_supplies, self.awi.my_suppliers, True),
            (self.awi.needed_sales, self.awi.my_consumers, False),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            if int(needs) <= 0:
                proposals.update({partner: None for partner in partners})
                continue
            proposals.update(self._simple_first_proposals(int(needs), partners, is_buy))

        self._record_sent_offers(proposals)
        return proposals

    def _simple_first_proposals(self, needs: int, partners: list[str], is_buy: bool):
        greedy_partners = self._greedy_partners_sorted(partners)
        if greedy_partners:
            return self._greedy_env_first_proposals(
                needs, partners, greedy_partners, is_buy
            )
        return self._nongreedy_env_first_proposals(needs, partners, is_buy)

    def _greedy_env_first_proposals(self, needs, partners, greedy_partners, is_buy):
        proposals = {partner: None for partner in partners}
        non_greedy = [partner for partner in partners if partner not in greedy_partners]

        if len(greedy_partners) >= 2:
            # @1 では 2 位 Greedy の確信度に応じて上限を変え、残りは他相手へ
            # 初手成功率で割って保険配分する。
            first = greedy_partners[0]
            second = greedy_partners[1]
            first_quantity = min(self.greedy_first_offer_cap, int(needs))
            proposals[first] = self._raw_offer(
                first, first_quantity, self._worst_price_for_me(first)
            )
            remaining = max(0, int(needs) - first_quantity)
            second_cap = (
                self._second_greedy_offer_cap(second)
                if self._is_process_one_agent() and is_buy
                else self.greedy_second_offer_cap
            )
            second_quantity = min(second_cap, remaining)
            if second_quantity > 0:
                proposals[second] = self._raw_offer(
                    second, second_quantity, self._worst_price_for_me(second)
                )
            if self._is_process_one_agent() and is_buy:
                remaining = max(0, remaining - second_quantity)
                rest = [partner for partner in partners if partner not in {first, second}]
                if remaining > 0 and rest:
                    scaled = self._success_scaled_quantity(remaining, rest)
                    self._fill_even(
                        proposals,
                        rest,
                        scaled,
                        lambda partner: self._good_price_for(partner, greedy_partners),
                    )
            return proposals

        # Greedy が 1 人: min(7, needs) を Greedy に。残りを Greedy 以外に
        # 「成功率で割った個数」で分配する。ただし最低でも残り+1は出す。
        greedy_partner = greedy_partners[0]
        greedy_quantity = min(self.greedy_single_offer_cap, int(needs))
        proposals[greedy_partner] = self._raw_offer(
            greedy_partner,
            greedy_quantity,
            self._worst_price_for_me(greedy_partner),
        )
        remaining = max(0, int(needs) - greedy_quantity)
        if remaining > 0 and non_greedy:
            scaled = max(
                self._success_scaled_quantity(remaining, non_greedy),
                remaining + 1,
            )
            self._fill_even(proposals, non_greedy, scaled, self._best_price_for_me)
        return proposals

    def _second_greedy_offer_cap(self, partner) -> int:
        greedy_probability = self.opponent_posteriors(partner).get(
            "GreedyOneShotAgent",
            0.0,
        )
        if greedy_probability < 0.65:
            return 2
        if greedy_probability < 0.75:
            return 3
        return 4

    def _nongreedy_env_first_proposals(self, needs, partners, is_buy: bool = True):
        proposals = {partner: None for partner in partners}
        needs = int(needs)

        # --- 主軸の選定 (ヒステリシス: 昇格0.75 / 降格0.70) ---
        # 在席している現主軸が降格閾値を割っていれば降格 (記憶を消す)。在席でなければ
        # 一時的な不在として記憶を残す。これにより「在席かつ閾値割れ」のときだけ降格し、
        # 1サンプルの揺れだけで主軸を失わない。再昇格は 0.75 以上が必要。
        cur_main = self._current_main.get(is_buy)
        if (
            cur_main is not None
            and cur_main in partners
            and self._success_rate(cur_main) < self.main_demote_threshold
        ):
            cur_main = None
            self._current_main[is_buy] = None

        main = None
        if cur_main is not None and cur_main in partners:
            # 現主軸を維持 (降格閾値以上)。
            main = cur_main
        else:
            strong = [
                partner
                for partner in partners
                if self._success_rate(partner) >= self.eighty_success_threshold
            ]
            if strong:
                main = max(strong, key=self._success_rate)
                self._current_main[is_buy] = main

        if main is not None:
            main_quantity = max(1, min(needs, round(needs * self.eighty_main_ratio)))
            proposals[main] = self._raw_offer(
                main, main_quantity, self._best_price_for_me(main)
            )
            remaining = max(0, needs - main_quantity)
            others = [partner for partner in partners if partner != main]
            if remaining > 0 and others:
                scaled = self._success_scaled_quantity(remaining, others)
                self._fill_even(proposals, others, scaled, self._best_price_for_me)
        else:
            # 主軸不在: 何人で契約成立を狙うかを target = round(人数 × 平均成功率) で
            # 決める。ただし2日確認ガードを通し、1日のノイズでは変えない。
            target = self._guarded_target(is_buy, partners)
            self._fill_by_target(
                proposals, partners, needs, target, self._best_price_for_me
            )

        # 適応キャップ: 提案総量を ceil(needs / 平均初手成功率) に頭打ちする。
        avg = max(self.success_cap_floor, self._average_success_rate(partners))
        self._cap_total_offer(proposals, partners, math.ceil(needs / avg))
        return proposals

    def _close_target_count(self, partners) -> int:
        """何人で契約成立を狙うか。

        各相手に必要量を配って期待成立件数 = 人数 × 平均成功率 とし、それを
        狙う成立人数とする。例 (下限なしのとき):
          - 4 人 / 平均 50% → round(2.0) = 2 人
          - 6 人 / 平均 30% → round(1.8) = 2 人
        案A: 平均成功率が実測より低めに出ると target が小さくなり総量が膨張する
        ため、target_success_floor で下限を設ける (過剰調達抑制)。
        """
        avg = max(self.target_success_floor, self._average_success_rate(partners))
        return max(1, min(len(partners), math.ceil(len(partners) * avg)))

    def _guarded_target(self, is_buy: bool, partners) -> int:
        """生 target (成功率由来) を「2日確認」してから採用 target に反映する。

        - 生 target が採用中 target と一致 → ズレ確認カウントをリセットして維持。
        - 異なる → そのズレの「向き」(+1=上げ / -1=下げ) が前回と同じなら連続日数を
          加算、違えば 1 から数え直す。
        - 同方向のズレが target_confirm_days(=2) 日連続した時点で、採用 target を
          その向きに target_change_step(=1) だけ動かし、カウントをリセットする。

        例: 採用 target=3 のとき、成功率が下がって生 target=2 になっても初日は
            維持 (様子見)。翌日も生 target≤2 のままなら 3→2 に下げる。途中で生
            target=3 に戻れば変更はキャンセルされる (1日のノイズでは下がらない)。
        """
        n = len(partners)
        raw = self._close_target_count(partners)
        active = self._active_target.get(is_buy)
        if active is None:
            self._active_target[is_buy] = raw
            self._pending_days[is_buy] = 0
            self._pending_dir[is_buy] = 0
            return raw
        if raw == active:
            self._pending_days[is_buy] = 0
            self._pending_dir[is_buy] = 0
            return active
        direction = 1 if raw > active else -1
        if self._pending_dir.get(is_buy, 0) == direction:
            self._pending_days[is_buy] = self._pending_days.get(is_buy, 0) + 1
        else:
            self._pending_dir[is_buy] = direction
            self._pending_days[is_buy] = 1
        if self._pending_days[is_buy] >= self.target_confirm_days:
            active = max(1, min(n, active + direction * self.target_change_step))
            self._active_target[is_buy] = active
            self._pending_days[is_buy] = 0
            self._pending_dir[is_buy] = 0
        return active

    def _cap_total_offer(self, proposals, partners, max_total: int):
        """提案総量が max_total を超える分を、成功率の低い相手から **1個ずつ
        ラウンドロビンで** まんべんなく削る。

        1人から大量に削るのではなく、成功率の低い順に1個ずつ巡回して削るため、
        各相手は (低成功率の相手が先に) ほぼ均等に減る。
        例: 4人で3個削減 → 下位3人から1つずつ (1,1,1,0)。
            4人で5個削減 → 全員1つ + 最下位にもう1つ (2,1,1,1)。
        """
        max_total = int(max_total)
        active = [p for p in partners if proposals.get(p) is not None]
        excess = sum(int(proposals[p][QUANTITY]) for p in active) - max_total
        if excess <= 0:
            return
        # 成功率の低い順 (同率は id で安定化)。
        order = sorted(active, key=lambda p: (self._success_rate(p), str(p)))
        qty = {p: int(proposals[p][QUANTITY]) for p in active}
        # 低成功率の相手から1個ずつ巡回して削る。
        while excess > 0:
            cut_this_cycle = False
            for partner in order:
                if excess <= 0:
                    break
                if qty[partner] > 0:
                    qty[partner] -= 1
                    excess -= 1
                    cut_this_cycle = True
            if not cut_this_cycle:
                break
        for partner in active:
            offer = proposals[partner]
            if qty[partner] <= 0:
                proposals[partner] = None
            elif qty[partner] != int(offer[QUANTITY]):
                proposals[partner] = self._raw_offer(
                    partner, qty[partner], int(offer[UNIT_PRICE])
                )

    def _success_scaled_quantity(self, quantity: int, partners) -> int:
        """quantity を相手の平均成功率で割って (= 期待充足量が quantity になる) 量に拡大。"""
        avg = self._average_success_rate(partners)
        return max(1, math.ceil(quantity / max(self.min_success_rate, avg)))

    def _fill_even(self, proposals, partners, total: int, price_getter):
        """total を partners に均等配分する。"""
        partners = list(partners)
        n = len(partners)
        if n <= 0 or total <= 0:
            return
        base = total // n
        remainder = total - base * n
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            if quantity <= 0:
                continue
            offer = self._raw_offer(partner, quantity, price_getter(partner))
            if offer is not None:
                proposals[partner] = offer

    def _fill_by_target(self, proposals, partners, needs: int, target: int, price_getter):
        """各相手に ceil/floor(needs/target) を配る (target 人で needs を満たす配分)。

        例 (needs=7, target=3, 6 人) → (3,2,2,3,2,2)。
        """
        partners = list(partners)
        target = max(1, int(target))
        if needs <= 0 or not partners:
            return
        base = needs // target
        remainder = needs % target
        for index, partner in enumerate(partners):
            quantity = base + (1 if (index % target) < remainder else 0)
            if quantity <= 0:
                continue
            offer = self._raw_offer(partner, quantity, price_getter(partner))
            if offer is not None:
                proposals[partner] = offer

    def _current_offer_responses(self, offers, states):
        # 親の counter_all が相手挙動の観測を済ませた上でこのメソッドを呼ぶ。
        t = self._relative_time(states)
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and len(offer) > UNIT_PRICE
            and offer[TIME] == self.awi.current_step
        }

        # 既定では全相手に「不要」を返し、各サイドで上書きする。
        responses = {partner: self._unneeded_response() for partner in offers}

        for needs, all_partners, is_sell in (
            (self.awi.needed_supplies, self.awi.my_suppliers, False),
            (self.awi.needed_sales, self.awi.my_consumers, True),
        ):
            side_partners = [
                partner for partner in all_partners if partner in current_offers
            ]
            if not side_partners:
                continue
            if int(needs) <= 0:
                for partner in side_partners:
                    responses[partner] = self._unneeded_response()
                continue
            responses.update(
                self._simple_side_responses(
                    int(needs),
                    side_partners,
                    current_offers,
                    t,
                    is_sell,
                )
            )

        self._record_response_offers(responses, relative_time=t)
        return responses

    def _simple_side_responses(self, needs, partners, offers, t, is_sell):
        greedy_partners = self._greedy_partners_sorted(partners)

        # 不足方向 (= 売り手の過剰契約) の受諾は禁止する。売り手は受諾合計が
        # 必要量を超えないよう、受諾候補を needs 以下に制限する。
        accept_cap = int(needs) if (is_sell and self.forbid_shortfall_accept) else None

        # 1. 最良の組み合わせ (誤差最小) を求める。誤差 0 なら必要量ちょうど → 受諾。
        best_subset, best_error = self._best_subset(partners, offers, needs)
        if best_error == 0 and best_subset:
            return self._accept_subset_and_counter(
                needs, partners, offers, best_subset, greedy_partners, t
            )

        # 2. floor(needs / 相手数) + 1 以上の良いオファーがあれば先に受諾し、
        #    残り必要量に対してカウンターを作る。同じ数量でカウンターする相手は
        #    _counter_or_accept_response() 側で受諾に変わる。
        partial = self._partial_accept_set(int(needs), partners, offers)
        if partial:
            return self._counter_side(needs, partners, offers, partial, greedy_partners, t)

        # 3. Greedy 環境 & 売り手 & ちょうどの組み合わせなし:
        #    最も Greedy らしい 1 人を除き、必要量以下で最大の組み合わせを受諾し、
        #    残りを Greedy にオファー。
        if greedy_partners and is_sell:
            plan = self._seller_greedy_fill_plan(partners, offers, int(needs))
            if plan is not None:
                accepted_partners, greedy_partner, remaining_needs = plan
                responses = {}
                for partner in partners:
                    if partner in accepted_partners:
                        responses[partner] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, offers[partner]
                        )
                    elif partner == greedy_partner and remaining_needs > 0:
                        counter = self._raw_offer(
                            partner,
                            remaining_needs,
                            self._worst_price_for_me(partner),
                        )
                        responses[partner] = (
                            self._unneeded_response()
                            if counter is None
                            else SAOResponse(ResponseType.REJECT_OFFER, counter)
                        )
                    else:
                        responses[partner] = self._unneeded_response()
                return responses

        # 4. t > forced_accept_time(0.925): 利益を最大化する組み合わせを強制受諾。
        #    売り手は needs を超える組み合わせを除外 (過剰契約=不足を防ぐ)。
        if t > self.forced_accept_time:
            return self._forced_profit_max_responses(partners, offers, max_total=accept_cap)

        # 5. 受諾閾値 (誤差÷残り必要数) 判定。売り手は needs 以下の最良部分集合で判定。
        accept_subset, accept_error = (
            (best_subset, best_error)
            if accept_cap is None
            else self._best_subset(partners, offers, needs, max_total=accept_cap)
        )
        relative_error = accept_error / max(1, int(needs))
        threshold = self._acceptance_threshold(
            len(partners),
            is_sell,
            has_greedy_partner=bool(greedy_partners),
        )
        if (
            accept_subset
            and relative_error <= threshold
        ):
            return self._accept_subset_and_counter(
                needs, partners, offers, accept_subset, greedy_partners, t
            )

        # 6. 受諾なしで残り全量をカウンターする。
        return self._counter_side(needs, partners, offers, set(), greedy_partners, t)

    def _acceptance_threshold(
        self, n_partners: int, is_sell: bool, has_greedy_partner: bool = False
    ) -> float:
        """受諾閾値 = 許容する「誤差÷残り必要数」。

        - 相手人数が多いほど厳しく (まだ良い相手を探せるため小さく)、少ないほど
          緩く (取りこぼしを避けるため大きく) する: ``base * 2 / (n + 1)``。
        - 需給: 自分に有利な市場なら厳しめ (選り好み)、不利なら緩め (取りに行く)。
          買い手は供給過多 (sell/buy 比が高い) が有利、売り手は需要過多
          (sell/buy 比が低い) が有利。
          ただしこの市場は通常 sell/buy ≈ 1.06 で供給超過が常態のため、中立帯を
          実測平均 (neutral_market_ratio) の上下 band% に置いている。例えば
          neutral=1.06, band=0.13 なら、買い手有利は ratio>=1.20、買い手不利は
          ratio<=0.92。供給が需要を下回りかける (比が常態 1.06 を割り込む) と
          買い手は早めに「取りに行く」側へ振れる。
        時間 t はここでは使わない。
        """
        threshold = self.accept_base_tolerance * (2.0 / (n_partners + 1.0))
        if n_partners == 1 and not has_greedy_partner:
            threshold *= self.single_partner_acceptance_factor

        ratio = self._input_market_sell_buy_ratio()
        if ratio is not None:
            if not is_sell:  # 買い手
                if ratio >= self.favorable_market_ratio:
                    threshold *= self.accept_favorable_factor
                elif ratio <= self.unfavorable_market_ratio:
                    threshold *= self.accept_unfavorable_factor
            else:  # 売り手
                if ratio <= self.unfavorable_market_ratio:
                    threshold *= self.accept_favorable_factor
                elif ratio >= self.favorable_market_ratio:
                    threshold *= self.accept_unfavorable_factor
        return threshold

    def _best_subset(self, partners, offers, needs: int, max_total: int | None = None):
        """|提供量合計 - needs| を最小化する組み合わせを返す。

        同点では自分にとっての価値が高く、人数が少ない方を優先する。
        ``max_total`` 指定時は合計がそれを超える組み合わせを除外する
        (不足方向＝過剰契約の受諾を禁止するために使用)。
        返り値は ``(partner の set, 誤差)``。
        """
        candidates = [
            partner for partner in partners if int(offers[partner][QUANTITY]) > 0
        ]
        if not candidates:
            return set(), int(needs)

        best_key = None
        best_set: set[str] = set()
        for subset in powerset(candidates):
            offered = sum(int(offers[partner][QUANTITY]) for partner in subset)
            if max_total is not None and offered > int(max_total):
                continue
            error = abs(offered - int(needs))
            value = self._subset_value_for_me(subset, offers)
            key = (error, -value, len(subset))
            if best_key is None or key < best_key:
                best_key = key
                best_set = set(subset)
        return best_set, (best_key[0] if best_key is not None else int(needs))

    def _partial_accept_set(self, needs: int, partners, offers):
        """部分受諾する組み合わせ (合計 <= needs) を返す。

        取引は完了しなくても、オファーの中で「割のよい部分」だけは受け取る。
        各相手の公平分 ``floor(needs / 人数)`` を 1 以上上回る量
        (= ``floor(needs / 人数) + 1`` 以上) のオファーを「良い部分」とみなし、
        その中から合計が needs を超えない範囲で最大になる組み合わせを受諾する。
        残りはカウンターで取りに行く。

        例: needs=10, 4 人, オファー (3,3,3,3) のとき
            閾値 = floor(10/4)+1 = 3 → 4 件とも該当。
            合計 <= 10 で最大は (3,3,3)=9 を受諾し、残り 1 をカウンター。
        """
        n = len(partners)
        if n <= 0 or int(needs) <= 0:
            return set()

        quality_threshold = int(needs) // n + 1
        good = [
            partner
            for partner in partners
            if int(offers[partner][QUANTITY]) >= quality_threshold
            and int(offers[partner][QUANTITY]) > 0
        ]
        if not good:
            return set()

        best_key = None
        best_set: set[str] = set()
        for subset in powerset(good):
            total = sum(int(offers[partner][QUANTITY]) for partner in subset)
            if total > int(needs):
                continue
            # 合計を最大化 (= 残りを最小化)。同点では自分の価値が高く、人数の
            # 少ない方を優先。
            key = (total, self._subset_value_for_me(subset, offers), -len(subset))
            if best_key is None or key > best_key:
                best_key = key
                best_set = set(subset)
        return best_set

    def _subset_value_for_me(self, partners, offers) -> float:
        value = 0.0
        for partner in partners:
            offer = offers[partner]
            quantity = int(offer[QUANTITY])
            price = float(offer[UNIT_PRICE])
            if self._is_seller_to(partner):
                value += quantity * price
            else:
                value -= quantity * price
        return value

    def _subset_utility(self, offers_subset) -> float:
        try:
            return float(self.ufun.from_offers(offers_subset))
        except Exception:
            total = 0.0
            for partner, offer in offers_subset.items():
                quantity = int(offer[QUANTITY])
                price = float(offer[UNIT_PRICE])
                if self._is_seller_to(partner):
                    total += quantity * price
                else:
                    total -= quantity * price
            return total

    def _forced_profit_max_responses(self, partners, offers, max_total: int | None = None):
        # 何も受けない (空集合) を基準に、効用を最大化する組み合わせを選ぶ。
        # max_total 指定時は合計がそれを超える組み合わせを除外 (過剰契約=不足の禁止)。
        best_set: set[str] = set()
        best_utility = self._subset_utility({})
        for subset in powerset(partners):
            if max_total is not None and sum(
                int(offers[p][QUANTITY]) for p in subset
            ) > int(max_total):
                continue
            subset_offers = {partner: offers[partner] for partner in subset}
            utility = self._subset_utility(subset_offers)
            if utility > best_utility:
                best_utility = utility
                best_set = set(subset)

        responses = {}
        for partner in partners:
            if partner in best_set:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offers[partner]
                )
            else:
                responses[partner] = self._unneeded_response()
        return responses

    def _accept_subset_and_counter(self, needs, partners, offers, accept_set, greedy_partners, t=0.0):
        return self._counter_side(needs, partners, offers, set(accept_set), greedy_partners, t)

    def _counter_side(self, needs, partners, offers, accepted, greedy_partners, t=0.0):
        responses = {}
        for partner in accepted:
            responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[partner])

        accepted_quantity = sum(int(offers[partner][QUANTITY]) for partner in accepted)
        remaining = max(0, int(needs) - accepted_quantity)
        counter_partners = [partner for partner in partners if partner not in accepted]

        if remaining <= 0:
            for partner in counter_partners:
                responses[partner] = self._unneeded_response()
            return responses

        accepted_offers = {partner: offers[partner] for partner in accepted}
        quantities = self._equal_counter_quantities(remaining, counter_partners)
        for index, partner in enumerate(counter_partners):
            quantity = quantities.get(partner, 0)
            if quantity <= 0:
                responses[partner] = self._unneeded_response()
                continue
            price = self._counter_price_for(partner, greedy_partners, index)
            # 時間譲歩: t に応じてカウンター数量を相手のオファー量へ寄せる
            # (損益分岐点ガード付き)。
            quantity = self._time_conceded_quantity(
                partner, quantity, offers[partner], t, price, accepted_offers
            )
            if quantity <= 0:
                responses[partner] = self._unneeded_response()
                continue
            counter = self._raw_offer(partner, quantity, price)
            # 同じ数量でカウンターするなら現在のオファーを受諾。
            responses[partner] = self._counter_or_accept_response(
                partner,
                offers[partner],
                counter,
            )
        return responses

    def _counter_price_for(self, partner, greedy_partners, index: int = 0) -> int:
        if (
            self._is_process_zero_agent()
            and partner in greedy_partners
        ):
            return self._worst_price_for_me(partner)
        if self._is_seller_to(partner) and self.awi.current_step < self.exploration_days:
            return self._exploration_probe_price(partner, index)
        return self._good_price_for(partner, greedy_partners)

    def _time_conceded_quantity(self, partner, desired_q, offer, t, price, accepted_offers):
        """カウンター数量を t に応じて相手のオファー量へ寄せる (B022 同様)。

        ただし **損益分岐点を守る**: 親の ``_max_floor_quantity`` で、
        その量を約定しても効用が無契約 (disagreement) 効用を下回らない範囲に
        制限する。売り手は需要超過 (= 不足ペナルティ) になる増量を許さない。
        """
        desired_q = int(desired_q)
        if offer is None or t <= self.quantity_concession_start or len(offer) <= QUANTITY:
            return desired_q
        opponent_q = int(offer[QUANTITY])
        if opponent_q == desired_q:
            return desired_q
        span = max(1e-9, 0.95 - self.quantity_concession_start)
        concession = max(0.0, min(1.0, (float(t) - self.quantity_concession_start) / span))
        target = int(round(desired_q + (opponent_q - desired_q) * concession))
        # 損益分岐点 (無契約効用) を下回らない量に制限。
        return self._max_floor_quantity(
            partner, desired_q, target, int(price), accepted_offers
        )
