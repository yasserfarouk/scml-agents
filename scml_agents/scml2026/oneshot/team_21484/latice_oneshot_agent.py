                     
\
\
\
\
\
\
   

from __future__ import annotations

import itertools
import json
import math
import os
import random

from negmas import Contract, ResponseType, SAOResponse
from scml.oneshot import *                

from .config import LatticeAgentConfig

__all__ = ["LatticeAgentConfig", "LatticeOneShotAgent"]


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal: bool = False,
    concentrated: bool = False,
    allow_zero: bool = False,
    concentrated_idx: list[int] = [],
) -> list[int]:
                                           
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        concentrated_lst = sorted(lst, reverse=True)[: len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i, x in zip(concentrated_idx, concentrated_lst):
            lst.insert(i, x)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]


def powerset(iterable):
             
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class LatticeOneShotAgent(OneShotSyncAgent):
    debug_log_enabled = os.getenv("LATTICE_DEBUG_LOG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    debug_log_file_path = os.getenv("LATTICE_DEBUG_LOG_FILE", "")

    def __init__(
        self,
        *args,
        config: LatticeAgentConfig | None = None,
        **kwargs,
    ):
        agent_kwargs = dict(kwargs)
        self.debug_log_enabled = bool(
            agent_kwargs.pop("debug_log_enabled", self.debug_log_enabled)
        )
        self.debug_log_file_path = str(
            agent_kwargs.pop("debug_log_file_path", self.debug_log_file_path)
        )
        config_field_names = LatticeAgentConfig.field_names()
        config_kwargs = {
            key: agent_kwargs.pop(key)
            for key in list(agent_kwargs)
            if key in config_field_names
        }
        if config is not None and config_kwargs:
            raise ValueError("Pass either config=... or config fields directly, not both.")

        self.config = config if config is not None else LatticeAgentConfig(**config_kwargs)

        self.equal_distribution = self.config.equal
        self.first_overordering_scale = self.config.first_overordering_scale
        self.first_overordering_use_sd_ratio = self.config.first_overordering_use_sd_ratio
        self.first_overordering_gap_scale = self.config.first_overordering_gap_scale
        self.first_overordering_gap_positive_only = (
            self.config.first_overordering_gap_positive_only
        )
        self.first_distribution_mode = self.config.first_distribution_mode
        self.first_distribution_gap_pressure_mode = (
            self.config.first_distribution_gap_pressure_mode
        )
        self.first_overshoot_lockin_penalty = (
            self.config.first_overshoot_lockin_penalty
        )
        self.first_counter_continuation_value = (
            self.config.first_counter_continuation_value
        )
        self.first_proposal_warmup_steps = self.config.first_proposal_warmup_steps
        self.first_trinary_outcome_eval_mode = (
            self.config.first_trinary_outcome_eval_mode
        )
        self.first_trinary_allocation_candidate_mode = (
            self.config.first_trinary_allocation_candidate_mode
        )
        self.counter_overordering_scale = self.config.counter_overordering_scale
        self.counter_overordering_exp = self.config.counter_overordering_exp
        self.counter_overordering_gap_scale = (
            self.config.counter_overordering_gap_scale
        )
        self.use_incoming_quantity_counter_distribution = (
            self.config.use_incoming_quantity_counter_distribution
        )
        self.counter_distribution_always_min_one = (
            self.config.counter_distribution_always_min_one
        )
        self.use_large_shortage_offer_acceptance = (
            self.config.use_large_shortage_offer_acceptance
        )
        self.large_offer_acceptance_margin = self.config.large_offer_acceptance_margin
        self.mismatch_exp = self.config.mismatch_exp
        self.undermismatch_sell = self.config.undermismatch_sell
        self.overmismatch_sell = self.config.overmismatch_sell
        self.undermismatch_buy = self.config.undermismatch_buy
        self.overmismatch_buy = self.config.overmismatch_buy
        self.overmismatch_sell_linear = self.config.overmismatch_sell_linear
        self.overmismatch_buy_linear = self.config.overmismatch_buy_linear
        self.use_cash_tiebreak = self.config.use_cash_tiebreak
        self.use_gap_price = self.config.use_gap_price
        self.counter_offer_price_mode = self.config.counter_offer_price_mode
        self.counter_price_warmup_steps = self.config.counter_price_warmup_steps
        self.counter_price_min_sample_per_side = (
            self.config.counter_price_min_sample_per_side
        )
        self.counter_price_accept_rate_margin = (
            self.config.counter_price_accept_rate_margin
        )
        self.first_price_warmup_steps = self.config.first_price_warmup_steps
        self.first_price_min_sample_per_side = (
            self.config.first_price_min_sample_per_side
        )
        self.first_price_accept_rate_margin = (
            self.config.first_price_accept_rate_margin
        )
        self.gap_d_scaler = self.config.gap_d_scaler
        self.no_change_distribute = self.config.no_change_distribute
        self.utility_fallback_relative_time = (
            self.config.utility_fallback_relative_time
        )
        self.deactivate_acceptanve_gate = self.config.deactivate_acceptanve_gate
        self.use_utility_acceptance_choice = (
            self.config.use_utility_acceptance_choice
        )
        self.use_future_partner_prior = self.config.use_future_partner_prior
        self.future_partner_prior_weight = self.config.future_partner_prior_weight
        self.counter_accept_matching_offer_tolerance = (
            self.config.counter_accept_matching_offer_tolerance
        )
        self.counter_acceptance_prior_weight = (
            self.config.counter_acceptance_prior_weight
        )
        self.counter_acceptance_rate_warmup_steps = (
            self.config.counter_acceptance_rate_warmup_steps
        )
        self.counter_acceptance_rate_prior_by_level_delta = (
            self.config.counter_acceptance_rate_prior_by_level_delta
        )
        self.use_counter_trinary_dp_decision = (
            self.config.use_counter_trinary_dp_decision
        )
        self.use_counter_trinary_dp_before_threshold = (
            self.config.use_counter_trinary_dp_before_threshold
        )
        self.use_counter_trinary_same_sign_smoothing = (
            self.config.use_counter_trinary_same_sign_smoothing
        )
        self.use_counter_trinary_neutral_quantity_shrink = (
            self.config.use_counter_trinary_neutral_quantity_shrink
        )
        self.counter_trinary_approx_candidate_mode = (
            self.config.counter_trinary_approx_candidate_mode
        )
        self.counter_trinary_dp_margin = self.config.counter_trinary_dp_margin
        self.counter_trinary_dp_lookahead_rounds = (
            self.config.counter_trinary_dp_lookahead_rounds
        )
        self.counter_trinary_total_quantity_mode = (
            self.config.counter_trinary_total_quantity_mode
        )
        self.counter_trinary_extended_total_quantity_multiplier = (
            self.config.counter_trinary_extended_total_quantity_multiplier
        )
        self.counter_trinary_rate_prior_by_level_delta = (
            self.config.counter_trinary_rate_prior_by_level_delta
        )
        self.counter_trinary_rate_prior_by_level_attempt_delta = (
            self.config.counter_trinary_rate_prior_by_level_attempt_delta
        )
        self.counter_trinary_neutral_lambda_prior = (
            self.config.counter_trinary_neutral_lambda_prior
        )
        self.counter_trinary_neutral_lambda_prior_by_level_side_delta = (
            self.config.counter_trinary_neutral_lambda_prior_by_level_side_delta
        )
        self.first_trinary_response_prior = self.config.first_trinary_response_prior
        self.first_trinary_response_prior_weight = (
            self.config.first_trinary_response_prior_weight
        )
        super().__init__(*args, **agent_kwargs)

    def log_message(
        self,
        *args: object,
        sep: str = " ",
        end: str = "\n",
        file: object | None = None,
        flush: bool = False,
    ) -> None:
\
\
\
\
           
        del file, flush
        if not self.debug_log_enabled:
            return
        msg = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            msg = f"{msg}{end}"
        if msg:
            self.awi.loginfo_agent(msg)
            if self.debug_log_file_path:
                try:
                    with open(self.debug_log_file_path, "a", encoding="utf-8") as f:
                        f.write(msg)
                        f.write("\n")
                except OSError:
                    pass

    def _debug_value(self, value):
        if isinstance(value, dict):
            return {str(k): self._debug_value(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(str(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [self._debug_value(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if value.__class__.__module__.startswith("numpy") and hasattr(value, "item"):
            return value.item()
        if hasattr(value, "_asdict"):
            return self._debug_value(value._asdict())
        if hasattr(value, "__attrs_attrs__"):
            return {
                attr.name: self._debug_value(getattr(value, attr.name))
                for attr in value.__attrs_attrs__
            }
        if hasattr(value, "__dict__"):
            return self._debug_value(vars(value))
        return str(value)

    def _debug_log_decision(self, event: str, **fields) -> None:
        if not self.debug_log_enabled:
            return
        payload = {
            "event": event,
            "agent_id": self.id,
            "step": self.awi.current_step,
            "is_first_level": self.awi.is_first_level,
            "relative_step": self.awi.current_step / max(1, self.awi.n_steps - 1),
            "gap_d": self.get_gap_d(),
            "gap_pressure": self.get_gap_pressure(),
        }
        payload.update({k: self._debug_value(v) for k, v in fields.items()})
        self.log_message(
            "LATTICE_DEBUG "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def _debug_offer_dict(self, offers: dict[str, tuple]) -> dict[str, dict[str, int]]:
        return {
            partner_id: {
                "quantity": offer[QUANTITY],
                "time": offer[TIME],
                "unit_price": offer[UNIT_PRICE],
            }
            for partner_id, offer in offers.items()
        }

    def _debug_response_dict(
        self, responses: dict[str, SAOResponse]
    ) -> dict[str, dict[str, object]]:
        return {
            partner_id: {
                "response": str(response.response),
                "outcome": (
                    None
                    if response.outcome is None
                    else {
                        "quantity": response.outcome[QUANTITY],
                        "time": response.outcome[TIME],
                        "unit_price": response.outcome[UNIT_PRICE],
                    }
                ),
            }
            for partner_id, response in responses.items()
        }

    def _debug_contract_summary(self, contract: Contract) -> dict[str, object]:
        partner_id = next((p for p in contract.partners if p != self.id), None)
        annotation = dict(getattr(contract, "annotation", {}) or {})
        agreement = dict(contract.agreement)
        return {
            "id": getattr(contract, "id", None),
            "partner_id": partner_id,
            "partners": list(contract.partners),
            "agreement": agreement,
            "annotation": annotation,
            "is_selling": annotation.get("seller") == self.id,
        }

    def _debug_partner_set(
        self, plist: list[tuple[str, ...]], index: int
    ) -> list[str] | None:
        if 0 <= index < len(plist):
            return list(plist[index])
        return None

    def _debug_offer_set_summary(
        self, offers: dict[str, tuple], partner_ids
    ) -> dict[str, object]:
        partner_ids = tuple(partner_ids)
        return {
            "partner_ids": list(partner_ids),
            "offers": self._debug_offer_dict(
                {partner_id: offers[partner_id] for partner_id in partner_ids}
            ),
            "total_quantity": sum(offers[p][QUANTITY] for p in partner_ids),
            "weighted_price": self._price_tiebreak_score(offers, partner_ids),
            "total_cash": self._partners_total_cash(partner_ids),
        }

    def _world_layout_key(self) -> tuple[int, int]:
                                                                                     
        if self.awi.is_first_level:
            l0_count = self.awi.n_competitors + 1
            l1_count = len([p for p in self.awi.my_consumers if p != "BUYER"])
        else:
            l0_count = len([p for p in self.awi.my_suppliers if p != "SELLER"])
            l1_count = self.awi.n_competitors + 1
        return int(l0_count), int(l1_count)

    def _layout_config_value(self, parameter_name: str, default):
        field_name = self.config.layout_parameter_fields.get(
            self.layout_key, {}
        ).get(parameter_name)
        if field_name is None:
            return default
        return getattr(self.config, field_name, default)

    def init(self):
        self.layout_key = self._world_layout_key()
        self.undermismatch_sell = self._layout_config_value(
            "undermismatch_sell",
            self.undermismatch_sell,
        )
        self.overmismatch_sell = self._layout_config_value(
            "overmismatch_sell",
            self.overmismatch_sell,
        )
        if self.undermismatch_sell != -1:
            self.undermismatch_sell *= self.awi.n_lines
        if self.overmismatch_sell != -1:
            self.overmismatch_sell *= self.awi.n_lines
        if self.undermismatch_buy != -1:
            self.undermismatch_buy *= self.awi.n_lines
        if self.overmismatch_buy != -1:
            self.overmismatch_buy *= self.awi.n_lines

        self.first_overordering_scale = self._layout_config_value(
            "first_overordering_scale",
            self.config.first_overordering_scale,
        )

        self.rounds_ave_offered = (
            [self.awi.n_lines / len(self.awi.my_consumers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_consumers)] * 9
            + [1] * 10
            if self.awi.my_suppliers == ["SELLER"]
            else [self.awi.n_lines / len(self.awi.my_suppliers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_suppliers)] * 9
            + [1] * 10
        )

        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self._debug_step_contracts = []
        self._debug_step_failures = []
        self.counter_acceptance_stats = {}
        self.counter_trinary_stats = {}
        self.counter_trinary_neutral_lambda_stats = {}
        self.counter_price_delta_stats = {}
        self.counter_price_delta_price_stats = {}
        self.first_trinary_stats = {}
        self.first_price_quantity_stats = {}
        self.first_price_quantity_price_stats = {}
        self.first_trinary_full_search_count = 0
        self.pending_first_attempts = {}
        self.pending_counter_attempts = {}
        self.counter_attempt_order_by_partner = {}
        self.counter_trinary_dp_decision_seq = 0
        self._debug_step_dp_decisions = []
        self._counter_acceptance_table_logged = False
        self._counter_trinary_table_logged = False
        self._counter_price_response_table_logged = False
        self._first_trinary_table_logged = False
        self._first_price_response_table_logged = False
        result = super().init()
        self._debug_log_decision(
            "agent_init",
            config=self.config.to_kwargs(),
            n_lines=self.awi.n_lines,
            n_steps=self.awi.n_steps,
            my_suppliers=self.awi.my_suppliers,
            my_consumers=self.awi.my_consumers,
            needed_supplies=self.awi.needed_supplies,
            needed_sales=self.awi.needed_sales,
            exogenous_contract_summary=self.awi.exogenous_contract_summary,
            rounds_ave_offered=self.rounds_ave_offered,
            layout_key=self.layout_key,
            layout_parameter_fields=self.config.layout_parameter_fields.get(
                self.layout_key, {}
            ),
            effective_first_overordering_scale=self.first_overordering_scale,
            effective_undermismatch_sell=self.undermismatch_sell,
            effective_overmismatch_sell=self.overmismatch_sell,
        )
        return result

    def before_step(self):
        super().before_step()
        self._debug_step_contracts = []
        self._debug_step_failures = []
        self._debug_step_dp_decisions = []
        self.counter_attempt_order_by_partner = {}
        self.pending_first_attempts = {}
        self._debug_log_decision(
            "step_begin",
            n_lines=self.awi.n_lines,
            current_exogenous_input_quantity=(
                self.awi.current_exogenous_input_quantity
            ),
            current_exogenous_output_quantity=(
                self.awi.current_exogenous_output_quantity
            ),
            current_exogenous_input_price=self.awi.current_exogenous_input_price,
            current_exogenous_output_price=self.awi.current_exogenous_output_price,
            current_shortfall_penalty=self.awi.current_shortfall_penalty,
            current_disposal_cost=self.awi.current_disposal_cost,
            current_storage_cost=self.awi.current_storage_cost,
            current_inventory=self.awi.current_inventory,
            current_balance=self.awi.current_balance,
            needed_supplies=self.awi.needed_supplies,
            needed_sales=self.awi.needed_sales,
        )

    def step(self):
        self._debug_log_step_summary()
        if self.awi.current_step >= self.awi.n_steps - 1:
            self._log_first_trinary_table_summary()
            self._log_first_price_response_table_summary()
            self._log_counter_acceptance_table_summary()
            self._log_counter_trinary_table_summary()
            self._log_counter_price_response_table_summary()
        return super().step()

    def distribute_needs(
        self,
        t: float,
        mx: int | None = None,
        equal: bool | None = None,
        allow_zero: bool | None = None,
        concentrated: bool = False,
        concentrated_ids: list[str] = [],
    ) -> dict[str, int]:
                                                                 

        if equal is None:
            equal = self.equal_distribution
        if allow_zero is None:
            allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners, n_partners = [], 0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            is_selling = all_partners == self.awi.my_consumers
            offering_quanitity = (
                int(needs * (1 + self._counter_overorder_ratio(t, is_selling)))
                if len(partners) > 1
                else needs
            )
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            n_partners,
                            mx=mx,
                            equal=equal,
                            concentrated=concentrated,
                            allow_zero=allow_zero,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            )
        return dist

    def _partner_latest_cash(self, partner_id: str) -> int:
                                                                                                  
        reports = self.awi.reports_of_agent(partner_id)
        if not reports:
            return 0
        return reports[max(reports)].cash

    def _partners_total_cash(self, partner_ids) -> int:
        return sum(self._partner_latest_cash(p) for p in partner_ids)

    def _price_tiebreak_score(self, offers, partner_ids) -> int:
        return sum(offers[p][QUANTITY] * offers[p][UNIT_PRICE] for p in partner_ids)

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        quantity = int(contract.agreement["quantity"])
        self.total_agreed_quantity[partner_id] = (
            self.total_agreed_quantity.get(partner_id, 0) + quantity
        )
        self._resolve_pending_first_attempt_accept(partner_id)
        self._resolve_pending_counter_attempt_success(
            partner_id,
            quantity,
            contract.agreement.get("unit_price"),
        )
        if self.debug_log_enabled:
            self._debug_step_contracts.append(contract)
        self._debug_log_decision(
            "negotiation_success",
            partner_id=partner_id,
            contract=self._debug_contract_summary(contract),
            mechanism_id=getattr(mechanism, "id", None),
            total_agreed_quantity=dict(self.total_agreed_quantity),
        )

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        for partner_id in partners:
            if partner_id != self.id:
                self._resolve_pending_first_attempt_reject(partner_id)
                self._resolve_pending_counter_attempt_failure(partner_id)
        failure = {
            "partners": list(partners),
            "annotation": dict(annotation or {}),
            "mechanism_id": getattr(mechanism, "id", None),
            "state_step": getattr(state, "step", None),
            "relative_time": getattr(state, "relative_time", None),
            "current_offer": getattr(state, "current_offer", None),
            "agreement": getattr(state, "agreement", None),
            "ended": getattr(state, "ended", None),
            "timedout": getattr(state, "timedout", None),
            "broken": getattr(state, "broken", None),
        }
        if self.debug_log_enabled:
            self._debug_step_failures.append(failure)
        self._debug_log_decision("negotiation_failure", **failure)

    def _debug_log_step_summary(self) -> None:
        if not self.debug_log_enabled:
            return
        contracts = list(getattr(self, "_debug_step_contracts", []))
        contract_summaries = [self._debug_contract_summary(c) for c in contracts]
        negotiated_supplies = sum(
            int(summary["agreement"].get("quantity", 0))
            for summary in contract_summaries
            if summary["is_selling"] is False
        )
        negotiated_sales = sum(
            int(summary["agreement"].get("quantity", 0))
            for summary in contract_summaries
            if summary["is_selling"] is True
        )

        exogenous_input = self.awi.current_exogenous_input_quantity
        exogenous_output = self.awi.current_exogenous_output_quantity
        if self.awi.is_first_level:
            available = exogenous_input
            target = negotiated_sales
        elif self.awi.is_last_level:
            available = negotiated_supplies
            target = exogenous_output
        else:
            available = min(exogenous_input + negotiated_supplies, exogenous_output)
            target = max(exogenous_output, negotiated_sales)
        estimated_shortfall = max(0, target - available)
        estimated_disposal = max(0, available - target)

        utility_with_exogenous = None
        utility_without_exogenous = None
        try:
            utility_with_exogenous = self.ufun.from_contracts(
                contracts,
                return_info=True,
                ignore_exogenous=False,
            )
        except Exception as exc:                                            
            utility_with_exogenous = {"error": repr(exc)}
        try:
            utility_without_exogenous = self.ufun.from_contracts(
                contracts,
                return_info=True,
                ignore_exogenous=True,
            )
        except Exception as exc:                                            
            utility_without_exogenous = {"error": repr(exc)}

        self._debug_log_counter_trinary_dp_realizations(
            contracts=contracts,
            contract_summaries=contract_summaries,
            failures=getattr(self, "_debug_step_failures", []),
            step_utility_with_exogenous=utility_with_exogenous,
            step_utility_without_exogenous=utility_without_exogenous,
        )

        self._debug_log_decision(
            "step_summary",
            contracts=contract_summaries,
            failures=getattr(self, "_debug_step_failures", []),
            negotiated_supplies=negotiated_supplies,
            negotiated_sales=negotiated_sales,
            exogenous_input_quantity=exogenous_input,
            exogenous_output_quantity=exogenous_output,
            estimated_available_quantity=available,
            estimated_target_quantity=target,
            estimated_shortfall_quantity=estimated_shortfall,
            estimated_disposal_quantity=estimated_disposal,
            needed_supplies_end=self.awi.needed_supplies,
            needed_sales_end=self.awi.needed_sales,
            current_score=self.awi.current_score,
            current_balance=self.awi.current_balance,
            current_inventory=self.awi.current_inventory,
            utility_with_exogenous=utility_with_exogenous,
            utility_without_exogenous=utility_without_exogenous,
        )

    def _debug_log_counter_trinary_dp_realizations(
        self,
        *,
        contracts: list[Contract],
        contract_summaries: list[dict[str, object]],
        failures: list[dict[str, object]],
        step_utility_with_exogenous,
        step_utility_without_exogenous,
    ) -> None:
        if not self.debug_log_enabled:
            return
        decisions = list(getattr(self, "_debug_step_dp_decisions", []))
        if not decisions:
            return

        step_utility = self._extract_utility_value(step_utility_with_exogenous)
        step_utility_without = self._extract_utility_value(
            step_utility_without_exogenous
        )
        for decision in decisions:
            active_partners = set(decision.get("active_partners", []))
            is_selling = bool(decision.get("is_selling", False))
            related_contracts = [
                contract
                for contract in contracts
                if self._contract_matches_dp_decision(
                    contract,
                    active_partners,
                    is_selling,
                )
            ]
            related_contract_summaries = [
                summary
                for summary in contract_summaries
                if summary.get("partner_id") in active_partners
                and bool(summary.get("is_selling")) == is_selling
            ]
            related_failures = [
                failure
                for failure in failures
                if self._failure_matches_dp_decision(
                    failure,
                    active_partners,
                    is_selling,
                )
            ]
            related_utility_with_exogenous = self._contracts_utility_payload(
                related_contracts,
                ignore_exogenous=False,
            )
            related_utility_without_exogenous = self._contracts_utility_payload(
                related_contracts,
                ignore_exogenous=True,
            )
            related_utility = self._extract_utility_value(
                related_utility_with_exogenous
            )
            dp_expected = decision.get("dp_expected")
            baseline_expected = decision.get("baseline_expected")
            self._debug_log_decision(
                "counter_trinary_dp_realization",
                counter_trinary_dp_decision_id=decision.get(
                    "counter_trinary_dp_decision_id"
                ),
                is_selling=is_selling,
                active_partners=sorted(active_partners),
                accepted_partner_ids=decision.get("accepted_partner_ids", []),
                counter_partner_ids=decision.get("counter_partner_ids", []),
                counter_quantities=decision.get("counter_quantities", {}),
                best_action_type=decision.get("best_action_type"),
                dp_expected=dp_expected,
                baseline_expected=baseline_expected,
                expected_improvement=(
                    None
                    if dp_expected is None or baseline_expected is None
                    else float(dp_expected) - float(baseline_expected)
                ),
                related_contracts=related_contract_summaries,
                related_failures=related_failures,
                related_utility_with_exogenous=related_utility_with_exogenous,
                related_utility_without_exogenous=related_utility_without_exogenous,
                step_utility_with_exogenous=step_utility_with_exogenous,
                step_utility_without_exogenous=step_utility_without_exogenous,
                related_realized_utility=related_utility,
                step_realized_utility=step_utility,
                step_realized_utility_without_exogenous=step_utility_without,
                related_prediction_error=(
                    None
                    if related_utility is None or dp_expected is None
                    else float(related_utility) - float(dp_expected)
                ),
                step_prediction_error=(
                    None
                    if step_utility is None or dp_expected is None
                    else float(step_utility) - float(dp_expected)
                ),
                realized_quantity=sum(
                    int(summary["agreement"].get("quantity", 0))
                    for summary in related_contract_summaries
                ),
            )

    def _extract_utility_value(self, payload) -> float | None:
        if isinstance(payload, dict):
            value = payload.get("utility")
            if isinstance(value, (int, float)):
                return float(value)
        if hasattr(payload, "utility"):
            value = getattr(payload, "utility")
            if isinstance(value, (int, float)):
                return float(value)
        if isinstance(payload, (int, float)):
            return float(payload)
        return None

    def _contracts_utility_payload(
        self,
        contracts: list[Contract],
        *,
        ignore_exogenous: bool,
    ):
        try:
            return self.ufun.from_contracts(
                contracts,
                return_info=True,
                ignore_exogenous=ignore_exogenous,
            )
        except Exception as exc:                                            
            return {"error": repr(exc)}

    def _contract_matches_dp_decision(
        self,
        contract: Contract,
        active_partners: set[str],
        is_selling: bool,
    ) -> bool:
        summary = self._debug_contract_summary(contract)
        return (
            summary.get("partner_id") in active_partners
            and bool(summary.get("is_selling")) == is_selling
        )

    def _failure_matches_dp_decision(
        self,
        failure: dict[str, object],
        active_partners: set[str],
        is_selling: bool,
    ) -> bool:
        partners = set(str(partner) for partner in failure.get("partners", []))
        annotation = dict(failure.get("annotation", {}) or {})
        partner_match = bool(partners.intersection(active_partners))
        seller_id = annotation.get("seller")
        side_match = (seller_id == self.id) == is_selling
        return partner_match and side_match

    def _counter_acceptance_table_payload(self) -> dict[str, object]:
        stats = getattr(self, "counter_acceptance_stats", {})
        table = []
        for (partner_id, delta_quantity), value in sorted(
            stats.items(),
            key=lambda item: (str(item[0][0]), int(item[0][1])),
        ):
            attempts = int(value["attempts"])
            successes = int(value["successes"])
            probability_detail = self._counter_acceptance_probability_detail(
                str(partner_id),
                int(delta_quantity),
            )
            table.append(
                {
                    "partner_id": partner_id,
                    "delta_quantity": int(delta_quantity),
                    "successes": successes,
                    "attempts": attempts,
                    "observed_acceptance_rate": (
                        successes / attempts if attempts > 0 else None
                    ),
                    "smoothed_p_accept": probability_detail["p_accept"],
                    "prior_rate": probability_detail["prior_rate"],
                    "prior_delta": probability_detail["prior_delta"],
                    "prior_successes": probability_detail["prior_successes"],
                    "prior_attempts": probability_detail["prior_attempts"],
                }
            )

        return {
            "event": "counter_acceptance_table_summary",
            "agent_id": self.id,
            "step": self.awi.current_step,
            "n_steps": self.awi.n_steps,
            "layout_key": getattr(self, "layout_key", None),
            "is_first_level": self.awi.is_first_level,
            "level": "L0" if self.awi.is_first_level else "L1",
            "n_lines": self.awi.n_lines,
            "n_competitors": self.awi.n_competitors,
            "my_suppliers": list(self.awi.my_suppliers),
            "my_consumers": list(self.awi.my_consumers),
            "exogenous_contract_summary": self.awi.exogenous_contract_summary,
            "current_exogenous_input_quantity": (
                self.awi.current_exogenous_input_quantity
            ),
            "current_exogenous_output_quantity": (
                self.awi.current_exogenous_output_quantity
            ),
            "current_shortfall_penalty": self.awi.current_shortfall_penalty,
            "current_disposal_cost": self.awi.current_disposal_cost,
            "use_counter_acceptance_rate_decision": (
                self.counter_acceptance_prior_weight >= 0
            ),
            "warmup_steps": self.counter_acceptance_rate_warmup_steps,
            "prior_weight": self.counter_acceptance_prior_weight,
            "prior_level": self._counter_acceptance_prior_level(),
            "prior_table": self.counter_acceptance_rate_prior_by_level_delta,
            "table": table,
            "pending_counter_attempts": list(
                getattr(self, "pending_counter_attempts", {}).values()
            ),
        }

    def _log_counter_acceptance_table_summary(self) -> None:
        if getattr(self, "_counter_acceptance_table_logged", False):
            return
        self._counter_acceptance_table_logged = True
        payload = self._debug_value(self._counter_acceptance_table_payload())
        self.log_message(
            "LATTICE_COUNTER_ACCEPTANCE_TABLE "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def _first_trinary_table_payload(self) -> dict[str, object]:
        stats = getattr(self, "first_trinary_stats", {})
        table = []
        for (partner_id, is_selling, quantity_bucket), value in sorted(
            stats.items(),
            key=lambda item: (str(item[0][0]), bool(item[0][1]), str(item[0][2])),
        ):
            accepts = int(value["accepts"])
            counters = int(value["counters"])
            rejects = int(value["rejects"])
            total = accepts + counters + rejects
            representative_quantity = {
                "0": 0,
                "1": 1,
                "2": 2,
                "3-4": 3,
                "5-7": 5,
                "8+": 8,
            }.get(str(quantity_bucket), 1)
            probability_detail = self._first_trinary_probability_detail(
                str(partner_id),
                bool(is_selling),
                representative_quantity,
            )
            table.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "quantity_bucket": str(quantity_bucket),
                    "accepts": accepts,
                    "counters": counters,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_counter_rate": counters / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                    "smoothed_p_accept": probability_detail["p_accept"],
                    "smoothed_p_counter": probability_detail["p_counter"],
                    "smoothed_p_reject": probability_detail["p_reject"],
                    "prior": probability_detail["prior"],
                    "prior_weight": probability_detail["prior_weight"],
                }
            )
        return {
            "event": "first_trinary_table_summary",
            "agent_id": self.id,
            "step": self.awi.current_step,
            "n_steps": self.awi.n_steps,
            "layout_key": getattr(self, "layout_key", None),
            "is_first_level": self.awi.is_first_level,
            "level": "L0" if self.awi.is_first_level else "L1",
            "n_lines": self.awi.n_lines,
            "my_suppliers": list(self.awi.my_suppliers),
            "my_consumers": list(self.awi.my_consumers),
            "first_overshoot_lockin_penalty": self.first_overshoot_lockin_penalty,
            "first_counter_continuation_value": (
                self.first_counter_continuation_value
            ),
            "first_proposal_warmup_steps": self.first_proposal_warmup_steps,
            "prior": self.first_trinary_response_prior,
            "prior_weight": self.first_trinary_response_prior_weight,
            "table": table,
            "pending_first_attempts": list(
                getattr(self, "pending_first_attempts", {}).values()
            ),
        }

    def _log_first_trinary_table_summary(self) -> None:
        if getattr(self, "_first_trinary_table_logged", False):
            return
        self._first_trinary_table_logged = True
        payload = self._debug_value(self._first_trinary_table_payload())
        self.log_message(
            "LATTICE_FIRST_TRINARY_TABLE "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def _first_price_response_table_payload(self) -> dict[str, object]:
        quantity_rows = []
        quantity_stats = getattr(self, "first_price_quantity_stats", {})
        for (partner_id, is_selling, quantity), value in sorted(
            quantity_stats.items(),
            key=lambda item: (str(item[0][0]), bool(item[0][1]), int(item[0][2])),
        ):
            accepts = int(value["accepts"])
            counters = int(value["counters"])
            rejects = int(value["rejects"])
            total = accepts + counters + rejects
            quantity_rows.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "quantity": int(quantity),
                    "quantity_bucket": self._first_quantity_bucket(int(quantity)),
                    "accepts": accepts,
                    "counters": counters,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_counter_rate": counters / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                }
            )

        quantity_price_rows = []
        quantity_price_stats = getattr(self, "first_price_quantity_price_stats", {})
        for (
            partner_id,
            is_selling,
            quantity,
            unit_price,
            partner_price_bucket,
        ), value in sorted(
            quantity_price_stats.items(),
            key=lambda item: (
                str(item[0][0]),
                bool(item[0][1]),
                int(item[0][2]),
                int(item[0][3]),
                str(item[0][4]),
            ),
        ):
            accepts = int(value["accepts"])
            counters = int(value["counters"])
            rejects = int(value["rejects"])
            total = accepts + counters + rejects
            quantity_price_rows.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "quantity": int(quantity),
                    "quantity_bucket": self._first_quantity_bucket(int(quantity)),
                    "unit_price": int(unit_price),
                    "partner_price_bucket": str(partner_price_bucket),
                    "accepts": accepts,
                    "counters": counters,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_counter_rate": counters / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                }
            )

        return {
            "event": "first_price_response_table_summary",
            "agent_id": self.id,
            "step": self.awi.current_step,
            "n_steps": self.awi.n_steps,
            "layout_key": getattr(self, "layout_key", None),
            "is_first_level": self.awi.is_first_level,
            "level": "L0" if self.awi.is_first_level else "L1",
            "n_lines": self.awi.n_lines,
            "n_competitors": self.awi.n_competitors,
            "my_suppliers": list(self.awi.my_suppliers),
            "my_consumers": list(self.awi.my_consumers),
            "first_price_warmup_steps": self.first_price_warmup_steps,
            "first_price_min_sample_per_side": (
                self.first_price_min_sample_per_side
            ),
            "first_price_accept_rate_margin": (
                self.first_price_accept_rate_margin
            ),
            "price_bucket_definition": {
                "self_favorable": "partner_favorability_score <= 1/3",
                "middle": "1/3 < partner_favorability_score < 2/3",
                "partner_favorable": "partner_favorability_score >= 2/3",
                "selling_first": "lower price is more favorable to partner",
                "buying_first": "higher price is more favorable to partner",
            },
            "quantity_table": quantity_rows,
            "quantity_price_table": quantity_price_rows,
            "pending_first_attempts": list(
                getattr(self, "pending_first_attempts", {}).values()
            ),
        }

    def _log_first_price_response_table_summary(self) -> None:
        if getattr(self, "_first_price_response_table_logged", False):
            return
        self._first_price_response_table_logged = True
        payload = self._debug_value(self._first_price_response_table_payload())
        self.log_message(
            "LATTICE_FIRST_PRICE_RESPONSE_TABLE "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def _counter_trinary_table_payload(self) -> dict[str, object]:
        stats = getattr(self, "counter_trinary_stats", {})
        table = []
        for (partner_id, is_selling, delta_bucket, attempt_bucket), value in sorted(
            stats.items(),
            key=lambda item: (
                str(item[0][0]),
                bool(item[0][1]),
                str(item[0][2]),
                str(item[0][3]),
            ),
        ):
            accepts = int(value["accepts"])
            neutrals = int(value["neutrals"])
            rejects = int(value["rejects"])
            total = accepts + neutrals + rejects
            representative_delta = self._counter_delta_bucket_representative(
                str(delta_bucket)
            )
            probability_detail = self._counter_trinary_probability_detail(
                str(partner_id),
                bool(is_selling),
                representative_delta,
                self._counter_attempt_index_from_bucket(str(attempt_bucket)),
            )
            table.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "delta_bucket": str(delta_bucket),
                    "attempt_bucket": str(attempt_bucket),
                    "accepts": accepts,
                    "neutrals": neutrals,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_neutral_rate": neutrals / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                    "smoothed_p_accept": probability_detail["p_accept"],
                    "smoothed_p_neutral": probability_detail["p_neutral"],
                    "smoothed_p_reject": probability_detail["p_reject"],
                    "prior_delta": probability_detail["prior_delta"],
                    "prior_accept": probability_detail["prior_accept"],
                    "prior_neutral": probability_detail["prior_neutral"],
                    "prior_reject": probability_detail["prior_reject"],
                    "prior_weight": probability_detail["prior_weight"],
                    "probability_source": probability_detail["source"],
                }
            )

        neutral_lambda_table = []
        neutral_lambda_stats = getattr(
            self,
            "counter_trinary_neutral_lambda_stats",
            {},
        )
        neutral_lambda_prior = getattr(
            self,
            "counter_trinary_neutral_lambda_prior",
            {},
        )
        neutral_lambda_prior_by_level_side_delta = getattr(
            self,
            "counter_trinary_neutral_lambda_prior_by_level_side_delta",
            {},
        )
        neutral_lambda_prior_weight = max(
            0.0,
            float(self.counter_acceptance_prior_weight),
        )
        for (
            partner_id,
            is_selling,
            delta_bucket,
            attempt_bucket,
        ), value in sorted(
            neutral_lambda_stats.items(),
            key=lambda item: (
                str(item[0][0]),
                bool(item[0][1]),
                str(item[0][2]),
                str(item[0][3]),
            ),
        ):
            counts = {float(k): float(v) for k, v in value.items()}
            online_total = sum(counts.values())
            prior_detail = self._counter_trinary_neutral_lambda_prior_detail(
                bool(is_selling),
                str(delta_bucket),
            )
            prior = prior_detail["prior"]
            numerator = sum(bucket * count for bucket, count in counts.items())
            denominator = online_total + neutral_lambda_prior_weight
            if neutral_lambda_prior_weight > 0.0:
                numerator += sum(
                    float(bucket) * float(probability) * neutral_lambda_prior_weight
                    for bucket, probability in prior.items()
                )
            smoothed_expected_lambda = (
                numerator / denominator
                if denominator > 0.0
                else sum(
                    float(bucket) * float(probability)
                    for bucket, probability in prior.items()
                )
            )
            neutral_lambda_table.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "delta_bucket": str(delta_bucket),
                    "attempt_bucket": str(attempt_bucket),
                    "counts": counts,
                    "online_total": online_total,
                    "observed_expected_lambda": (
                        sum(bucket * count for bucket, count in counts.items())
                        / online_total
                        if online_total > 0.0
                        else None
                    ),
                    "smoothed_expected_lambda": smoothed_expected_lambda,
                    "prior": prior,
                    "prior_source": prior_detail["prior_source"],
                    "prior_level": prior_detail["prior_level"],
                    "prior_side": prior_detail["prior_side"],
                    "prior_delta_bucket": prior_detail["prior_delta_bucket"],
                    "prior_weight": neutral_lambda_prior_weight,
                }
            )

        return {
            "event": "counter_trinary_table_summary",
            "agent_id": self.id,
            "step": self.awi.current_step,
            "n_steps": self.awi.n_steps,
            "layout_key": getattr(self, "layout_key", None),
            "is_first_level": self.awi.is_first_level,
            "level": "L0" if self.awi.is_first_level else "L1",
            "n_lines": self.awi.n_lines,
            "n_competitors": self.awi.n_competitors,
            "my_suppliers": list(self.awi.my_suppliers),
            "my_consumers": list(self.awi.my_consumers),
            "use_counter_trinary_dp_decision": self.use_counter_trinary_dp_decision,
            "lookahead_rounds": self.counter_trinary_dp_lookahead_rounds,
            "total_quantity_mode": self.counter_trinary_total_quantity_mode,
            "prior_weight": self.counter_acceptance_prior_weight,
            "prior_level": self._counter_acceptance_prior_level(),
            "prior_table": self.counter_trinary_rate_prior_by_level_attempt_delta,
            "neutral_lambda_prior": neutral_lambda_prior,
            "neutral_lambda_prior_by_level_side_delta": (
                neutral_lambda_prior_by_level_side_delta
            ),
            "neutral_lambda_prior_weight": neutral_lambda_prior_weight,
            "neutral_lambda_table": neutral_lambda_table,
            "table": table,
            "pending_counter_attempts": list(
                getattr(self, "pending_counter_attempts", {}).values()
            ),
        }

    def _log_counter_trinary_table_summary(self) -> None:
        if getattr(self, "_counter_trinary_table_logged", False):
            return
        self._counter_trinary_table_logged = True
        payload = self._debug_value(self._counter_trinary_table_payload())
        self.log_message(
            "LATTICE_COUNTER_TRINARY_TABLE "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def _counter_price_response_table_payload(self) -> dict[str, object]:
        delta_rows = []
        delta_stats = getattr(self, "counter_price_delta_stats", {})
        for (partner_id, is_selling, delta_quantity), value in sorted(
            delta_stats.items(),
            key=lambda item: (str(item[0][0]), bool(item[0][1]), int(item[0][2])),
        ):
            accepts = int(value["accepts"])
            counters = int(value["counters"])
            rejects = int(value["rejects"])
            total = accepts + counters + rejects
            delta_rows.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "delta_quantity": int(delta_quantity),
                    "delta_bucket": self._counter_delta_bucket(int(delta_quantity)),
                    "accepts": accepts,
                    "counters": counters,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_counter_rate": counters / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                }
            )

        delta_price_rows = []
        delta_price_stats = getattr(self, "counter_price_delta_price_stats", {})
        for (
            partner_id,
            is_selling,
            delta_quantity,
            unit_price,
            partner_price_bucket,
        ), value in sorted(
            delta_price_stats.items(),
            key=lambda item: (
                str(item[0][0]),
                bool(item[0][1]),
                int(item[0][2]),
                int(item[0][3]),
                str(item[0][4]),
            ),
        ):
            accepts = int(value["accepts"])
            counters = int(value["counters"])
            rejects = int(value["rejects"])
            total = accepts + counters + rejects
            delta_price_rows.append(
                {
                    "partner_id": partner_id,
                    "is_selling": bool(is_selling),
                    "delta_quantity": int(delta_quantity),
                    "delta_bucket": self._counter_delta_bucket(int(delta_quantity)),
                    "unit_price": int(unit_price),
                    "partner_price_bucket": str(partner_price_bucket),
                    "accepts": accepts,
                    "counters": counters,
                    "rejects": rejects,
                    "observed_accept_rate": accepts / total if total else None,
                    "observed_counter_rate": counters / total if total else None,
                    "observed_reject_rate": rejects / total if total else None,
                }
            )

        return {
            "event": "counter_price_response_table_summary",
            "agent_id": self.id,
            "step": self.awi.current_step,
            "n_steps": self.awi.n_steps,
            "layout_key": getattr(self, "layout_key", None),
            "is_first_level": self.awi.is_first_level,
            "level": "L0" if self.awi.is_first_level else "L1",
            "n_lines": self.awi.n_lines,
            "n_competitors": self.awi.n_competitors,
            "my_suppliers": list(self.awi.my_suppliers),
            "my_consumers": list(self.awi.my_consumers),
            "counter_offer_price_mode": self.counter_offer_price_mode,
            "counter_price_warmup_steps": self.counter_price_warmup_steps,
            "counter_price_min_sample_per_side": (
                self.counter_price_min_sample_per_side
            ),
            "counter_price_accept_rate_margin": (
                self.counter_price_accept_rate_margin
            ),
            "price_bucket_definition": {
                "self_favorable": "partner_favorability_score <= 1/3",
                "middle": "1/3 < partner_favorability_score < 2/3",
                "partner_favorable": "partner_favorability_score >= 2/3",
                "selling_counter": "lower price is more favorable to partner",
                "buying_counter": "higher price is more favorable to partner",
            },
            "delta_table": delta_rows,
            "delta_price_table": delta_price_rows,
            "pending_counter_attempts": list(
                getattr(self, "pending_counter_attempts", {}).values()
            ),
        }

    def _log_counter_price_response_table_summary(self) -> None:
        if getattr(self, "_counter_price_response_table_logged", False):
            return
        self._counter_price_response_table_logged = True
        payload = self._debug_value(self._counter_price_response_table_payload())
        self.log_message(
            "LATTICE_COUNTER_PRICE_RESPONSE_TABLE "
            + json.dumps(payload, sort_keys=True, ensure_ascii=True)
        )

    def first_proposals(self):
        s, price = self._step_and_price(best_price=True)
        issues = (
            self.awi.current_output_issues
            if self.awi.is_first_level
            else self.awi.current_input_issues
        )
        my_negotiators, not_negotiators = [], []
        if self.awi.my_suppliers == ["SELLER"]:
            for k in self.awi.my_consumers:
                if self.awi.is_bankrupt(k) or (
                    self._is_late_phase()
                    and self.total_agreed_quantity.get(k, 0) == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            first_overorder_ratio = (
                self._first_overorder_ratio(is_selling=True)
                if len(my_negotiators) > 1
                else None
            )
            offering_quantity = (
                int(self.awi.needed_sales * (1 + first_overorder_ratio))
                if len(my_negotiators) > 1
                else self.awi.needed_sales
            )
            needs = self.awi.needed_sales
            is_selling = True
        else:
            for k in self.awi.my_suppliers:
                if self.awi.is_bankrupt(k) or (
                    self._is_late_phase()
                    and self.total_agreed_quantity.get(k, 0) == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            first_overorder_ratio = (
                self._first_overorder_ratio(is_selling=False)
                if len(my_negotiators) > 1
                else None
            )
            offering_quantity = (
                int(self.awi.needed_supplies * (1 + first_overorder_ratio))
                if len(my_negotiators) > 1
                else self.awi.needed_supplies
            )
            needs = self.awi.needed_supplies
            is_selling = False

        d = {}
        distribution = {}
        if len(my_negotiators) > 0:
            distribution = self._first_proposal_distribution(
                my_negotiators=my_negotiators,
                offering_quantity=offering_quantity,
                is_l1_first_proposer=self.awi.my_suppliers != ["SELLER"],
                needs=needs,
                price=price,
                issues=issues,
                is_selling=is_selling,
            )

            proposal_prices = {}
            for k, q in distribution.items():
                if q > 0 or self.awi.allow_zero_quantity:
                    proposal_price = self._first_offer_price(
                        issues,
                        is_selling,
                        partner_id=k,
                        quantity=int(q),
                        default_price=price,
                    )
                    proposal_prices[k] = proposal_price
                    d[k] = (q, s, proposal_price)
                else:
                    proposal_prices[k] = None
                    d[k] = None
            self._record_pending_first_attempts(d, is_selling)
        else:
            proposal_prices = {}
        d |= {k: None for k in not_negotiators}
        self._debug_log_decision(
            "first_proposals",
            is_selling=is_selling,
            needs=needs,
            first_overorder_ratio=first_overorder_ratio,
            offering_quantity=offering_quantity,
            default_price=price,
            proposal_prices=proposal_prices,
            my_negotiators=my_negotiators,
            not_negotiators=not_negotiators,
            distribution=distribution,
            proposals=d,
            late_phase=self._is_late_phase(),
            first_distribution_mode=self.first_distribution_mode,
            first_distribution_gap_pressure_mode=(
                self.first_distribution_gap_pressure_mode
            ),
            no_change_distribute=self.no_change_distribute,
        )
        return d

    def _first_proposal_distribution(
        self,
        *,
        my_negotiators: list[str],
        offering_quantity: int,
        is_l1_first_proposer: bool,
        needs: int,
        price: int,
        issues,
        is_selling: bool,
    ) -> dict[str, int]:
        if self._use_first_trinary_distribution():
            trinary_distribution = self._first_trinary_distribution(
                my_negotiators=my_negotiators,
                offering_quantity=offering_quantity,
                needs=needs,
                price=price,
                issues=issues,
                is_selling=is_selling,
                is_l1_first_proposer=is_l1_first_proposer,
            )
            if trinary_distribution is not None:
                return trinary_distribution

        return self._legacy_first_proposal_distribution(
            my_negotiators=my_negotiators,
            offering_quantity=offering_quantity,
            is_l1_first_proposer=is_l1_first_proposer,
        )

    def _legacy_first_proposal_distribution(
        self,
        *,
        my_negotiators: list[str],
        offering_quantity: int,
        is_l1_first_proposer: bool,
        log_decision: bool = True,
    ) -> dict[str, int]:
        if self._use_equal_first_distribution_by_gap_pressure():
            quantities = distribute(
                offering_quantity,
                len(my_negotiators),
                equal=True,
            )
            distribution = dict(zip(my_negotiators, quantities))
            if log_decision:
                self._debug_log_decision(
                    "first_distribution",
                    method="gap_pressure_equal",
                    my_negotiators=my_negotiators,
                    offering_quantity=offering_quantity,
                    distribution=distribution,
                    first_distribution_gap_pressure_mode=(
                        self.first_distribution_gap_pressure_mode
                    ),
                )
            return distribution

        concentrated_idx: list[int] = []
        concentrated_ids: list[str] = []
        method = "normal"
        use_concentration = False

        if is_l1_first_proposer and self.first_distribution_mode == 1:
            concentrated_idx = [random.randrange(len(my_negotiators))]
            method = "first_mode_random_concentration"
            use_concentration = True
        elif is_l1_first_proposer and self.first_distribution_mode == 2:
            if self._is_late_phase():
                concentrated_ids = self._concentrated_ids(my_negotiators)
                concentrated_idx = [
                    i for i, k in enumerate(my_negotiators) if k in concentrated_ids
                ]
                method = "first_mode_late_performance_concentration"
            else:
                concentrated_idx = [random.randrange(len(my_negotiators))]
                method = "first_mode_early_random_concentration"
            use_concentration = bool(concentrated_idx)
        elif self._is_late_phase() and not self.no_change_distribute:
            concentrated_ids = self._concentrated_ids(my_negotiators)
            concentrated_idx = [
                i for i, k in enumerate(my_negotiators) if k in concentrated_ids
            ]
            method = "late_performance_concentration"
            use_concentration = bool(concentrated_idx)

        if use_concentration:
            quantities = distribute(
                offering_quantity,
                len(my_negotiators),
                mx=self.awi.n_lines,
                concentrated=True,
                concentrated_idx=concentrated_idx,
            )
        else:
            quantities = distribute(offering_quantity, len(my_negotiators))
        distribution = dict(zip(my_negotiators, quantities))
        if log_decision:
            self._debug_log_decision(
                "first_distribution",
                method=method,
                my_negotiators=my_negotiators,
                offering_quantity=offering_quantity,
                use_concentration=use_concentration,
                concentrated_ids=concentrated_ids,
                concentrated_idx=concentrated_idx,
                distribution=distribution,
                first_distribution_mode=self.first_distribution_mode,
                no_change_distribute=self.no_change_distribute,
                late_phase=self._is_late_phase(),
            )
        return distribution

    def _use_equal_first_distribution_by_gap_pressure(self) -> bool:
        if self.first_distribution_gap_pressure_mode == 1:
            return self.get_gap_pressure() > -1
        if self.first_distribution_gap_pressure_mode == 2:
            return self.get_gap_pressure() < -1
        return False

    def _use_first_trinary_distribution(self) -> bool:
        if (
            self.first_overshoot_lockin_penalty < 0
            and self.first_counter_continuation_value < 0
        ):
            return False
        if self.awi.current_step < self.first_proposal_warmup_steps:
            self._debug_log_decision(
                "first_trinary_distribution_skip",
                reason="warmup",
                current_step=self.awi.current_step,
                warmup_steps=self.first_proposal_warmup_steps,
            )
            return False
        return True

    def _first_trinary_distribution(
        self,
        *,
        my_negotiators: list[str],
        offering_quantity: int,
        needs: int,
        price: int,
        issues,
        is_selling: bool,
        is_l1_first_proposer: bool,
    ) -> dict[str, int] | None:
        if not my_negotiators:
            return None
        use_full_search = self._use_first_trinary_full_search(
            int(offering_quantity),
            len(my_negotiators),
        )
        if use_full_search:
            allocations = self._bounded_integer_allocations(
                int(offering_quantity),
                len(my_negotiators),
                self.awi.n_lines,
            )
        else:
            allocations = self._first_trinary_candidate_allocations(
                my_negotiators=my_negotiators,
                offering_quantity=int(offering_quantity),
                is_l1_first_proposer=is_l1_first_proposer,
            )
        if not allocations:
            self._debug_log_decision(
                "first_trinary_distribution_skip",
                reason="no_allocations",
                offering_quantity=offering_quantity,
                n_partners=len(my_negotiators),
            )
            return None
        outcome_multiplier = (
            1
            if int(self.first_trinary_outcome_eval_mode) == 2
            else 3 ** len(my_negotiators)
        )
        estimated_candidates = len(allocations) * outcome_multiplier
        max_candidates = 250000
        if estimated_candidates > max_candidates:
            self._debug_log_decision(
                "first_trinary_distribution_skip",
                reason="candidate_limit",
                offering_quantity=offering_quantity,
                n_partners=len(my_negotiators),
                allocation_count=len(allocations),
                outcome_multiplier=outcome_multiplier,
                estimated_candidates=estimated_candidates,
                max_candidates=max_candidates,
            )
            return None

        utility_by_quantity = self._first_quantity_utility_table(
            max_quantity=int(offering_quantity),
            partners=my_negotiators,
            price=price,
        )
        unit_scale = self._first_trinary_unit_scale(
            needs,
            my_negotiators,
            price,
        )
        best_score = -float("inf")
        best_allocation: dict[str, int] | None = None
        best_summary: dict[str, object] | None = None
        evaluated = 0
        outcome_eval_compare_count = 0
        outcome_eval_mismatch_count = 0
        outcome_eval_max_abs_diff = 0.0
        for allocation_tuple in allocations:
            allocation = {
                partner_id: int(quantity)
                for partner_id, quantity in zip(my_negotiators, allocation_tuple)
            }
            score, summary = self._evaluate_first_trinary_allocation(
                allocation=allocation,
                needs=needs,
                price=price,
                issues=issues,
                is_selling=is_selling,
                unit_scale=unit_scale,
                utility_by_quantity=utility_by_quantity,
            )
            evaluated += 1
            if "aggregate_dp_abs_diff" in summary:
                outcome_eval_compare_count += 1
                abs_diff = float(summary["aggregate_dp_abs_diff"])
                outcome_eval_max_abs_diff = max(outcome_eval_max_abs_diff, abs_diff)
                if abs_diff > 1e-9:
                    outcome_eval_mismatch_count += 1
            if math.isfinite(score) and score > best_score:
                best_score = score
                best_allocation = allocation
                best_summary = summary

        if best_allocation is None:
            self._debug_log_decision(
                "first_trinary_distribution_skip",
                reason="no_finite_score",
                offering_quantity=offering_quantity,
                allocation_count=len(allocations),
            )
            return None

        self._debug_log_decision(
            "first_trinary_distribution",
            selected=True,
            needs=needs,
            offering_quantity=offering_quantity,
            price=price,
            first_price_accept_rate_margin=self.first_price_accept_rate_margin,
            first_price_warmup_steps=self.first_price_warmup_steps,
            is_selling=is_selling,
            unit_scale=unit_scale,
            first_overshoot_lockin_penalty=self.first_overshoot_lockin_penalty,
            first_counter_continuation_value=self.first_counter_continuation_value,
            first_proposal_warmup_steps=self.first_proposal_warmup_steps,
            first_trinary_outcome_eval_mode=self.first_trinary_outcome_eval_mode,
            first_trinary_allocation_candidate_mode=(
                self.first_trinary_allocation_candidate_mode
            ),
            candidate_generation=(
                "full_integer_allocations"
                if use_full_search
                else "legacy_equal_concentrated_random_neighbors"
            ),
            first_trinary_full_search_count=self.first_trinary_full_search_count,
            allocation_count=len(allocations),
            evaluated_allocations=evaluated,
            outcome_eval_compare_count=outcome_eval_compare_count,
            outcome_eval_mismatch_count=outcome_eval_mismatch_count,
            outcome_eval_max_abs_diff=outcome_eval_max_abs_diff,
            best_score=best_score,
            best_allocation=best_allocation,
            best_summary=best_summary,
        )
        return best_allocation

    def _use_first_trinary_full_search(
        self,
        offering_quantity: int,
        n_partners: int,
    ) -> bool:
        mode = int(self.first_trinary_allocation_candidate_mode)
        if mode <= 0:
            return True
        if mode >= 2:
            return False
        full_candidate_count = self._bounded_integer_allocation_count(
            int(offering_quantity),
            int(n_partners),
            self.awi.n_lines,
        )
        return full_candidate_count <= 256

    def _first_trinary_candidate_allocations(
        self,
        *,
        my_negotiators: list[str],
        offering_quantity: int,
        is_l1_first_proposer: bool,
    ) -> list[tuple[int, ...]]:
        n_partners = len(my_negotiators)
        max_per_partner = int(self.awi.n_lines)
        max_candidates = 64
        if n_partners <= 0:
            return []
        if offering_quantity < 0 or offering_quantity > n_partners * max_per_partner:
            return []

        candidates: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add_candidate(values) -> None:
            if len(candidates) >= max_candidates:
                return
            allocation = tuple(int(v) for v in values)
            if len(allocation) != n_partners:
                return
            if sum(allocation) != int(offering_quantity):
                return
            if any(v < 0 or v > max_per_partner for v in allocation):
                return
            if allocation in seen:
                return
            seen.add(allocation)
            candidates.append(allocation)

        base_allocations: list[tuple[int, ...]] = []

        legacy_distribution = self._legacy_first_proposal_distribution(
            my_negotiators=my_negotiators,
            offering_quantity=offering_quantity,
            is_l1_first_proposer=is_l1_first_proposer,
            log_decision=False,
        )
        base_allocations.append(
            tuple(int(legacy_distribution.get(p, 0)) for p in my_negotiators)
        )

        base_allocations.append(
            self._balanced_integer_allocation(
                offering_quantity,
                n_partners,
                max_per_partner,
            )
        )

        concentrated_ids = self._concentrated_ids(my_negotiators)
        if concentrated_ids:
            concentrated_order = [
                i for i, partner_id in enumerate(my_negotiators)
                if partner_id in concentrated_ids
            ]
            base_allocations.append(
                self._balanced_integer_allocation(
                    offering_quantity,
                    n_partners,
                    max_per_partner,
                    preferred_indices=concentrated_order,
                )
            )

        for focus_index in range(n_partners):
            base_allocations.append(
                self._balanced_integer_allocation(
                    offering_quantity,
                    n_partners,
                    max_per_partner,
                    preferred_indices=[focus_index],
                )
            )

        random_attempts = min(12, 2 * max(1, n_partners))
        for _ in range(random_attempts):
            base_allocations.append(
                tuple(
                    distribute(
                        offering_quantity,
                        n_partners,
                        mx=max_per_partner,
                        allow_zero=True,
                    )
                )
            )

        for allocation in base_allocations:
            add_candidate(allocation)

        for allocation in list(candidates):
            if len(candidates) >= max_candidates:
                break
            for source_index, quantity in enumerate(allocation):
                if quantity <= 0:
                    continue
                for target_index, target_quantity in enumerate(allocation):
                    if source_index == target_index:
                        continue
                    if target_quantity >= max_per_partner:
                        continue
                    neighbor = list(allocation)
                    neighbor[source_index] -= 1
                    neighbor[target_index] += 1
                    add_candidate(neighbor)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

        self._debug_log_decision(
            "first_trinary_candidate_allocations",
            offering_quantity=offering_quantity,
            n_partners=n_partners,
            candidate_count=len(candidates),
            max_candidates=max_candidates,
            candidates=candidates,
        )
        return candidates

    def _balanced_integer_allocation(
        self,
        total_quantity: int,
        n_bins: int,
        max_per_bin: int,
        preferred_indices: list[int] | None = None,
    ) -> tuple[int, ...]:
        if n_bins <= 0:
            return ()
        allocation = [0] * n_bins
        remaining = int(total_quantity)
        preferred = [
            i for i in (preferred_indices or [])
            if 0 <= int(i) < n_bins
        ]
        for index in preferred:
            if remaining <= 0:
                break
            assigned = min(max_per_bin, remaining)
            allocation[index] += assigned
            remaining -= assigned

        preferred_set = set(preferred)
        order = preferred + [i for i in range(n_bins) if i not in preferred_set]
        if not order:
            order = list(range(n_bins))
        while remaining > 0:
            changed = False
            for index in order:
                if remaining <= 0:
                    break
                if allocation[index] >= max_per_bin:
                    continue
                allocation[index] += 1
                remaining -= 1
                changed = True
            if not changed:
                break
        return tuple(allocation)

    def _first_quantity_utility_table(
        self,
        *,
        max_quantity: int,
        partners: list[str],
        price: int,
    ) -> dict[int, float]:
        return {
            quantity: self._utility_from_offer_dict(
                self._first_virtual_offer_dict(quantity, partners, price)
            )
            for quantity in range(0, max(0, int(max_quantity)) + 1)
        }

    def _evaluate_first_trinary_allocation(
        self,
        *,
        allocation: dict[str, int],
        needs: int,
        price: int,
        issues,
        is_selling: bool,
        unit_scale: float,
        utility_by_quantity: dict[int, float],
    ) -> tuple[float, dict[str, object]]:
        active_ids = tuple(
            partner_id
            for partner_id, quantity in allocation.items()
            if int(quantity) > 0
        )
        probability_details = {
            partner_id: (
                self._first_trinary_probability_detail(
                    partner_id,
                    is_selling,
                    int(allocation[partner_id]),
                    price_bucket=self._first_price_bucket_for_trinary_probability(
                        is_selling,
                        partner_id,
                        int(allocation[partner_id]),
                    ),
                )
            )
            for partner_id in active_ids
        }
        if int(self.first_trinary_outcome_eval_mode) == 2:
            return self._evaluate_first_trinary_allocation_aggregate_dp(
                allocation=allocation,
                active_ids=active_ids,
                needs=needs,
                unit_scale=unit_scale,
                utility_by_quantity=utility_by_quantity,
                probability_details=probability_details,
            )

        legacy_score, legacy_summary = self._evaluate_first_trinary_allocation_legacy(
            allocation=allocation,
            active_ids=active_ids,
            needs=needs,
            unit_scale=unit_scale,
            utility_by_quantity=utility_by_quantity,
            probability_details=probability_details,
        )
        dp_score, dp_summary = self._evaluate_first_trinary_allocation_aggregate_dp(
            allocation=allocation,
            active_ids=active_ids,
            needs=needs,
            unit_scale=unit_scale,
            utility_by_quantity=utility_by_quantity,
            probability_details=probability_details,
        )
        abs_diff = abs(float(legacy_score) - float(dp_score))
        legacy_summary["outcome_eval_mode"] = "legacy_with_aggregate_dp_compare"
        legacy_summary["legacy_expected_score"] = legacy_score
        legacy_summary["aggregate_dp_expected_score"] = dp_score
        legacy_summary["aggregate_dp_abs_diff"] = abs_diff
        legacy_summary["aggregate_dp_summary"] = dp_summary
        return legacy_score, legacy_summary

    def _evaluate_first_trinary_allocation_legacy(
        self,
        *,
        allocation: dict[str, int],
        active_ids: tuple[str, ...],
        needs: int,
        unit_scale: float,
        utility_by_quantity: dict[int, float],
        probability_details: dict[str, dict[str, object]],
    ) -> tuple[float, dict[str, object]]:
        if not active_ids:
            return self._utility_from_offer_dict({}), {
                "expected_accepted_quantity": 0.0,
                "p_overshoot": 0.0,
                "expected_overshoot": 0.0,
                "expected_counter_recoverable": 0.0,
                "accepted_quantity_distribution": {0: 1.0},
                "probabilities_by_partner": probability_details,
                "outcome_eval_mode": "legacy_enumeration",
            }
        expected_score = 0.0
        expected_accepted_quantity = 0.0
        p_overshoot = 0.0
        expected_overshoot = 0.0
        expected_counter_recoverable = 0.0
        quantity_distribution: dict[int, float] = {}

        for outcome in itertools.product(
            ("accept", "counter", "reject"),
            repeat=len(active_ids),
        ):
            probability = 1.0
            accepted_partners = []
            counter_partners = []
            for partner_id, state in zip(active_ids, outcome):
                detail = probability_details[partner_id]
                if state == "accept":
                    probability *= float(detail["p_accept"])
                    accepted_partners.append(partner_id)
                elif state == "counter":
                    probability *= float(detail["p_counter"])
                    counter_partners.append(partner_id)
                else:
                    probability *= float(detail["p_reject"])
            if probability <= 0.0:
                continue

            accepted_quantity = sum(int(allocation[p]) for p in accepted_partners)
            utility = utility_by_quantity[accepted_quantity]
            delta = accepted_quantity - int(needs)
            recoverable = 0.0
            if delta > 0 and self.first_overshoot_lockin_penalty >= 0:
                utility -= float(unit_scale) * float(
                    self.first_overshoot_lockin_penalty
                ) * (0.5 + float(delta))
            elif delta < 0 and self.first_counter_continuation_value >= 0:
                counter_capacity = sum(
                    int(allocation[p]) * 0.5 for p in counter_partners
                )
                recoverable = min(float(-delta), float(counter_capacity))
                utility += (
                    float(unit_scale)
                    * float(self.first_counter_continuation_value)
                    * recoverable
                )

            expected_score += probability * utility
            expected_accepted_quantity += probability * accepted_quantity
            quantity_distribution[accepted_quantity] = (
                quantity_distribution.get(accepted_quantity, 0.0) + probability
            )
            if delta > 0:
                p_overshoot += probability
                expected_overshoot += probability * delta
            expected_counter_recoverable += probability * recoverable

        return expected_score, {
            "expected_accepted_quantity": expected_accepted_quantity,
            "p_overshoot": p_overshoot,
            "expected_overshoot": expected_overshoot,
            "expected_counter_recoverable": expected_counter_recoverable,
            "accepted_quantity_distribution": quantity_distribution,
            "probabilities_by_partner": probability_details,
            "outcome_eval_mode": "legacy_enumeration",
        }

    def _evaluate_first_trinary_allocation_aggregate_dp(
        self,
        *,
        allocation: dict[str, int],
        active_ids: tuple[str, ...],
        needs: int,
        unit_scale: float,
        utility_by_quantity: dict[int, float],
        probability_details: dict[str, dict[str, object]],
    ) -> tuple[float, dict[str, object]]:
        if not active_ids:
            return self._utility_from_offer_dict({}), {
                "expected_accepted_quantity": 0.0,
                "p_overshoot": 0.0,
                "expected_overshoot": 0.0,
                "expected_counter_recoverable": 0.0,
                "accepted_quantity_distribution": {0: 1.0},
                "aggregate_state_distribution": {(0, 0): 1.0},
                "probabilities_by_partner": probability_details,
                "outcome_eval_mode": "aggregate_dp",
            }

        state_distribution: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for partner_id in active_ids:
            quantity = int(allocation[partner_id])
            detail = probability_details[partner_id]
            next_distribution: dict[tuple[int, int], float] = {}
            for (accepted_quantity, counter_capacity_units), probability in (
                state_distribution.items()
            ):
                accept_probability = probability * float(detail["p_accept"])
                if accept_probability > 0.0:
                    key = (accepted_quantity + quantity, counter_capacity_units)
                    next_distribution[key] = (
                        next_distribution.get(key, 0.0) + accept_probability
                    )

                counter_probability = probability * float(detail["p_counter"])
                if counter_probability > 0.0:
                    key = (accepted_quantity, counter_capacity_units + quantity)
                    next_distribution[key] = (
                        next_distribution.get(key, 0.0) + counter_probability
                    )

                reject_probability = probability * float(detail["p_reject"])
                if reject_probability > 0.0:
                    key = (accepted_quantity, counter_capacity_units)
                    next_distribution[key] = (
                        next_distribution.get(key, 0.0) + reject_probability
                    )
            state_distribution = next_distribution

        expected_score = 0.0
        expected_accepted_quantity = 0.0
        p_overshoot = 0.0
        expected_overshoot = 0.0
        expected_counter_recoverable = 0.0
        quantity_distribution: dict[int, float] = {}
        for (
            accepted_quantity,
            counter_capacity_units,
        ), probability in state_distribution.items():
            if probability <= 0.0:
                continue
            utility = utility_by_quantity[accepted_quantity]
            delta = accepted_quantity - int(needs)
            recoverable = 0.0
            if delta > 0 and self.first_overshoot_lockin_penalty >= 0:
                utility -= float(unit_scale) * float(
                    self.first_overshoot_lockin_penalty
                ) * (0.5 + float(delta))
            elif delta < 0 and self.first_counter_continuation_value >= 0:
                counter_capacity = float(counter_capacity_units) * 0.5
                recoverable = min(float(-delta), counter_capacity)
                utility += (
                    float(unit_scale)
                    * float(self.first_counter_continuation_value)
                    * recoverable
                )

            expected_score += probability * utility
            expected_accepted_quantity += probability * accepted_quantity
            quantity_distribution[accepted_quantity] = (
                quantity_distribution.get(accepted_quantity, 0.0) + probability
            )
            if delta > 0:
                p_overshoot += probability
                expected_overshoot += probability * delta
            expected_counter_recoverable += probability * recoverable

        return expected_score, {
            "expected_accepted_quantity": expected_accepted_quantity,
            "p_overshoot": p_overshoot,
            "expected_overshoot": expected_overshoot,
            "expected_counter_recoverable": expected_counter_recoverable,
            "accepted_quantity_distribution": quantity_distribution,
            "aggregate_state_distribution": state_distribution,
            "probabilities_by_partner": probability_details,
            "outcome_eval_mode": "aggregate_dp",
        }

    def _first_trinary_unit_scale(
        self,
        needs: int,
        partners: list[str],
        price: int,
    ) -> float:
        fallback = max(abs(float(price)), 1e-6)
        try:
            u_exact = self._utility_from_offer_dict(
                self._first_virtual_offer_dict(needs, partners, price)
            )
            u_under = self._utility_from_offer_dict(
                self._first_virtual_offer_dict(max(0, needs - 1), partners, price)
            )
            u_over = self._utility_from_offer_dict(
                self._first_virtual_offer_dict(needs + 1, partners, price)
            )
            return max(
                abs(float(u_exact) - float(u_under)),
                abs(float(u_exact) - float(u_over)),
                fallback,
                1e-6,
            )
        except Exception as exc:                                         
            self._debug_log_decision(
                "first_trinary_unit_scale_fallback",
                reason=repr(exc),
                fallback=fallback,
            )
            return fallback

    def _first_virtual_offer_dict(
        self,
        quantity: int,
        partners: list[str],
        price: int,
    ) -> dict[str, tuple]:
        if quantity <= 0 or not partners:
            return {}
        quantities = distribute(
            int(quantity),
            len(partners),
            mx=self.awi.n_lines,
            equal=True,
            allow_zero=True,
        )
        return {
            partner_id: (int(q), self.awi.current_step, price)
            for partner_id, q in zip(partners, quantities)
            if int(q) > 0
        }

    def _first_quantity_bucket(self, quantity: int) -> str:
        q = int(quantity)
        if q <= 0:
            return "0"
        if q == 1:
            return "1"
        if q == 2:
            return "2"
        if q <= 4:
            return "3-4"
        if q <= 7:
            return "5-7"
        return "8+"

    def _first_trinary_probability_detail(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
        price_bucket: str | None = None,
    ) -> dict[str, object]:
        quantity_bucket = self._first_quantity_bucket(quantity)
        probability_source = "all_prices"
        if price_bucket in ("partner_favorable", "self_favorable"):
            value = self._first_price_probability_counts(
                partner_id,
                bool(is_selling),
                int(quantity),
                str(price_bucket),
            )
            probability_source = "selected_price_bucket"
        else:
            stats = getattr(self, "first_trinary_stats", {})
            value = stats.get(
                (partner_id, bool(is_selling), quantity_bucket),
                {"accepts": 0, "counters": 0, "rejects": 0},
            )
        accepts = float(value.get("accepts", 0))
        counters = float(value.get("counters", 0))
        rejects = float(value.get("rejects", 0))
        online_total = accepts + counters + rejects
        prior = getattr(self, "first_trinary_response_prior", {})
        prior_weight = max(0.0, float(self.first_trinary_response_prior_weight))
        prior_accept = float(prior.get("accept", 1.0 / 3.0))
        prior_counter = float(prior.get("counter", 1.0 / 3.0))
        prior_reject = float(prior.get("reject", 1.0 / 3.0))
        prior_total = prior_accept + prior_counter + prior_reject
        if prior_total <= 0.0:
            prior_accept = prior_counter = prior_reject = 1.0 / 3.0
        else:
            prior_accept /= prior_total
            prior_counter /= prior_total
            prior_reject /= prior_total
        total = online_total + prior_weight
        if total <= 0.0:
            p_accept, p_counter, p_reject = prior_accept, prior_counter, prior_reject
        else:
            p_accept = (accepts + prior_accept * prior_weight) / total
            p_counter = (counters + prior_counter * prior_weight) / total
            p_reject = (rejects + prior_reject * prior_weight) / total
        return {
            "p_accept": p_accept,
            "p_counter": p_counter,
            "p_reject": p_reject,
            "quantity_bucket": quantity_bucket,
            "price_bucket": price_bucket,
            "probability_source": probability_source,
            "online_accepts": accepts,
            "online_counters": counters,
            "online_rejects": rejects,
            "online_total": online_total,
            "prior": prior,
            "prior_weight": prior_weight,
        }

    def _first_offer_price(
        self,
        issues,
        is_selling: bool,
        *,
        partner_id: str,
        quantity: int,
        default_price: int,
    ) -> int:
        if int(quantity) <= 0 or self.first_price_accept_rate_margin < 0:
            self._debug_log_decision(
                "first_offer_price",
                is_selling=is_selling,
                partner_id=partner_id,
                quantity=int(quantity),
                selected_price=int(default_price),
                adaptive_enabled=False,
                first_price_accept_rate_margin=self.first_price_accept_rate_margin,
            )
            return int(default_price)

        if self.awi.current_step < self.first_price_warmup_steps:
            selected_bucket = random.choice(("partner_favorable", "self_favorable"))
            selected_price = self._counter_price_for_bucket(
                issues,
                is_selling,
                selected_bucket,
            )
            self._debug_log_decision(
                "first_offer_price",
                is_selling=is_selling,
                partner_id=partner_id,
                quantity=int(quantity),
                default_price=int(default_price),
                selected_price=selected_price,
                selected_bucket=selected_bucket,
                adaptive_enabled=False,
                reason="warmup_exploration",
                current_step=self.awi.current_step,
                first_price_warmup_steps=self.first_price_warmup_steps,
            )
            return selected_price

        decision = self._first_price_adaptive_decision(
            partner_id,
            is_selling,
            int(quantity),
        )
        selected_bucket = (
            "partner_favorable"
            if decision["use_partner_favorable"]
            else "self_favorable"
        )
        selected_price = self._counter_price_for_bucket(
            issues,
            is_selling,
            selected_bucket,
        )
        self._debug_log_decision(
            "first_offer_price",
            is_selling=is_selling,
            partner_id=partner_id,
            quantity=int(quantity),
            default_price=int(default_price),
            selected_price=selected_price,
            selected_bucket=selected_bucket,
            adaptive_enabled=True,
            decision=decision,
        )
        return selected_price

    def _first_price_bucket_for_trinary_probability(
        self,
        is_selling: bool,
        partner_id: str,
        quantity: int,
    ) -> str | None:
        if int(quantity) <= 0:
            return None
        if self.first_price_accept_rate_margin < 0:
            return None
        if self.awi.current_step < self.first_price_warmup_steps:
            return None
        decision = self._first_price_adaptive_decision(
            partner_id,
            is_selling,
            int(quantity),
        )
        return (
            "partner_favorable"
            if decision["use_partner_favorable"]
            else "self_favorable"
        )

    def _first_price_adaptive_decision(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
    ) -> dict[str, object]:
        min_samples = max(0, int(self.first_price_min_sample_per_side))
        margin = float(self.first_price_accept_rate_margin)
        for scope in ("exact_quantity", "quantity_bucket", "partner_all_quantities"):
            counts = self._first_price_bucket_counts(
                partner_id,
                bool(is_selling),
                int(quantity),
                scope,
            )
            partner_total = self._counter_price_count_total(
                counts["partner_favorable"]
            )
            self_total = self._counter_price_count_total(counts["self_favorable"])
            if partner_total < min_samples or self_total < min_samples:
                continue
            partner_rate = (
                counts["partner_favorable"]["accepts"] / partner_total
                if partner_total > 0
                else 0.0
            )
            self_rate = (
                counts["self_favorable"]["accepts"] / self_total
                if self_total > 0
                else 0.0
            )
            rate_diff = partner_rate - self_rate
            return {
                "scope": scope,
                "use_partner_favorable": rate_diff >= margin,
                "partner_favorable_accept_rate": partner_rate,
                "self_favorable_accept_rate": self_rate,
                "accept_rate_diff": rate_diff,
                "partner_favorable_samples": partner_total,
                "self_favorable_samples": self_total,
                "min_sample_per_side": min_samples,
                "margin": margin,
                "counts": counts,
            }
        return {
            "scope": "insufficient_samples",
            "use_partner_favorable": False,
            "min_sample_per_side": min_samples,
            "margin": margin,
        }

    def _first_price_probability_counts(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
        price_bucket: str,
    ) -> dict[str, int]:
        counts = {"accepts": 0, "counters": 0, "rejects": 0}
        stats = getattr(self, "first_price_quantity_price_stats", {})
        target_bucket = self._first_quantity_bucket(quantity)
        for (
            observed_partner,
            observed_is_selling,
            observed_quantity,
            _unit_price,
            observed_price_bucket,
        ), value in stats.items():
            if observed_partner != partner_id:
                continue
            if bool(observed_is_selling) != bool(is_selling):
                continue
            if str(observed_price_bucket) != str(price_bucket):
                continue
            if (
                int(observed_quantity) != int(quantity)
                and self._first_quantity_bucket(int(observed_quantity))
                != target_bucket
            ):
                continue
            for field_name in ("accepts", "counters", "rejects"):
                counts[field_name] += int(value.get(field_name, 0))
        if self._counter_price_count_total(counts) > 0:
            return counts
        for (
            observed_partner,
            observed_is_selling,
            _observed_quantity,
            _unit_price,
            observed_price_bucket,
        ), value in stats.items():
            if observed_partner != partner_id:
                continue
            if bool(observed_is_selling) != bool(is_selling):
                continue
            if str(observed_price_bucket) != str(price_bucket):
                continue
            for field_name in ("accepts", "counters", "rejects"):
                counts[field_name] += int(value.get(field_name, 0))
        return counts

    def _first_price_bucket_counts(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
        scope: str,
    ) -> dict[str, dict[str, int]]:
        counts = {
            "partner_favorable": {"accepts": 0, "counters": 0, "rejects": 0},
            "self_favorable": {"accepts": 0, "counters": 0, "rejects": 0},
        }
        stats = getattr(self, "first_price_quantity_price_stats", {})
        target_bucket = self._first_quantity_bucket(quantity)
        for (
            observed_partner,
            observed_is_selling,
            observed_quantity,
            _unit_price,
            partner_price_bucket,
        ), value in stats.items():
            if observed_partner != partner_id:
                continue
            if bool(observed_is_selling) != bool(is_selling):
                continue
            if partner_price_bucket not in counts:
                continue
            observed_quantity = int(observed_quantity)
            if scope == "exact_quantity" and observed_quantity != int(quantity):
                continue
            if (
                scope == "quantity_bucket"
                and self._first_quantity_bucket(observed_quantity) != target_bucket
            ):
                continue
            for field_name in ("accepts", "counters", "rejects"):
                counts[str(partner_price_bucket)][field_name] += int(
                    value.get(field_name, 0)
                )
        return counts

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        self._clear_pending_first_attempts_for_received_offers(offers)
        self._clear_pending_counter_attempts_for_received_offers(offers)
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers
            price = self._counter_offer_price(issues, is_selling)
            self._debug_log_decision(
                "counter_context",
                is_selling=is_selling,
                needs=needs,
                all_partners=list(all_partners),
                active_partners=partners,
                future_partners=future_partners,
                offers=self._debug_offer_dict(offers),
                price=price,
                counter_offer_price_mode=self.counter_offer_price_mode,
                use_incoming_quantity_counter_distribution=(
                    self.use_incoming_quantity_counter_distribution
                ),
                counter_distribution_always_min_one=(
                    self.counter_distribution_always_min_one
                ),
                relative_times={
                    p: states[p].relative_time for p in partners if p in states
                },
            )

            if len(partners) > 0:
                neg_step = min(state.step for state in states.values())
                update_rate = self.config.round_offer_update_rate
                self.rounds_ave_offered[neg_step] = (
                    (1 - update_rate) * self.rounds_ave_offered[neg_step]
                    + update_rate
                    * sum([offers[p][QUANTITY] for p in partners])
                    / len(partners)
                )
                self._debug_log_decision(
                    "round_average_offer_update",
                    is_selling=is_selling,
                    neg_step=neg_step,
                    update_rate=update_rate,
                    partners=partners,
                    offered_quantities={p: offers[p][QUANTITY] for p in partners},
                    updated_round_average=self.rounds_ave_offered[neg_step],
                )

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            plist = list(powerset(partners))[::-1]
            plus_best_diff, _plus_best_expected_diff, plus_best_indx = (
                float("inf"),
                float("inf"),
                -1,
            )
            minus_best_diff, _minus_best_expected_diff, minus_best_indx = (
                -float("inf"),
                -float("inf"),
                -1,
            )
            best_diff, best_indx = float("inf"), -1

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff >= 0:
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        new_price = self._price_tiebreak_score(offers, partner_ids)
                        cur_price = self._price_tiebreak_score(
                            offers, plist[plus_best_indx]
                        )
                        price_better = new_price > cur_price if is_selling else new_price < cur_price
                        price_equal = new_price == cur_price
                        if price_better:
                            plus_best_diff, plus_best_indx = diff, i
                        elif price_equal and self.use_cash_tiebreak:
                            new_cash = self._partners_total_cash(partner_ids)
                            cur_cash = self._partners_total_cash(plist[plus_best_indx])
                            if new_cash < cur_cash:
                                        
                                                                                                             
                                                                                              
                                   
                                plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(
                            plist[minus_best_indx]
                        ):
                            new_price = self._price_tiebreak_score(offers, partner_ids)
                            cur_price = self._price_tiebreak_score(
                                offers, plist[minus_best_indx]
                            )
                            price_better = new_price > cur_price if is_selling else new_price < cur_price
                            price_equal = new_price == cur_price
                            if price_better:
                                minus_best_diff, minus_best_indx = diff, i
                            elif price_equal and self.use_cash_tiebreak:
                                new_cash = self._partners_total_cash(partner_ids)
                                cur_cash = self._partners_total_cash(plist[minus_best_indx])
                                if new_cash < cur_cash:
                                            
                                                                                                                  
                                                                                                   
                                       
                                    minus_best_diff, minus_best_indx = diff, i

            th_min_plus, th_max_plus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[plus_best_indx]).union(future_partners)),
                is_selling,
            )
            th_min_minus, th_max_minus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[minus_best_indx]).union(future_partners)),
                is_selling,
            )
            if self.debug_log_enabled:
                plus_summary = (
                    self._debug_offer_set_summary(offers, plist[plus_best_indx])
                    if 0 <= plus_best_indx < len(plist)
                    else None
                )
                minus_summary = (
                    self._debug_offer_set_summary(offers, plist[minus_best_indx])
                    if 0 <= minus_best_indx < len(plist)
                    else None
                )
                self._debug_log_decision(
                    "counter_candidate_scan",
                    is_selling=is_selling,
                    needs=needs,
                    plus_best_diff=plus_best_diff,
                    plus_best_indx=plus_best_indx,
                    plus_best_offer_set=plus_summary,
                    minus_best_diff=minus_best_diff,
                    minus_best_indx=minus_best_indx,
                    minus_best_offer_set=minus_summary,
                    th_min_plus=th_min_plus,
                    th_max_plus=th_max_plus,
                    th_min_minus=th_min_minus,
                    th_max_minus=th_max_minus,
                    plus_within_threshold=plus_best_diff <= th_max_plus,
                    minus_within_threshold=th_min_minus <= minus_best_diff,
                    use_utility_acceptance_choice=(
                        self.use_utility_acceptance_choice
                    ),
                )
            if (
                self.use_counter_trinary_dp_before_threshold
                and self.awi.current_step >= self.counter_acceptance_rate_warmup_steps
            ):
                t = min(_.relative_time for _ in states.values())
                dp_partners = list(partners.union(future_partners))
                dp_counter_overorder_ratio = (
                    self._counter_overorder_ratio(t, is_selling)
                    if len(dp_partners) > 1
                    else None
                )
                dp_offering_quantity = (
                    int(needs * (1 + dp_counter_overorder_ratio))
                    if len(dp_partners) > 1
                    else needs
                )
                dp_distribution = self._counter_quantity_distribution(
                    partners=dp_partners,
                    offering_quantity=dp_offering_quantity,
                    shortage_quantity=needs,
                    offers=offers,
                )
                trinary_dp_response = self._counter_trinary_dp_decision_response(
                    offers=offers,
                    distribution=dp_distribution,
                    offering_quantity=dp_offering_quantity,
                    partners=dp_partners,
                    price=price,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                )
                if trinary_dp_response is not None:
                    self._record_pending_counter_responses(
                        trinary_dp_response,
                        offers,
                        source="counter_trinary_dp_before_threshold",
                    )
                    response.update(trinary_dp_response)
                    continue
            if th_min_minus <= minus_best_diff or plus_best_diff <= th_max_plus:
                accept_source = "unknown"
                plus_utility = None
                minus_utility = None
                if th_min_minus <= minus_best_diff and plus_best_diff <= th_max_plus:
                    if self.use_utility_acceptance_choice:
                        plus_utility = self._utility_from_offer_dict(
                            {p: offers[p] for p in plist[plus_best_indx]}
                        )
                        minus_utility = self._utility_from_offer_dict(
                            {p: offers[p] for p in plist[minus_best_indx]}
                        )
                        if plus_utility > minus_utility:
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                            accept_source = "threshold_both_utility_plus"
                        elif minus_utility > plus_utility:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                            accept_source = "threshold_both_utility_minus"
                        else:
                            best_diff, best_indx = self._quantity_acceptance_choice(
                                plus_best_diff=plus_best_diff,
                                plus_best_indx=plus_best_indx,
                                minus_best_diff=minus_best_diff,
                                minus_best_indx=minus_best_indx,
                                plist=plist,
                                partners=partners,
                                future_partners=future_partners,
                                is_selling=is_selling,
                            )
                            accept_source = "threshold_both_utility_equal_quantity"
                    else:
                        best_diff, best_indx = self._quantity_acceptance_choice(
                            plus_best_diff=plus_best_diff,
                            plus_best_indx=plus_best_indx,
                            minus_best_diff=minus_best_diff,
                            minus_best_indx=minus_best_indx,
                            plist=plist,
                            partners=partners,
                            future_partners=future_partners,
                            is_selling=is_selling,
                        )
                        accept_source = "threshold_both_quantity"
                elif minus_best_diff < th_min_minus and plus_best_diff <= th_max_plus:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                    accept_source = "threshold_plus_only"
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx
                    accept_source = "threshold_minus_only"

                partner_ids = plist[best_indx]
                self._debug_log_decision(
                    "counter_acceptance_choice",
                    is_selling=is_selling,
                    source=accept_source,
                    best_diff=best_diff,
                    best_indx=best_indx,
                    partner_ids=partner_ids,
                    plus_utility=plus_utility,
                    minus_utility=minus_utility,
                )
                if partner_ids:
                    branch_response = self._responses_for_accepted_offer_set(
                        offers=offers,
                        partner_ids=partner_ids,
                        best_diff=best_diff,
                        partners=partners,
                        future_partners=future_partners,
                        states=states,
                        issues=issues,
                        is_selling=is_selling,
                        unneeded_response=unneeded_response,
                        source=accept_source,
                    )
                    self._record_pending_counter_responses(
                        branch_response,
                        offers,
                        source=accept_source,
                    )
                    response |= branch_response
                    continue

            large_partner_ids = (
                ()
                if self.deactivate_acceptanve_gate in (2, 3)
                else self._large_shortage_offer_acceptance_partners(
                    offers=offers,
                    plist=plist,
                    minus_best_indx=minus_best_indx,
                    minus_best_diff=minus_best_diff,
                    plus_best_diff=plus_best_diff,
                    is_selling=is_selling,
                )
            )
            if self.deactivate_acceptanve_gate in (2, 3):
                self._debug_log_decision(
                    "large_shortage_offer_gate",
                    reason="deactivated_by_deactivate_acceptanve_gate",
                    deactivate_acceptanve_gate=self.deactivate_acceptanve_gate,
                )
            if large_partner_ids:
                best_diff = (
                    sum(offers[p][QUANTITY] for p in large_partner_ids) - needs
                )
                self._debug_log_decision(
                    "counter_acceptance_choice",
                    is_selling=is_selling,
                    source="large_shortage_offer",
                    best_diff=best_diff,
                    partner_ids=large_partner_ids,
                    threshold=self._large_offer_acceptance_threshold(),
                )
                branch_response = self._responses_for_accepted_offer_set(
                    offers=offers,
                    partner_ids=large_partner_ids,
                    best_diff=best_diff,
                    partners=partners,
                    future_partners=future_partners,
                    states=states,
                    issues=issues,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                    source="large_shortage_offer",
                )
                self._record_pending_counter_responses(
                    branch_response,
                    offers,
                    source="large_shortage_offer",
                )
                response |= branch_response
                continue

            fallback_candidate = (
                None
                if self.deactivate_acceptanve_gate in (1, 3)
                else self._utility_fallback_acceptance_candidate(
                    plist,
                    offers,
                    states,
                    response,
                    needs,
                )
            )
            if self.deactivate_acceptanve_gate in (1, 3):
                self._debug_log_decision(
                    "utility_fallback_gate",
                    reason="deactivated_by_deactivate_acceptanve_gate",
                    deactivate_acceptanve_gate=self.deactivate_acceptanve_gate,
                )
            if fallback_candidate is not None:
                best_diff, best_indx = fallback_candidate
                partner_ids = plist[best_indx]
                self._debug_log_decision(
                    "counter_acceptance_choice",
                    is_selling=is_selling,
                    source="utility_fallback",
                    best_diff=best_diff,
                    best_indx=best_indx,
                    partner_ids=partner_ids,
                )
                branch_response = self._responses_for_accepted_offer_set(
                    offers=offers,
                    partner_ids=partner_ids,
                    best_diff=best_diff,
                    partners=partners,
                    future_partners=future_partners,
                    states=states,
                    issues=issues,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                    source="utility_fallback",
                )
                self._record_pending_counter_responses(
                    branch_response,
                    offers,
                    source="utility_fallback",
                )
                response |= branch_response
                continue

            t = min(_.relative_time for _ in states.values())

            partners = partners.union(future_partners)
            partners = list(partners)
            counter_overorder_ratio = (
                self._counter_overorder_ratio(t, is_selling)
                if len(partners) > 1
                else None
            )
            offering_quanitity = (
                int(needs * (1 + counter_overorder_ratio))
                if len(partners) > 1
                else needs
            )
            distribution = self._counter_quantity_distribution(
                partners=partners,
                offering_quantity=offering_quanitity,
                shortage_quantity=needs,
                offers=offers,
            )
            trinary_dp_response = self._counter_trinary_dp_decision_response(
                offers=offers,
                distribution=distribution,
                offering_quantity=offering_quanitity,
                partners=partners,
                price=price,
                is_selling=is_selling,
                unneeded_response=unneeded_response,
            )
            if trinary_dp_response is not None:
                self._record_pending_counter_responses(
                    trinary_dp_response,
                    offers,
                    source="counter_trinary_dp_decision",
                )
                response.update(trinary_dp_response)
                continue
            acceptance_rate_response = (
                self._counter_acceptance_rate_decision_response(
                    offers=offers,
                    distribution=distribution,
                    partners=partners,
                    price=price,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                )
            )
            if acceptance_rate_response is not None:
                self._record_pending_counter_responses(
                    acceptance_rate_response,
                    offers,
                    source="counter_acceptance_rate_decision",
                )
                response.update(acceptance_rate_response)
                continue
            matching_partner_ids = self._matching_counter_offer_acceptance_partners(
                offers=offers,
                distribution=distribution,
                needs=needs,
            )
            if matching_partner_ids:
                best_diff = (
                    sum(offers[p][QUANTITY] for p in matching_partner_ids) - needs
                )
                self._debug_log_decision(
                    "counter_acceptance_choice",
                    is_selling=is_selling,
                    source="matching_counter_offer",
                    best_diff=best_diff,
                    partner_ids=matching_partner_ids,
                    tolerance=self.counter_accept_matching_offer_tolerance,
                    distribution=distribution,
                )
                branch_response = self._responses_for_accepted_offer_set(
                    offers=offers,
                    partner_ids=matching_partner_ids,
                    best_diff=best_diff,
                    partners=set(partners).difference(future_partners),
                    future_partners=future_partners,
                    states=states,
                    issues=issues,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                    source="matching_counter_offer",
                )
                self._record_pending_counter_responses(
                    branch_response,
                    offers,
                    source="matching_counter_offer",
                )
                response |= branch_response
                continue
            self._debug_log_decision(
                "counter_reject_and_propose",
                is_selling=is_selling,
                reason="no_acceptance_gate_matched",
                needs=needs,
                relative_time=t,
                counter_overorder_ratio=counter_overorder_ratio,
                offering_quantity=offering_quanitity,
                price=price,
                partners=partners,
                distribution=distribution,
            )

            counter_response = {
                k: (
                    unneeded_response
                    if q == 0
                    else self._counter_reject_response(
                        partner_id=k,
                        counter_quantity=int(q),
                        default_price=int(price),
                        offers=offers,
                        is_selling=is_selling,
                        unneeded_response=unneeded_response,
                    )
                )
                for k, q in distribution.items()
            }
            self._record_pending_counter_responses(
                counter_response,
                offers,
                source="normal_counter",
            )
            response.update(counter_response)
        self._debug_log_decision(
            "counter_all_summary",
            final_responses=self._debug_response_dict(response),
            accepted_offers=self._debug_offer_dict(
                self._accepted_offers_from_response(response)
            ),
            response_partner_ids=list(response.keys()),
        )
        return response

    def _counter_trinary_dp_decision_response(
        self,
        *,
        offers: dict[str, tuple],
        distribution: dict[str, int],
        offering_quantity: int,
        partners: list[str],
        price: int,
        is_selling: bool,
        unneeded_response: SAOResponse,
    ) -> dict[str, SAOResponse] | None:
        if not self.use_counter_trinary_dp_decision:
            return None
        if self.counter_acceptance_prior_weight < 0:
            self._debug_log_decision(
                "counter_trinary_dp_decision_skip",
                reason="prior_weight_disabled",
                prior_weight=self.counter_acceptance_prior_weight,
            )
            return None
        if self.awi.current_step < self.counter_acceptance_rate_warmup_steps:
            self._debug_log_decision(
                "counter_trinary_dp_decision_skip",
                reason="warmup",
                warmup_steps=self.counter_acceptance_rate_warmup_steps,
            )
            return None
        horizon = int(self.counter_trinary_dp_lookahead_rounds)
        if horizon <= 0:
            self._debug_log_decision(
                "counter_trinary_dp_decision_skip",
                reason="non_positive_horizon",
                horizon=horizon,
            )
            return None

        active_partners = [
            partner_id
            for partner_id in partners
            if partner_id in offers and int(distribution.get(partner_id, 0)) > 0
        ]
        if not active_partners:
            self._debug_log_decision(
                "counter_trinary_dp_decision_skip",
                reason="no_current_offer_counter_candidates",
                distribution=distribution,
            )
            return None

        decision_id = self._next_counter_trinary_dp_decision_id()
        memo: dict[tuple, float] = {}
        utility_cache: dict[tuple, float] = {}
        counter = {"candidate_count": 0}
        max_candidates = 200000
        attempt_index_by_partner = {
            partner_id: min(
                4,
                int(
                    getattr(self, "counter_attempt_order_by_partner", {}).get(
                        partner_id,
                        0,
                    )
                )
                + 1,
            )
            for partner_id in active_partners
        }
        try:
            baseline_expected, baseline_probabilities = (
                self._counter_trinary_action_score(
                    base_offers={},
                    active_offers={
                        partner_id: offers[partner_id]
                        for partner_id in active_partners
                    },
                    counter_quantities={
                        partner_id: int(distribution[partner_id])
                        for partner_id in active_partners
                    },
                    h=horizon,
                    price=price,
                    is_selling=is_selling,
                    offering_quantity=int(offering_quantity),
                    attempt_index_by_partner=attempt_index_by_partner,
                    memo=memo,
                    utility_cache=utility_cache,
                    counter=counter,
                    max_candidates=max_candidates,
                )
            )
            best_action = self._counter_trinary_dp_best_action(
                base_offers={},
                active_offers={partner_id: offers[partner_id] for partner_id in active_partners},
                h=horizon,
                price=price,
                is_selling=is_selling,
                offering_quantity=int(offering_quantity),
                attempt_index_by_partner=attempt_index_by_partner,
                memo=memo,
                utility_cache=utility_cache,
                counter=counter,
                max_candidates=max_candidates,
            )
        except RuntimeError as exc:
            self._debug_log_decision(
                "counter_trinary_dp_decision",
                counter_trinary_dp_decision_id=decision_id,
                is_selling=is_selling,
                active_partners=active_partners,
                horizon=horizon,
                candidate_count=counter["candidate_count"],
                fallback_used=True,
                fallback_reason=repr(exc),
                total_quantity_mode=self.counter_trinary_total_quantity_mode,
                counter_trinary_approx_candidate_mode=(
                    self.counter_trinary_approx_candidate_mode
                ),
                attempt_index_by_partner=attempt_index_by_partner,
            )
            return None
        except Exception as exc:                                         
            self._debug_log_decision(
                "counter_trinary_dp_decision",
                counter_trinary_dp_decision_id=decision_id,
                is_selling=is_selling,
                active_partners=active_partners,
                horizon=horizon,
                candidate_count=counter["candidate_count"],
                fallback_used=True,
                fallback_reason="utility_or_dp_error",
                error=repr(exc),
                total_quantity_mode=self.counter_trinary_total_quantity_mode,
                counter_trinary_approx_candidate_mode=(
                    self.counter_trinary_approx_candidate_mode
                ),
                attempt_index_by_partner=attempt_index_by_partner,
            )
            return None

        dp_expected = float(best_action["expected_score"])
        dp_margin = float(self.counter_trinary_dp_margin)
        if dp_expected <= float(baseline_expected) + dp_margin:
            self._debug_log_decision(
                "counter_trinary_dp_decision",
                counter_trinary_dp_decision_id=decision_id,
                is_selling=is_selling,
                active_partners=active_partners,
                horizon=horizon,
                candidate_count=counter["candidate_count"],
                baseline_expected=baseline_expected,
                dp_expected=dp_expected,
                dp_margin=dp_margin,
                baseline_probabilities_by_partner=baseline_probabilities,
                best_action_type=best_action["action_type"],
                best_probabilities_by_partner=best_action["probabilities_by_partner"],
                attempt_index_by_partner=attempt_index_by_partner,
                total_quantity_mode=self.counter_trinary_total_quantity_mode,
                counter_trinary_approx_candidate_mode=(
                    self.counter_trinary_approx_candidate_mode
                ),
                fallback_used=True,
                fallback_reason="dp_not_above_baseline_margin",
            )
            return None

        accepted_partner_ids = set(best_action["accepted_partner_ids"])
        counter_quantities = {
            str(partner_id): int(quantity)
            for partner_id, quantity in best_action["counter_quantities"].items()
        }
        active_partner_set = set(active_partners)
        chosen_response: dict[str, SAOResponse] = {}
        for partner_id, quantity in distribution.items():
            if partner_id in accepted_partner_ids:
                chosen_response[partner_id] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    offers[partner_id],
                )
            elif partner_id in counter_quantities:
                counter_quantity = int(counter_quantities[partner_id])
                chosen_response[partner_id] = (
                    unneeded_response
                    if counter_quantity <= 0
                    else self._counter_reject_response(
                        partner_id=partner_id,
                        counter_quantity=counter_quantity,
                        default_price=int(price),
                        offers=offers,
                        is_selling=is_selling,
                        unneeded_response=unneeded_response,
                    )
                )
            elif partner_id in active_partner_set:
                chosen_response[partner_id] = unneeded_response
            elif int(quantity) == 0:
                chosen_response[partner_id] = unneeded_response
            else:
                chosen_response[partner_id] = self._counter_reject_response(
                    partner_id=partner_id,
                    counter_quantity=int(quantity),
                    default_price=int(price),
                    offers=offers,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                )

        self._record_counter_trinary_dp_debug_decision(
            counter_trinary_dp_decision_id=decision_id,
            is_selling=is_selling,
            active_partners=active_partners,
            accepted_partner_ids=sorted(accepted_partner_ids),
            counter_partner_ids=sorted(counter_quantities),
            counter_quantities=counter_quantities,
            best_action_type=str(best_action["action_type"]),
            dp_expected=dp_expected,
            baseline_expected=float(baseline_expected),
        )
        self._debug_log_decision(
            "counter_trinary_dp_decision",
            counter_trinary_dp_decision_id=decision_id,
            is_selling=is_selling,
            active_partners=active_partners,
            horizon=horizon,
            candidate_count=counter["candidate_count"],
            baseline_expected=baseline_expected,
            dp_expected=dp_expected,
            dp_margin=dp_margin,
            best_expected_score=best_action["expected_score"],
            best_action_type=best_action["action_type"],
            accepted_partner_ids=sorted(accepted_partner_ids),
            counter_partner_ids=sorted(counter_quantities),
            counter_quantities=counter_quantities,
            probabilities_by_partner=best_action["probabilities_by_partner"],
            baseline_probabilities_by_partner=baseline_probabilities,
            attempt_index_by_partner=attempt_index_by_partner,
            total_quantity_mode=self.counter_trinary_total_quantity_mode,
            counter_trinary_approx_candidate_mode=(
                self.counter_trinary_approx_candidate_mode
            ),
            fallback_used=False,
            fallback_reason=None,
        )
        return chosen_response

    def _next_counter_trinary_dp_decision_id(self) -> str:
        self.counter_trinary_dp_decision_seq = (
            int(getattr(self, "counter_trinary_dp_decision_seq", 0)) + 1
        )
        return (
            f"{self.id}:{self.awi.current_step}:"
            f"{self.counter_trinary_dp_decision_seq}"
        )

    def _record_counter_trinary_dp_debug_decision(
        self,
        *,
        counter_trinary_dp_decision_id: str,
        is_selling: bool,
        active_partners: list[str],
        accepted_partner_ids: list[str],
        counter_partner_ids: list[str],
        counter_quantities: dict[str, int],
        best_action_type: str,
        dp_expected: float,
        baseline_expected: float,
    ) -> None:
        if not self.debug_log_enabled:
            return
        self._debug_step_dp_decisions.append(
            {
                "counter_trinary_dp_decision_id": counter_trinary_dp_decision_id,
                "is_selling": bool(is_selling),
                "active_partners": list(active_partners),
                "accepted_partner_ids": list(accepted_partner_ids),
                "counter_partner_ids": list(counter_partner_ids),
                "counter_quantities": dict(counter_quantities),
                "best_action_type": best_action_type,
                "dp_expected": float(dp_expected),
                "baseline_expected": float(baseline_expected),
            }
        )

    def _normal_counter_response(
        self,
        *,
        distribution: dict[str, int],
        price: int,
        offers: dict[str, tuple] | None = None,
        is_selling: bool | None = None,
        unneeded_response: SAOResponse,
    ) -> dict[str, SAOResponse]:
        return {
            partner_id: (
                unneeded_response
                if int(quantity) == 0
                else self._counter_reject_response(
                    partner_id=partner_id,
                    counter_quantity=int(quantity),
                    default_price=int(price),
                    offers=offers or {},
                    is_selling=bool(is_selling),
                    unneeded_response=unneeded_response,
                )
            )
            for partner_id, quantity in distribution.items()
        }

    def _counter_reject_response(
        self,
        *,
        partner_id: str,
        counter_quantity: int,
        default_price: int,
        offers: dict[str, tuple],
        is_selling: bool,
        unneeded_response: SAOResponse,
    ) -> SAOResponse:
        if int(counter_quantity) <= 0:
            return unneeded_response
        issues = (
            self.awi.current_output_issues
            if is_selling
            else self.awi.current_input_issues
        )
        current_quantity = (
            int(offers[partner_id][QUANTITY])
            if partner_id in offers
            else None
        )
        price = self._counter_offer_price(
            issues,
            is_selling,
            partner_id=partner_id if current_quantity is not None else None,
            current_quantity=current_quantity,
            counter_quantity=int(counter_quantity),
            default_price=int(default_price),
        )
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            (int(counter_quantity), self.awi.current_step, price),
        )

    def _counter_trinary_dp_best_action(
        self,
        *,
        base_offers: dict[str, tuple],
        active_offers: dict[str, tuple],
        h: int,
        price: int,
        is_selling: bool,
        offering_quantity: int,
        attempt_index_by_partner: dict[str, int],
        memo: dict[tuple, float],
        utility_cache: dict[tuple, float],
        counter: dict[str, int],
        max_candidates: int,
    ) -> dict[str, object]:
        best_accept_utility, best_accept_ids = self._best_accept_subset(
            base_offers,
            active_offers,
            utility_cache=utility_cache,
        )
        best_action: dict[str, object] = {
            "expected_score": best_accept_utility,
            "action_type": "accept_only",
            "accepted_partner_ids": set(best_accept_ids),
            "counter_quantities": {},
            "probabilities_by_partner": {},
        }

        active_ids = tuple(active_offers)
        total_quantity_candidates = self._counter_trinary_total_quantity_candidates(
            offering_quantity,
            len(active_ids),
        )
        for counter_ids_tuple in powerset(active_ids):
            if not counter_ids_tuple:
                continue
            counter_ids = tuple(counter_ids_tuple)
            use_approx = self._use_counter_trinary_approx_for_subset(
                len(counter_ids),
                total_quantity_candidates,
            )
            subset_total_quantity_candidates = total_quantity_candidates
            if use_approx:
                subset_total_quantity_candidates = (
                    self._counter_trinary_approx_total_quantity_candidates(
                        offering_quantity,
                        active_offers,
                        total_quantity_candidates,
                    )
                )
            for total_quantity in subset_total_quantity_candidates:
                allocations = (
                    self._counter_trinary_approx_allocations(
                        counter_ids=counter_ids,
                        active_offers=active_offers,
                        total_quantity=int(total_quantity),
                        is_selling=is_selling,
                        attempt_index_by_partner=attempt_index_by_partner,
                    )
                    if use_approx
                    else self._bounded_integer_allocations(
                        int(total_quantity),
                        len(counter_ids),
                        self.awi.n_lines,
                    )
                )
                for allocation in allocations:
                    counter["candidate_count"] += 1
                    if counter["candidate_count"] > max_candidates:
                        raise RuntimeError("counter_trinary_dp_candidate_limit")
                    counter_quantities = {
                        partner_id: int(quantity)
                        for partner_id, quantity in zip(counter_ids, allocation)
                    }
                    expected_score, probabilities = self._counter_trinary_action_score(
                        base_offers=base_offers,
                        active_offers=active_offers,
                        counter_quantities=counter_quantities,
                        h=h,
                        price=price,
                        is_selling=is_selling,
                        offering_quantity=offering_quantity,
                        attempt_index_by_partner=attempt_index_by_partner,
                        memo=memo,
                        utility_cache=utility_cache,
                        counter=counter,
                        max_candidates=max_candidates,
                    )
                    if expected_score > float(best_action["expected_score"]):
                        best_action = {
                            "expected_score": expected_score,
                            "action_type": "counter",
                            "accepted_partner_ids": set(active_ids).difference(
                                counter_quantities
                            ),
                            "counter_quantities": counter_quantities,
                            "probabilities_by_partner": probabilities,
                        }
        return best_action

    def _counter_trinary_dp_value(
        self,
        *,
        base_offers: dict[str, tuple],
        active_offers: dict[str, tuple],
        h: int,
        price: int,
        is_selling: bool,
        offering_quantity: int,
        attempt_index_by_partner: dict[str, int],
        memo: dict[tuple, float],
        utility_cache: dict[tuple, float],
        counter: dict[str, int],
        max_candidates: int,
    ) -> float:
        if h <= 0 or not active_offers:
            utility, _ = self._best_accept_subset(
                base_offers,
                active_offers,
                utility_cache=utility_cache,
            )
            return utility
        key = (
            h,
            tuple(sorted((partner_id, tuple(offer)) for partner_id, offer in base_offers.items())),
            tuple(sorted((partner_id, tuple(offer)) for partner_id, offer in active_offers.items())),
            price,
            is_selling,
            offering_quantity,
            tuple(sorted(attempt_index_by_partner.items())),
        )
        cached = memo.get(key)
        if cached is not None:
            return cached
        best_action = self._counter_trinary_dp_best_action(
            base_offers=base_offers,
            active_offers=active_offers,
            h=h,
            price=price,
            is_selling=is_selling,
            offering_quantity=offering_quantity,
            attempt_index_by_partner=attempt_index_by_partner,
            memo=memo,
            utility_cache=utility_cache,
            counter=counter,
            max_candidates=max_candidates,
        )
        value = float(best_action["expected_score"])
        memo[key] = value
        return value

    def _counter_trinary_action_score(
        self,
        *,
        base_offers: dict[str, tuple],
        active_offers: dict[str, tuple],
        counter_quantities: dict[str, int],
        h: int,
        price: int,
        is_selling: bool,
        offering_quantity: int,
        attempt_index_by_partner: dict[str, int],
        memo: dict[tuple, float],
        utility_cache: dict[tuple, float],
        counter: dict[str, int],
        max_candidates: int,
    ) -> tuple[float, dict[str, dict[str, object]]]:
        counter_ids = tuple(
            partner_id
            for partner_id, quantity in counter_quantities.items()
            if int(quantity) > 0
        )
        accepted_offers = {
            partner_id: active_offers[partner_id]
            for partner_id in active_offers
            if partner_id not in counter_quantities
        }
        counter_offers = {
            partner_id: (
                int(counter_quantity),
                self.awi.current_step,
                price,
            )
            for partner_id, counter_quantity in counter_quantities.items()
            if int(counter_quantity) > 0
        }
        probability_details = {}
        for partner_id, counter_offer in counter_offers.items():
            delta_quantity = int(counter_offer[QUANTITY]) - int(
                active_offers[partner_id][QUANTITY]
            )
            probability_details[partner_id] = self._counter_trinary_probability_detail(
                partner_id,
                is_selling,
                delta_quantity,
                int(attempt_index_by_partner.get(partner_id, 1)),
            )

        expected_score = 0.0
        for outcome in itertools.product(
            ("accept", "neutral", "reject"),
            repeat=len(counter_ids),
        ):
            probability = 1.0
            realized_offers = base_offers | accepted_offers
            neutral_offers: dict[str, tuple] = {}
            accepted_now_quantity = sum(
                int(offer[QUANTITY]) for offer in accepted_offers.values()
            )
            for partner_id, state in zip(counter_ids, outcome):
                detail = probability_details[partner_id]
                if state == "accept":
                    probability *= float(detail["p_accept"])
                    realized_offers[partner_id] = counter_offers[partner_id]
                    accepted_now_quantity += int(
                        counter_offers[partner_id][QUANTITY]
                    )
                elif state == "neutral":
                    probability *= float(detail["p_neutral"])
                    neutral_offers[partner_id] = (
                        self._counter_trinary_neutral_offer(
                            active_offers[partner_id],
                            counter_offers[partner_id],
                            partner_id,
                            is_selling,
                            int(attempt_index_by_partner.get(partner_id, 1)),
                        )
                    )
                else:
                    probability *= float(detail["p_reject"])
            if probability <= 0.0:
                continue
            if neutral_offers:
                next_offering_quantity = max(
                    0,
                    int(offering_quantity) - int(accepted_now_quantity),
                )
                next_attempt_index_by_partner = dict(attempt_index_by_partner)
                for partner_id in neutral_offers:
                    next_attempt_index_by_partner[partner_id] = min(
                        4,
                        int(next_attempt_index_by_partner.get(partner_id, 1)) + 1,
                    )
                utility = self._counter_trinary_dp_value(
                    base_offers=realized_offers,
                    active_offers=neutral_offers,
                    h=h - 1,
                    price=price,
                    is_selling=is_selling,
                    offering_quantity=next_offering_quantity,
                    attempt_index_by_partner=next_attempt_index_by_partner,
                    memo=memo,
                    utility_cache=utility_cache,
                    counter=counter,
                    max_candidates=max_candidates,
                )
            else:
                utility = self._utility_from_offer_dict_cached(
                    realized_offers,
                    utility_cache,
                )
            expected_score += probability * utility

        probability_summary = {
            partner_id: {
                "current_quantity": int(active_offers[partner_id][QUANTITY]),
                "counter_quantity": int(counter_offers[partner_id][QUANTITY]),
                "delta_quantity": int(counter_offers[partner_id][QUANTITY])
                - int(active_offers[partner_id][QUANTITY]),
                "p_accept": detail["p_accept"],
                "p_neutral": detail["p_neutral"],
                "p_reject": detail["p_reject"],
                "probability_source": detail["source"],
                "sample_weight": detail["sample_weight"],
                "exact_sample_weight": detail["exact_sample_weight"],
                "same_sign_sample_weight": detail["same_sign_sample_weight"],
                "delta_bucket": detail["delta_bucket"],
                "attempt_index": detail["attempt_index"],
                "attempt_bucket": detail["attempt_bucket"],
                "prior_attempt_bucket": detail["prior_attempt_bucket"],
                "prior_delta": detail["prior_delta"],
            }
            for partner_id, detail in probability_details.items()
        }
        for partner_id, quantity in counter_quantities.items():
            if int(quantity) > 0:
                continue
            probability_summary[partner_id] = {
                "current_quantity": int(active_offers[partner_id][QUANTITY]),
                "counter_quantity": 0,
                "delta_quantity": -int(active_offers[partner_id][QUANTITY]),
                "p_accept": 0.0,
                "p_neutral": 0.0,
                "p_reject": 1.0,
                "probability_source": "zero_counter_reject",
                "sample_weight": 0.0,
                "exact_sample_weight": 0.0,
                "same_sign_sample_weight": 0.0,
                "delta_bucket": self._counter_delta_bucket(
                    -int(active_offers[partner_id][QUANTITY])
                ),
                "attempt_index": int(attempt_index_by_partner.get(partner_id, 1)),
                "attempt_bucket": self._counter_attempt_bucket(
                    int(attempt_index_by_partner.get(partner_id, 1))
                ),
                "prior_attempt_bucket": None,
                "prior_delta": None,
            }
        return expected_score, probability_summary

    def _counter_trinary_neutral_offer(
        self,
        active_offer: tuple,
        counter_offer: tuple,
        partner_id: str | None = None,
        is_selling: bool | None = None,
        attempt_index: int = 1,
    ) -> tuple:
        if not self.use_counter_trinary_neutral_quantity_shrink:
            return active_offer
        current_quantity = int(active_offer[QUANTITY])
        counter_quantity = int(counter_offer[QUANTITY])
        delta = counter_quantity - current_quantity
        if delta == 0:
            return active_offer
        lambda_value = self._counter_trinary_neutral_lambda_expected(
            partner_id,
            bool(is_selling) if is_selling is not None else False,
            delta,
            attempt_index,
        )
        shift = int(math.floor(abs(delta) * lambda_value + 0.5))
        next_quantity = current_quantity + (shift if delta > 0 else -shift)
        next_quantity = max(0, min(self.awi.n_lines, next_quantity))
        next_offer = list(active_offer)
        next_offer[QUANTITY] = next_quantity
        return tuple(next_offer)

    def _counter_trinary_neutral_lambda_bucket(self, lambda_value: float) -> float:
        value = max(0.0, min(1.0, float(lambda_value)))
        if value < 0.1:
            return 0.0
        if value < 0.3:
            return 0.2
        if value < 0.5:
            return 0.4
        return 0.6

    def _record_counter_trinary_neutral_lambda(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
        attempt_index: int,
        current_quantity: int,
        counter_quantity: int,
        next_quantity: int,
        *,
        source: str,
    ) -> None:
        stats = getattr(self, "counter_trinary_neutral_lambda_stats", None)
        if stats is None:
            return
        denominator = int(counter_quantity) - int(current_quantity)
        if denominator == 0:
            return
        raw_lambda = (int(next_quantity) - int(current_quantity)) / denominator
        lambda_bucket = self._counter_trinary_neutral_lambda_bucket(raw_lambda)
        delta_bucket = self._counter_delta_bucket(delta_quantity)
        attempt_bucket = self._counter_attempt_bucket(attempt_index)
        key = (partner_id, bool(is_selling), delta_bucket, attempt_bucket)
        value = stats.setdefault(key, {0.0: 0, 0.2: 0, 0.4: 0, 0.6: 0})
        value[lambda_bucket] = int(value.get(lambda_bucket, 0)) + 1
        self._debug_log_decision(
            "counter_trinary_neutral_lambda_result",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            delta_quantity=delta_quantity,
            delta_bucket=delta_bucket,
            attempt_index=int(attempt_index),
            attempt_bucket=attempt_bucket,
            current_quantity=int(current_quantity),
            counter_quantity=int(counter_quantity),
            next_quantity=int(next_quantity),
            raw_lambda=raw_lambda,
            lambda_bucket=lambda_bucket,
            counts=value,
            source=source,
        )

    def _counter_trinary_neutral_lambda_counts(
        self,
        partner_id: str,
        is_selling: bool,
        delta_bucket: str,
        attempt_bucket: str | None,
    ) -> tuple[dict[float, float], float]:
        stats = getattr(self, "counter_trinary_neutral_lambda_stats", {})
        counts = {0.0: 0.0, 0.2: 0.0, 0.4: 0.0, 0.6: 0.0}
        total = 0.0
        for (
            observed_partner,
            observed_is_selling,
            observed_delta_bucket,
            observed_attempt_bucket,
        ), value in stats.items():
            if observed_partner != partner_id or bool(observed_is_selling) != is_selling:
                continue
            if observed_delta_bucket != delta_bucket:
                continue
            if attempt_bucket is not None and observed_attempt_bucket != attempt_bucket:
                continue
            for bucket in counts:
                amount = float(value.get(bucket, 0.0))
                counts[bucket] += amount
                total += amount
        return counts, total

    def _counter_trinary_neutral_lambda_prior_detail(
        self,
        is_selling: bool,
        delta_bucket: str,
    ) -> dict[str, object]:
        fallback_prior = getattr(self, "counter_trinary_neutral_lambda_prior", {})
        level = self._counter_acceptance_prior_level()
        side = "sell" if is_selling else "buy"
        table = getattr(
            self,
            "counter_trinary_neutral_lambda_prior_by_level_side_delta",
            {},
        )
        side_table = table.get(level, {}).get(side, {})
        if delta_bucket in side_table:
            prior = side_table[delta_bucket]
            source = "level_side_delta_prior"
            prior_delta_bucket = delta_bucket
        elif "__overall__" in side_table:
            prior = side_table["__overall__"]
            source = "level_side_overall_prior"
            prior_delta_bucket = "__overall__"
        else:
            prior = fallback_prior
            source = "global_fixed_prior"
            prior_delta_bucket = None
        return {
            "prior": prior,
            "prior_source": source,
            "prior_level": level,
            "prior_side": side,
            "prior_delta_bucket": prior_delta_bucket,
        }

    def _counter_trinary_neutral_lambda_expected(
        self,
        partner_id: str | None,
        is_selling: bool,
        delta_quantity: int,
        attempt_index: int = 1,
    ) -> float:
        prior_weight = max(0.0, float(self.counter_acceptance_prior_weight))
        counts = {0.0: 0.0, 0.2: 0.0, 0.4: 0.0, 0.6: 0.0}
        online_total = 0.0
        source = "neutral_lambda_prior"
        delta_bucket = self._counter_delta_bucket(delta_quantity)
        attempt_bucket = self._counter_attempt_bucket(attempt_index)
        prior_detail = self._counter_trinary_neutral_lambda_prior_detail(
            bool(is_selling),
            delta_bucket,
        )
        prior = prior_detail["prior"]
        if partner_id is not None:
            counts, online_total = self._counter_trinary_neutral_lambda_counts(
                partner_id,
                bool(is_selling),
                delta_bucket,
                attempt_bucket,
            )
            source = "partner_side_delta_bucket_attempt_with_prior"
            if online_total <= 0.0:
                counts, online_total = self._counter_trinary_neutral_lambda_counts(
                    partner_id,
                    bool(is_selling),
                    delta_bucket,
                    None,
                )
                source = "partner_side_delta_bucket_all_attempts_with_prior"
        numerator = 0.0
        denominator = online_total + prior_weight
        for bucket, count in counts.items():
            numerator += float(bucket) * count
        if prior_weight > 0.0:
            for bucket, probability in prior.items():
                numerator += float(bucket) * float(probability) * prior_weight
        if denominator <= 0.0:
            prior_total = sum(float(probability) for probability in prior.values())
            if prior_total <= 0.0:
                expected_lambda = 0.2
            else:
                expected_lambda = sum(
                    float(bucket) * float(probability)
                    for bucket, probability in prior.items()
                ) / prior_total
        else:
            expected_lambda = numerator / denominator
        self._debug_log_decision(
            "counter_trinary_neutral_lambda_expected",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            delta_quantity=int(delta_quantity),
            delta_bucket=delta_bucket,
            attempt_index=int(attempt_index),
            attempt_bucket=attempt_bucket,
            expected_lambda=expected_lambda,
            online_counts=counts,
            online_total=online_total,
            prior=prior,
            prior_weight=prior_weight,
            prior_source=prior_detail["prior_source"],
            prior_level=prior_detail["prior_level"],
            prior_side=prior_detail["prior_side"],
            prior_delta_bucket=prior_detail["prior_delta_bucket"],
            source=source,
        )
        return max(0.0, min(1.0, float(expected_lambda)))

    def _best_accept_subset(
        self,
        base_offers: dict[str, tuple],
        active_offers: dict[str, tuple],
        *,
        utility_cache: dict[tuple, float] | None = None,
    ) -> tuple[float, tuple[str, ...]]:
        best_utility = -float("inf")
        best_ids: tuple[str, ...] = ()
        active_ids = tuple(active_offers)
        for partner_ids in powerset(active_ids):
            utility = self._utility_from_offer_dict_cached(
                base_offers | {partner_id: active_offers[partner_id] for partner_id in partner_ids},
                utility_cache,
            )
            if utility > best_utility:
                best_utility = utility
                best_ids = tuple(partner_ids)
        return best_utility, best_ids

    def _use_counter_trinary_approx_for_subset(
        self,
        n_counter_ids: int,
        total_quantity_candidates: tuple[int, ...],
    ) -> bool:
        mode = int(self.counter_trinary_approx_candidate_mode)
        if mode <= 0:
            return False
        if mode >= 2:
            return True

        exact_candidate_count = sum(
            self._bounded_integer_allocation_count(
                int(total_quantity),
                int(n_counter_ids),
                self.awi.n_lines,
            )
            for total_quantity in total_quantity_candidates
        )
        return exact_candidate_count > 256

    def _bounded_integer_allocation_count(
        self,
        total_quantity: int,
        n_bins: int,
        max_per_bin: int,
    ) -> int:
        memo: dict[tuple[int, int], int] = {}

        def count(remaining: int, bins_left: int) -> int:
            if bins_left <= 0:
                return 1 if remaining == 0 else 0
            if remaining < 0 or remaining > bins_left * max_per_bin:
                return 0
            key = (remaining, bins_left)
            cached = memo.get(key)
            if cached is not None:
                return cached
            total = 0
            for value in range(0, min(max_per_bin, remaining) + 1):
                total += count(remaining - value, bins_left - 1)
            memo[key] = total
            return total

        return count(int(total_quantity), int(n_bins))

    def _counter_trinary_approx_total_quantity_candidates(
        self,
        offering_quantity: int,
        active_offers: dict[str, tuple],
        full_candidates: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not full_candidates:
            return ()
        full_set = {int(value) for value in full_candidates}
        current_total = sum(int(offer[QUANTITY]) for offer in active_offers.values())
        raw_candidates = (
            0,
            int(offering_quantity) - 2,
            int(offering_quantity) - 1,
            int(offering_quantity),
            int(offering_quantity) + 1,
            int(offering_quantity) + 2,
            current_total,
        )
        candidates = [
            int(value)
            for value in raw_candidates
            if int(value) in full_set
        ]
        if not candidates:
            candidates = [min(full_set, key=lambda value: abs(value - int(offering_quantity)))]
        return tuple(dict.fromkeys(candidates))

    def _counter_trinary_approx_allocations(
        self,
        *,
        counter_ids: tuple[str, ...],
        active_offers: dict[str, tuple],
        total_quantity: int,
        is_selling: bool,
        attempt_index_by_partner: dict[str, int],
    ) -> list[tuple[int, ...]]:
        n_partners = len(counter_ids)
        if n_partners <= 0:
            return []
        max_per_partner = self.awi.n_lines
        if total_quantity < 0 or total_quantity > n_partners * max_per_partner:
            return []

        max_candidates = 96
        candidates: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add_candidate(values) -> None:
            if len(candidates) >= max_candidates:
                return
            allocation = tuple(int(value) for value in values)
            if len(allocation) != n_partners:
                return
            if any(value < 0 or value > max_per_partner for value in allocation):
                return
            if sum(allocation) != int(total_quantity):
                return
            if allocation in seen:
                return
            seen.add(allocation)
            candidates.append(allocation)

        current_quantities = [
            int(active_offers[partner_id][QUANTITY])
            for partner_id in counter_ids
        ]
        probability_weights = []
        for partner_id in counter_ids:
            current_quantity = int(active_offers[partner_id][QUANTITY])
            detail = self._counter_trinary_probability_detail(
                partner_id,
                is_selling,
                0,
                int(attempt_index_by_partner.get(partner_id, 1)),
            )
            probability_weights.append(max(0.0, float(detail["p_accept"])))

        add_candidate(
            self._balanced_integer_allocation(
                int(total_quantity),
                n_partners,
                max_per_partner,
            )
        )
        add_candidate(
            self._counter_trinary_weighted_allocation(
                int(total_quantity),
                current_quantities,
                max_per_partner,
            )
        )
        add_candidate(
            self._counter_trinary_weighted_allocation(
                int(total_quantity),
                probability_weights,
                max_per_partner,
            )
        )
        combined_weights = [
            max(0.0, probability) * max(1, current_quantity)
            for probability, current_quantity in zip(
                probability_weights,
                current_quantities,
            )
        ]
        add_candidate(
            self._counter_trinary_weighted_allocation(
                int(total_quantity),
                combined_weights,
                max_per_partner,
            )
        )

        ranked_indices = sorted(
            range(n_partners),
            key=lambda index: (
                probability_weights[index],
                current_quantities[index],
            ),
            reverse=True,
        )
        for index in ranked_indices:
            add_candidate(
                self._balanced_integer_allocation(
                    int(total_quantity),
                    n_partners,
                    max_per_partner,
                    preferred_indices=[index],
                )
            )
        if len(ranked_indices) >= 2:
            add_candidate(
                self._balanced_integer_allocation(
                    int(total_quantity),
                    n_partners,
                    max_per_partner,
                    preferred_indices=ranked_indices[:2],
                )
            )

        if sum(current_quantities) == int(total_quantity):
            add_candidate(current_quantities)

        for allocation in list(candidates):
            if len(candidates) >= max_candidates:
                break
            for source_index, quantity in enumerate(allocation):
                if quantity <= 0:
                    continue
                for target_index, target_quantity in enumerate(allocation):
                    if source_index == target_index:
                        continue
                    if target_quantity >= max_per_partner:
                        continue
                    neighbor = list(allocation)
                    neighbor[source_index] -= 1
                    neighbor[target_index] += 1
                    add_candidate(neighbor)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

        return candidates

    def _counter_trinary_weighted_allocation(
        self,
        total_quantity: int,
        weights: list[float] | list[int],
        max_per_bin: int,
    ) -> tuple[int, ...]:
        n_bins = len(weights)
        if n_bins <= 0:
            return ()
        total_quantity = int(total_quantity)
        if total_quantity <= 0:
            return tuple(0 for _ in range(n_bins))
        positive_weights = [max(0.0, float(weight)) for weight in weights]
        if sum(positive_weights) <= 0.0:
            return self._balanced_integer_allocation(
                total_quantity,
                n_bins,
                max_per_bin,
            )

        allocation = [0] * n_bins
        remaining = total_quantity
        total_weight = sum(positive_weights)
        fractional_parts: list[tuple[float, int]] = []
        for index, weight in enumerate(positive_weights):
            ideal = total_quantity * weight / total_weight
            assigned = min(max_per_bin, int(math.floor(ideal)))
            allocation[index] = assigned
            remaining -= assigned
            fractional_parts.append((ideal - assigned, index))

        for _fraction, index in sorted(fractional_parts, reverse=True):
            if remaining <= 0:
                break
            if allocation[index] >= max_per_bin:
                continue
            allocation[index] += 1
            remaining -= 1

        while remaining > 0:
            changed = False
            for index in sorted(
                range(n_bins),
                key=lambda item: positive_weights[item],
                reverse=True,
            ):
                if remaining <= 0:
                    break
                if allocation[index] >= max_per_bin:
                    continue
                allocation[index] += 1
                remaining -= 1
                changed = True
            if not changed:
                break
        return tuple(allocation)

    def _counter_trinary_total_quantity_candidates(
        self,
        offering_quantity: int,
        n_partners: int,
    ) -> tuple[int, ...]:
        if n_partners <= 0:
            return ()
        if offering_quantity <= 0:
            return (0,)
        if self.counter_trinary_total_quantity_mode == 0:
            return tuple(dict.fromkeys((0, max(1, int(offering_quantity)))))
        if self.counter_trinary_total_quantity_mode == 1:
            return tuple(range(0, self.awi.n_lines + 1))
        extended_total_cap = int(
            math.ceil(
                self.awi.n_lines
                * float(self.counter_trinary_extended_total_quantity_multiplier)
            )
        )
        total_cap = min(
            n_partners * self.awi.n_lines,
            max(int(offering_quantity), extended_total_cap),
        )
        return tuple(range(0, max(1, total_cap) + 1))

    def _bounded_integer_allocations(
        self,
        total_quantity: int,
        n_bins: int,
        max_per_bin: int,
    ) -> list[tuple[int, ...]]:
        if n_bins <= 0:
            return []
        if total_quantity < 0 or total_quantity > n_bins * max_per_bin:
            return []
        if n_bins == 1:
            return [(total_quantity,)] if total_quantity <= max_per_bin else []
        allocations: list[tuple[int, ...]] = []
        max_first = min(max_per_bin, total_quantity)
        for first in range(0, max_first + 1):
            for rest in self._bounded_integer_allocations(
                total_quantity - first,
                n_bins - 1,
                max_per_bin,
            ):
                allocations.append((first,) + rest)
        return allocations

    def _counter_acceptance_rate_decision_response(
        self,
        *,
        offers: dict[str, tuple],
        distribution: dict[str, int],
        partners: list[str],
        price: int,
        is_selling: bool,
        unneeded_response: SAOResponse,
    ) -> dict[str, SAOResponse] | None:
        if self.counter_acceptance_prior_weight < 0:
            return None
        if self.awi.current_step < self.counter_acceptance_rate_warmup_steps:
            self._debug_log_decision(
                "counter_acceptance_rate_decision_skip",
                reason="warmup",
                warmup_steps=self.counter_acceptance_rate_warmup_steps,
            )
            return None

        active_partners = [
            partner_id
            for partner_id in partners
            if partner_id in offers and int(distribution.get(partner_id, 0)) > 0
        ]
        if not active_partners:
            self._debug_log_decision(
                "counter_acceptance_rate_decision_skip",
                reason="no_current_offer_counter_candidates",
                distribution=distribution,
            )
            return None

        utility_cache: dict[tuple, float] = {}
        baseline_utility = None
        if self.debug_log_enabled:
            try:
                baseline_utility = self._utility_from_offer_dict_cached(
                    {partner_id: offers[partner_id] for partner_id in active_partners},
                    utility_cache,
                )
            except Exception as exc:                                            
                baseline_utility = {"error": repr(exc)}

        best_score = -float("inf")
        best_counter_ids: set[str] = set()
        best_details: list[dict[str, object]] = []
        best_candidate_debug: dict[str, object] | None = None
        candidate_summaries: list[dict[str, object]] = []

        for counter_ids_tuple in powerset(active_partners):
            counter_ids = set(counter_ids_tuple)
            accepted_ids = [p for p in active_partners if p not in counter_ids]
            accepted_offers = {p: offers[p] for p in accepted_ids}
            counter_offers: dict[str, tuple] = {}
            counter_probabilities: dict[str, float] = {}
            counter_details: list[dict[str, object]] = []

            for partner_id in counter_ids:
                counter_offer = (
                    int(distribution[partner_id]),
                    self.awi.current_step,
                    price,
                )
                delta_quantity = int(distribution[partner_id]) - int(
                    offers[partner_id][QUANTITY]
                )
                probability_detail = self._counter_acceptance_probability_detail(
                    partner_id,
                    delta_quantity,
                )
                p_accept = float(probability_detail["p_accept"])
                counter_offers[partner_id] = counter_offer
                counter_probabilities[partner_id] = p_accept
                counter_details.append(
                    {
                        "partner_id": partner_id,
                        "current_quantity": int(offers[partner_id][QUANTITY]),
                        "counter_quantity": int(distribution[partner_id]),
                        "delta_quantity": delta_quantity,
                        "p_accept": p_accept,
                        "probability_source": probability_detail["source"],
                        "probability_sample_weight": probability_detail[
                            "sample_weight"
                        ],
                    }
                )

            try:
                expected_score, subset_debug = (
                    self._expected_counter_acceptance_utility(
                        accepted_offers=accepted_offers,
                        counter_ids=tuple(counter_ids),
                        counter_offers=counter_offers,
                        counter_probabilities=counter_probabilities,
                        utility_cache=utility_cache,
                    )
                )
            except Exception as exc:                                         
                candidate_summaries.append(
                    {
                        "counter_ids": sorted(counter_ids),
                        "accepted_ids": sorted(accepted_ids),
                        "error": repr(exc),
                    }
                )
                continue
            candidate_summary = {
                "counter_ids": sorted(counter_ids),
                "accepted_ids": sorted(accepted_ids),
                "expected_score": expected_score,
                "subset_count": 2 ** len(counter_ids),
            }
            candidate_summaries.append(candidate_summary)
            if expected_score > best_score:
                best_score = expected_score
                best_counter_ids = counter_ids
                best_details = counter_details
                best_candidate_debug = candidate_summary | {
                    "success_subset_samples": subset_debug
                }

        if best_score == -float("inf"):
            self._debug_log_decision(
                "counter_acceptance_rate_decision_skip",
                reason="all_candidate_utility_errors",
            )
            return None

        accept_ids = set(active_partners).difference(best_counter_ids)
        chosen_response = {}
        for partner_id, quantity in distribution.items():
            if partner_id in accept_ids:
                chosen_response[partner_id] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    offers[partner_id],
                )
            elif quantity == 0:
                chosen_response[partner_id] = unneeded_response
            else:
                chosen_response[partner_id] = self._counter_reject_response(
                    partner_id=partner_id,
                    counter_quantity=int(quantity),
                    default_price=int(price),
                    offers=offers,
                    is_selling=is_selling,
                    unneeded_response=unneeded_response,
                )

        self._debug_log_decision(
            "counter_acceptance_rate_decision",
            is_selling=is_selling,
            active_partners=active_partners,
            distribution=distribution,
            baseline_utility=baseline_utility,
            best_expected_score=best_score,
            accepted_partner_ids=sorted(accept_ids),
            counter_partner_ids=sorted(best_counter_ids),
            counter_details=best_details,
            best_candidate=best_candidate_debug,
            candidate_summaries=candidate_summaries,
        )
        return chosen_response

    def _expected_counter_acceptance_utility(
        self,
        *,
        accepted_offers: dict[str, tuple],
        counter_ids: tuple[str, ...],
        counter_offers: dict[str, tuple],
        counter_probabilities: dict[str, float],
        utility_cache: dict[tuple, float] | None = None,
    ) -> tuple[float, list[dict[str, object]]]:
        expected_score = 0.0
        subset_debug: list[dict[str, object]] = []
        for success_ids_tuple in powerset(counter_ids):
            success_ids = set(success_ids_tuple)
            probability = 1.0
            for partner_id in counter_ids:
                p_accept = counter_probabilities[partner_id]
                probability *= p_accept if partner_id in success_ids else 1 - p_accept
            realized_offers = accepted_offers | {
                partner_id: counter_offers[partner_id]
                for partner_id in success_ids
            }
            utility = self._utility_from_offer_dict_cached(
                realized_offers,
                utility_cache,
            )
            expected_score += probability * utility
            if len(subset_debug) < 8:
                subset_debug.append(
                    {
                        "success_ids": sorted(success_ids),
                        "probability": probability,
                        "utility": utility,
                    }
                )
        return expected_score, subset_debug

    def _counter_acceptance_probability_detail(
        self,
        partner_id: str,
        delta_quantity: int,
    ) -> dict[str, object]:
        stats = getattr(self, "counter_acceptance_stats", {})
        exact = stats.get((partner_id, int(delta_quantity)), {})
        successes = float(exact.get("successes", 0.0))
        attempts = float(exact.get("attempts", 0.0))
        prior_detail = self._counter_acceptance_prior_detail(delta_quantity)
        prior_successes = float(prior_detail["prior_successes"])
        prior_attempts = float(prior_detail["prior_attempts"])
        p_accept = (
            (prior_successes + successes) / (prior_attempts + attempts)
            if prior_attempts + attempts > 0
            else float(prior_detail["prior_rate"])
        )
        return {
            "p_accept": p_accept,
            "source": (
                "partner_exact_delta_with_level_prior"
                if attempts > 0
                else "level_delta_prior"
            ),
            "sample_weight": attempts,
            "online_successes": successes,
            "online_attempts": attempts,
            **prior_detail,
        }

    def _counter_acceptance_prior_level(self) -> str:
        return "L0" if self.awi.is_first_level else "L1"

    def _counter_acceptance_prior_detail(self, delta_quantity: int) -> dict[str, object]:
        level = self._counter_acceptance_prior_level()
        table = self.counter_acceptance_rate_prior_by_level_delta.get(level, {})
        if not table:
            prior_attempts = max(0.0, float(self.counter_acceptance_prior_weight))
            return {
                "prior_level": level,
                "prior_delta": int(delta_quantity),
                "prior_rate": 0.5,
                "prior_successes": 0.5 * prior_attempts,
                "prior_attempts": prior_attempts,
                "prior_delta_clamped": False,
            }
        delta = int(delta_quantity)
        min_delta = min(table)
        max_delta = max(table)
        prior_delta = max(min_delta, min(max_delta, delta))
        prior_rate = float(table[prior_delta])
        prior_attempts = max(0.0, float(self.counter_acceptance_prior_weight))
        return {
            "prior_level": level,
            "prior_delta": prior_delta,
            "prior_rate": prior_rate,
            "prior_successes": prior_rate * prior_attempts,
            "prior_attempts": prior_attempts,
            "prior_delta_clamped": prior_delta != delta,
        }

    def _counter_attempt_bucket(self, attempt_index: int | str | None) -> str:
        try:
            value = int(attempt_index)
        except (TypeError, ValueError):
            return "1"
        if value >= 4:
            return "4+"
        return str(max(1, value))

    def _counter_attempt_index_from_bucket(self, attempt_bucket: str) -> int:
        return 4 if attempt_bucket == "4+" else max(1, int(attempt_bucket))

    def _counter_delta_bucket(self, delta_quantity: int) -> str:
        delta = int(delta_quantity)
        if delta <= -5:
            return "large_decrease"
        if delta <= -1:
            return "small_decrease"
        if delta == 0:
            return "same_quantity"
        if delta <= 2:
            return "small_increase"
        if delta <= 4:
            return "medium_increase"
        return "large_increase"

    def _counter_delta_bucket_representative(self, delta_bucket: str) -> int:
        return {
            "large_decrease": -7,
            "small_decrease": -2,
            "same_quantity": 0,
            "small_increase": 1,
            "medium_increase": 3,
            "large_increase": 7,
        }.get(delta_bucket, 0)

    def _counter_trinary_observation_counts(
        self,
        stats: dict[tuple[str, bool, str, str], dict[str, int]],
        partner_id: str,
        is_selling: bool,
        delta_bucket: str,
        attempt_bucket: str | None = None,
    ) -> tuple[float, float, float, float]:
        accepts = neutrals = rejects = total = 0.0
        for (
            observed_partner,
            observed_is_selling,
            observed_delta_bucket,
            observed_attempt_bucket,
        ), value in stats.items():
            if observed_partner != partner_id or bool(observed_is_selling) != is_selling:
                continue
            if observed_delta_bucket != delta_bucket:
                continue
            if attempt_bucket is not None and observed_attempt_bucket != attempt_bucket:
                continue
            accepts += float(value.get("accepts", 0))
            neutrals += float(value.get("neutrals", 0))
            rejects += float(value.get("rejects", 0))
        total = accepts + neutrals + rejects
        return accepts, neutrals, rejects, total

    def _counter_trinary_probability_detail(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
        attempt_index: int = 1,
    ) -> dict[str, object]:
        stats = getattr(self, "counter_trinary_stats", {})
        delta_bucket = self._counter_delta_bucket(delta_quantity)
        attempt_bucket = self._counter_attempt_bucket(attempt_index)
        accepts, neutrals, rejects, exact_total = (
            self._counter_trinary_observation_counts(
                stats,
                partner_id,
                bool(is_selling),
                delta_bucket,
                attempt_bucket,
            )
        )
        online_source = "partner_side_delta_bucket_attempt"
        if exact_total <= 0.0:
            accepts, neutrals, rejects, exact_total = (
                self._counter_trinary_observation_counts(
                    stats,
                    partner_id,
                    bool(is_selling),
                    delta_bucket,
                    None,
                )
            )
            online_source = "partner_side_delta_bucket_all_attempts"
        same_sign_weight = 0.0
        same_sign_accepts = 0.0
        same_sign_neutrals = 0.0
        same_sign_rejects = 0.0
        if self.use_counter_trinary_same_sign_smoothing:
            (
                same_sign_accepts,
                same_sign_neutrals,
                same_sign_rejects,
                same_sign_weight,
            ) = self._same_sign_counter_trinary_observation_counts(
                stats,
                partner_id,
                bool(is_selling),
                int(delta_quantity),
                attempt_bucket,
            )
            accepts += same_sign_accepts
            neutrals += same_sign_neutrals
            rejects += same_sign_rejects
        online_total = accepts + neutrals + rejects
        prior_detail = self._counter_trinary_prior_detail(
            delta_quantity,
            attempt_index,
        )
        prior_weight = float(prior_detail["prior_weight"])
        prior_accepts = float(prior_detail["prior_accept"]) * prior_weight
        prior_neutrals = float(prior_detail["prior_neutral"]) * prior_weight
        prior_rejects = float(prior_detail["prior_reject"]) * prior_weight
        total = online_total + prior_weight
        if total <= 0.0:
            p_accept = float(prior_detail["prior_accept"])
            p_neutral = float(prior_detail["prior_neutral"])
            p_reject = float(prior_detail["prior_reject"])
        else:
            p_accept = (accepts + prior_accepts) / total
            p_neutral = (neutrals + prior_neutrals) / total
            p_reject = (rejects + prior_rejects) / total
        total_p = p_accept + p_neutral + p_reject
        if total_p > 0.0:
            p_accept /= total_p
            p_neutral /= total_p
            p_reject /= total_p
        return {
            "p_accept": p_accept,
            "p_neutral": p_neutral,
            "p_reject": p_reject,
            "source": (
                f"{online_source}_same_sign_with_attempt_prior"
                if same_sign_weight > 0
                else f"{online_source}_with_attempt_prior"
                if exact_total > 0
                else prior_detail["prior_source"]
            ),
            "sample_weight": online_total,
            "exact_sample_weight": exact_total,
            "same_sign_sample_weight": same_sign_weight,
            "online_accepts": accepts,
            "online_neutrals": neutrals,
            "online_rejects": rejects,
            "same_sign_accepts": same_sign_accepts,
            "same_sign_neutrals": same_sign_neutrals,
            "same_sign_rejects": same_sign_rejects,
            "delta_bucket": delta_bucket,
            "attempt_index": int(attempt_index),
            "attempt_bucket": attempt_bucket,
            **prior_detail,
        }

    def _same_sign_counter_trinary_observation_counts(
        self,
        stats: dict[tuple[str, bool, str, str], dict[str, int]],
        partner_id: str,
        is_selling: bool,
        target_delta: int,
        attempt_bucket: str,
    ) -> tuple[float, float, float, float]:
        if target_delta == 0:
            return 0.0, 0.0, 0.0, 0.0
        target_positive = target_delta > 0
        target_delta_bucket = self._counter_delta_bucket(target_delta)
        weighted_accepts = 0.0
        weighted_neutrals = 0.0
        weighted_rejects = 0.0
        total_weight = 0.0
        for (
            observed_partner,
            observed_is_selling,
            observed_delta_bucket,
            observed_attempt_bucket,
        ), value in stats.items():
            if observed_partner != partner_id or bool(observed_is_selling) != is_selling:
                continue
            if observed_attempt_bucket != attempt_bucket:
                continue
            if observed_delta_bucket == target_delta_bucket:
                continue
            observed_delta = self._counter_delta_bucket_representative(
                str(observed_delta_bucket)
            )
            if (observed_delta > 0) != target_positive:
                continue
            observations = (
                float(value.get("accepts", 0))
                + float(value.get("neutrals", 0))
                + float(value.get("rejects", 0))
            )
            if observations <= 0.0:
                continue
            weight = 1.0 / (1.0 + abs(observed_delta - target_delta))
            weighted_accepts += float(value.get("accepts", 0)) * weight
            weighted_neutrals += float(value.get("neutrals", 0)) * weight
            weighted_rejects += float(value.get("rejects", 0)) * weight
            total_weight += observations * weight
        return weighted_accepts, weighted_neutrals, weighted_rejects, total_weight

    def _counter_trinary_prior_detail(
        self,
        delta_quantity: int,
        attempt_index: int = 1,
    ) -> dict[str, object]:
        level = self._counter_acceptance_prior_level()
        prior_weight = max(0.0, float(self.counter_acceptance_prior_weight))
        attempt_bucket = self._counter_attempt_bucket(attempt_index)
        attempt_table = self.counter_trinary_rate_prior_by_level_attempt_delta.get(
            level,
            {},
        )
        table = attempt_table.get(attempt_bucket, {})
        prior_source = "level_attempt_delta_prior"
        prior_attempt_bucket = attempt_bucket
        if not table:
            table = self.counter_trinary_rate_prior_by_level_delta.get(level, {})
            prior_source = "level_delta_prior"
            prior_attempt_bucket = None
        if not table:
            overall_rates = [
                rates
                for attempt_rates in attempt_table.values()
                for rates in attempt_rates.values()
            ]
            prior_source = "level_overall_prior"
            prior_attempt_bucket = None
            if overall_rates:
                prior_accept = sum(
                    float(rates["accept"]) for rates in overall_rates
                ) / len(overall_rates)
                prior_neutral = sum(
                    float(rates["neutral"]) for rates in overall_rates
                ) / len(overall_rates)
                prior_reject = sum(
                    float(rates["reject"]) for rates in overall_rates
                ) / len(overall_rates)
                table = {int(delta_quantity): {
                    "accept": prior_accept,
                    "neutral": prior_neutral,
                    "reject": prior_reject,
                }}
        if not table:
            return {
                "prior_level": level,
                "prior_delta": int(delta_quantity),
                "prior_attempt_bucket": prior_attempt_bucket,
                "prior_accept": 1.0 / 3.0,
                "prior_neutral": 1.0 / 3.0,
                "prior_reject": 1.0 / 3.0,
                "prior_weight": prior_weight,
                "prior_delta_clamped": False,
                "prior_source": "uniform_prior",
            }
        delta = int(delta_quantity)
        prior_delta = min(table, key=lambda candidate: (abs(candidate - delta), candidate))
        rates = table[prior_delta]
        prior_accept = float(rates["accept"])
        prior_neutral = float(rates["neutral"])
        prior_reject = float(rates["reject"])
        total = prior_accept + prior_neutral + prior_reject
        if total <= 0.0:
            prior_accept = prior_neutral = prior_reject = 1.0 / 3.0
        else:
            prior_accept /= total
            prior_neutral /= total
            prior_reject /= total
        return {
            "prior_level": level,
            "prior_delta": prior_delta,
            "prior_attempt_bucket": prior_attempt_bucket,
            "prior_accept": prior_accept,
            "prior_neutral": prior_neutral,
            "prior_reject": prior_reject,
            "prior_weight": prior_weight,
            "prior_delta_clamped": prior_delta != delta,
            "prior_source": prior_source,
        }

    def _smoothed_counter_acceptance_rate(
        self,
        delta_quantity: int,
        successes: float,
        attempts: float,
    ) -> float:
        prior_detail = self._counter_acceptance_prior_detail(delta_quantity)
        prior_successes = float(prior_detail["prior_successes"])
        prior_attempts = float(prior_detail["prior_attempts"])
        return (successes + prior_successes) / (attempts + prior_attempts)

    def _record_pending_first_attempts(
        self,
        proposals: dict[str, tuple | None],
        is_selling: bool,
    ) -> None:
        pending = getattr(self, "pending_first_attempts", None)
        if pending is None:
            return
        for partner_id, offer in proposals.items():
            if offer is None:
                continue
            quantity = int(offer[QUANTITY])
            if quantity <= 0:
                continue
            price_context = self._counter_price_bucket_detail(
                is_selling=bool(is_selling),
                unit_price=int(offer[UNIT_PRICE]),
            )
            attempt = {
                "partner_id": partner_id,
                "is_selling": bool(is_selling),
                "quantity": quantity,
                "quantity_bucket": self._first_quantity_bucket(quantity),
                "price": int(offer[UNIT_PRICE]),
                **price_context,
                "step": self.awi.current_step,
            }
            pending[partner_id] = attempt
            self._debug_log_decision(
                "first_trinary_attempt_pending",
                **attempt,
            )

    def _clear_pending_first_attempts_for_received_offers(
        self,
        offers: dict[str, tuple],
    ) -> None:
        pending = getattr(self, "pending_first_attempts", {})
        for partner_id in list(offers):
            attempt = pending.pop(partner_id, None)
            if attempt is not None:
                self._record_first_trinary_result(
                    partner_id,
                    bool(attempt["is_selling"]),
                    int(attempt["quantity"]),
                    unit_price=int(attempt["price"]),
                    partner_price_bucket=str(
                        attempt.get("partner_price_bucket", "unknown")
                    ),
                    outcome="counter",
                    source="partner_sent_counter",
                )

    def _resolve_pending_first_attempt_accept(self, partner_id: str) -> None:
        pending = getattr(self, "pending_first_attempts", {})
        attempt = pending.pop(partner_id, None)
        if attempt is None:
            return
        self._record_first_trinary_result(
            partner_id,
            bool(attempt["is_selling"]),
            int(attempt["quantity"]),
            unit_price=int(attempt["price"]),
            partner_price_bucket=str(attempt.get("partner_price_bucket", "unknown")),
            outcome="accept",
            source="direct_contract",
        )

    def _resolve_pending_first_attempt_reject(self, partner_id: str) -> None:
        pending = getattr(self, "pending_first_attempts", {})
        attempt = pending.pop(partner_id, None)
        if attempt is None:
            return
        self._record_first_trinary_result(
            partner_id,
            bool(attempt["is_selling"]),
            int(attempt["quantity"]),
            unit_price=int(attempt["price"]),
            partner_price_bucket=str(attempt.get("partner_price_bucket", "unknown")),
            outcome="reject",
            source="negotiation_failure",
        )

    def _record_first_trinary_result(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
        *,
        unit_price: int | None = None,
        partner_price_bucket: str | None = None,
        outcome: str,
        source: str,
    ) -> None:
        stats = getattr(self, "first_trinary_stats", None)
        if stats is None:
            return
        quantity_bucket = self._first_quantity_bucket(quantity)
        key = (partner_id, bool(is_selling), quantity_bucket)
        value = stats.setdefault(key, {"accepts": 0, "counters": 0, "rejects": 0})
        field_name = {
            "accept": "accepts",
            "counter": "counters",
            "reject": "rejects",
        }[outcome]
        value[field_name] += 1
        if unit_price is not None and partner_price_bucket is not None:
            self._record_first_price_response_result(
                partner_id,
                bool(is_selling),
                int(quantity),
                int(unit_price),
                str(partner_price_bucket),
                outcome=outcome,
                source=source,
            )
        probability_detail = self._first_trinary_probability_detail(
            partner_id,
            bool(is_selling),
            int(quantity),
            price_bucket=(
                str(partner_price_bucket)
                if partner_price_bucket in ("partner_favorable", "self_favorable")
                else None
            ),
        )
        self._debug_log_decision(
            "first_trinary_result",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            quantity=int(quantity),
            quantity_bucket=quantity_bucket,
            unit_price=unit_price,
            partner_price_bucket=partner_price_bucket,
            outcome=outcome,
            source=source,
            accepts=value["accepts"],
            counters=value["counters"],
            rejects=value["rejects"],
            p_accept=probability_detail["p_accept"],
            p_counter=probability_detail["p_counter"],
            p_reject=probability_detail["p_reject"],
        )

    def _record_first_price_response_result(
        self,
        partner_id: str,
        is_selling: bool,
        quantity: int,
        unit_price: int,
        partner_price_bucket: str,
        *,
        outcome: str,
        source: str,
    ) -> None:
        quantity_stats = getattr(self, "first_price_quantity_stats", None)
        quantity_price_stats = getattr(self, "first_price_quantity_price_stats", None)
        if quantity_stats is None or quantity_price_stats is None:
            return
        field_name = {
            "accept": "accepts",
            "counter": "counters",
            "reject": "rejects",
        }[outcome]
        quantity_key = (partner_id, bool(is_selling), int(quantity))
        quantity_value = quantity_stats.setdefault(
            quantity_key,
            {"accepts": 0, "counters": 0, "rejects": 0},
        )
        quantity_value[field_name] += 1

        quantity_price_key = (
            partner_id,
            bool(is_selling),
            int(quantity),
            int(unit_price),
            str(partner_price_bucket),
        )
        quantity_price_value = quantity_price_stats.setdefault(
            quantity_price_key,
            {"accepts": 0, "counters": 0, "rejects": 0},
        )
        quantity_price_value[field_name] += 1
        self._debug_log_decision(
            "first_price_response_result",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            quantity=int(quantity),
            quantity_bucket=self._first_quantity_bucket(quantity),
            unit_price=int(unit_price),
            partner_price_bucket=str(partner_price_bucket),
            outcome=outcome,
            source=source,
            quantity_accepts=quantity_value["accepts"],
            quantity_counters=quantity_value["counters"],
            quantity_rejects=quantity_value["rejects"],
            quantity_price_accepts=quantity_price_value["accepts"],
            quantity_price_counters=quantity_price_value["counters"],
            quantity_price_rejects=quantity_price_value["rejects"],
        )

    def _clear_pending_counter_attempts_for_received_offers(
        self,
        offers: dict[str, tuple],
    ) -> None:
        pending = getattr(self, "pending_counter_attempts", {})
        for partner_id in list(offers):
            attempt = pending.pop(partner_id, None)
            if attempt is not None:
                self._record_counter_trinary_neutral_lambda(
                    partner_id,
                    bool(
                        attempt.get(
                            "is_selling",
                            self._is_selling_partner(partner_id),
                        )
                    ),
                    int(attempt["delta_quantity"]),
                    int(attempt.get("attempt_index", 1)),
                    int(attempt["current_quantity"]),
                    int(attempt["counter_quantity"]),
                    int(offers[partner_id][QUANTITY]),
                    source="partner_sent_further_counter",
                )
                self._record_counter_trinary_result(
                    partner_id,
                    bool(
                        attempt.get(
                            "is_selling",
                            self._is_selling_partner(partner_id),
                        )
                    ),
                    int(attempt["delta_quantity"]),
                    int(attempt.get("attempt_index", 1)),
                    outcome="neutral",
                    source="partner_sent_further_counter",
                )
                self._record_counter_price_response_result_from_attempt(
                    attempt,
                    outcome="counter",
                    source="partner_sent_further_counter",
                )
                self._debug_log_decision(
                    "counter_acceptance_attempt_neutralized",
                    reason="partner_sent_further_counter",
                    partner_id=partner_id,
                    attempt=attempt,
                    received_offer=offers[partner_id],
                )

    def _record_pending_counter_responses(
        self,
        responses: dict[str, SAOResponse],
        offers: dict[str, tuple],
        *,
        source: str,
    ) -> None:
        pending = getattr(self, "pending_counter_attempts", None)
        if pending is None:
            return
        for partner_id, response in responses.items():
            if partner_id not in offers:
                continue
            if response.response != ResponseType.REJECT_OFFER:
                continue
            if response.outcome is None:
                continue
            counter_quantity = int(response.outcome[QUANTITY])
            if counter_quantity <= 0:
                continue
            current_quantity = int(offers[partner_id][QUANTITY])
            unit_price = int(response.outcome[UNIT_PRICE])
            is_selling = self._is_selling_partner(partner_id)
            price_context = self._counter_price_bucket_detail(
                is_selling=is_selling,
                unit_price=unit_price,
            )
            attempt_order = getattr(self, "counter_attempt_order_by_partner", {})
            attempt_index = min(4, int(attempt_order.get(partner_id, 0)) + 1)
            attempt_order[partner_id] = attempt_index
            attempt = {
                "partner_id": partner_id,
                "is_selling": is_selling,
                "delta_quantity": counter_quantity - current_quantity,
                "delta_bucket": self._counter_delta_bucket(
                    counter_quantity - current_quantity
                ),
                "attempt_index": attempt_index,
                "attempt_bucket": self._counter_attempt_bucket(attempt_index),
                "current_quantity": current_quantity,
                "counter_quantity": counter_quantity,
                "unit_price": unit_price,
                **price_context,
                "step": self.awi.current_step,
                "source": source,
            }
            pending[partner_id] = attempt
            self._debug_log_decision(
                "counter_acceptance_attempt_pending",
                **attempt,
            )

    def _resolve_pending_counter_attempt_success(
        self,
        partner_id: str,
        agreement_quantity: int,
        agreement_unit_price: int | None = None,
    ) -> None:
        pending = getattr(self, "pending_counter_attempts", {})
        attempt = pending.pop(partner_id, None)
        if attempt is None:
            return
        price_matches = (
            agreement_unit_price is None
            or int(agreement_unit_price)
            == int(attempt.get("unit_price", agreement_unit_price))
        )
        if int(agreement_quantity) == int(attempt["counter_quantity"]):
            self._record_counter_acceptance_result(
                partner_id,
                int(attempt["delta_quantity"]),
                accepted=True,
                source="direct_contract",
            )
            self._record_counter_trinary_result(
                partner_id,
                bool(
                    attempt.get(
                        "is_selling",
                        self._is_selling_partner(partner_id),
                    )
                ),
                int(attempt["delta_quantity"]),
                int(attempt.get("attempt_index", 1)),
                outcome="accept",
                source="direct_contract",
            )
            self._record_counter_price_response_result_from_attempt(
                attempt,
                outcome="accept" if price_matches else "counter",
                source=(
                    "direct_contract"
                    if price_matches
                    else "contract_price_differs_from_counter"
                ),
            )
            return
        self._record_counter_trinary_result(
            partner_id,
            bool(
                attempt.get(
                    "is_selling",
                    self._is_selling_partner(partner_id),
                )
            ),
            int(attempt["delta_quantity"]),
            int(attempt.get("attempt_index", 1)),
            outcome="neutral",
            source="contract_quantity_differs_from_counter",
        )
        self._record_counter_price_response_result_from_attempt(
            attempt,
            outcome="counter",
            source="contract_quantity_differs_from_counter",
        )
        self._debug_log_decision(
            "counter_acceptance_attempt_neutralized",
            reason="contract_quantity_differs_from_counter",
            partner_id=partner_id,
            attempt=attempt,
            agreement_quantity=agreement_quantity,
            agreement_unit_price=agreement_unit_price,
        )

    def _resolve_pending_counter_attempt_failure(self, partner_id: str) -> None:
        pending = getattr(self, "pending_counter_attempts", {})
        attempt = pending.pop(partner_id, None)
        if attempt is None:
            return
        self._record_counter_acceptance_result(
            partner_id,
            int(attempt["delta_quantity"]),
            accepted=False,
            source="negotiation_failure",
        )
        self._record_counter_trinary_result(
            partner_id,
            bool(
                attempt.get(
                    "is_selling",
                    self._is_selling_partner(partner_id),
                )
            ),
            int(attempt["delta_quantity"]),
            int(attempt.get("attempt_index", 1)),
            outcome="reject",
            source="negotiation_failure",
        )
        self._record_counter_price_response_result_from_attempt(
            attempt,
            outcome="reject",
            source="negotiation_failure",
        )

    def _record_counter_acceptance_result(
        self,
        partner_id: str,
        delta_quantity: int,
        *,
        accepted: bool,
        source: str,
    ) -> None:
        stats = getattr(self, "counter_acceptance_stats", None)
        if stats is None:
            return
        key = (partner_id, int(delta_quantity))
        value = stats.setdefault(key, {"successes": 0, "attempts": 0})
        value["attempts"] += 1
        if accepted:
            value["successes"] += 1
        self._debug_log_decision(
            "counter_acceptance_result",
            partner_id=partner_id,
            delta_quantity=delta_quantity,
            accepted=accepted,
            source=source,
            successes=value["successes"],
            attempts=value["attempts"],
            p_accept=self._smoothed_counter_acceptance_rate(
                delta_quantity,
                value["successes"],
                value["attempts"],
            ),
        )

    def _record_counter_trinary_result(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
        attempt_index: int = 1,
        *,
        outcome: str,
        source: str,
    ) -> None:
        stats = getattr(self, "counter_trinary_stats", None)
        if stats is None:
            return
        delta_bucket = self._counter_delta_bucket(delta_quantity)
        attempt_bucket = self._counter_attempt_bucket(attempt_index)
        key = (partner_id, bool(is_selling), delta_bucket, attempt_bucket)
        value = stats.setdefault(
            key,
            {"accepts": 0, "neutrals": 0, "rejects": 0},
        )
        field_name = {
            "accept": "accepts",
            "neutral": "neutrals",
            "reject": "rejects",
        }[outcome]
        value[field_name] += 1
        probability_detail = self._counter_trinary_probability_detail(
            partner_id,
            bool(is_selling),
            int(delta_quantity),
            int(attempt_index),
        )
        self._debug_log_decision(
            "counter_trinary_result",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            delta_quantity=delta_quantity,
            delta_bucket=delta_bucket,
            attempt_index=int(attempt_index),
            attempt_bucket=attempt_bucket,
            outcome=outcome,
            source=source,
            accepts=value["accepts"],
            neutrals=value["neutrals"],
            rejects=value["rejects"],
            p_accept=probability_detail["p_accept"],
            p_neutral=probability_detail["p_neutral"],
            p_reject=probability_detail["p_reject"],
        )

    def _counter_price_bucket_detail(
        self,
        *,
        is_selling: bool,
        unit_price: int,
    ) -> dict[str, object]:
        issues = (
            self.awi.current_output_issues
            if is_selling
            else self.awi.current_input_issues
        )
        price_min = int(issues[UNIT_PRICE].min_value)
        price_max = int(issues[UNIT_PRICE].max_value)
        if price_max <= price_min:
            return {
                "price_min": price_min,
                "price_max": price_max,
                "price_position": None,
                "partner_price_score": None,
                "partner_price_bucket": "flat",
            }

        price_position = (float(unit_price) - price_min) / (price_max - price_min)
        partner_price_score = 1.0 - price_position if is_selling else price_position
        if partner_price_score >= (2.0 / 3.0):
            partner_price_bucket = "partner_favorable"
        elif partner_price_score <= (1.0 / 3.0):
            partner_price_bucket = "self_favorable"
        else:
            partner_price_bucket = "middle"
        return {
            "price_min": price_min,
            "price_max": price_max,
            "price_position": price_position,
            "partner_price_score": partner_price_score,
            "partner_price_bucket": partner_price_bucket,
        }

    def _record_counter_price_response_result_from_attempt(
        self,
        attempt: dict[str, object],
        *,
        outcome: str,
        source: str,
    ) -> None:
        partner_id = str(attempt["partner_id"])
        is_selling = bool(
            attempt.get("is_selling", self._is_selling_partner(partner_id))
        )
        unit_price = int(attempt.get("unit_price", 0))
        delta_quantity = int(attempt["delta_quantity"])
        partner_price_bucket = str(
            attempt.get("partner_price_bucket", "unknown")
        )
        self._record_counter_price_response_result(
            partner_id,
            is_selling,
            delta_quantity,
            unit_price,
            partner_price_bucket,
            outcome=outcome,
            source=source,
        )

    def _record_counter_price_response_result(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
        unit_price: int,
        partner_price_bucket: str,
        *,
        outcome: str,
        source: str,
    ) -> None:
        field_name = {
            "accept": "accepts",
            "counter": "counters",
            "reject": "rejects",
        }[outcome]
        delta_stats = getattr(self, "counter_price_delta_stats", None)
        delta_price_stats = getattr(self, "counter_price_delta_price_stats", None)
        if delta_stats is None or delta_price_stats is None:
            return

        delta_key = (partner_id, bool(is_selling), int(delta_quantity))
        delta_value = delta_stats.setdefault(
            delta_key,
            {"accepts": 0, "counters": 0, "rejects": 0},
        )
        delta_value[field_name] += 1

        delta_price_key = (
            partner_id,
            bool(is_selling),
            int(delta_quantity),
            int(unit_price),
            str(partner_price_bucket),
        )
        delta_price_value = delta_price_stats.setdefault(
            delta_price_key,
            {"accepts": 0, "counters": 0, "rejects": 0},
        )
        delta_price_value[field_name] += 1
        self._debug_log_decision(
            "counter_price_response_result",
            partner_id=partner_id,
            is_selling=bool(is_selling),
            delta_quantity=int(delta_quantity),
            delta_bucket=self._counter_delta_bucket(int(delta_quantity)),
            unit_price=int(unit_price),
            partner_price_bucket=str(partner_price_bucket),
            outcome=outcome,
            source=source,
            delta_accepts=delta_value["accepts"],
            delta_counters=delta_value["counters"],
            delta_rejects=delta_value["rejects"],
            delta_price_accepts=delta_price_value["accepts"],
            delta_price_counters=delta_price_value["counters"],
            delta_price_rejects=delta_price_value["rejects"],
        )

    def _is_selling_partner(self, partner_id: str) -> bool:
        return partner_id in set(self.awi.my_consumers)

    def _matching_counter_offer_acceptance_partners(
        self,
        *,
        offers: dict[str, tuple],
        distribution: dict[str, int],
        needs: int,
    ) -> tuple[str, ...]:
        tolerance = self.counter_accept_matching_offer_tolerance
        if tolerance < 0 or needs <= 0:
            return ()

        candidates: list[tuple[int, int, str]] = []
        for partner_id, planned_quantity in distribution.items():
            if planned_quantity <= 0 or partner_id not in offers:
                continue
            offered_quantity = int(offers[partner_id][QUANTITY])
            if offered_quantity <= 0:
                continue
            quantity_error = abs(offered_quantity - int(planned_quantity))
            if quantity_error <= tolerance:
                candidates.append((quantity_error, -offered_quantity, partner_id))

        accepted: list[str] = []
        accepted_quantity = 0
        for _, neg_offered_quantity, partner_id in sorted(candidates):
            offered_quantity = -neg_offered_quantity
            if accepted_quantity + offered_quantity > needs:
                continue
            accepted.append(partner_id)
            accepted_quantity += offered_quantity

        if accepted:
            self._debug_log_decision(
                "matching_counter_offer_acceptance_scan",
                tolerance=tolerance,
                needs=needs,
                distribution=distribution,
                candidates=[
                    {
                        "partner_id": partner_id,
                        "planned_quantity": distribution[partner_id],
                        "offered_quantity": offers[partner_id][QUANTITY],
                        "quantity_error": abs(
                            offers[partner_id][QUANTITY]
                            - distribution[partner_id]
                        ),
                    }
                    for _, _, partner_id in sorted(candidates)
                ],
                accepted_partner_ids=accepted,
                accepted_quantity=accepted_quantity,
            )
        return tuple(accepted)

    def _counter_quantity_distribution(
        self,
        *,
        partners: list[str],
        offering_quantity: int,
        shortage_quantity: int,
        offers: dict[str, tuple],
    ) -> dict[str, int]:
        if not self.use_incoming_quantity_counter_distribution:
            if self._is_late_phase() and len(partners) > 0:
                concentrated_ids = self._concentrated_ids(partners)
                concentrated_idx = [
                    i for i, p in enumerate(partners) if p in concentrated_ids
                ]
                quantities = distribute(
                    offering_quantity,
                    len(partners),
                    mx=self.awi.n_lines,
                    concentrated=True,
                    concentrated_idx=concentrated_idx,
                )
                method = "late_performance_concentration"
            else:
                quantities = distribute(
                    offering_quantity, len(partners), mx=self.awi.n_lines
                )
                concentrated_ids = []
                concentrated_idx = []
                method = "normal"
            distribution = dict(zip(partners, quantities))
            self._debug_log_decision(
                "counter_quantity_distribution",
                method=method,
                partners=partners,
                offering_quantity=offering_quantity,
                shortage_quantity=shortage_quantity,
                distribution=distribution,
                concentrated_ids=concentrated_ids,
                concentrated_idx=concentrated_idx,
                use_incoming_quantity_counter_distribution=False,
            )
            return distribution

        distribution = self._incoming_quantity_scaled_counter_distribution(
            partners=partners,
            offering_quantity=offering_quantity,
            shortage_quantity=shortage_quantity,
            offers=offers,
        )
        self._debug_log_decision(
            "counter_quantity_distribution",
            method="incoming_quantity_scaled",
            partners=partners,
            offering_quantity=offering_quantity,
            shortage_quantity=shortage_quantity,
            distribution=distribution,
            use_incoming_quantity_counter_distribution=True,
            counter_distribution_always_min_one=(
                self.counter_distribution_always_min_one
            ),
        )
        return distribution

    def _incoming_quantity_scaled_counter_distribution(
        self,
        *,
        partners: list[str],
        offering_quantity: int,
        shortage_quantity: int,
        offers: dict[str, tuple],
    ) -> dict[str, int]:
        if not partners:
            return {}
        if offering_quantity <= 0:
            return {partner: 0 for partner in partners}

        weights = []
        for partner_id in partners:
            if partner_id in offers:
                weights.append(max(0.0, float(offers[partner_id][QUANTITY])))
            elif self.use_future_partner_prior:
                weights.append(max(0.0, float(self.future_partner_prior_weight)))
            else:
                weights.append(0.0)
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0] * len(partners)
            total_weight = len(partners)

        guarantee_min_one = self.counter_distribution_always_min_one or (
            shortage_quantity >= len(partners)
        )
        quantities = [1 if guarantee_min_one else 0 for _ in partners]
        remaining_capacity = [self.awi.n_lines - quantity for quantity in quantities]
        remaining = offering_quantity - sum(quantities)
        if remaining <= 0:
            distribution = dict(zip(partners, quantities))
            self._debug_log_decision(
                "incoming_quantity_scaled_counter_distribution",
                partners=partners,
                offering_quantity=offering_quantity,
                shortage_quantity=shortage_quantity,
                weights=weights,
                total_weight=total_weight,
                use_future_partner_prior=self.use_future_partner_prior,
                future_partner_prior_weight=self.future_partner_prior_weight,
                future_partners_without_offer=[
                    partner_id for partner_id in partners if partner_id not in offers
                ],
                guarantee_min_one=guarantee_min_one,
                initial_quantities=quantities,
                distribution=distribution,
                remaining=remaining,
            )
            return distribution

        raw_quantities = [remaining * weight / total_weight for weight in weights]
        additions = [
            min(remaining_capacity[i], int(quantity))
            for i, quantity in enumerate(raw_quantities)
        ]
        quantities = [
            quantity + addition
            for quantity, addition in zip(quantities, additions)
        ]
        remaining -= sum(additions)
        remainders = sorted(
            range(len(partners)),
            key=lambda i: raw_quantities[i] - int(raw_quantities[i]),
            reverse=True,
        )
        while remaining > 0 and remainders:
            changed = False
            for i in remainders:
                if quantities[i] >= self.awi.n_lines:
                    continue
                quantities[i] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
            if not changed:
                break
        distribution = dict(zip(partners, quantities))
        self._debug_log_decision(
            "incoming_quantity_scaled_counter_distribution",
            partners=partners,
            offering_quantity=offering_quantity,
            shortage_quantity=shortage_quantity,
            weights=weights,
            total_weight=total_weight,
            use_future_partner_prior=self.use_future_partner_prior,
            future_partner_prior_weight=self.future_partner_prior_weight,
            future_partners_without_offer=[
                partner_id for partner_id in partners if partner_id not in offers
            ],
            guarantee_min_one=guarantee_min_one,
            raw_quantities=raw_quantities,
            additions=additions,
            distribution=distribution,
            remaining=remaining,
        )
        return distribution

    def _large_shortage_offer_acceptance_partners(
        self,
        *,
        offers: dict[str, tuple],
        plist: list[tuple[str, ...]],
        minus_best_indx: int,
        minus_best_diff: int,
        plus_best_diff: int,
        is_selling: bool,
    ) -> tuple[str, ...]:
        if not self.use_large_shortage_offer_acceptance:
            self._debug_log_decision(
                "large_shortage_offer_gate",
                enabled=False,
                skip_reason="feature_disabled",
            )
            return ()
        if not self.awi.is_first_level or not is_selling:
            self._debug_log_decision(
                "large_shortage_offer_gate",
                enabled=True,
                skip_reason="not_l0_seller",
                is_selling=is_selling,
                is_first_level=self.awi.is_first_level,
            )
            return ()
        if minus_best_indx < 0 or minus_best_diff >= 0:
            self._debug_log_decision(
                "large_shortage_offer_gate",
                enabled=True,
                skip_reason="not_shortage_minus_best",
                minus_best_indx=minus_best_indx,
                minus_best_diff=minus_best_diff,
            )
            return ()
        if -minus_best_diff > plus_best_diff:
            self._debug_log_decision(
                "large_shortage_offer_gate",
                enabled=True,
                skip_reason="shortage_worse_than_plus_candidate",
                minus_best_diff=minus_best_diff,
                plus_best_diff=plus_best_diff,
            )
            return ()

        threshold = self._large_offer_acceptance_threshold()
        candidate_partner_ids = tuple(
            partner_id
            for partner_id in plist[minus_best_indx]
            if offers[partner_id][QUANTITY] >= threshold
        )
        self._debug_log_decision(
            "large_shortage_offer_gate",
            enabled=True,
            skip_reason=None if candidate_partner_ids else "no_offer_reaches_threshold",
            threshold=threshold,
            large_offer_acceptance_margin=self.large_offer_acceptance_margin,
            minus_best_indx=minus_best_indx,
            minus_best_diff=minus_best_diff,
            plus_best_diff=plus_best_diff,
            minus_best_partner_ids=plist[minus_best_indx],
            offer_quantities={
                partner_id: offers[partner_id][QUANTITY]
                for partner_id in plist[minus_best_indx]
            },
            candidate_partner_ids=candidate_partner_ids,
        )
        return candidate_partner_ids

    def _large_offer_acceptance_threshold(self) -> float:
        summary = self.awi.exogenous_contract_summary
        if len(summary) < 2:
            return float("inf")

        opponent_exogenous_quantity = summary[-1][0]
        opponent_agent_count = len(self.awi.my_consumers)
        own_agent_count = self.get_num_same_level_agents()
        if opponent_agent_count <= 0 or own_agent_count <= 0:
            return float("inf")

        expected_offer_quantity = (
            opponent_exogenous_quantity / opponent_agent_count / own_agent_count
        )
        return expected_offer_quantity + self.large_offer_acceptance_margin

    def _responses_for_accepted_offer_set(
        self,
        *,
        offers: dict[str, tuple],
        partner_ids: tuple[str, ...],
        best_diff: int,
        partners: set[str],
        future_partners: set[str],
        states: dict,
        issues,
        is_selling: bool,
        unneeded_response: SAOResponse,
        source: str = "unknown",
    ) -> dict[str, SAOResponse]:
        others = list(partners.difference(partner_ids).union(future_partners))
        response = {
            k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
            for k in partner_ids
        } | {k: unneeded_response for k in others}
        self._debug_log_decision(
            "accepted_offer_response",
            source=source,
            is_selling=is_selling,
            accepted_partner_ids=partner_ids,
            other_partner_ids=others,
            best_diff=best_diff,
            accepted_offers=self._debug_offer_dict(
                {partner_id: offers[partner_id] for partner_id in partner_ids}
            ),
        )

        if best_diff < 0 and len(others) > 0:
            p = self._counter_offer_price(issues, is_selling)
            relative_times = [states[p].relative_time for p in others if p in states]
            t = (
                min(relative_times)
                if relative_times
                else min(state.relative_time for state in states.values())
            )
            counter_overorder_ratio = (
                self._counter_overorder_ratio(t, is_selling)
                if len(others) > 1
                else None
            )
            offering_quanitity = (
                int(-best_diff * (1 + counter_overorder_ratio))
                if len(others) > 1
                else -best_diff
            )
            distribution = self._counter_quantity_distribution(
                partners=others,
                offering_quantity=offering_quanitity,
                shortage_quantity=-best_diff,
                offers=offers,
            )
            self._debug_log_decision(
                "accepted_offer_residual_counter",
                source=source,
                is_selling=is_selling,
                best_diff=best_diff,
                relative_time=t,
                counter_overorder_ratio=counter_overorder_ratio,
                offering_quantity=offering_quanitity,
                price=p,
                other_partner_ids=others,
                distribution=distribution,
            )
            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else self._counter_reject_response(
                            partner_id=k,
                            counter_quantity=int(q),
                            default_price=int(p),
                            offers=offers,
                            is_selling=is_selling,
                            unneeded_response=unneeded_response,
                        )
                    )
                    for k, q in distribution.items()
                }
            )

        return response

    def _accepted_offers_from_response(
        self, response: dict[str, SAOResponse]
    ) -> dict[str, tuple]:
        return {
            partner_id: sao_response.outcome
            for partner_id, sao_response in response.items()
            if sao_response.response == ResponseType.ACCEPT_OFFER
            and sao_response.outcome is not None
        }

    def _utility_from_offer_dict(self, offers: dict[str, tuple]) -> float:
        return self.ufun.from_offers(
            offers,
            return_info=False,
            ignore_signed_contracts=False,
        )

    def _utility_from_offer_dict_cached(
        self,
        offers: dict[str, tuple],
        utility_cache: dict[tuple, float] | None,
    ) -> float:
        if utility_cache is None:
            return self._utility_from_offer_dict(offers)
        key = tuple(
            sorted((partner_id, tuple(offer)) for partner_id, offer in offers.items())
        )
        cached = utility_cache.get(key)
        if cached is not None:
            return cached
        utility = self._utility_from_offer_dict(offers)
        utility_cache[key] = utility
        return utility

    def _quantity_acceptance_choice(
        self,
        *,
        plus_best_diff: int,
        plus_best_indx: int,
        minus_best_diff: int,
        minus_best_indx: int,
        plist: list[tuple[str, ...]],
        partners: set[str],
        future_partners: set[str],
        is_selling: bool,
    ) -> tuple[int, int]:
        if -minus_best_diff == plus_best_diff:
            if is_selling:
                self._debug_log_decision(
                    "quantity_acceptance_choice",
                    reason="equal_abs_diff_seller_prefers_minus",
                    selected_diff=minus_best_diff,
                    selected_indx=minus_best_indx,
                )
                return minus_best_diff, minus_best_indx
            self._debug_log_decision(
                "quantity_acceptance_choice",
                reason="equal_abs_diff_buyer_prefers_plus",
                selected_diff=plus_best_diff,
                selected_indx=plus_best_indx,
            )
            return plus_best_diff, plus_best_indx

        if -minus_best_diff < plus_best_diff:
            if (
                not is_selling
                and len(
                    partners.difference(plist[minus_best_indx]).union(
                        future_partners
                    )
                )
                == 0
            ):
                self._debug_log_decision(
                    "quantity_acceptance_choice",
                    reason="minus_closer_but_buyer_has_no_other_partners",
                    selected_diff=plus_best_diff,
                    selected_indx=plus_best_indx,
                )
                return plus_best_diff, plus_best_indx
            self._debug_log_decision(
                "quantity_acceptance_choice",
                reason="minus_closer",
                selected_diff=minus_best_diff,
                selected_indx=minus_best_indx,
            )
            return minus_best_diff, minus_best_indx

        self._debug_log_decision(
            "quantity_acceptance_choice",
            reason="plus_closer",
            selected_diff=plus_best_diff,
            selected_indx=plus_best_indx,
        )
        return plus_best_diff, plus_best_indx

    def _utility_fallback_acceptance_candidate(
        self,
        plist: list[tuple[str, ...]],
        offers: dict[str, tuple],
        states: dict,
        response: dict[str, SAOResponse],
        needs: int,
    ) -> tuple[int, int] | None:
        relative_time = min(state.relative_time for state in states.values())
        if relative_time < self.utility_fallback_relative_time:
            self._debug_log_decision(
                "utility_fallback_gate",
                enabled=False,
                skip_reason="too_early",
                relative_time=relative_time,
                threshold=self.utility_fallback_relative_time,
            )
            return None

        accepted_offers = self._accepted_offers_from_response(response)
        base_utility = self._utility_from_offer_dict(accepted_offers)
        best_utility = base_utility
        best_indx = -1
        best_diff = 0
        evaluated_candidates = []

        for i, partner_ids in enumerate(plist):
            if not partner_ids:
                continue

            candidate_offers = dict(accepted_offers)
            candidate_offers.update({p: offers[p] for p in partner_ids})
            utility = self._utility_from_offer_dict(candidate_offers)
            if self.debug_log_enabled:
                evaluated_candidates.append(
                    {
                        "index": i,
                        "partner_ids": list(partner_ids),
                        "diff": sum(offers[p][QUANTITY] for p in partner_ids)
                        - needs,
                        "utility": utility,
                    }
                )
            if utility > best_utility + 1e-9:
                best_utility = utility
                best_indx = i
                best_diff = sum(offers[p][QUANTITY] for p in partner_ids) - needs

        if best_indx < 0:
            self._debug_log_decision(
                "utility_fallback_gate",
                enabled=True,
                accepted=False,
                skip_reason="no_utility_improvement",
                relative_time=relative_time,
                threshold=self.utility_fallback_relative_time,
                base_accepted_offers=self._debug_offer_dict(accepted_offers),
                base_utility=base_utility,
                evaluated_candidates=evaluated_candidates,
            )
            return None
        self._debug_log_decision(
            "utility_fallback_gate",
            enabled=True,
            accepted=True,
            relative_time=relative_time,
            threshold=self.utility_fallback_relative_time,
            base_accepted_offers=self._debug_offer_dict(accepted_offers),
            base_utility=base_utility,
            best_indx=best_indx,
            best_diff=best_diff,
            best_partner_ids=plist[best_indx],
            best_utility=best_utility,
            evaluated_candidates=evaluated_candidates,
        )
        return best_diff, best_indx

    def _calculate_losses(
        self, offers: dict[str, tuple], world=None
    ) -> tuple[float, float]:
        if not offers:
            return (0.0, 0.0)
        sample_offer = next(iter(offers.values()))
        _, step, price = sample_offer
        qty_list, util_list = [], []
        for qty in range(0, 13):
            virtual_offer = (qty, step, price)
            u = self.ufun.from_offers(
                (virtual_offer,),
                (self.awi.is_first_level,),
                return_info=False,
                ignore_signed_contracts=True,
            )
            qty_list.append(qty)
            util_list.append(u)
        shortfall_loss = util_list[1] - util_list[0]
        disposal_loss = util_list[-2] - util_list[-1]
        return shortfall_loss, disposal_loss
    
    def _get_sd_ratio(self) -> float:
        return self._get_sd_ratio2()

    def _get_sd_ratio2(self) -> float:
        offers = self.awi.current_offers
        if offers:
            shortfall_loss, disposal_loss = self._calculate_losses(offers)
            source = "current_offers"
        else:
            is_selling = self.awi.is_first_level
            issues = (
                self.awi.current_output_issues
                if is_selling
                else self.awi.current_input_issues
            )
            price = self.calc_price(issues, is_selling, best_price=True)
            shortfall_loss, disposal_loss = self._calculate_losses(
                {"__virtual_sd_ratio_offer__": (0, self.awi.current_step, price)}
            )
            source = "virtual_offer"
        raw_ratio = shortfall_loss / (disposal_loss + 1e-9)
        ratio = min(10.0, max(0.1, raw_ratio))
        self._debug_log_decision(
            "sd_ratio",
            source=source,
            shortfall_loss=shortfall_loss,
            disposal_loss=disposal_loss,
            raw_ratio=raw_ratio,
            ratio=ratio,
        )
        return ratio

    def get_gap_d(self) -> int:
                                                                               
        summary = self.awi.exogenous_contract_summary
        if len(summary) < 2:
            return 0
        supply = summary[0][0]
        demand = summary[-1][0]
        return int(demand - supply)

    def get_num_same_level_agents(self) -> int:
                                                                          
        return self.awi.n_competitors + 1

    def get_gap_pressure(self) -> float:
                                                                         
        num_same_level_agents = self.get_num_same_level_agents()
        if num_same_level_agents <= 0:
            return 0.0
        return self.get_gap_d() / num_same_level_agents

    def calc_price(
        self,
        issues,
        is_selling: bool,
        best_price: bool = True,
        gap_price: bool | None = None,
    ) -> int:
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        use_gap_price = self.use_gap_price if gap_price is None else gap_price
        if use_gap_price and self.get_gap_pressure() < -1:
            price = pmin
            reason = "gap_pressure_low"
        elif best_price:
            price = pmax if is_selling else pmin
            reason = "best_price"
        else:
            price = issues[UNIT_PRICE].rand()
            reason = "random"
        self._debug_log_decision(
            "calc_price",
            is_selling=is_selling,
            best_price=best_price,
            gap_price=gap_price,
            effective_use_gap_price=use_gap_price,
            price_min=pmin,
            price_max=pmax,
            selected_price=price,
            reason=reason,
        )
        return price

    def _counter_offer_price(
        self,
        issues,
        is_selling: bool,
        partner_id: str | None = None,
        current_quantity: int | None = None,
        counter_quantity: int | None = None,
        default_price: int | None = None,
    ) -> int:
        if default_price is not None:
            mode_price = int(default_price)
            mode_name = "precomputed_default"
        else:
            mode_price, mode_name = self._counter_offer_price_by_mode(
                issues,
                is_selling,
            )
        if (
            partner_id is None
            or current_quantity is None
            or counter_quantity is None
            or self.counter_price_accept_rate_margin < 0
        ):
            self._debug_log_decision(
                "counter_offer_price",
                is_selling=is_selling,
                partner_id=partner_id,
                counter_offer_price_mode=self.counter_offer_price_mode,
                mode_name=mode_name,
                selected_price=mode_price,
                adaptive_enabled=False,
                counter_price_accept_rate_margin=(
                    self.counter_price_accept_rate_margin
                ),
            )
            return mode_price
        if self.awi.current_step < self.counter_price_warmup_steps:
            self._debug_log_decision(
                "counter_offer_price",
                is_selling=is_selling,
                partner_id=partner_id,
                counter_offer_price_mode=self.counter_offer_price_mode,
                mode_name=mode_name,
                selected_price=mode_price,
                adaptive_enabled=False,
                reason="warmup",
                current_step=self.awi.current_step,
                counter_price_warmup_steps=self.counter_price_warmup_steps,
            )
            return mode_price

        delta_quantity = int(counter_quantity) - int(current_quantity)
        decision = self._counter_price_adaptive_decision(
            partner_id,
            is_selling,
            delta_quantity,
        )
        selected_bucket = (
            "partner_favorable"
            if decision["use_partner_favorable"]
            else "self_favorable"
        )
        selected_price = self._counter_price_for_bucket(
            issues,
            is_selling,
            selected_bucket,
        )
        self._debug_log_decision(
            "counter_offer_price",
            is_selling=is_selling,
            partner_id=partner_id,
            current_quantity=int(current_quantity),
            counter_quantity=int(counter_quantity),
            delta_quantity=delta_quantity,
            counter_offer_price_mode=self.counter_offer_price_mode,
            mode_name="adaptive_accept_rate",
            default_mode_name=mode_name,
            default_price=mode_price,
            selected_price=selected_price,
            selected_bucket=selected_bucket,
            adaptive_enabled=True,
            decision=decision,
        )
        return selected_price

    def _counter_offer_price_by_mode(self, issues, is_selling: bool) -> tuple[int, str]:
        if self.counter_offer_price_mode == 1:
            price = self.calc_price(
                issues, is_selling, best_price=True, gap_price=False
            )
            mode_name = "best_price"
        elif self.counter_offer_price_mode == 2:
            price = self.calc_price(
                issues, is_selling, best_price=False, gap_price=True
            )
            mode_name = "gap_pressure_low_or_random"
        else:
            price = self.calc_price(
                issues, is_selling, best_price=False, gap_price=False
            )
            mode_name = "random"
        return int(price), mode_name

    def _counter_price_for_bucket(
        self,
        issues,
        is_selling: bool,
        price_bucket: str,
    ) -> int:
        pmin = int(issues[UNIT_PRICE].min_value)
        pmax = int(issues[UNIT_PRICE].max_value)
        if price_bucket == "partner_favorable":
            return pmin if is_selling else pmax
        return pmax if is_selling else pmin

    def _counter_price_adaptive_decision(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
    ) -> dict[str, object]:
        min_samples = max(0, int(self.counter_price_min_sample_per_side))
        margin = float(self.counter_price_accept_rate_margin)
        for scope in ("exact_delta", "delta_bucket", "delta_sign"):
            counts = self._counter_price_bucket_counts(
                partner_id,
                bool(is_selling),
                int(delta_quantity),
                scope,
            )
            partner_total = self._counter_price_count_total(
                counts["partner_favorable"]
            )
            self_total = self._counter_price_count_total(counts["self_favorable"])
            if partner_total < min_samples or self_total < min_samples:
                continue
            partner_rate = (
                counts["partner_favorable"]["accepts"] / partner_total
                if partner_total > 0
                else 0.0
            )
            self_rate = (
                counts["self_favorable"]["accepts"] / self_total
                if self_total > 0
                else 0.0
            )
            rate_diff = partner_rate - self_rate
            return {
                "scope": scope,
                "use_partner_favorable": rate_diff >= margin,
                "partner_favorable_accept_rate": partner_rate,
                "self_favorable_accept_rate": self_rate,
                "accept_rate_diff": rate_diff,
                "partner_favorable_samples": partner_total,
                "self_favorable_samples": self_total,
                "min_sample_per_side": min_samples,
                "margin": margin,
                "counts": counts,
            }
        return {
            "scope": "insufficient_samples",
            "use_partner_favorable": False,
            "min_sample_per_side": min_samples,
            "margin": margin,
        }

    def _counter_price_bucket_counts(
        self,
        partner_id: str,
        is_selling: bool,
        delta_quantity: int,
        scope: str,
    ) -> dict[str, dict[str, int]]:
        counts = {
            "partner_favorable": {"accepts": 0, "counters": 0, "rejects": 0},
            "self_favorable": {"accepts": 0, "counters": 0, "rejects": 0},
        }
        stats = getattr(self, "counter_price_delta_price_stats", {})
        target_bucket = self._counter_delta_bucket(delta_quantity)
        target_sign = self._counter_price_delta_sign(delta_quantity)
        for (
            observed_partner,
            observed_is_selling,
            observed_delta,
            _unit_price,
            partner_price_bucket,
        ), value in stats.items():
            if observed_partner != partner_id:
                continue
            if bool(observed_is_selling) != bool(is_selling):
                continue
            if partner_price_bucket not in counts:
                continue
            observed_delta = int(observed_delta)
            if scope == "exact_delta" and observed_delta != int(delta_quantity):
                continue
            if (
                scope == "delta_bucket"
                and self._counter_delta_bucket(observed_delta) != target_bucket
            ):
                continue
            if (
                scope == "delta_sign"
                and self._counter_price_delta_sign(observed_delta) != target_sign
            ):
                continue
            for field_name in ("accepts", "counters", "rejects"):
                counts[str(partner_price_bucket)][field_name] += int(
                    value.get(field_name, 0)
                )
        return counts

    def _counter_price_count_total(self, counts: dict[str, int]) -> int:
        return int(counts["accepts"]) + int(counts["counters"]) + int(counts["rejects"])

    def _counter_price_delta_sign(self, delta_quantity: int) -> str:
        if int(delta_quantity) < 0:
            return "decrease"
        if int(delta_quantity) > 0:
            return "increase"
        return "zero"

    def _allowed_mismatch(self, r: float, n_others: int, is_selling: bool):
        if is_selling:
            use_dynamic_under = self.undermismatch_sell != -1
            use_dynamic_over = self.overmismatch_sell != -1
            use_linear_over = self.overmismatch_sell_linear != -1
            sd_ratio = (
                self._get_sd_ratio() ** 0.5
                if use_dynamic_under
                or use_dynamic_over
                or use_linear_over
                else 1.0
            )
            undermismatch_min = (
                -self.undermismatch_sell * sd_ratio
                if use_dynamic_under
                else -0.5 * self.awi.n_lines
            )
            if self.gap_d_scaler != -1:
                undermismatch_min += self.get_gap_pressure() * self.gap_d_scaler
            overmismatch_max = (
                self.overmismatch_sell / sd_ratio
                if use_dynamic_over
                else 0.0
            )
        else:
            use_dynamic_under = self.undermismatch_buy != -1
            use_dynamic_over = self.overmismatch_buy != -1
            use_linear_over = self.overmismatch_buy_linear != -1
            sd_ratio = (
                self._get_sd_ratio() ** 0.5
                if use_dynamic_under
                or use_dynamic_over
                or use_linear_over
                else 1.0
            )
            undermismatch_min = (
                -self.undermismatch_buy / sd_ratio
                if use_dynamic_under
                else -0.2 * self.awi.n_lines
            )
            overmismatch_max = (
                self.overmismatch_buy * sd_ratio
                if use_dynamic_over
                else 0.2 * self.awi.n_lines
            )
        undermismatch_threshold = undermismatch_min * ((1 - r) ** self.mismatch_exp)
        if use_linear_over:
            overmismatch_threshold = self._linear_overmismatch_threshold(
                r, sd_ratio, is_selling
            )
            over_threshold_type = "linear"
        else:
            overmismatch_threshold = overmismatch_max * (
                r ** (1 / self.mismatch_exp)
            )
            over_threshold_type = "curve"
        self._debug_log_decision(
            "allowed_mismatch",
            is_selling=is_selling,
            relative_time=r,
            n_others=n_others,
            mismatch_exp=self.mismatch_exp,
            sd_ratio=sd_ratio,
            use_dynamic_under=use_dynamic_under,
            use_dynamic_over=use_dynamic_over,
            use_linear_over=use_linear_over,
            over_threshold_type=over_threshold_type,
            undermismatch_min=undermismatch_min,
            overmismatch_max=overmismatch_max,
            undermismatch_threshold=undermismatch_threshold,
            overmismatch_threshold=overmismatch_threshold,
            gap_d_scaler=self.gap_d_scaler,
        )
        return undermismatch_threshold, overmismatch_threshold

    def _linear_overmismatch_threshold(
        self, r: float, sd_ratio: float, is_selling: bool
    ) -> float:
        if is_selling:
            return self.overmismatch_sell_linear * self.awi.n_lines / sd_ratio * r
        return self.overmismatch_buy_linear * self.awi.n_lines * sd_ratio * r

    def _is_late_phase(self) -> bool:
        return self.awi.current_step > min(
            self.awi.n_steps * self.config.late_phase_fraction,
            self.config.late_phase_step_cap,
        )

    def _concentrated_ids(self, partner_ids: list[str]) -> list[str]:
        top_k = max(0, min(len(partner_ids), self.config.concentration_top_k))
        if top_k == 0:
            return []
        return sorted(
            partner_ids,
            key=lambda partner_id: self.total_agreed_quantity.get(partner_id, 0),
            reverse=True,
        )[:top_k]

    def _first_overorder_ratio(self, is_selling: bool) -> float:
        if is_selling:
            self._debug_log_decision(
                "first_overorder_ratio",
                is_selling=is_selling,
                ratio=0.0,
                reason="seller_side_disabled",
            )
            return 0.0
        ratio = self.first_overordering_scale
        sd_multiplier = 1.0
        if self.first_overordering_use_sd_ratio:
            sd_multiplier = self._get_sd_ratio2() ** 0.5
            ratio *= sd_multiplier
        gap_pressure = self.get_gap_pressure()
        raw_gap_pressure = gap_pressure
        if self.first_overordering_gap_positive_only:
            gap_pressure = max(gap_pressure, 0.0)
        ratio += self.first_overordering_gap_scale * gap_pressure
        final_ratio = max(0.0, ratio)
        self._debug_log_decision(
            "first_overorder_ratio",
            is_selling=is_selling,
            first_overordering_scale=self.first_overordering_scale,
            layout_key=getattr(self, "layout_key", None),
            layout_specific_field=(
                self.config.layout_parameter_fields.get(
                    getattr(self, "layout_key", None), {}
                ).get("first_overordering_scale")
            ),
            first_overordering_use_sd_ratio=self.first_overordering_use_sd_ratio,
            sd_multiplier=sd_multiplier,
            first_overordering_gap_scale=self.first_overordering_gap_scale,
            first_overordering_gap_positive_only=(
                self.first_overordering_gap_positive_only
            ),
            raw_gap_pressure=raw_gap_pressure,
            effective_gap_pressure=gap_pressure,
            unclipped_ratio=ratio,
            ratio=final_ratio,
        )
        return final_ratio

    def _counter_overorder_ratio(self, t: float, is_selling: bool) -> float:
        if is_selling:
            self._debug_log_decision(
                "counter_overorder_ratio",
                is_selling=is_selling,
                relative_time=t,
                ratio=0.0,
                reason="seller_side_disabled",
            )
            return 0.0
        sd_multiplier = self._get_sd_ratio() ** 0.5
        time_multiplier = 1 - t**self.counter_overordering_exp
        ratio = (
            self.counter_overordering_scale
            * time_multiplier
            * sd_multiplier
        )
        raw_gap_pressure = self.get_gap_pressure()
        effective_gap_pressure = max(raw_gap_pressure, 0.0)
        ratio += self.counter_overordering_gap_scale * effective_gap_pressure
        final_ratio = max(0.0, ratio)
        self._debug_log_decision(
            "counter_overorder_ratio",
            is_selling=is_selling,
            relative_time=t,
            counter_overordering_scale=self.counter_overordering_scale,
            counter_overordering_exp=self.counter_overordering_exp,
            time_multiplier=time_multiplier,
            sd_multiplier=sd_multiplier,
            counter_overordering_gap_scale=self.counter_overordering_gap_scale,
            raw_gap_pressure=raw_gap_pressure,
            effective_gap_pressure=effective_gap_pressure,
            unclipped_ratio=ratio,
            ratio=final_ratio,
        )
        return final_ratio

    def _step_and_price(self, best_price: bool = False):
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price or self.use_gap_price:
            return s, self.calc_price(issues, seller, best_price=best_price)
        price = random.randint(pmin, pmax)
        self._debug_log_decision(
            "step_and_price",
            is_selling=seller,
            best_price=best_price,
            use_gap_price=self.use_gap_price,
            price_min=pmin,
            price_max=pmax,
            selected_price=price,
            reason="random_int",
        )
        return s, price


