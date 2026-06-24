                     
              
\
\
\
\
\
\
   

from __future__ import annotations

import random
from collections import Counter
from itertools import chain, combinations, repeat
from math import ceil, floor, isfinite

                     
from negmas import *
from numpy.random import choice

                          
from scml.std import *

from .config import HorizonAwareAgentConfig, PenguinAgentConfig
from .debug_tools import DebugMixin

__all__ = [
    "HorizonAwareAgent",
    "HorizonAwareAgentConfig",
    "PenguinAgent",
    "PenguinAgentConfig",
]

def distribute(q: int, n: int) -> list[int]:
                                                                                        
                          
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


             
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class HorizonAwareAgent(DebugMixin, StdSyncAgent):
\
                                      

                     
    def __init__(
        self,
        *args,
        config: HorizonAwareAgentConfig | None = None,
        threshold=None,
        ptoday=(0.5, 0.5),
        productivity=0.7,
        lastlevel_inventory_buffer_days=1,
        last_level_negotiation_mode=3,
        adjust_last_level_proposal_price=False,
        middle_level_negotiation_mode=9,
        middlelevel_inventory_buffer_days=0,
        first_level_sell_mode=3,
        first_level_future_offer_divisor=1,
        first_level_staggered_future_sell=0,
        sell_partner_filter_enabled=True,
        use_bankruptcy_forecast=True,
        allowable_excess_sell_margin=1,
        debug_log_enabled=None,
        debug_log_file_path=None,
        **kwargs,
    ):
        if debug_log_enabled is None:
            debug_log_enabled = self.debug_log_enabled
        self.debug_log_enabled = bool(debug_log_enabled)
        if debug_log_file_path is None:
            debug_log_file_path = self.debug_log_file_path
        self.debug_log_file_path = str(debug_log_file_path or "")
        kwargs.pop("middle_level_inventory_cap", None)
        kwargs.pop("last_level_inventory_cap", None)
        legacy_first_level_sell_mode = kwargs.pop("new_first_level_sell_mode", None)
        if legacy_first_level_sell_mode is not None and "first_level_sell_mode" not in kwargs:
            first_level_sell_mode = 1 if legacy_first_level_sell_mode else 0

        direct_config = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in HorizonAwareAgentConfig.field_names()
        }
        if direct_config:
            explicit_values = {
                "threshold": threshold,
                "ptoday": ptoday,
                "productivity": productivity,
                "lastlevel_inventory_buffer_days": lastlevel_inventory_buffer_days,
                "last_level_negotiation_mode": last_level_negotiation_mode,
                "adjust_last_level_proposal_price": adjust_last_level_proposal_price,
                "middle_level_negotiation_mode": middle_level_negotiation_mode,
                "middlelevel_inventory_buffer_days": middlelevel_inventory_buffer_days,
                "first_level_sell_mode": first_level_sell_mode,
                "first_level_future_offer_divisor": (
                    first_level_future_offer_divisor
                ),
                "first_level_staggered_future_sell": (
                    first_level_staggered_future_sell
                ),
                "sell_partner_filter_enabled": sell_partner_filter_enabled,
                "use_bankruptcy_forecast": use_bankruptcy_forecast,
                "allowable_excess_sell_margin": allowable_excess_sell_margin,
            }
            explicit_values.update(direct_config)
            if config is not None:
                raise ValueError("Pass either config or direct HorizonAwareAgentConfig fields, not both")
            config = HorizonAwareAgentConfig.from_mapping(explicit_values)
        elif config is None:
            config = HorizonAwareAgentConfig(
                threshold=threshold,
                ptoday=ptoday,
                productivity=productivity,
                lastlevel_inventory_buffer_days=lastlevel_inventory_buffer_days,
                last_level_negotiation_mode=last_level_negotiation_mode,
                adjust_last_level_proposal_price=adjust_last_level_proposal_price,
                middle_level_negotiation_mode=middle_level_negotiation_mode,
                middlelevel_inventory_buffer_days=middlelevel_inventory_buffer_days,
                first_level_sell_mode=first_level_sell_mode,
                first_level_future_offer_divisor=first_level_future_offer_divisor,
                first_level_staggered_future_sell=first_level_staggered_future_sell,
                sell_partner_filter_enabled=sell_partner_filter_enabled,
                use_bankruptcy_forecast=use_bankruptcy_forecast,
                allowable_excess_sell_margin=allowable_excess_sell_margin,
            )

        super().__init__(*args, **kwargs)
        threshold = config.threshold
        if threshold is None:
            threshold = 1                             
        self._threshold = threshold
        self._ptoday_schedule = self._normalize_ptoday_schedule(config.ptoday)
        self._productivity = config.productivity
        self._lastlevel_inventory_buffer_days = config.lastlevel_inventory_buffer_days
        self._last_level_negotiation_mode = config.last_level_negotiation_mode
        self._adjust_last_level_proposal_price = config.adjust_last_level_proposal_price
        self._middle_level_negotiation_mode = int(config.middle_level_negotiation_mode)
        self._middlelevel_inventory_buffer_days = (
            config.middlelevel_inventory_buffer_days
        )
        self._middle_level_mode4_buy_cap = self._middle_level_negotiation_mode == 4
        self._first_level_sell_mode = int(config.first_level_sell_mode)
        self._first_level_future_offer_divisor = max(
            1, int(config.first_level_future_offer_divisor)
        )
        self._first_level_staggered_future_sell = int(
            config.first_level_staggered_future_sell
        )
        self._sell_partner_filter_enabled = config.sell_partner_filter_enabled
        self._use_bankruptcy_forecast = bool(
            getattr(config, "use_bankruptcy_forecast", False)
        )
        self._allowable_excess_sell_margin = int(
            getattr(config, "allowable_excess_sell_margin", 10)
        )
        self.config = config
        self._last_strategy_metrics: dict[str, object] = {}
        self._strategy_metrics_history: list[dict[str, object]] = []
        self._first_level_exogenous_input_quantities: list[int] = []
        self._last_level_exogenous_output_quantities: list[int] = []
        self._middle_level_expected_output_quantities: list[float] = []
        self._contract_value_totals: dict[str, dict[str, float]] = {
            "buy": {"quantity": 0.0, "value": 0.0},
            "sell": {"quantity": 0.0, "value": 0.0},
        }
        self._sell_contract_steps_by_partner: dict[str, list[int]] = {}
        self._sell_contract_quantities_by_step_partner: dict[int, dict[str, int]] = {}
        self._logged_bankrupt_partners: set[str] = set()
        self._debug_world_context_logged = False
        self._middle_level_mode5_phase = "initial_sell_probe"
        self._middle_level_mode8_sell_offer_history: dict[str, list[dict[str, object]]] = {}
        self._middle_level_mode8_learning_summary_logged = False
        self._first_level_mode2_phase_start_step: int | None = None
        self._first_level_mode2_sell_offer_history: dict[str, list[dict[str, object]]] = {}
        self._first_level_mode2_learning_summary_logged = False
        self._cash_report_history: dict[str, dict[int, float]] = {}

    def _normalize_ptoday_schedule(self, value) -> tuple[float, float]:
        if isinstance(value, (int, float)):
            v = max(0.0, min(1.0, float(value)))
            return v, v
        try:
            start, end = value
        except Exception:
            return 0.7, 0.7
        return (
            max(0.0, min(1.0, float(start))),
            max(0.0, min(1.0, float(end))),
        )

    def _ptoday(self) -> float:
        start, end = self._ptoday_schedule
        n_steps = max(1, int(self._awi_value("n_steps", 1) or 1))
        current_step = max(0, int(self._awi_value("current_step", 0) or 0))
        if n_steps <= 1:
            return end
        progress = max(0.0, min(1.0, current_step / (n_steps - 1)))
        return start + (end - start) * progress

    def _awi_value(self, name: str, default=None):
        try:
            return getattr(self.awi, name)
        except Exception:
            return default

    def _call_awi(self, name: str, *args, default=None):
        value = self._awi_value(name, None)
        if value is None:
            return default
        try:
            return value(*args) if callable(value) else value
        except Exception:
            return default


    def _debug_value(self, value):
        if isinstance(value, dict):
            return {str(k): self._debug_value(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(str(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [self._debug_value(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if value.__class__.__module__.startswith("numpy"):
            if hasattr(value, "tolist"):
                return self._debug_value(value.tolist())
            if hasattr(value, "item"):
                try:
                    return value.item()
                except ValueError:
                    return str(value)
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









    def _debug_contract_summary(self, contract: Contract) -> dict[str, object]:
        annotation = dict(getattr(contract, "annotation", {}) or {})
        agreement = getattr(contract, "agreement", {}) or {}
        partner_id = next(
            (partner for partner in getattr(contract, "partners", []) if partner != self._my_agent_id()),
            None,
        )
        return {
            "id": getattr(contract, "id", None),
            "partner_id": partner_id,
            "partners": list(getattr(contract, "partners", []) or []),
            "annotation": annotation,
            "agreement": self._debug_value(agreement),
            "quantity": self._agreement_value(agreement, "quantity", QUANTITY, None),
            "time": self._agreement_value(agreement, "time", TIME, None),
            "unit_price": self._agreement_value(agreement, "unit_price", UNIT_PRICE, None),
            "is_selling": annotation.get("seller") == self._my_agent_id(),
        }



    def _agreement_value(self, agreement, name, index, default=None):
        try:
            if hasattr(agreement, "get"):
                value = agreement.get(name, None)
                if value is not None:
                    return value
        except Exception:
            pass
        try:
            return agreement[index]
        except Exception:
            return default

    def _my_agent_id(self):
        return (
            getattr(self, "id", None)
            or self._awi_value("agent_id", None)
            or self._awi_value("id", None)
        )

    def _is_partner_bankrupt(self, partner) -> bool:
        for name in ("is_bankrupt", "is_bankrupt_agent"):
            value = self._awi_value(name, None)
            if callable(value):
                try:
                    if value(partner):
                        return True
                except Exception:
                    pass

        for name in ("bankrupt_agents", "bankruptcies"):
            value = self._awi_value(name, None)
            if value is None:
                continue
            try:
                if partner in value:
                    return True
            except Exception:
                pass
        return False


    def _log_bankrupt_partners_once(self) -> None:
        partners = []
        try:
            partners.extend(self.awi.my_suppliers)
        except Exception:
            pass
        try:
            partners.extend(self.awi.my_consumers)
        except Exception:
            pass
        for partner in partners:
            if self._is_partner_bankrupt(partner):
                self._log_bankrupt_partner_once(partner)

    def _known_financial_report_agent_ids(self) -> list[str]:
        agent_ids: set[str] = set()
        for name in ("all_suppliers", "all_consumers"):
            value = self._awi_value(name, None)
            if value is None:
                continue
            try:
                for group in value:
                    if isinstance(group, str):
                        agent_ids.add(group)
                    else:
                        agent_ids.update(str(agent) for agent in group)
            except Exception:
                pass
        for name in ("my_suppliers", "my_consumers"):
            value = self._awi_value(name, None)
            if value is None:
                continue
            try:
                agent_ids.update(str(agent) for agent in value)
            except Exception:
                pass
        my_id = self._my_agent_id()
        if my_id is not None:
            agent_ids.add(str(my_id))
        return sorted(agent_ids)

    def _record_cash_report(self, agent_id, report) -> bool:
        try:
            cash = float(getattr(report, "cash"))
        except Exception:
            return False
        if not isfinite(cash):
            return False
        try:
            report_step = int(getattr(report, "step"))
        except Exception:
            report_step = int(self._awi_value("current_step", 0) or 0)
        self._cash_report_history.setdefault(str(agent_id), {})[report_step] = cash
        return True

    def _record_cash_reports(self) -> int:
        current_step = int(self._awi_value("current_step", 0) or 0)
        recorded = 0
        reports = self._call_awi("reports_at_step", current_step, default=None)
        if reports:
            try:
                for agent_id, report in reports.items():
                    if self._record_cash_report(agent_id, report):
                        recorded += 1
            except Exception:
                pass

        if recorded:
            return recorded

        for agent_id in self._known_financial_report_agent_ids():
            reports = self._call_awi("reports_of_agent", agent_id, default=None)
            if not reports:
                continue
            try:
                for _, report in reports.items():
                    if self._record_cash_report(agent_id, report):
                        recorded += 1
            except Exception:
                pass
        return recorded

    def _calc_cash_bankruptcy_forecast(self) -> dict[str, dict[str, object]]:
        forecasts: dict[str, dict[str, object]] = {}
        for agent_id, history in sorted(self._cash_report_history.items()):
            points = sorted(history.items())[-3:]
            if not points:
                continue
            payload: dict[str, object] = {
                "points": [[int(step), cash] for step, cash in points],
                "n_points": len(points),
                "latest_step": int(points[-1][0]),
                "latest_cash": points[-1][1],
                "slope": None,
                "predicted_bankruptcy_step": None,
            }
            if len(points) < 3:
                forecasts[agent_id] = payload
                continue
            xs = [float(step) for step, _ in points]
            ys = [float(cash) for _, cash in points]
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            denominator = sum((x - x_mean) ** 2 for x in xs)
            if denominator <= 0:
                forecasts[agent_id] = payload
                continue
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False)) / denominator
            payload["slope"] = slope
            if slope < 0:
                intercept = y_mean - slope * x_mean
                predicted_step = -intercept / slope
                if isfinite(predicted_step):
                    payload["predicted_bankruptcy_step"] = floor(predicted_step)
                    payload["predicted_bankruptcy_step_raw"] = predicted_step
            forecasts[agent_id] = payload
        return forecasts

    def _predicted_bankruptcy_steps(self) -> dict[str, int]:
        forecasts = self._calc_cash_bankruptcy_forecast()
        return {
            agent_id: int(forecast["predicted_bankruptcy_step"])
            for agent_id, forecast in forecasts.items()
            if forecast.get("predicted_bankruptcy_step") is not None
        }

    def _future_sell_step_allowed_by_bankruptcy_forecast(
        self, partner, step: int
    ) -> bool:
        if not self._use_bankruptcy_forecast:
            return True
        predicted_step = self._predicted_bankruptcy_steps().get(str(partner))
        if predicted_step is None:
            return True
        return int(step) <= int(predicted_step)

    def _filter_future_sell_steps_by_bankruptcy_forecast(
        self, partner, steps: list[int]
    ) -> list[int]:
        if not self._use_bankruptcy_forecast:
            return steps
        return [
            int(step)
            for step in steps
            if self._future_sell_step_allowed_by_bankruptcy_forecast(partner, int(step))
        ]


    def _record_sell_contract_partner(self, contract: Contract) -> None:
        annotation = getattr(contract, "annotation", {}) or {}
        agreement = getattr(contract, "agreement", {}) or {}
        seller = annotation.get("seller")
        buyer = annotation.get("buyer")
        if buyer is None or buyer not in self.awi.my_consumers:
            return
        agent_id = self._my_agent_id()
        if agent_id is not None and seller != agent_id:
            return
        step = int(self._awi_value("current_step", 0) or 0)
        self._sell_contract_steps_by_partner.setdefault(buyer, []).append(step)
        try:
            delivery_step = int(self._agreement_value(agreement, "time", TIME, step))
            quantity = int(self._agreement_value(agreement, "quantity", QUANTITY, 0))
        except Exception:
            return
        if quantity <= 0:
            return
        by_partner = self._sell_contract_quantities_by_step_partner.setdefault(
            delivery_step, {}
        )
        by_partner[str(buyer)] = by_partner.get(str(buyer), 0) + quantity

    def _record_contract_price_history(self, contract: Contract) -> None:
        agreement = getattr(contract, "agreement", {}) or {}
        annotation = getattr(contract, "annotation", {}) or {}
        agent_id = self._my_agent_id()
        buyer = annotation.get("buyer")
        seller = annotation.get("seller")

        if buyer == agent_id:
            side = "buy"
        elif seller == agent_id:
            side = "sell"
        else:
            return

        quantity = self._agreement_value(agreement, "quantity", QUANTITY, 0)
        unit_price = self._agreement_value(agreement, "unit_price", UNIT_PRICE, 0)
        try:
            quantity = float(quantity)
            unit_price = float(unit_price)
        except Exception:
            return
        if quantity <= 0 or unit_price <= 0:
            return

        totals = self._contract_value_totals[side]
        totals["quantity"] += quantity
        totals["value"] += quantity * unit_price

    def _has_recent_sell_contract_with(self, partner, window: int = 10) -> bool:
        current_step = int(self._awi_value("current_step", 0) or 0)
        min_step = max(0, current_step - window)
        steps = self._sell_contract_steps_by_partner.get(partner, [])
        return any(min_step <= step < current_step for step in steps)

    def _is_active_sell_target(self, partner) -> bool:
        if not self._sell_partner_filter_enabled or not self.is_consumer(partner):
            return True
        if self._is_partner_bankrupt(partner):
            self._log_bankrupt_partner_once(partner)
            return False
        if int(self._awi_value("current_step", 0) or 0) < 30:
            return True
        return self._has_recent_sell_contract_with(partner, window=20)

    def _filter_sell_targets(self, partners):
        return [partner for partner in partners if self._is_active_sell_target(partner)]

    def _level_label(self) -> str:
        if self._awi_value("is_first_level", False):
            return "first"
        if self._awi_value("is_last_level", False):
            return "last"
        return "middle"

    def _middle_level_mode5_family(self) -> bool:
        return self._middle_level_negotiation_mode in (5, 6, 7, 8, 9)

    def _middle_level_mode6_family(self) -> bool:
        return self._middle_level_negotiation_mode in (6, 7, 8, 9)

    def _middle_level_mode8_family(self) -> bool:
        return self._middle_level_negotiation_mode in (8, 9)

    def _middle_level_mode9_enabled(self) -> bool:
        return self._middle_level_negotiation_mode == 9

    def _active_sold_quantity_at(self, step: int) -> int:
        by_partner = self._sell_contract_quantities_by_step_partner.get(int(step), {})
        awi_total = int(self._call_awi("total_sales_at", step, default=0) or 0)
        if not by_partner:
            return awi_total
        active_total = 0
        tracked_total = 0
        for partner, quantity in by_partner.items():
            tracked_total += int(quantity)
            if not self._is_partner_bankrupt(partner):
                active_total += int(quantity)
        untracked_total = max(0, awi_total - tracked_total)
        return max(0, active_total + untracked_total)

    def _middle_level_mode9_active_sold_quantity_at(self, step: int) -> int:
                                                                                  
        return self._active_sold_quantity_at(step)

    def _middle_level_mode8_recorded_sales_at(self, step: int) -> int:
        if self._middle_level_mode9_enabled():
            return self._middle_level_mode9_active_sold_quantity_at(step)
        return int(self._call_awi("total_sales_at", step, default=0) or 0)

    def _sell_strategy_recorded_sales_at(self, step: int) -> int:
        if self._middle_level_mode9_enabled() or self._first_level_mode3_enabled():
            return self._active_sold_quantity_at(step)
        return int(self._call_awi("total_sales_at", step, default=0) or 0)

    def _same_level_agent_count(self) -> int | None:
        value = self._awi_value("n_competitors", None)
        if isinstance(value, int) and value > 0:
            return value

        agents_per_process = self._awi_value("n_agents_per_process", None)
        try:
            process = int(self._awi_value("my_input_product", 0) or 0)
            value = agents_per_process[process]
            if isinstance(value, int) and value > 0:
                return value
        except Exception:
            pass

        value = self._awi_value("n_processes", None)
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _same_level_factory_count(self) -> int:
        value = self._awi_value("n_competitors", None)
        if isinstance(value, int) and value >= 0:
            return value + 1

        agents_per_process = self._awi_value("n_agents_per_process", None)
        try:
            process = int(self._awi_value("my_input_product", 0) or 0)
            value = agents_per_process[process]
            if isinstance(value, int) and value > 0:
                return value
        except Exception:
            pass

        return 1

    def _current_exogenous_input_quantity(self) -> int:
        return int(self._call_awi("current_exogenous_input_quantity", default=0) or 0)

    def _current_exogenous_output_quantity(self) -> int:
        return int(self._call_awi("current_exogenous_output_quantity", default=0) or 0)

    def _record_first_level_exogenous_input_quantity(self) -> None:
        if self._level_label() != "first":
            return
        self._first_level_exogenous_input_quantities.append(
            self._current_exogenous_input_quantity()
        )

    def _record_last_level_exogenous_output_quantity(self) -> None:
        if self._level_label() != "last":
            return
        self._last_level_exogenous_output_quantities.append(
            self._current_exogenous_output_quantity()
        )

    def _record_middle_level_expected_output_quantity(self) -> None:
        if self._level_label() != "middle":
            return
        if self._middle_level_negotiation_mode == 0:
            return
        self._middle_level_expected_output_quantities.append(
            self._calc_current_middle_level_expected_output_quantity()
        )

    def _calc_first_level_average_exogenous_input(self) -> float:
        quantities = self._first_level_exogenous_input_quantities
        if not quantities:
            return float(self._current_exogenous_input_quantity())
        return sum(quantities) / len(quantities)

    def _calc_last_level_average_exogenous_output(self) -> float:
                                                                                      
        quantities = self._last_level_exogenous_output_quantities
        if not quantities:
            return float(self._current_exogenous_output_quantity())
        return sum(quantities) / len(quantities)

    def _calc_last_level_peak_exogenous_output(self) -> float:
        quantities = self._last_level_exogenous_output_quantities
        if not quantities:
            return float(self._current_exogenous_output_quantity())
        return max(quantities)

    def _calc_last_level_inventory_buffer_quantity(self) -> float:
                                                                          
        if self._level_label() != "last":
            return 0.0
        return (
            self._calc_last_level_average_exogenous_output()
            * self._lastlevel_inventory_buffer_days
        )

    def _calc_last_level_required_input_quantity(self, step: int | None = None) -> int:
        if self._last_level_negotiation_mode == 2:
            return self._calc_last_level_peak_required_input_quantity(step)
        if self._last_level_negotiation_mode == 3:
            return self._calc_last_level_capped_peak_required_input_quantity(step)
        if self._last_level_negotiation_mode == 4:
            return self._calc_last_level_hard_capped_peak_required_input_quantity(step)
        if self._last_level_negotiation_mode == 5:
            return self._calc_last_level_inventory_capped_peak_input_quantity(step)
        return self._calc_last_level_buffer_required_input_quantity(step)

    def _calc_last_level_buffer_required_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                                           
        if self._level_label() != "last":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step

        if step == awi.current_step:
            exogenous_output = self._current_exogenous_output_quantity()
            output_inventory = int(self._awi_value("current_inventory_output", 0) or 0)
        else:
            exogenous_output = self._calc_last_level_average_exogenous_output()
            output_inventory = 0

        output_shortfall = max(0.0, exogenous_output - output_inventory)
        target_input_inventory = (
            output_shortfall + self._calc_last_level_inventory_buffer_quantity()
        )
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        return max(
            0,
            ceil(target_input_inventory - current_input_inventory - contracted_supplies),
        )

    def _calc_last_level_peak_required_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                                     
        return self._calc_last_level_peak_required_input_quantity_impl(
            step, cap_to_n_lines=False
        )

    def _calc_last_level_capped_peak_required_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                                  
        return self._calc_last_level_peak_required_input_quantity_impl(
            step, cap_to_n_lines=True
        )

    def _calc_last_level_hard_capped_peak_required_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                               
        return self._calc_last_level_peak_required_input_quantity_impl(
            step, cap_to_n_lines=True, subtract_contracted_supplies=True
        )

    def _calc_last_level_inventory_capped_peak_input_quantity(
        self, step: int | None = None
    ) -> int:
        if self._level_label() != "last":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step
        if step != awi.current_step:
            return 0
        target_inventory = min(
            float(awi.n_lines),
            float(self._calc_last_level_peak_exogenous_output()),
        )
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        return max(
            0,
            ceil(target_inventory - current_input_inventory - contracted_supplies),
        )

    def _calc_last_level_peak_required_input_quantity_impl(
        self,
        step: int | None = None,
        *,
        cap_to_n_lines: bool,
        subtract_contracted_supplies: bool = False,
    ) -> int:
        if self._level_label() != "last":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step
        if step != awi.current_step:
            return 0

        exogenous_output = float(self._current_exogenous_output_quantity())
        if cap_to_n_lines:
            exogenous_output = min(float(awi.n_lines), exogenous_output)
        output_inventory = int(self._awi_value("current_inventory_output", 0) or 0)
        output_shortfall = max(0.0, exogenous_output - output_inventory)
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        peak_exogenous_output = float(self._calc_last_level_peak_exogenous_output())
        if cap_to_n_lines:
            peak_exogenous_output = min(float(awi.n_lines), peak_exogenous_output)
        if current_input_inventory >= peak_exogenous_output:
            return 0
        contracted_supplies = 0
        if subtract_contracted_supplies:
            contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        return max(
            0,
            ceil(
                peak_exogenous_output
                + output_shortfall
                - current_input_inventory
                - contracted_supplies
            ),
        )

    def _calc_current_middle_level_expected_output_quantity(self) -> float:
        summary = self._awi_value("exogenous_contract_summary", None) or []
        try:
            final_exogenous_output_quantity = float(summary[-1][0])
        except Exception:
            final_exogenous_output_quantity = 0.0
        return final_exogenous_output_quantity / self._same_level_factory_count()

    def _calc_middle_level_average_expected_output(self) -> float:
        quantities = self._middle_level_expected_output_quantities
        if not quantities:
            return self._calc_current_middle_level_expected_output_quantity()
        return sum(quantities) / len(quantities)

    def _calc_middle_level_expected_output_quantity(
        self, step: int | None = None
    ) -> float:
        if step is None or step == self.awi.current_step:
            return self._calc_current_middle_level_expected_output_quantity()
        return self._calc_middle_level_average_expected_output()

    def _calc_middle_level_inventory_buffer_quantity(self) -> float:
                                                                                         
        if self._level_label() != "middle":
            return 0.0
        return (
            self._calc_middle_level_average_expected_output()
            * self._middlelevel_inventory_buffer_days
        )

    def _calc_middle_level_required_input_quantity(self, step: int | None = None) -> int:
                                                                            
        if self._level_label() != "middle":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step

        committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
        expected_output = self._calc_middle_level_expected_output_quantity(step)
        target_output = max(float(committed_sales), expected_output)
        output_inventory = (
            int(self._awi_value("current_inventory_output", 0) or 0)
            if step == awi.current_step
            else 0
        )
        producible_output_need = max(0.0, target_output - output_inventory)
        planned_input_use = min(float(awi.n_lines), producible_output_need)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        return max(
            0,
            ceil(planned_input_use - current_input_inventory - contracted_supplies),
        )

    def _calc_middle_level_buffered_required_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                                      
        if self._level_label() != "middle":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step

        committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
        expected_output = self._calc_middle_level_expected_output_quantity(step)
        current_output_inventory = (
            int(self._awi_value("current_inventory_output", 0) or 0)
            if step == awi.current_step
            else 0
        )
        daily_output_target = min(
            float(awi.n_lines), max(float(committed_sales), expected_output)
        )
        target_output_coverage = (
            daily_output_target + self._calc_middle_level_inventory_buffer_quantity()
        )
        target_input_inventory = max(
            0.0, target_output_coverage - current_output_inventory
        )
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        return max(
            0,
            ceil(target_input_inventory - current_input_inventory - contracted_supplies),
        )

    def _calc_middle_level_full_capacity_input_quantity(
        self, step: int | None = None
    ) -> int:
                                                                                 
        if self._level_label() != "middle":
            return 0
        if self._middle_level_mode5_family():
            phase = self._middle_level_mode5_current_phase()
            if phase == "initial_sell_probe":
                return 0
            if phase == "input_fill":
                return self._calc_middle_level_mode5_fill_input_quantity(step)
        awi = self.awi
        if step is None:
            step = awi.current_step
        if step != awi.current_step:
            return 0

        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        if current_input_inventory >= awi.n_lines:
            return 0

        today_sales = self._call_awi("total_sales_at", awi.current_step, default=0) or 0
        return max(
            0,
            ceil(float(awi.n_lines) + float(today_sales) - current_input_inventory),
        )

    def _calc_middle_level_mode5_fill_input_quantity(
        self, step: int | None = None
    ) -> int:
        if self._level_label() != "middle":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step
        if step != awi.current_step:
            return 0
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        return max(0, ceil(float(awi.n_lines) - current_input_inventory - contracted_supplies))

    def _input_inventory_cap_enabled(self) -> bool:
        level = self._level_label()
        if level == "last":
            return self._last_level_negotiation_mode in (4, 5)
        if level == "middle":
            return self._middle_level_mode4_buy_cap
        return False

    def _cap_input_buy_need(self, step: int, need: int) -> int:
        if not self._input_inventory_cap_enabled():
            return need
        if step != self.awi.current_step:
            return 0

        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        remaining_capacity = int(
            self.awi.n_lines - current_input_inventory - contracted_supplies
        )
        return max(0, min(need, remaining_capacity))

    def _supplier_need_at(self, step: int | None = None) -> int:
        awi = self.awi
        if step is None:
            step = awi.current_step
        level = self._level_label()
        if level == "last" and self._last_level_negotiation_mode != 0:
            need = self._calc_last_level_required_input_quantity(step)
            return self._cap_input_buy_need(step, need)
        if level == "middle" and self._middle_level_negotiation_mode == 1:
            need = self._calc_middle_level_required_input_quantity(step)
            return self._cap_input_buy_need(step, need)
        if level == "middle" and self._middle_level_negotiation_mode == 2:
            need = self._calc_middle_level_buffered_required_input_quantity(step)
            return self._cap_input_buy_need(step, need)
        if level == "middle" and self._middle_level_negotiation_mode in (3, 4, 5, 6, 7, 8, 9):
            need = self._calc_middle_level_full_capacity_input_quantity(step)
            return self._cap_input_buy_need(step, need)
        return self._legacy_supplier_need_at(step)

    def _legacy_supplier_need_at(self, step: int | None = None) -> int:
        awi = self.awi
        if step is None:
            step = awi.current_step
        day_production = awi.n_lines * self._productivity
        return int(
            day_production
            - awi.current_inventory_input
            - awi.total_supplies_at(step)
        )

    def _legacy_consumer_need_at(self, step: int | None = None) -> int:
        awi = self.awi
        if step is None:
            step = awi.current_step
        day_production = awi.n_lines * self._productivity
        return int(
            max(
                0,
                min(awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(step),
            )
        )

    def _first_level_future_offer_divisor_value(self) -> int:
        if self._level_label() == "first":
            return self._first_level_future_offer_divisor
        return 3

    def _calc_middle_level_consumer_need(self, step: int | None = None) -> int:
                                                                                                
        if self._level_label() != "middle":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step

        expected_output = self._calc_middle_level_expected_output_quantity(step)
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        secured_input = max(0.0, float(current_input_inventory + contracted_supplies))
        sell_target = min(float(awi.n_lines), expected_output, secured_input)
        committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
        return max(0, int(sell_target - committed_sales))

    def _calc_middle_level_buffered_consumer_need(self, step: int | None = None) -> int:
                                                                                  
        if self._level_label() != "middle":
            return 0
        awi = self.awi
        if step is None:
            step = awi.current_step

        expected_output = self._calc_middle_level_expected_output_quantity(step)
        current_output_inventory = (
            int(self._awi_value("current_inventory_output", 0) or 0)
            if step == awi.current_step
            else 0
        )
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        contracted_supplies = self._call_awi("total_supplies_at", step, default=0) or 0
        secured_input = max(0.0, float(current_input_inventory + contracted_supplies))
        producible_output = min(float(awi.n_lines), secured_input)
        available_output = max(0.0, float(current_output_inventory) + producible_output)
        buffer_quantity = self._calc_middle_level_inventory_buffer_quantity()
        safe_available_output = max(0.0, available_output - buffer_quantity)
        sell_target = min(float(awi.n_lines), expected_output, safe_available_output)
        committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
        return max(0, int(sell_target - committed_sales))

    def _calc_middle_level_initial_fill_consumer_need(
        self, step: int | None = None
    ) -> int:
                                                                                        
        if self._level_label() != "middle":
            return 0
        if self._middle_level_mode5_family():
            phase = self._middle_level_mode5_current_phase()
            if phase == "initial_sell_probe":
                if self._middle_level_mode8_family():
                    return self._middle_level_mode8_fill_sell_quantity(step)
                return self._middle_level_mode5_initial_sell_quantity()
            if phase == "input_fill":
                if not self._middle_level_mode5_future_sell_enabled():
                    return 0
                if self._middle_level_mode8_family():
                    return self._middle_level_mode8_fill_sell_quantity(step)
                return self._middle_level_mode7_cap_sell_need(
                    self._middle_level_mode5_future_sell_quantity(step), step
                )
            awi = self.awi
            if step is None:
                step = awi.current_step
            committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
            current_input_inventory = int(
                self._awi_value("current_inventory_input", 0) or 0
            )
            sell_target = min(int(awi.n_lines), current_input_inventory)
            need = max(0, int(sell_target - committed_sales))
            return self._middle_level_mode7_cap_sell_need(need, step)
        awi = self.awi
        if step is None:
            step = awi.current_step
        if step <= 1:
            return 0

        committed_sales = self._call_awi("total_sales_at", step, default=0) or 0
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        sell_target = min(int(awi.n_lines), int(current_input_inventory * 1.2))
        return max(0, int(sell_target - committed_sales))

    def _middle_level_mode5_current_phase(self) -> str:
        if (
            self._level_label() != "middle"
            or not self._middle_level_mode5_family()
        ):
            return "normal"
        if (
            self._middle_level_mode6_family()
            and self._middle_level_mode5_phase == "initial_sell_probe"
        ):
            self._middle_level_mode5_phase = self._middle_level_mode6_initial_phase()
        if self._middle_level_mode5_phase == "input_fill":
            current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
            if current_input_inventory >= int(self._awi_value("n_lines", 0) or 0):
                self._middle_level_mode5_phase = "normal"
        return self._middle_level_mode5_phase

    def _middle_level_mode5_advance_after_sell_response(self) -> None:
        if (
            self._level_label() == "middle"
            and self._middle_level_mode5_family()
        ):
            if self._middle_level_mode5_phase == "initial_sell_probe":
                self._middle_level_mode5_phase = "input_fill"

    def _middle_level_mode6_initial_phase(self) -> str:
        n_processes = int(self._awi_value("n_processes", 0) or 0)
        process_index = int(self._awi_value("my_input_product", 0) or 0)
        if n_processes == 4 and process_index == 1:
            return "initial_sell_probe"
        return "input_fill"

    def _middle_level_mode5_future_sell_enabled(self) -> bool:
        if self._middle_level_negotiation_mode == 5:
            return True
        if not self._middle_level_mode6_family():
            return True
        n_processes = int(self._awi_value("n_processes", 0) or 0)
        process_index = int(self._awi_value("my_input_product", 0) or 0)
        return not (n_processes == 4 and process_index == 2)

    def _middle_level_mode5_initial_sell_quantity(self) -> int:
        return max(1, int(self._awi_value("n_lines", 1) or 1) // 2)

    def _middle_level_mode8_fill_sell_quantity(self, step: int | None = None) -> int:
        del step
        return max(1, int(self._awi_value("n_lines", 1) or 1))

    def _middle_level_mode8_buy_price(self, partner) -> int:
        _pmin, pmax = self._unit_price_bounds(partner)
        return int(pmax)

    def _middle_level_mode5_future_sell_quantity(self, step: int | None = None) -> int:
        if step is None:
            step = self.awi.current_step
        current_step = int(self._awi_value("current_step", 0) or 0)
        if step <= current_step:
            return 0
        if self._middle_level_mode6_family():
            n_processes = int(self._awi_value("n_processes", 0) or 0)
            if n_processes == 3:
                return max(
                    self._middle_level_mode5_initial_sell_quantity(),
                    self._middle_level_mode5_inventory_based_future_sell_quantity(step),
                )
        return self._middle_level_mode5_inventory_based_future_sell_quantity(step)

    def _middle_level_mode7_cap_sell_need(
        self, need: int, step: int | None = None
    ) -> int:
        if self._middle_level_negotiation_mode != 7:
            return need
        return max(0, min(int(need), self._middle_level_mode7_remaining_sell_capacity(step)))

    def _middle_level_mode7_remaining_sell_capacity(
        self, step: int | None = None
    ) -> int:
        if step is None:
            step = self.awi.current_step
        already_sold = self._call_awi("total_sales_at", step, default=0) or 0
        return max(0, int(self.awi.n_lines - already_sold))

    def _middle_level_mode5_inventory_based_future_sell_quantity(
        self, step: int
    ) -> int:
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        already_sold = self._middle_level_mode8_recorded_sales_at(step)
        return max(0, int(current_input_inventory - already_sold))

    def _middle_level_mode5_ranked_far_consumer_steps(self) -> list[int]:
        awi = self.awi
        min_time, max_time = self._time_issue_bounds_for_consumer_future_sell()
        min_step = max(awi.current_step + 1, min_time)
        max_step = min(awi.n_steps - 1, max_time)
        if min_step > max_step:
            return []
        steps = list(range(min_step, max_step + 1))
        steps.sort(
            key=lambda step: (
                self._middle_level_mode8_recorded_sales_at(step),
                -step,
            )
        )
        return steps

    def _middle_level_mode5_far_consumer_steps(self, n_partners: int) -> list[int]:
        if n_partners <= 0:
            return []
        steps = self._middle_level_mode5_ranked_far_consumer_steps()
        return sorted(steps[:n_partners], reverse=True)

    def _middle_level_mode5_far_consumer_offers(self, partners, quantity_fn):
        partners = self._filter_sell_targets(list(partners))
        response = {}
        candidate_steps = self._middle_level_mode5_ranked_far_consumer_steps()
        used_steps: set[int] = set()
        for partner in partners:
            partner_steps = self._filter_future_sell_steps_by_bankruptcy_forecast(
                partner,
                [step for step in candidate_steps if step not in used_steps],
            )
            if not partner_steps:
                continue
            partner_steps = partner_steps[: max(1, len(partners))]
            step = partner_steps[0]
            quantity = int(quantity_fn(step))
            if quantity <= 0:
                continue
            offer = (quantity, step, self.best_price(partner))
            response[partner] = offer
            used_steps.add(step)
            self._middle_level_mode8_record_sell_proposal(partner, offer)
        return response

    def _middle_level_mode5_is_late_consumer_step(
        self, step: int, n_partners: int
    ) -> bool:
        late_steps = set(
            self._middle_level_mode5_far_consumer_steps(max(1, int(n_partners)))
        )
        return step in late_steps

    def _middle_level_mode8_active_sell_phase(self) -> bool:
        return (
            self._level_label() == "middle"
            and self._middle_level_mode8_family()
            and self._middle_level_mode5_current_phase() == "normal"
        )

    def _middle_level_mode8_exploration_steps(self) -> int:
        return 10

    def _middle_level_mode8_acceptance_distance(self) -> int:
        return 2

    def _middle_level_mode8_offer_distance(self) -> int:
        return 1

    def _middle_level_mode8_quantity_grid(self, broad: bool = False) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        ratios = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0) if broad else (0.6, 0.8, 1.0)
        return sorted({max(1, min(n_lines, ceil(n_lines * ratio))) for ratio in ratios})

    def _middle_level_mode9_quantity_grid(self) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        return sorted({max(1, min(n_lines, quantity)) for quantity in (1, 2, 5, 8, 10)})

    def _middle_level_mode8_price_grid(self, partner) -> list[int]:
        pmin, pmax = self._unit_price_bounds(partner)
        pmin = int(pmin)
        pmax = int(pmax)
        if pmin >= pmax:
            return [pmin]
        return sorted(
            {
                pmin,
                round(pmin + (pmax - pmin) * 0.50),
                pmax,
            }
        )

    def _middle_level_mode9_counter_probe_start_step(self) -> int:
        return self._middle_level_mode8_exploration_steps()

    def _middle_level_mode9_counter_probe_end_step(self) -> int:
        return self._middle_level_mode9_counter_probe_start_step() + 10

    def _middle_level_mode9_history(self, partner) -> list[dict[str, object]]:
        return self._middle_level_mode8_sell_offer_history.get(str(partner), [])

    def _middle_level_mode9_is_high_price_record(
        self, partner, record: dict[str, object]
    ) -> bool:
        try:
            proposed_at = int(record.get("step", -1))
            quantity = int(record.get("quantity", 0))
            price = int(record.get("price", 0))
        except Exception:
            return False
        price_grid = self._middle_level_mode8_price_grid(partner)
        min_high_quantity = min(self._middle_level_mode8_quantity_grid())
        return (
            proposed_at < self._middle_level_mode8_exploration_steps()
            and quantity >= min_high_quantity
            and bool(record.get("accepted"))
            and bool(price_grid)
            and price >= price_grid[-1]
        )

    def _middle_level_mode9_has_high_price_acceptance(self, partner) -> bool:
        if not self._middle_level_mode9_enabled():
            return False
        return any(
            self._middle_level_mode9_is_high_price_record(partner, record)
            for record in self._middle_level_mode9_history(partner)
        )

    def _middle_level_mode9_counter_probe_active(self, partner) -> bool:
        if not self._middle_level_mode9_enabled():
            return False
        current_step = int(self._awi_value("current_step", 0) or 0)
        return (
            self._middle_level_mode9_counter_probe_start_step()
            <= current_step
            < self._middle_level_mode9_counter_probe_end_step()
            and not self._middle_level_mode9_has_high_price_acceptance(partner)
        )

    def _middle_level_mode9_counter_probe_accepted(self, partner) -> bool:
        start = self._middle_level_mode9_counter_probe_start_step()
        end = self._middle_level_mode9_counter_probe_end_step()
        for record in self._middle_level_mode9_history(partner):
            try:
                proposed_at = int(record.get("step", -1))
            except Exception:
                continue
            if start <= proposed_at < end and bool(record.get("accepted")):
                return True
        return False

    def _middle_level_mode9_partner_classification(self, partner) -> str:
        if not self._middle_level_mode9_enabled():
            return "not_mode9"
        current_step = int(self._awi_value("current_step", 0) or 0)
        if self._middle_level_mode9_has_high_price_acceptance(partner):
            return "high_price_acceptor"
        if current_step < self._middle_level_mode9_counter_probe_start_step():
            return "high_price_probe_pending"
        if current_step < self._middle_level_mode9_counter_probe_end_step():
            if self._middle_level_mode9_counter_probe_accepted(partner):
                return "counter_acceptor"
            return "counter_probe_pending"
        if self._middle_level_mode9_counter_probe_accepted(partner):
            return "counter_acceptor"
        return "counter_non_acceptor"

    def _middle_level_mode9_classification_complete(self) -> bool:
        return (
            self._middle_level_mode9_enabled()
            and int(self._awi_value("current_step", 0) or 0)
            >= self._middle_level_mode9_counter_probe_end_step()
        )

    def _middle_level_mode9_counter_price_grid(self, partner) -> list[int]:
        price_grid = self._middle_level_mode8_price_grid(partner)
        if len(price_grid) <= 2:
            return price_grid
        return [price_grid[0], price_grid[1]]

    def _middle_level_mode9_exploration_probability(self, partner) -> float:
        accepted_count = len(self._middle_level_mode8_accepted_points(partner))
        return 1.0 / (1.0 + max(0, accepted_count))

    def _middle_level_mode9_counter_acceptor_candidates(
        self, partner, explore: bool
    ) -> list[tuple[int, int]]:
        quantity_grid = self._middle_level_mode9_quantity_grid()
        price_grid = self._middle_level_mode9_counter_price_grid(partner)
        if explore:
            return [
                (quantity, price)
                for quantity in quantity_grid
                for price in price_grid
            ]

        candidates = []
        for quantity in quantity_grid:
            for price in price_grid:
                if self._middle_level_mode8_near_pareto_point(partner, quantity, price):
                    candidates.append((quantity, price))
        return candidates or [
            (quantity, price)
            for quantity in quantity_grid
            for price in price_grid
        ]

    def _middle_level_mode9_preserve_profile_price(self, partner) -> bool:
        if not self._middle_level_mode9_classification_complete():
            return False
        return (
            self._middle_level_mode9_partner_classification(partner)
            == "counter_acceptor"
        )

    def _middle_level_mode9_classification_summary(self) -> dict[str, object]:
        partners = sorted(
            set(self._middle_level_mode8_sell_offer_history.keys())
            | set(str(partner) for partner in self.awi.my_consumers)
        )
        by_partner = {
            partner: self._middle_level_mode9_partner_classification(partner)
            for partner in partners
        }
        counts = Counter(by_partner.values())
        return {
            "probe_start_step": self._middle_level_mode9_counter_probe_start_step(),
            "probe_end_step": self._middle_level_mode9_counter_probe_end_step(),
            "by_partner": by_partner,
            "counts": dict(counts),
        }

    def _middle_level_mode9_log_final_classification(self) -> None:
        if self._level_label() != "middle" or not self._middle_level_mode9_enabled():
            return
        summary = self._middle_level_mode9_classification_summary()
        by_partner = summary.get("by_partner", {})
        grouped: dict[str, list[str]] = {}
        if isinstance(by_partner, dict):
            for partner, status in by_partner.items():
                grouped.setdefault(str(status), []).append(str(partner))
        self.log_message(
            "MIDDLE_MODE9_FINAL_CLASSIFICATION",
            f"agent_id={self._my_agent_id()}",
            f"step={self._awi_value('current_step', None)}",
            f"probe_start={summary.get('probe_start_step')}",
            f"probe_end={summary.get('probe_end_step')}",
            f"counts={summary.get('counts')}",
        )
        for status in (
            "high_price_acceptor",
            "counter_acceptor",
            "counter_non_acceptor",
            "counter_probe_pending",
            "high_price_probe_pending",
        ):
            partners = sorted(grouped.get(status, []))
            self.log_message(
                "MIDDLE_MODE9_FINAL_CLASSIFICATION_DETAIL",
                f"status={status}",
                f"count={len(partners)}",
                f"partners={','.join(partners) if partners else '-'}",
            )

    def _middle_level_mode8_record_sell_proposal(self, partner, offer) -> None:
        if (
            self._level_label() != "middle"
            or not self._middle_level_mode8_family()
            or offer is None
            or not self.is_consumer(partner)
        ):
            return
        try:
            quantity, step, price = int(offer[QUANTITY]), int(offer[TIME]), int(offer[UNIT_PRICE])
        except Exception:
            return
        self._middle_level_mode8_sell_offer_history.setdefault(str(partner), []).append(
            {
                "quantity": quantity,
                "time": step,
                "price": price,
                "step": int(self._awi_value("current_step", 0) or 0),
                "accepted": False,
            }
        )

    def _middle_level_mode8_record_accepted_contract(
        self, contract_summary: dict[str, object]
    ) -> None:
        if self._level_label() != "middle" or not self._middle_level_mode8_family():
            return
        partner = contract_summary.get("partner_id")
        if partner is None:
            return
        try:
            quantity = int(contract_summary.get("quantity"))
            step = int(contract_summary.get("time"))
            price = int(contract_summary.get("unit_price"))
        except Exception:
            return
        history = self._middle_level_mode8_sell_offer_history.get(str(partner), [])
        for record in reversed(history):
            if record.get("accepted"):
                continue
            if (
                int(record.get("quantity", -1)) == quantity
                and int(record.get("time", -1)) == step
                and int(record.get("price", -1)) == price
            ):
                record["accepted"] = True
                record["accepted_at_step"] = int(self._awi_value("current_step", 0) or 0)
                return

    def _middle_level_mode8_accepts_initial_sell_offer(
        self, partner, offer, planned_quantities: Counter | None = None
    ) -> bool:
        try:
            quantity = int(offer[QUANTITY])
            step = int(offer[TIME])
            price = int(offer[UNIT_PRICE])
        except Exception:
            return False
        if (
            step != int(self.awi.current_step)
            and not self._middle_level_mode5_future_sell_enabled()
        ):
            return False
        if not self._middle_level_mode8_matches_full_quantity_policy(
            partner, quantity, step, planned_quantities
        ):
            return False
        return (
            quantity >= 8
            and step >= int(self.awi.current_step) + 6
            and step < int(self.awi.n_steps)
            and (
                self._future_sell_offer_within_allowable_excess(
                    step, quantity, price, planned_quantities
                )
                if self._middle_level_mode9_enabled()
                else quantity
                <= self._middle_level_mode8_remaining_sell_capacity(
                    step, planned_quantities
                )
            )
        )

    def _middle_level_mode8_log_learning_summary_if_finished(self) -> None:
        if self._middle_level_mode8_learning_summary_logged:
            return
        if self._level_label() != "middle" or not self._middle_level_mode8_family():
            return
        current_step = int(self._awi_value("current_step", 0) or 0)
        n_steps = int(self._awi_value("n_steps", 0) or 0)
        if n_steps <= 0 or current_step < n_steps - 1:
            return
        self._middle_level_mode8_learning_summary_logged = True
        self._middle_level_mode9_log_final_classification()
        self._debug_log_decision(
            "middle_mode8_learning_summary",
            learning_summary=self._middle_level_mode8_learning_summary(),
        )

    def _middle_level_mode8_learning_summary(self) -> dict[str, object]:
        partners = sorted(
            set(self._middle_level_mode8_sell_offer_history.keys())
            | set(str(partner) for partner in self.awi.my_consumers)
        )
        summary = {}
        for partner in partners:
            history = self._middle_level_mode8_sell_offer_history.get(partner, [])
            accepted_records = [record for record in history if record.get("accepted")]
            quantity_grid = (
                self._middle_level_mode9_quantity_grid()
                if self._middle_level_mode9_enabled()
                else self._middle_level_mode8_quantity_grid(broad=True)
            )
            try:
                price_grid = self._middle_level_mode8_price_grid(partner)
            except Exception:
                price_grid = []
            summary[partner] = {
                "proposed_count": len(history),
                "accepted_count": len(accepted_records),
                "mode9_classification": (
                    self._middle_level_mode9_partner_classification(partner)
                    if self._middle_level_mode9_enabled()
                    else None
                ),
                "quantity_grid": quantity_grid,
                "price_grid": price_grid,
                "accepted_points": [
                    {
                        "quantity": record.get("quantity"),
                        "price": record.get("price"),
                        "time": record.get("time"),
                        "proposed_at_step": record.get("step"),
                        "accepted_at_step": record.get("accepted_at_step"),
                    }
                    for record in accepted_records
                ],
                "pareto_points": self._middle_level_mode8_pareto_points(partner),
                "frontier_sample": {
                    str(quantity): self._middle_level_mode8_frontier_price(
                        partner, quantity
                    )
                    for quantity in quantity_grid
                },
            }
        return {
            "mode": self._middle_level_negotiation_mode,
            "phase": self._middle_level_mode5_current_phase(),
            "n_processes": self._awi_value("n_processes", None),
            "process_index": self._awi_value("my_input_product", None),
            "partners": summary,
        }

    def _middle_level_mode8_accepted_points(self, partner) -> list[tuple[int, int]]:
        return [
            (int(record["quantity"]), int(record["price"]))
            for record in self._middle_level_mode8_sell_offer_history.get(partner, [])
            if record.get("accepted")
        ]

    def _middle_level_mode8_has_accepted_full_quantity(self, partner) -> bool:
        n_lines = int(self._awi_value("n_lines", 0) or 0)
        return any(
            quantity >= n_lines
            for quantity, _price in self._middle_level_mode8_accepted_points(partner)
        )

    def _middle_level_mode8_matches_full_quantity_policy(
        self, partner, quantity: int, step: int, planned_quantities: Counter | None
    ) -> bool:
        if not self._middle_level_mode8_has_accepted_full_quantity(partner):
            return True
        n_lines = int(self._awi_value("n_lines", 1) or 1)
        remaining_capacity = self._middle_level_mode8_remaining_sell_capacity(
            step, planned_quantities
        )
        return quantity == min(n_lines, remaining_capacity)

    def _middle_level_mode8_pareto_points(self, partner) -> list[tuple[int, int]]:
        best_by_quantity: dict[int, int] = {}
        for quantity, price in self._middle_level_mode8_accepted_points(partner):
            best_by_quantity[quantity] = max(price, best_by_quantity.get(quantity, price))
        points = sorted(best_by_quantity.items())
        pareto = []
        for quantity, price in points:
            dominated = any(
                other_q >= quantity
                and other_p >= price
                and (other_q > quantity or other_p > price)
                for other_q, other_p in points
            )
            if not dominated:
                pareto.append((quantity, price))
        return sorted(pareto)

    def _middle_level_mode8_frontier_price(self, partner, quantity: int) -> float | None:
        points = self._middle_level_mode8_pareto_points(partner)
        if not points:
            return None
        if len(points) == 1:
            return float(points[0][1])
        if quantity <= points[0][0]:
            return float(points[0][1])
        if quantity >= points[-1][0]:
            return float(points[-1][1])
        for (q1, p1), (q2, p2) in zip(points, points[1:]):
            if q1 <= quantity <= q2:
                if q1 == q2:
                    return float(max(p1, p2))
                ratio = (quantity - q1) / (q2 - q1)
                return p1 + (p2 - p1) * ratio
        return float(points[-1][1])

    def _middle_level_mode8_grid_distance_to_frontier(
        self, partner, quantity: int, price: int
    ) -> int | None:
        pareto = self._middle_level_mode8_pareto_points(partner)
        if not pareto:
            return None
        quantity_grid = self._middle_level_mode8_quantity_grid(broad=True)
        price_grid = self._middle_level_mode8_price_grid(partner)
        q_index = min(
            range(len(quantity_grid)),
            key=lambda index: abs(quantity_grid[index] - quantity),
        )
        p_index = min(
            range(len(price_grid)),
            key=lambda index: abs(price_grid[index] - price),
        )
        distances = []
        for frontier_q, frontier_p in pareto:
            fq_index = min(
                range(len(quantity_grid)),
                key=lambda index: abs(quantity_grid[index] - frontier_q),
            )
            fp_index = min(
                range(len(price_grid)),
                key=lambda index: abs(price_grid[index] - frontier_p),
            )
            distances.append(abs(q_index - fq_index) + abs(p_index - fp_index))
        return min(distances)

    def _middle_level_mode8_near_pareto_point(
        self, partner, quantity: int, price: int
    ) -> bool:
        pareto = self._middle_level_mode8_pareto_points(partner)
        if not pareto:
            return False
        quantity_grid = self._middle_level_mode8_quantity_grid(broad=True)
        price_grid = self._middle_level_mode8_price_grid(partner)
        q_index = min(
            range(len(quantity_grid)),
            key=lambda index: abs(quantity_grid[index] - quantity),
        )
        p_index = min(
            range(len(price_grid)),
            key=lambda index: abs(price_grid[index] - price),
        )
        for frontier_q, frontier_p in pareto:
            fq_index = min(
                range(len(quantity_grid)),
                key=lambda index: abs(quantity_grid[index] - frontier_q),
            )
            fp_index = min(
                range(len(price_grid)),
                key=lambda index: abs(price_grid[index] - frontier_p),
            )
            if (
                abs(q_index - fq_index) <= self._middle_level_mode8_offer_distance()
                and abs(p_index - fp_index) <= self._middle_level_mode8_offer_distance()
            ):
                return True
        return False

    def _middle_level_mode8_offer_candidates(self, partner) -> list[tuple[int, int]]:
        accepted_count = len(self._middle_level_mode8_accepted_points(partner))
        current_step = int(self._awi_value("current_step", 0) or 0)
        price_grid = self._middle_level_mode8_price_grid(partner)
        fixed_full_quantity = self._middle_level_mode8_has_accepted_full_quantity(
            partner
        )
        quantity_grid = (
            [int(self._awi_value("n_lines", 1) or 1)]
            if fixed_full_quantity
            else (
                self._middle_level_mode9_quantity_grid()
                if self._middle_level_mode9_enabled()
                else self._middle_level_mode8_quantity_grid(broad=True)
            )
        )
        if (
            self._middle_level_mode9_enabled()
            and self._middle_level_mode9_has_high_price_acceptance(partner)
        ):
            if fixed_full_quantity:
                return [(quantity_grid[0], price_grid[-1])]
            return [
                (quantity, price_grid[-1])
                for quantity in self._middle_level_mode8_quantity_grid()
            ]
        if current_step < self._middle_level_mode8_exploration_steps():
            if fixed_full_quantity:
                return [(quantity_grid[0], price_grid[-1])]
            return [
                (quantity, price_grid[-1])
                for quantity in self._middle_level_mode8_quantity_grid()
            ]
        if self._middle_level_mode9_counter_probe_active(partner):
            return [(quantity, price_grid[0]) for quantity in quantity_grid]
        if (
            self._middle_level_mode9_classification_complete()
            and self._middle_level_mode9_partner_classification(partner)
            == "counter_acceptor"
        ):
            explore = (
                random.random()
                < self._middle_level_mode9_exploration_probability(partner)
            )
            return self._middle_level_mode9_counter_acceptor_candidates(
                partner, explore
            )
        if accepted_count < 3:
            return [
                (quantity, price_grid[0])
                for quantity in quantity_grid
            ]

        candidates = []
        for quantity in quantity_grid:
            for price in price_grid:
                if self._middle_level_mode8_near_pareto_point(partner, quantity, price):
                    candidates.append((quantity, price))
        return candidates or [
            (quantity, price)
            for quantity in quantity_grid
            for price in price_grid
        ]

    def _middle_level_mode8_partner_offer_profile(
        self, partner
    ) -> tuple[int, int] | None:
        candidates = self._middle_level_mode8_offer_candidates(partner)
        if not candidates:
            return None
        accepted_count = len(self._middle_level_mode8_accepted_points(partner))
        current_step = int(self._awi_value("current_step", 0) or 0)
        if (
            self._middle_level_mode9_classification_complete()
            and self._middle_level_mode9_partner_classification(partner)
            == "counter_acceptor"
        ):
            return random.choice(candidates)
        if (
            current_step < self._middle_level_mode8_exploration_steps()
            or accepted_count < 3
        ):
            return random.choice(candidates)
        if self._middle_level_mode8_has_accepted_full_quantity(partner):
            max_quantity = int(self._awi_value("n_lines", 1) or 1)
            prices = [price for quantity, price in candidates if quantity == max_quantity]
            if prices:
                return max_quantity, random.choice(prices)
        return random.choice(candidates)

    def _middle_level_mode8_offer_price_for_quantity(
        self, partner, quantity: int, fallback_price: int
    ) -> int:
        if self._middle_level_mode9_preserve_profile_price(partner):
            return int(fallback_price)
        candidates = self._middle_level_mode8_offer_candidates(partner)
        exact_prices = [price for q, price in candidates if q == quantity]
        if exact_prices:
            return max(exact_prices)
        frontier_price = self._middle_level_mode8_frontier_price(partner, quantity)
        price_grid = self._middle_level_mode8_price_grid(partner)
        if frontier_price is not None:
            affordable_prices = [price for price in price_grid if price <= frontier_price]
            if affordable_prices:
                return max(affordable_prices)
        return int(fallback_price)

    def _middle_level_mode8_consumer_steps(
        self, include_current_step: bool = True
    ) -> list[int]:
        issues = self._awi_value("current_output_issues", None)
        min_step = self.awi.current_step if include_current_step else self.awi.current_step + 1
        max_step = self.awi.n_steps - 1
        if issues is not None:
            try:
                issue = issues[TIME]
                min_step = max(min_step, int(issue.min_value))
                max_step = min(max_step, int(issue.max_value))
            except Exception:
                pass
        if min_step > max_step:
            return []
        return list(range(min_step, max_step + 1))

    def _middle_level_mode8_remaining_sell_capacity(
        self, step: int, planned_quantities: Counter | None = None
    ) -> int:
        planned = 0 if planned_quantities is None else planned_quantities[step]
        already_sold = self._middle_level_mode8_recorded_sales_at(step)
        return max(0, int(self.awi.n_lines - already_sold - planned))

    def _middle_level_mode8_future_horizon_steps(self) -> list[int]:
        issues = self._awi_value("current_output_issues", None)
        current_step = int(self.awi.current_step)
        min_step = current_step + 1
        horizon_offset = 9 if self._middle_level_mode9_enabled() else 5
        max_step = min(int(self.awi.n_steps) - 1, current_step + horizon_offset)
        if issues is not None:
            try:
                issue = issues[TIME]
                min_step = max(min_step, int(issue.min_value))
                max_step = min(max_step, int(issue.max_value))
            except Exception:
                pass
        if min_step > max_step:
            return []
        return list(range(min_step, max_step + 1))

    def _future_sell_allowed_issue_steps(self) -> list[int]:
        issues = self._awi_value("current_output_issues", None)
        current_step = int(self.awi.current_step)
        min_step = current_step + 1
        max_step = int(self.awi.n_steps) - 1
        if issues is not None:
            try:
                issue = issues[TIME]
                min_step = max(min_step, int(issue.min_value))
                max_step = min(max_step, int(issue.max_value))
            except Exception:
                pass
        if min_step > max_step:
            return []
        return list(range(min_step, max_step + 1))

    def _future_sell_fill_ratio_for_excess(
        self, planned_quantities: Counter | None = None
    ) -> float:
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        steps = self._future_sell_allowed_issue_steps()
        if not steps:
            return 0.0
        n_lines = max(1, int(self.awi.n_lines))
        filled = 0
        for step in steps:
            sold = self._sell_strategy_recorded_sales_at(step)
            sold += int(planned_quantities[step])
            filled += min(n_lines, max(0, int(sold)))
        return max(0.0, min(1.0, filled / float(len(steps) * n_lines)))

    def _future_sell_base_target_amount(
        self, planned_quantities: Counter | None = None
    ) -> int:
        n_lines = max(1, int(self.awi.n_lines))
        fill_ratio = self._future_sell_fill_ratio_for_excess(planned_quantities)
        return int(floor(fill_ratio * n_lines))

    def _sell_score_at_total_quantity(
        self,
        step: int,
        total_quantity: int,
        price: float,
        planned_quantities: Counter | None = None,
    ) -> float:
        available, _committed = self._middle_level_mode8_sellable_units_before_offer_at(
            step, planned_quantities
        )
        total_quantity = max(0, int(total_quantity))
        fulfillable_capacity = min(float(self.awi.n_lines), available)
        fulfilled = min(fulfillable_capacity, float(total_quantity))
        shortfall = max(0.0, float(total_quantity) - fulfillable_capacity)
        return (
            fulfilled * (float(price) - self._production_cost())
            - shortfall * self._middle_level_mode8_shortfall_penalty_per_unit(price)
        )

    def _allowable_excess_sell_quantity(
        self,
        step: int,
        price: float,
        planned_quantities: Counter | None = None,
        upper_total_quantity: int | None = None,
    ) -> int:
        n_lines = max(1, int(self.awi.n_lines))
        if upper_total_quantity is None:
            upper_total_quantity = n_lines
        upper_total_quantity = max(n_lines, int(upper_total_quantity))
        base_sell_target = self._future_sell_base_target_amount(planned_quantities)
        base_quantity = min(
            n_lines,
            max(0, base_sell_target + int(self._allowable_excess_sell_margin)),
        )
        base_score = self._sell_score_at_total_quantity(
            step, base_quantity, price, planned_quantities
        )
        allowable_excess = 0
        for total_quantity in range(n_lines, upper_total_quantity + 1):
            score = self._sell_score_at_total_quantity(
                step, total_quantity, price, planned_quantities
            )
            if score >= base_score:
                allowable_excess = max(allowable_excess, total_quantity - n_lines)
        return max(0, int(allowable_excess))

    def _future_sell_offer_within_allowable_excess(
        self,
        step: int,
        quantity: int,
        price: float,
        planned_quantities: Counter | None = None,
    ) -> bool:
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        n_lines = max(1, int(self.awi.n_lines))
        current_total = self._sell_strategy_recorded_sales_at(step)
        current_total += int(planned_quantities[step])
        candidate_total = int(current_total) + int(quantity)
        if candidate_total <= n_lines:
            return True
        allowable_excess = self._allowable_excess_sell_quantity(
            step,
            price,
            planned_quantities,
            upper_total_quantity=candidate_total,
        )
        return candidate_total <= n_lines + allowable_excess

    def _middle_level_mode8_future_sales_total(
        self, planned_quantities: Counter | None = None
    ) -> int:
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        total = 0
        for step in self._middle_level_mode8_future_horizon_steps():
            total += self._middle_level_mode8_recorded_sales_at(step)
            total += int(planned_quantities[step])
        return total

    def _middle_level_mode8_low_future_fill(
        self, planned_quantities: Counter | None = None
    ) -> bool:
        steps = self._middle_level_mode8_future_horizon_steps()
        if not steps:
            return False
        capacity = len(steps) * int(self.awi.n_lines)
        return self._middle_level_mode8_future_sales_total(planned_quantities) <= (
            capacity * 0.4
        )

    def _weighted_average_unit_price(
        self, totals: dict[str, float], fallback: float
    ) -> float:
        quantity = totals.get("quantity", 0.0)
        if quantity <= 0:
            return fallback
        return totals.get("value", 0.0) / quantity

    def _catalog_price(self, product: int | None = None) -> float:
        if product is None:
            product = int(self._awi_value("my_output_product", 0) or 0)
        prices = self._awi_value("catalog_prices", None)
        try:
            return float(prices[product])
        except Exception:
            return 0.0

    def _average_buy_unit_price(self) -> float:
        input_product = int(self._awi_value("my_input_product", 0) or 0)
        return self._weighted_average_unit_price(
            self._contract_value_totals["buy"],
            self._catalog_price(input_product),
        )

    def _production_cost(self) -> float:
        profile = self._awi_value("profile", None)
        value = getattr(profile, "cost", None)
        if value is None:
            value = self._call_awi("production_cost", default=0)
        if not isinstance(value, (str, bytes)) and hasattr(value, "__getitem__"):
            try:
                value = value[int(self._awi_value("my_input_product", 0) or 0)]
            except Exception:
                try:
                    value = value[0]
                except Exception:
                    value = 0
        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def _middle_level_mode8_expected_sell_unit_margin(self, price: float) -> float:
        return float(price) - self._average_buy_unit_price() - self._production_cost()

    def _middle_level_mode8_extra_input_available_at(
        self, step: int, planned_quantities: Counter | None = None
    ) -> int:
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        current_step = int(self.awi.current_step)
        inventory = float(self._awi_value("current_inventory_input", 0) or 0)
        if step < current_step:
            return 0
        for target_step in range(current_step, step + 1):
            inventory += float(
                self._call_awi("total_supplies_at", target_step, default=0) or 0
            )
            committed_sales = float(
                self._sell_strategy_recorded_sales_at(target_step)
            ) + float(planned_quantities[target_step])
            inventory = max(0.0, inventory - committed_sales)
        return max(0, int(inventory))

    def _middle_level_mode8_profitable_sell_capacity(
        self, step: int, price: float, planned_quantities: Counter | None = None
    ) -> int:
        if self._middle_level_mode8_expected_sell_unit_margin(price) < 0:
            return 0
        return min(
            self._middle_level_mode8_remaining_sell_capacity(step, planned_quantities),
            self._middle_level_mode8_extra_input_available_at(step, planned_quantities),
        )

    def _middle_level_mode8_sellable_units_before_offer_at(
        self, step: int, planned_quantities: Counter | None = None
    ) -> tuple[float, float]:
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        current_step = int(self.awi.current_step)
        step = int(step)
        if step < current_step:
            return 0.0, 0.0
        available = (
            float(self._awi_value("current_inventory_output", 0) or 0)
            + float(self._awi_value("current_inventory_input", 0) or 0)
        )
        for target_step in range(current_step, step):
            available += float(
                self._call_awi("total_supplies_at", target_step, default=0) or 0
            )
            committed_before = (
                float(self._sell_strategy_recorded_sales_at(target_step))
                + float(planned_quantities[target_step])
            )
            available = max(0.0, available - committed_before)
        available += float(self._call_awi("total_supplies_at", step, default=0) or 0)
        committed = (
            float(self._sell_strategy_recorded_sales_at(step))
            + float(planned_quantities[step])
        )
        return max(0.0, available), max(0.0, committed)

    def _middle_level_mode8_current_sellable_units_before_offer(
        self, planned_quantities: Counter | None = None
    ) -> tuple[float, float]:
        return self._middle_level_mode8_sellable_units_before_offer_at(
            int(self.awi.current_step), planned_quantities
        )

    def _middle_level_mode8_shortfall_penalty_per_unit(self, unit_price: float) -> float:
        value = self._call_awi("current_shortfall_penalty", default=0) or 0
        multiplier = self._debug_penalty_multiplier(False, unit_price)
        try:
            return max(0.0, float(value) * float(multiplier))
        except Exception:
            return 0.0

    def _middle_level_mode8_current_sell_score_delta(
        self,
        quantity: int,
        price: float,
        planned_quantities: Counter | None = None,
    ) -> float:
        return self._middle_level_mode8_sell_score_delta_at(
            int(self.awi.current_step), quantity, price, planned_quantities
        )

    def _middle_level_mode8_sell_score_delta_at(
        self,
        step: int,
        quantity: int,
        price: float,
        planned_quantities: Counter | None = None,
    ) -> float:
        available, committed = self._middle_level_mode8_sellable_units_before_offer_at(
            step,
            planned_quantities
        )
        fulfillable_capacity = min(available, float(self.awi.n_lines))
        before_fulfilled = min(fulfillable_capacity, committed)
        after_fulfilled = min(fulfillable_capacity, committed + quantity)
        additional_fulfilled = max(0.0, after_fulfilled - before_fulfilled)

        before_shortfall = max(0.0, committed - fulfillable_capacity)
        after_shortfall = max(0.0, committed + quantity - fulfillable_capacity)
        additional_shortfall = max(0.0, after_shortfall - before_shortfall)

        return (
            additional_fulfilled * (float(price) - self._production_cost())
            - additional_shortfall
            * self._middle_level_mode8_shortfall_penalty_per_unit(price)
        )

    def _middle_level_mode8_state_relative_time(self, state) -> float:
        value = getattr(state, "relative_time", None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _middle_level_mode8_state_round(self, state) -> int:
        for name in ("step", "n_steps", "round", "current_step"):
            value = getattr(state, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if value is None:
                continue
            try:
                return max(0, int(value))
            except Exception:
                continue
        return 0

    def _middle_level_mode8_price_acceptance_threshold(self, partner, state) -> float:
        pmin, pmax = self._unit_price_bounds(partner)
        relative_time = self._middle_level_mode8_state_relative_time(state)
        return float(pmax) - (float(pmax) - float(pmin)) * relative_time

    def _middle_level_mode8_quantity_acceptance_bounds(self, state) -> tuple[int, int]:
        center = min(10, max(1, int(self._awi_value("n_lines", 10) or 10)))
        width = self._middle_level_mode8_state_round(state)
        return max(1, center - width), center + width

    def _middle_level_mode8_low_history_current_quantity_bounds(
        self, state
    ) -> tuple[int, int]:
        center = min(10, max(1, int(self._awi_value("n_lines", 10) or 10)))
        base_width = 3
        extra_width = int(self._middle_level_mode8_state_relative_time(state) * center)
        width = base_width + max(0, extra_width)
        return max(1, center - width), center + width

    def _middle_level_mode9_classified_accepts_offer(
        self,
        partner,
        step: int,
        quantity: int,
        price: int,
        planned_quantities: Counter,
    ) -> bool | None:
        if not self._middle_level_mode9_classification_complete():
            return None
        classification = self._middle_level_mode9_partner_classification(partner)
        if classification == "high_price_acceptor":
            return None
        is_current_step_offer = step == int(self.awi.current_step)
        if (
            not is_current_step_offer
            and not self._future_sell_offer_within_allowable_excess(
                step, quantity, price, planned_quantities
            )
        ):
            return False
        score_improves = (
            self._middle_level_mode8_sell_score_delta_at(
                step, quantity, price, planned_quantities
            )
            > 0
        )
        if classification == "counter_acceptor":
            if quantity <= 2:
                return False
            return score_improves
        if classification == "counter_non_acceptor":
            return score_improves
        return None

    def _middle_level_mode8_pick_sell_step(
        self,
        candidate_steps: list[int],
        planned_quantities: Counter,
        target_quantity: int | None = None,
    ) -> int | None:
        available_steps = [
            step
            for step in candidate_steps
            if self._middle_level_mode8_remaining_sell_capacity(step, planned_quantities) > 0
        ]
        if not available_steps:
            return None
        if target_quantity is not None and target_quantity > 0:
            full_capacity_steps = [
                step
                for step in available_steps
                if self._middle_level_mode8_remaining_sell_capacity(
                    step, planned_quantities
                )
                >= target_quantity
            ]
            if full_capacity_steps:
                available_steps = full_capacity_steps
        ranked = sorted(
            available_steps,
            key=lambda step: (
                self._middle_level_mode8_recorded_sales_at(step)
                + planned_quantities[step],
                step,
            ),
        )
        return random.choice(ranked[: min(2, len(ranked))])

    def _middle_level_mode8_sell_offers(
        self,
        partners,
        planned_quantities: Counter | None = None,
        include_current_step: bool = True,
    ) -> dict[str, tuple[int, int, int]]:
        if (
            not include_current_step
            and self._level_label() == "middle"
            and self._middle_level_mode5_family()
            and not self._middle_level_mode5_future_sell_enabled()
        ):
            return {}
        partners = self._filter_sell_targets(list(partners))
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        profiles = []
        for partner in partners:
            profile = self._middle_level_mode8_partner_offer_profile(partner)
            if profile is None:
                continue
            quantity, price = profile
            profiles.append((partner, quantity, price))
        profiles.sort(key=lambda item: item[1], reverse=True)

        response = {}
        candidate_steps = self._middle_level_mode8_consumer_steps(
            include_current_step=include_current_step
        )
        for partner, quantity, price in profiles:
            current_step = int(self._awi_value("current_step", 0) or 0)
            partner_candidate_steps = [
                step
                for step in candidate_steps
                if step <= current_step
                or self._future_sell_step_allowed_by_bankruptcy_forecast(partner, step)
            ]
            step = self._middle_level_mode8_pick_sell_step(
                partner_candidate_steps, planned_quantities, int(quantity)
            )
            if step is None:
                continue
            offer_quantity = min(
                int(quantity),
                self._middle_level_mode8_remaining_sell_capacity(step, planned_quantities),
            )
            if offer_quantity <= 0:
                continue
            price = self._middle_level_mode8_offer_price_for_quantity(
                partner, offer_quantity, price
            )
            offer = (offer_quantity, step, int(price))
            response[partner] = offer
            planned_quantities[step] += offer_quantity
            self._middle_level_mode8_record_sell_proposal(partner, offer)
        return response

    def _middle_level_mode8_accepts_offer(
        self, partner, offer, planned_quantities: Counter, state=None
    ) -> bool:
        try:
            quantity = int(offer[QUANTITY])
            step = int(offer[TIME])
            price = int(offer[UNIT_PRICE])
        except Exception:
            return False
        if step < self.awi.current_step or step >= self.awi.n_steps:
            return False
        if quantity <= 0:
            return False
        is_current_step_offer = step == int(self.awi.current_step)
        if (
            not is_current_step_offer
            and not self._middle_level_mode5_future_sell_enabled()
        ):
            return False
        classified_decision = self._middle_level_mode9_classified_accepts_offer(
            partner, step, quantity, price, planned_quantities
        )
        if classified_decision is not None:
            return classified_decision
        low_future_fill_offer = (
            not is_current_step_offer
            and self._middle_level_mode8_low_future_fill(planned_quantities)
        )
        if (
            not is_current_step_offer
            and not low_future_fill_offer
            and (
                (
                    self._middle_level_mode9_enabled()
                    and not self._future_sell_offer_within_allowable_excess(
                        step, quantity, price, planned_quantities
                    )
                )
                or (
                    not self._middle_level_mode9_enabled()
                    and quantity
                    > self._middle_level_mode8_remaining_sell_capacity(
                        step, planned_quantities
                    )
                )
            )
        ):
            return False
        if (
            not is_current_step_offer
            and not low_future_fill_offer
            and not self._middle_level_mode8_matches_full_quantity_policy(
                partner, quantity, step, planned_quantities
            )
        ):
            return False

        accepted_count = len(self._middle_level_mode8_accepted_points(partner))
        low_history_current_offer = is_current_step_offer and accepted_count < 3
        if low_future_fill_offer:
            return (
                self._middle_level_mode8_sell_score_delta_at(
                    step, quantity, price, planned_quantities
                )
                > 0
                and (
                    not self._middle_level_mode9_enabled()
                    or self._future_sell_offer_within_allowable_excess(
                        step, quantity, price, planned_quantities
                    )
                )
            )
        if low_history_current_offer:
            min_quantity, max_quantity = (
                self._middle_level_mode8_low_history_current_quantity_bounds(state)
            )
        else:
            price_threshold = self._middle_level_mode8_price_acceptance_threshold(
                partner, state
            )
            if price < price_threshold:
                return False
            min_quantity, max_quantity = (
                self._middle_level_mode8_quantity_acceptance_bounds(state)
            )
        if quantity < min_quantity or quantity > max_quantity:
            return False

        if is_current_step_offer:
            return (
                self._middle_level_mode8_current_sell_score_delta(
                    quantity, price, planned_quantities
                )
                > 0
            )
        if self._middle_level_mode9_enabled():
            return (
                self._middle_level_mode8_sell_score_delta_at(
                    step, quantity, price, planned_quantities
                )
                > 0
                and self._future_sell_offer_within_allowable_excess(
                    step, quantity, price, planned_quantities
                )
            )
        return quantity <= self._middle_level_mode8_profitable_sell_capacity(
            step, price, planned_quantities
        )

    def _consumer_need_at(self, step: int | None = None) -> int:
        level = self._level_label()
        if level == "middle" and self._middle_level_negotiation_mode == 1:
            return self._calc_middle_level_consumer_need(step)
        if level == "middle" and self._middle_level_negotiation_mode == 2:
            return self._calc_middle_level_buffered_consumer_need(step)
        if level == "middle" and self._middle_level_negotiation_mode in (3, 4, 5, 6, 7, 8, 9):
            return self._calc_middle_level_initial_fill_consumer_need(step)
        return self._legacy_consumer_need_at(step)

    def _current_gap_pressure(self) -> float:
        denominator = self._same_level_agent_count() or 1
        return (
            self._current_exogenous_input_quantity()
            - self._current_exogenous_output_quantity()
        ) / denominator

    def _storage_cost_per_unit(self) -> float:
        profile = self._awi_value("profile", None)
        storage_cost = getattr(profile, "storage_cost", None)
        if callable(storage_cost):
            storage_cost = storage_cost()
        if storage_cost is None:
            storage_cost = self._call_awi("storage_cost", default=0)

        if not isinstance(storage_cost, (str, bytes)) and hasattr(
            storage_cost, "__getitem__"
        ):
            product = int(self._awi_value("my_output_product", 0) or 0)
            try:
                storage_cost = storage_cost[product]
            except Exception:
                try:
                    storage_cost = storage_cost[0]
                except Exception:
                    storage_cost = 0

        try:
            return max(0.0, float(storage_cost))
        except Exception:
            return 0.0

    def _debug_penalty_multiplier(
        self, is_input: bool, unit_price: float | None = None
    ) -> float:
        value = self._call_awi(
            "penalty_multiplier", is_input, unit_price, default=1.0
        )
        try:
            return max(0.0, float(value))
        except Exception:
            return 1.0

    def _debug_current_storage_cost_per_unit(self) -> float:
        value = self._call_awi("current_storage_cost", default=None)
        if value is None:
            value = self._storage_cost_per_unit()
        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def _debug_current_shortfall_penalty_per_unit(self) -> float:
        value = self._call_awi("current_shortfall_penalty", default=0) or 0
        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def _debug_step_cost_summary(self) -> dict[str, object]:
        input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        output_inventory = int(self._awi_value("current_inventory_output", 0) or 0)
        storage_units = max(0, input_inventory) + max(0, output_inventory)
        storage_cost_per_unit = self._debug_current_storage_cost_per_unit()
        storage_multiplier = self._debug_penalty_multiplier(True, None)

        shortfall_units = max(0, int(self._awi_value("needed_supplies", 0) or 0))
        shortfall_penalty_per_unit = self._debug_current_shortfall_penalty_per_unit()
        shortfall_multiplier = self._debug_penalty_multiplier(False, None)

        return {
            "storage_units": storage_units,
            "storage_input_units": max(0, input_inventory),
            "storage_output_units": max(0, output_inventory),
            "storage_cost_per_unit": storage_cost_per_unit,
            "storage_multiplier": storage_multiplier,
            "estimated_storage_cost": (
                storage_units * storage_cost_per_unit * storage_multiplier
            ),
            "shortfall_units": shortfall_units,
            "shortfall_penalty_per_unit": shortfall_penalty_per_unit,
            "shortfall_multiplier": shortfall_multiplier,
            "estimated_shortfall_penalty": (
                shortfall_units
                * shortfall_penalty_per_unit
                * shortfall_multiplier
            ),
            "needed_sales_proxy_units": max(
                0, int(self._awi_value("needed_sales", 0) or 0)
            ),
        }

    def _current_strategy_metrics(self) -> dict[str, object]:
        step = self._awi_value("current_step", None)
        input_inventory = self._awi_value("current_inventory_input", None)
        output_inventory = self._awi_value("current_inventory_output", None)
        storage_cost = self._storage_cost_per_unit()

        return {
            "step": step,
            "level": self._level_label(),
            "ptoday_schedule": self._ptoday_schedule,
            "ptoday_current": self._ptoday(),
            "gap_pressure": self._current_gap_pressure(),
            "current_exogenous_input_quantity": self._current_exogenous_input_quantity(),
            "current_exogenous_output_quantity": self._current_exogenous_output_quantity(),
            "needed_supplies": self._awi_value("needed_supplies", None),
            "needed_sales": self._awi_value("needed_sales", None),
            "current_inventory_input": input_inventory,
            "current_inventory_output": output_inventory,
            "middle_level_mode4_buy_cap": self._middle_level_mode4_buy_cap,
            "input_inventory_cap_enabled": self._input_inventory_cap_enabled(),
            "first_level_future_offer_divisor": (
                self._first_level_future_offer_divisor
            ),
            "first_level_staggered_future_sell": (
                self._first_level_staggered_future_sell
            ),
            "sell_partner_filter_enabled": self._sell_partner_filter_enabled,
            "use_bankruptcy_forecast": self._use_bankruptcy_forecast,
            "allowable_excess_sell_margin": self._allowable_excess_sell_margin,
            "storage_cost": storage_cost,
            "total_supplies_at_current_step": self._call_awi(
                "total_supplies_at", step, default=None
            )
            if step is not None
            else None,
            "total_sales_at_current_step": self._call_awi(
                "total_sales_at", step, default=None
            )
            if step is not None
            else None,
        }

    def _record_strategy_metrics(self) -> None:
        self._record_first_level_exogenous_input_quantity()
        self._record_last_level_exogenous_output_quantity()
        self._record_middle_level_expected_output_quantity()
        metrics = self._current_strategy_metrics()
        if self._level_label() == "first":
            metrics["first_level_sell_mode"] = self._first_level_sell_mode
            if self._first_level_sell_mode in (2, 3):
                metrics["first_level_mode2_phase"] = self._first_level_mode2_phase()
                metrics["first_level_mode2_phase_start_step"] = (
                    self._first_level_mode2_phase_start_step
                )
            metrics["first_level_exogenous_input_average"] = (
                self._calc_first_level_average_exogenous_input()
            )
            metrics["active_consumer_targets"] = len(
                self._filter_sell_targets(self.awi.my_consumers)
            )
        if self._level_label() == "last":
            metrics["last_level_average_exogenous_output_quantity"] = (
                self._calc_last_level_average_exogenous_output()
            )
            metrics["last_level_peak_exogenous_output_quantity"] = (
                self._calc_last_level_peak_exogenous_output()
            )
            metrics["last_level_inventory_buffer_quantity"] = (
                self._calc_last_level_inventory_buffer_quantity()
            )
            metrics["last_level_required_input_quantity"] = (
                self._calc_last_level_required_input_quantity()
            )
            metrics["last_level_capped_supplier_need_quantity"] = (
                self._supplier_need_at()
            )
            metrics["last_level_negotiation_mode"] = self._last_level_negotiation_mode
            metrics["adjust_last_level_proposal_price"] = (
                self._adjust_last_level_proposal_price
            )
        if self._level_label() == "middle":
            metrics["middle_level_negotiation_mode"] = self._middle_level_negotiation_mode
            metrics["middle_level_expected_output_quantity"] = (
                self._calc_middle_level_expected_output_quantity()
            )
            metrics["middle_level_average_expected_output_quantity"] = (
                self._calc_middle_level_average_expected_output()
            )
            metrics["middle_level_inventory_buffer_days"] = (
                self._middlelevel_inventory_buffer_days
            )
            metrics["middle_level_inventory_buffer_quantity"] = (
                self._calc_middle_level_inventory_buffer_quantity()
            )
            metrics["middle_level_required_input_quantity"] = (
                self._supplier_need_at()
            )
            metrics["middle_level_consumer_need_quantity"] = (
                self._consumer_need_at()
            )
            metrics["middle_level_mode5_phase"] = (
                self._middle_level_mode5_current_phase()
            )
        self._last_strategy_metrics = metrics
        self._strategy_metrics_history.append(metrics)

    @property
    def strategy_metrics_history(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(metrics) for metrics in self._strategy_metrics_history)

    def before_step(self):
        self._record_strategy_metrics()
        self._log_bankrupt_partners_once()
        recorded_cash_reports = self._record_cash_reports()
        self._log_cash_bankruptcy_forecast(recorded_cash_reports)
        if not self._debug_world_context_logged:
            self._debug_world_context_logged = True
            self._debug_log_decision(
                "world_context",
                world_structure=self._debug_world_structure(),
            )
        self._debug_log_decision(
            "before_step",
            strategy_metrics=self._last_strategy_metrics,
            suppliers=list(self.awi.my_suppliers),
            consumers=list(self.awi.my_consumers),
        )
        return super().before_step()

    def step(self):
        self._debug_log_decision("step_start")
        super().step()
        self._threshold = self.awi.n_lines * 0.1        
        self._debug_log_decision(
            "step_end",
            threshold=self._threshold,
            step_costs=self._debug_step_cost_summary(),
        )
        self._middle_level_mode8_log_learning_summary_if_finished()
        self._first_level_mode2_log_learning_summary_if_finished()

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        self._record_contract_price_history(contract)
        self._record_sell_contract_partner(contract)
        contract_summary = self._debug_contract_summary(contract)
        if contract_summary.get("is_selling"):
            self._middle_level_mode5_advance_after_sell_response()
            self._middle_level_mode8_record_accepted_contract(contract_summary)
            self._first_level_mode2_record_accepted_contract(contract_summary)
        self._debug_log_decision(
            "negotiation_success",
            contract=contract_summary,
        )
        return super().on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        result = super().on_negotiation_failure(partners, annotation, mechanism, state)
        self._debug_log_decision(
            "negotiation_failure",
            partners=list(partners) if partners is not None else None,
            annotation=dict(annotation or {}),
            state=self._debug_value(state),
        )
        return result

    def first_proposals(self):
        level = self._level_label()
        if level == "first":
            proposals = self._first_level_first_proposals()
        elif level == "last":
            proposals = self._last_level_first_proposals()
        else:
            proposals = self._middle_level_first_proposals()
        self._debug_log_decision(
            "first_proposals",
            proposals=self._debug_offer_dict(proposals),
        )
        return proposals

    def _first_level_first_proposals(self):
        if self._first_level_sell_mode in (2, 3):
            self._debug_log_decision(
                "first_level_first_proposals_route",
                first_level_sell_mode=self._first_level_sell_mode,
                route="first_level_mode2_family",
            )
            partners = [
                partner
                for partner in self.negotiators.keys()
                if self.is_consumer(partner)
            ]
            return self._first_level_mode2_sell_offers(partners)
        if self._first_level_sell_mode != 0:
            self._debug_log_decision(
                "first_level_first_proposals_route",
                first_level_sell_mode=self._first_level_sell_mode,
                route="new_first_level",
            )
            return self._new_first_level_first_proposals()
        self._debug_log_decision(
            "first_level_first_proposals_route",
            first_level_sell_mode=self._first_level_sell_mode,
            route="legacy",
        )
        return self._legacy_first_proposals()

    def _new_first_level_first_proposals(self):
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()
        current_offers = {}
        future_supplier_partners = []
        future_consumer_partners = []

        for partner, quantity in distribution.items():
            if quantity > 0:
                current_offers[partner] = (quantity, s, self.best_price(partner))
            elif self.is_supplier(partner):
                future_supplier_partners.append(partner)
            elif self.is_consumer(partner):
                future_consumer_partners.append(partner)

        response = {}
        response |= current_offers
        response |= self.future_supplie_offer(future_supplier_partners)
        response |= self._new_first_level_future_consumer_offers(
            future_consumer_partners
        )
        return response

    def _new_first_level_future_consumer_offers(self, partners):
        partners = self._filter_sell_targets(partners)
        if self._first_level_staggered_future_sell == 0:
            return self.future_consume_offer(partners)
        if self._first_level_staggered_future_sell == 2:
            return self._new_first_level_low_sales_day_future_consumer_offers(partners)
        return self._new_first_level_staggered_future_consumer_offers(partners)

    def _first_level_future_sell_quantity(self, step: int) -> int:
        exogenous_input_average = self._calc_first_level_average_exogenous_input()
        current_output_inventory = int(
            self._awi_value("current_inventory_output", 0) or 0
        )
        already_sold = self._call_awi("total_sales_at", step, default=0) or 0
        target_quantity = exogenous_input_average + current_output_inventory - already_sold
        return int(
            min(
                self.awi.n_lines,
                max(0.0, target_quantity) * random.uniform(0.8, 1.0),
            )
        )

    def _new_first_level_staggered_future_consumer_offers(self, partners):
        response = {}
        awi = self.awi
        for offset, partner in enumerate(partners, start=1):
            step = awi.current_step + offset
            if step >= awi.n_steps:
                continue
            if not self._future_sell_step_allowed_by_bankruptcy_forecast(partner, step):
                continue
            quantity = self._first_level_future_sell_quantity(step)
            if quantity <= 0:
                continue
            response[partner] = (quantity, step, self.best_price(partner))
        return response

    def _new_first_level_low_sales_day_future_consumer_offers(self, partners):
        response = {}
        steps = self._first_level_sorted_future_sell_steps()
        used_steps: set[int] = set()
        for partner in partners:
            partner_steps = self._filter_future_sell_steps_by_bankruptcy_forecast(
                partner,
                [step for step in steps if step not in used_steps],
            )
            if not partner_steps:
                continue
            step = partner_steps[0]
            quantity = self._first_level_future_sell_quantity(step)
            if quantity <= 0:
                continue
            response[partner] = (quantity, step, self.best_price(partner))
            used_steps.add(step)
        return response

    def _first_level_sorted_future_sell_steps(self) -> list[int]:
        awi = self.awi
        min_time, max_time = self._time_issue_bounds_for_consumer_future_sell()
        min_step = max(awi.current_step + 1, min_time)
        max_step = min(awi.n_steps - 1, max_time)
        if min_step > max_step:
            return []
        steps = list(range(min_step, max_step + 1))
        return sorted(
            steps,
            key=lambda step: (
                self._call_awi("total_sales_at", step, default=0) or 0,
                step,
            ),
        )

    def _time_issue_bounds_for_consumer_future_sell(self) -> tuple[int, int]:
        issues = self._awi_value("current_output_issues", None)
        if issues is None:
            return self.awi.current_step + 1, self.awi.n_steps - 1
        try:
            issue = issues[TIME]
            return int(issue.min_value), int(issue.max_value)
        except Exception:
            return self.awi.current_step + 1, self.awi.n_steps - 1

    def _first_level_mode2_enabled(self) -> bool:
        return self._level_label() == "first" and self._first_level_sell_mode in (2, 3)

    def _first_level_mode3_enabled(self) -> bool:
        return self._level_label() == "first" and self._first_level_sell_mode == 3

    def _first_level_mode2_quantity_grid(self) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        return sorted({max(1, min(n_lines, quantity)) for quantity in (1, 2, 5, 8, 10)})

    def _first_level_mode3_quantity_grid(self) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        return sorted({max(1, min(n_lines, quantity)) for quantity in (1, 3, 5, 7, 10)})

    def _first_level_mode2_active_quantity_grid(self) -> list[int]:
        if self._first_level_mode3_enabled():
            return self._first_level_mode3_quantity_grid()
        return self._first_level_mode2_quantity_grid()

    def _first_level_mode2_high_quantity_grid(self) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        return sorted({max(1, min(n_lines, quantity)) for quantity in (8, 10)})

    def _first_level_mode3_high_quantity_grid(self) -> list[int]:
        n_lines = max(1, int(self._awi_value("n_lines", 1) or 1))
        return sorted({max(1, min(n_lines, quantity)) for quantity in (7, 10)})

    def _first_level_mode2_active_high_quantity_grid(self) -> list[int]:
        if self._first_level_mode3_enabled():
            return self._first_level_mode3_high_quantity_grid()
        return self._first_level_mode2_high_quantity_grid()

    def _first_level_mode2_price_grid(self, partner) -> list[int]:
        return self._middle_level_mode8_price_grid(partner)

    def _first_level_mode2_phase(self) -> str:
        if not self._first_level_mode2_enabled():
            return "inactive"
        current_input_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        n_lines = int(self._awi_value("n_lines", 0) or 0)
        current_step = int(self._awi_value("current_step", 0) or 0)
        if self._first_level_mode2_phase_start_step is None:
            if current_input_inventory < n_lines:
                return "inventory_fill"
            self._first_level_mode2_phase_start_step = current_step
        elapsed = current_step - int(self._first_level_mode2_phase_start_step)
        if elapsed < self._middle_level_mode8_exploration_steps():
            return "high_price_probe"
        if elapsed < self._middle_level_mode8_exploration_steps() + 10:
            return "counter_probe"
        return "classified"

    def _first_level_mode2_counter_probe_bounds(self) -> tuple[int, int] | None:
        if self._first_level_mode2_phase_start_step is None:
            return None
        start = (
            int(self._first_level_mode2_phase_start_step)
            + self._middle_level_mode8_exploration_steps()
        )
        return start, start + 10

    def _first_level_mode2_record_sell_proposal(self, partner, offer) -> None:
        if (
            not self._first_level_mode2_enabled()
            or offer is None
            or not self.is_consumer(partner)
        ):
            return
        try:
            quantity, step, price = (
                int(offer[QUANTITY]),
                int(offer[TIME]),
                int(offer[UNIT_PRICE]),
            )
        except Exception:
            return
        self._first_level_mode2_sell_offer_history.setdefault(str(partner), []).append(
            {
                "quantity": quantity,
                "time": step,
                "price": price,
                "step": int(self._awi_value("current_step", 0) or 0),
                "phase": self._first_level_mode2_phase(),
                "accepted": False,
            }
        )

    def _first_level_mode2_record_accepted_contract(
        self, contract_summary: dict[str, object]
    ) -> None:
        if not self._first_level_mode2_enabled():
            return
        partner = contract_summary.get("partner_id")
        if partner is None:
            return
        try:
            quantity = int(contract_summary.get("quantity"))
            step = int(contract_summary.get("time"))
            price = int(contract_summary.get("unit_price"))
        except Exception:
            return
        history = self._first_level_mode2_sell_offer_history.get(str(partner), [])
        for record in reversed(history):
            if record.get("accepted"):
                continue
            if (
                int(record.get("quantity", -1)) == quantity
                and int(record.get("time", -1)) == step
                and int(record.get("price", -1)) == price
            ):
                record["accepted"] = True
                record["accepted_at_step"] = int(self._awi_value("current_step", 0) or 0)
                return

    def _first_level_mode2_history(self, partner) -> list[dict[str, object]]:
        return self._first_level_mode2_sell_offer_history.get(str(partner), [])

    def _first_level_mode2_accepted_points(self, partner) -> list[tuple[int, int]]:
        return [
            (int(record["quantity"]), int(record["price"]))
            for record in self._first_level_mode2_history(partner)
            if record.get("accepted")
        ]

    def _first_level_mode2_has_accepted_full_quantity(self, partner) -> bool:
        n_lines = int(self._awi_value("n_lines", 0) or 0)
        return any(
            quantity >= n_lines
            for quantity, _price in self._first_level_mode2_accepted_points(partner)
        )

    def _first_level_mode2_is_high_price_record(
        self, partner, record: dict[str, object]
    ) -> bool:
        try:
            quantity = int(record.get("quantity", 0))
            price = int(record.get("price", 0))
            phase = str(record.get("phase", ""))
        except Exception:
            return False
        price_grid = self._first_level_mode2_price_grid(partner)
        return (
            phase in {"inventory_fill", "high_price_probe"}
            and quantity >= min(self._first_level_mode2_active_high_quantity_grid())
            and bool(record.get("accepted"))
            and bool(price_grid)
            and price >= price_grid[-1]
        )

    def _first_level_mode2_has_high_price_acceptance(self, partner) -> bool:
        return any(
            self._first_level_mode2_is_high_price_record(partner, record)
            for record in self._first_level_mode2_history(partner)
        )

    def _first_level_mode2_counter_probe_accepted(self, partner) -> bool:
        bounds = self._first_level_mode2_counter_probe_bounds()
        if bounds is None:
            return False
        start, end = bounds
        for record in self._first_level_mode2_history(partner):
            try:
                proposed_at = int(record.get("step", -1))
            except Exception:
                continue
            if start <= proposed_at < end and bool(record.get("accepted")):
                return True
        return False

    def _first_level_mode2_partner_classification(self, partner) -> str:
        phase = self._first_level_mode2_phase()
        if self._first_level_mode2_has_high_price_acceptance(partner):
            return "high_price_acceptor"
        if phase in {"inventory_fill", "high_price_probe"}:
            return "high_price_probe_pending"
        if phase == "counter_probe":
            if self._first_level_mode2_counter_probe_accepted(partner):
                return "counter_acceptor"
            return "counter_probe_pending"
        if self._first_level_mode2_counter_probe_accepted(partner):
            return "counter_acceptor"
        return "counter_non_acceptor"

    def _first_level_mode2_exploration_probability(self, partner) -> float:
        accepted_count = len(self._first_level_mode2_accepted_points(partner))
        return 1.0 / (1.0 + max(0, accepted_count))

    def _first_level_mode2_pareto_points(self, partner) -> list[tuple[int, int]]:
        best_by_quantity: dict[int, int] = {}
        for quantity, price in self._first_level_mode2_accepted_points(partner):
            best_by_quantity[quantity] = max(price, best_by_quantity.get(quantity, price))
        points = sorted(best_by_quantity.items())
        pareto = []
        for quantity, price in points:
            dominated = any(
                other_q >= quantity
                and other_p >= price
                and (other_q > quantity or other_p > price)
                for other_q, other_p in points
            )
            if not dominated:
                pareto.append((quantity, price))
        return sorted(pareto)

    def _first_level_mode2_near_pareto_point(
        self, partner, quantity: int, price: int
    ) -> bool:
        pareto = self._first_level_mode2_pareto_points(partner)
        if not pareto:
            return False
        quantity_grid = self._first_level_mode2_active_quantity_grid()
        price_grid = self._first_level_mode2_price_grid(partner)
        q_index = min(
            range(len(quantity_grid)),
            key=lambda index: abs(quantity_grid[index] - quantity),
        )
        p_index = min(
            range(len(price_grid)),
            key=lambda index: abs(price_grid[index] - price),
        )
        for frontier_q, frontier_p in pareto:
            fq_index = min(
                range(len(quantity_grid)),
                key=lambda index: abs(quantity_grid[index] - frontier_q),
            )
            fp_index = min(
                range(len(price_grid)),
                key=lambda index: abs(price_grid[index] - frontier_p),
            )
            if (
                abs(q_index - fq_index) <= self._middle_level_mode8_offer_distance()
                and abs(p_index - fp_index) <= self._middle_level_mode8_offer_distance()
            ):
                return True
        return False

    def _first_level_mode2_future_steps(self) -> list[int]:
        awi = self.awi
        min_time, max_time = self._time_issue_bounds_for_consumer_future_sell()
        min_step = max(int(awi.current_step) + 1, int(min_time))
        max_step = min(int(awi.n_steps) - 1, int(max_time))
        if not self._first_level_mode3_enabled():
            max_step = min(max_step, int(awi.current_step) + 9)
        if min_step > max_step:
            return []
        return list(range(min_step, max_step + 1))

    def _first_level_mode2_recorded_sales_at(self, step: int) -> int:
        if self._first_level_mode3_enabled():
            return self._active_sold_quantity_at(step)
        return int(self._call_awi("total_sales_at", step, default=0) or 0)

    def _first_level_mode2_remaining_sell_capacity(
        self, step: int, planned_quantities: Counter | None = None
    ) -> int:
        planned = 0 if planned_quantities is None else planned_quantities[step]
        already_sold = self._first_level_mode2_recorded_sales_at(step)
        return max(0, int(self.awi.n_lines - already_sold - planned))

    def _first_level_mode2_pick_sell_step(
        self,
        candidate_steps: list[int],
        planned_quantities: Counter,
        target_quantity: int | None = None,
        prefer_far: bool = False,
        randomize: bool = False,
    ) -> int | None:
        available_steps = [
            step
            for step in candidate_steps
            if self._first_level_mode2_remaining_sell_capacity(step, planned_quantities) > 0
        ]
        if not available_steps:
            return None
        if target_quantity is not None and target_quantity > 0:
            full_capacity_steps = [
                step
                for step in available_steps
                if self._first_level_mode2_remaining_sell_capacity(
                    step, planned_quantities
                )
                >= target_quantity
            ]
            if full_capacity_steps:
                available_steps = full_capacity_steps
        if randomize:
            return random.choice(available_steps)
        ranked = sorted(
            available_steps,
            key=lambda step: (
                -step if prefer_far else (
                    self._first_level_mode2_recorded_sales_at(step)
                ),
                step,
            ),
        )
        return random.choice(ranked[: min(2, len(ranked))])

    def _first_level_mode2_offer_candidates(self, partner) -> list[tuple[int, int]]:
        price_grid = self._first_level_mode2_price_grid(partner)
        if not price_grid:
            return []
        phase = self._first_level_mode2_phase()
        classification = self._first_level_mode2_partner_classification(partner)
        if (
            classification == "high_price_acceptor"
            or phase in {"inventory_fill", "high_price_probe"}
        ):
            quantity_grid = (
                [int(self._awi_value("n_lines", 1) or 1)]
                if self._first_level_mode2_has_accepted_full_quantity(partner)
                else self._first_level_mode2_active_high_quantity_grid()
            )
            return [(quantity, price_grid[-1]) for quantity in quantity_grid]
        if phase == "counter_probe":
            return [
                (quantity, price_grid[0])
                for quantity in self._first_level_mode2_active_quantity_grid()
            ]
        if classification == "counter_acceptor":
            price_candidates = price_grid[: min(2, len(price_grid))]
            explore = (
                random.random()
                < self._first_level_mode2_exploration_probability(partner)
            )
            candidates = []
            for quantity in self._first_level_mode2_active_quantity_grid():
                for price in price_candidates:
                    if explore or self._first_level_mode2_near_pareto_point(
                        partner, quantity, price
                    ):
                        candidates.append((quantity, price))
            if not candidates:
                candidates = [
                    (quantity, price)
                    for quantity in self._first_level_mode2_active_quantity_grid()
                    for price in price_candidates
                ]
            return candidates
        if classification == "counter_non_acceptor":
            return [
                (quantity, price)
                for quantity in self._first_level_mode2_active_quantity_grid()
                for price in price_grid
            ]
        return [
            (quantity, price_grid[0])
            for quantity in self._first_level_mode2_active_quantity_grid()
        ]

    def _first_level_mode2_preserve_profile_price(self, partner) -> bool:
        return (
            self._first_level_mode2_phase() == "classified"
            and self._first_level_mode2_partner_classification(partner)
            == "counter_acceptor"
        )

    def _first_level_mode2_offer_price_for_quantity(
        self, partner, quantity: int, fallback_price: int
    ) -> int:
        if not self._first_level_mode3_enabled():
            return int(fallback_price)
        if self._first_level_mode2_preserve_profile_price(partner):
            return int(fallback_price)
        candidates = self._first_level_mode2_offer_candidates(partner)
        exact_prices = [price for q, price in candidates if q == quantity]
        if exact_prices:
            return max(exact_prices)
        return int(fallback_price)

    def _first_level_mode2_partner_offer_profile(
        self, partner
    ) -> tuple[int, int] | None:
        candidates = self._first_level_mode2_offer_candidates(partner)
        if not candidates:
            return None
        return random.choice(candidates)

    def _first_level_mode2_sell_offers(
        self,
        partners,
        planned_quantities: Counter | None = None,
    ) -> dict[str, tuple[int, int, int]]:
        partners = self._filter_sell_targets([p for p in partners if self.is_consumer(p)])
        planned_quantities = Counter() if planned_quantities is None else planned_quantities
        phase = self._first_level_mode2_phase()
        candidate_steps = self._first_level_mode2_future_steps()
        profiles = []
        for partner in partners:
            profile = self._first_level_mode2_partner_offer_profile(partner)
            if profile is None:
                continue
            quantity, price = profile
            profiles.append((partner, quantity, price))
        profiles.sort(key=lambda item: item[1], reverse=True)

        response = {}
        for partner, quantity, price in profiles:
            classification = self._first_level_mode2_partner_classification(partner)
            partner_candidate_steps = self._filter_future_sell_steps_by_bankruptcy_forecast(
                partner, candidate_steps
            )
            step = self._first_level_mode2_pick_sell_step(
                partner_candidate_steps,
                planned_quantities,
                int(quantity),
                prefer_far=phase == "inventory_fill",
                randomize=classification == "counter_non_acceptor",
            )
            if step is None:
                continue
            offer_quantity = min(
                int(quantity),
                self._first_level_mode2_remaining_sell_capacity(step, planned_quantities),
            )
            if offer_quantity <= 0:
                continue
            price = self._first_level_mode2_offer_price_for_quantity(
                partner, offer_quantity, price
            )
            offer = (offer_quantity, step, int(price))
            response[partner] = offer
            planned_quantities[step] += offer_quantity
            self._first_level_mode2_record_sell_proposal(partner, offer)
        return response

    def _first_level_mode2_accepts_initial_sell_offer(
        self, partner, offer, planned_quantities: Counter | None = None
    ) -> bool:
        del partner
        try:
            quantity = int(offer[QUANTITY])
            step = int(offer[TIME])
            price = int(offer[UNIT_PRICE])
        except Exception:
            return False
        return (
            quantity >= min(self._first_level_mode2_active_high_quantity_grid())
            and step >= int(self.awi.current_step) + 6
            and step < int(self.awi.n_steps)
            and self._future_sell_offer_within_allowable_excess(
                step, quantity, price, planned_quantities
            )
        )

    def _first_level_mode2_accepts_offer(
        self, partner, offer, planned_quantities: Counter, state=None
    ) -> bool:
        del state
        try:
            quantity = int(offer[QUANTITY])
            step = int(offer[TIME])
            price = int(offer[UNIT_PRICE])
        except Exception:
            return False
        if step < self.awi.current_step or step >= self.awi.n_steps or quantity <= 0:
            return False
        phase = self._first_level_mode2_phase()
        if phase == "inventory_fill":
            if step == int(self.awi.current_step):
                return False
            return self._first_level_mode2_accepts_initial_sell_offer(
                partner, offer, planned_quantities
            )

        classification = self._first_level_mode2_partner_classification(partner)
        is_current_step_offer = step == int(self.awi.current_step)
        if (
            not is_current_step_offer
            and not self._future_sell_offer_within_allowable_excess(
                step, quantity, price, planned_quantities
            )
        ):
            return False

        score_improves = (
            self._middle_level_mode8_sell_score_delta_at(
                step, quantity, price, planned_quantities
            )
            > 0
        )
        if phase == "classified":
            if classification == "counter_acceptor" and quantity <= 2:
                return False
            if classification in {"counter_acceptor", "counter_non_acceptor"}:
                return score_improves
        if is_current_step_offer:
            return score_improves
        return (
            score_improves
            and self._future_sell_offer_within_allowable_excess(
                step, quantity, price, planned_quantities
            )
        )

    def _first_level_mode2_counter_all(self, offers, states):
        response = {}
        planned_quantities = Counter()
        consumers = [
            partner
            for partner in self.awi.my_consumers
            if partner in self.negotiators.keys()
        ]
        remaining_partners = []
        for partner in consumers:
            offer = offers.get(partner)
            if offer is None:
                if partner in offers:
                    remaining_partners.append(partner)
                continue
            if not self.is_valid_price(offer[UNIT_PRICE], partner):
                remaining_partners.append(partner)
                continue
            if self._first_level_mode2_accepts_offer(
                partner,
                offer,
                planned_quantities,
                None if states is None else states.get(partner),
            ):
                response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                planned_quantities[int(offer[TIME])] += int(offer[QUANTITY])
            else:
                remaining_partners.append(partner)

        remaining_partners = [
            partner
            for partner in consumers
            if partner not in response.keys()
        ]
        for partner, offer in self._first_level_mode2_sell_offers(
            remaining_partners,
            planned_quantities,
        ).items():
            response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        for partner in remaining_partners:
            if partner not in response:
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)

        for partner, offer in offers.items():
            if partner not in response and offer is not None:
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return response

    def _first_level_mode2_classification_summary(self) -> dict[str, object]:
        partners = sorted(
            set(self._first_level_mode2_sell_offer_history.keys())
            | set(str(partner) for partner in self.awi.my_consumers)
        )
        by_partner = {
            partner: self._first_level_mode2_partner_classification(partner)
            for partner in partners
        }
        return {
            "phase": self._first_level_mode2_phase(),
            "phase_start_step": self._first_level_mode2_phase_start_step,
            "by_partner": by_partner,
            "counts": dict(Counter(by_partner.values())),
        }

    def _first_level_mode2_log_learning_summary_if_finished(self) -> None:
        if (
            self._first_level_mode2_learning_summary_logged
            or not self._first_level_mode2_enabled()
        ):
            return
        current_step = int(self._awi_value("current_step", 0) or 0)
        n_steps = int(self._awi_value("n_steps", 0) or 0)
        if n_steps <= 0 or current_step < n_steps - 1:
            return
        self._first_level_mode2_learning_summary_logged = True
        summary = self._first_level_mode2_classification_summary()
        self.log_message(
            "FIRST_MODE2_FINAL_CLASSIFICATION",
            f"agent_id={self._my_agent_id()}",
            f"step={current_step}",
            f"phase={summary.get('phase')}",
            f"phase_start={summary.get('phase_start_step')}",
            f"counts={summary.get('counts')}",
        )
        grouped: dict[str, list[str]] = {}
        by_partner = summary.get("by_partner", {})
        if isinstance(by_partner, dict):
            for partner, status in by_partner.items():
                grouped.setdefault(str(status), []).append(str(partner))
        for status in (
            "high_price_acceptor",
            "counter_acceptor",
            "counter_non_acceptor",
            "counter_probe_pending",
            "high_price_probe_pending",
        ):
            partners = sorted(grouped.get(status, []))
            self.log_message(
                "FIRST_MODE2_FINAL_CLASSIFICATION_DETAIL",
                f"status={status}",
                f"count={len(partners)}",
                f"partners={','.join(partners) if partners else '-'}",
            )
        self._debug_log_decision(
            "first_mode2_learning_summary",
            classification=summary,
            history=self._first_level_mode2_sell_offer_history,
        )

    def _middle_level_first_proposals(self):
        if self._middle_level_negotiation_mode == 0:
            return self._legacy_first_proposals()
        response = {}
        response.update(self._middle_level_edge_first_proposals("supplier"))
        response.update(self._middle_level_edge_first_proposals("consumer"))
        return response

    def _middle_level_edge_first_proposals(self, edge):
        if self._middle_level_edge_mode(edge) == 0:
            return self._middle_level_legacy_edge_first_proposals(edge)
        if self._middle_level_mode5_family():
            phase = self._middle_level_mode5_current_phase()
            if phase == "initial_sell_probe":
                if edge == "supplier":
                    return {}
                partners = [
                    p for p in self.negotiators.keys() if self.is_consumer(p)
                ]
                partners = self._filter_sell_targets(partners)
                quantity_fn = (
                    self._middle_level_mode8_fill_sell_quantity
                    if self._middle_level_mode8_family()
                    else lambda step: self._middle_level_mode5_initial_sell_quantity()
                )
                return self._middle_level_mode5_far_consumer_offers(
                    partners, quantity_fn
                )
            if edge == "consumer":
                if not self._middle_level_mode5_future_sell_enabled():
                    return {}
                partners = [
                    p for p in self.negotiators.keys() if self.is_consumer(p)
                ]
                partners = self._filter_sell_targets(partners)
                if self._middle_level_mode8_family() and phase == "normal":
                    return self._middle_level_mode8_sell_offers(partners)
                if self._middle_level_mode8_family():
                    return self._middle_level_mode5_far_consumer_offers(
                        partners, self._middle_level_mode8_fill_sell_quantity
                    )
                return self._middle_level_mode5_far_consumer_offers(
                    partners, self._middle_level_mode5_future_sell_quantity
                )
        return self._middle_level_expected_edge_first_proposals(edge)

    def _middle_level_legacy_edge_first_proposals(self, edge):
        partners = [
            p
            for p in self.negotiators.keys()
            if (self.is_supplier(p) if edge == "supplier" else self.is_consumer(p))
        ]
        if edge == "consumer":
            partners = self._filter_sell_targets(partners)
        distribution = self.distribute_todays_needs(partners)

        response = {}
        future_partners = []
        for partner, quantity in distribution.items():
            if quantity > 0:
                price = self.best_price(partner)
                if edge == "supplier" and self._middle_level_mode8_family():
                    price = self._middle_level_mode8_buy_price(partner)
                response[partner] = (
                    quantity,
                    self.awi.current_step,
                    price,
                )
            else:
                future_partners.append(partner)

        if edge == "supplier":
            response.update(self.future_supplie_offer(future_partners))
        else:
            response.update(self.future_consume_offer(future_partners))
        return response

    def _middle_level_expected_edge_first_proposals(self, edge):
        partners = [
            p
            for p in self.negotiators.keys()
            if (self.is_supplier(p) if edge == "supplier" else self.is_consumer(p))
        ]
        if edge == "consumer":
            partners = self._filter_sell_targets(partners)

        distribution = dict(zip(partners, repeat(0)))
        needs = self._middle_level_edge_need(edge, self.awi.current_step)
        if partners and needs > 0:
            distribution |= self._middle_level_distribute_edge_needs(
                edge, partners, needs
            )

        response = {}
        future_partners = []
        for partner, quantity in distribution.items():
            if quantity > 0:
                response[partner] = (
                    quantity,
                    self.awi.current_step,
                    self.best_price(partner),
                )
            else:
                future_partners.append(partner)

        response.update(self._middle_level_future_edge_offers(edge, future_partners))
        return response

    def _last_level_first_proposals(self):
        if self._last_level_negotiation_mode == 0:
            return self._legacy_first_proposals()
        return self._last_level_buffer_first_proposals()

    def _last_level_buffer_first_proposals(self):
        suppliers = [p for p in self.negotiators.keys() if self.is_supplier(p)]
        distribution = dict(zip(suppliers, repeat(0)))
        needs = self._supplier_need_at(self.awi.current_step)
        if suppliers and needs > 0:
            distribution |= self._last_level_distribute_supplier_needs(
                suppliers, needs
            )

        response = {}
        future_supplier_partners = []
        for partner, quantity in distribution.items():
            if quantity > 0:
                response[partner] = (
                    quantity,
                    self.awi.current_step,
                    self._last_level_supplier_first_price(partner),
                )
            elif not self._last_level_uses_current_supplier_only_mode():
                future_supplier_partners.append(partner)
        if not self._last_level_uses_current_supplier_only_mode():
            response |= self._last_level_future_supplier_offers(
                future_supplier_partners
            )
        return response

    def _legacy_first_proposals(self):
                                                                                      
        self.negotiators.keys()
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

                                       
        first_dict = dict()
        future_supplie_partner = []
        future_consume_partner = []

        for k, q in distribution.items():
            if q > 0:
                first_dict[k] = (q, s, self.best_price(k))
            elif self.is_supplier(k):
                future_supplie_partner.append(k)
            elif self.is_consumer(k):
                future_consume_partner.append(k)

        response = dict()
        future_supplie_dict = dict()
        future_consume_dict = dict()
        response |= first_dict
        future_supplie_dict |= self.future_supplie_offer(future_supplie_partner)
        response |= future_supplie_dict
        future_consume_dict |= self.future_consume_offer(future_consume_partner)
        response |= future_consume_dict

        return response

    def counter_all(self, offers, states):
        self._debug_log_decision(
            "counter_all_incoming",
            offers=self._debug_offer_dict(offers),
            states=self._debug_value(states),
        )
        level = self._level_label()
        if level == "first":
            responses = self._first_level_counter_all(offers, states)
        elif level == "last":
            responses = self._last_level_counter_all(offers, states)
        else:
            responses = self._middle_level_counter_all(offers, states)
        self._debug_log_decision(
            "counter_all_responses",
            responses=self._debug_response_dict(responses),
        )
        return responses

    def _first_level_counter_all(self, offers, states):
        if self._first_level_sell_mode in (2, 3):
            self._debug_log_decision(
                "first_level_counter_all_route",
                first_level_sell_mode=self._first_level_sell_mode,
                route="first_level_mode2_family",
            )
            return self._first_level_mode2_counter_all(offers, states)
        if self._first_level_sell_mode != 0:
            self._debug_log_decision(
                "first_level_counter_all_route",
                first_level_sell_mode=self._first_level_sell_mode,
                route="new_first_level",
            )
            return self._new_first_level_counter_all(offers, states)
        self._debug_log_decision(
            "first_level_counter_all_route",
            first_level_sell_mode=self._first_level_sell_mode,
            route="legacy",
        )
        return self._legacy_counter_all(offers, states)

    def _new_first_level_counter_all(self, offers, states):
        response = dict()
        awi = self.awi
        for edge_needs, all_partners, issues in [
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
            needs = 0
            if self.is_supplier(all_partners[0]):
                needs = self._legacy_supplier_need_at(awi.current_step)
                if edge_needs <= 0:
                    needs = 0
            elif self.is_consumer(all_partners[0]):
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = self._legacy_consumer_need_at(awi.current_step)

            partners = {_ for _ in all_partners if _ in offers.keys()}
            current_step_offers = {}
            future_step_offers = {}

            for partner in partners:
                offer = offers[partner]
                if offer is None:
                    continue
                if offer[TIME] == self.awi.current_step and self.is_valid_price(
                    offer[UNIT_PRICE], partner
                ):
                    current_step_offers[partner] = offer
                elif offer[TIME] != self.awi.current_step and self.is_valid_price(
                    offer[UNIT_PRICE], partner
                ):
                    future_step_offers[partner] = offer

            current_step_partners = {
                partner for partner in partners if partner in current_step_offers.keys()
            }

            duplicate_list = [0 for _ in range(awi.n_steps)]
            for partner, offer in future_step_offers.items():
                step = offer[TIME]
                if step <= awi.n_steps:
                    if (
                        offer[QUANTITY] + duplicate_list[step - 1]
                        <= self.needs_at(step, partner)
                    ):
                        response[partner] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, offer
                        )
                        duplicate_list[step - 1] += offer[QUANTITY]

            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))
            best_indx = -1
            for i, partner_ids in enumerate(plist):
                offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if offered - needs >= 0:
                    if diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = diff, i
                else:
                    if diff < best_minus_diff and offered > 0:
                        best_minus_diff, best_index_minus = diff, i

            has_accept_offer = True
            if (
                best_index_plus >= 0
                and best_plus_diff <= self._threshold
                and len(plist[best_index_plus]) > 0
            ):
                best_indx = best_index_plus
            elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
                best_indx = best_index_minus
            else:
                has_accept_offer = False

            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = plist[best_indx]
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))
                other_suppliers = []
                other_consumers = []
                response.update(
                    {
                        partner: SAOResponse(
                            ResponseType.ACCEPT_OFFER, current_step_offers[partner]
                        )
                        for partner in partner_ids
                    }
                )
                for partner in others:
                    if self.is_supplier(partner):
                        other_suppliers.append(partner)
                    if self.is_consumer(partner):
                        other_consumers.append(partner)

                for partner, offer in self.future_supplie_offer(other_suppliers).items():
                    response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

                for partner, offer in self._new_first_level_future_consumer_offers(
                    other_consumers
                ).items():
                    response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

                flag = 1

            if flag != 1:
                other_partners = {
                    partner
                    for partner in all_partners
                    if partner not in response.keys()
                    and partner in self.negotiators.keys()
                }
                distribution = self.distribute_todays_needs(other_partners)
                future_supplier_partners = []
                future_consumer_partners = []

                for partner, quantity in distribution.items():
                    if quantity > 0:
                        response[partner] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (quantity, self.awi.current_step, self.price(partner)),
                        )
                    elif self.is_supplier(partner):
                        future_supplier_partners.append(partner)
                    elif self.is_consumer(partner):
                        future_consumer_partners.append(partner)

                for partner, offer in self.future_supplie_offer(
                    future_supplier_partners
                ).items():
                    response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

                for partner, offer in self._new_first_level_future_consumer_offers(
                    future_consumer_partners
                ).items():
                    response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        return response

    def _middle_level_counter_all(self, offers, states):
        if self._middle_level_negotiation_mode == 0:
            return self._legacy_counter_all(offers, states)
        response = {}
        for edge in ("supplier", "consumer"):
            if self._middle_level_edge_mode(edge) == 0:
                self._middle_level_legacy_update_edge_responses(
                    edge, offers, response
                )
            else:
                self._middle_level_update_edge_responses(edge, offers, response, states)
        return response

    def _last_level_counter_all(self, offers, states):
        if self._last_level_negotiation_mode == 0:
            return self._legacy_counter_all(offers, states)
        return self._last_level_buffer_counter_all(offers, states)

    def _last_level_buffer_counter_all(self, offers, states):
        response = {}
        awi = self.awi
        suppliers = [
            p
            for p in awi.my_suppliers
            if p in self.negotiators.keys() and p in offers.keys()
        ]

        current_step_offers = {}
        future_step_offers = {}
        for partner in suppliers:
            offer = offers[partner]
            if offer is None:
                continue
            if not self.is_valid_price(offer[UNIT_PRICE], partner):
                continue
            if offer[TIME] == awi.current_step:
                current_step_offers[partner] = offer
            else:
                future_step_offers[partner] = offer

        if self._last_level_negotiation_mode == 1:
            accepted_future_quantities = Counter()
            for partner, offer in future_step_offers.items():
                if self._input_inventory_cap_enabled():
                    response[partner] = SAOResponse(
                        ResponseType.END_NEGOTIATION, None
                    )
                    continue
                step = offer[TIME]
                if step < awi.current_step or step >= awi.n_steps:
                    continue
                remaining_need = (
                    self._supplier_need_at(step) - accepted_future_quantities[step]
                )
                if 0 < offer[QUANTITY] <= remaining_need:
                    response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    accepted_future_quantities[step] += offer[QUANTITY]

        current_need = self._supplier_need_at(awi.current_step)
        current_step_partners = {
            p for p in current_step_offers.keys() if p not in response.keys()
        }
        plist = list(powerset(current_step_partners))
        best_index_plus = -1
        best_plus_diff = float("inf")
        best_index_minus = -1
        best_minus_diff = float("inf")

        for i, partner_ids in enumerate(plist):
            offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
            diff = abs(offered - current_need)
            if offered - current_need >= 0:
                if (
                    (not self._input_inventory_cap_enabled() or offered <= current_need)
                    and diff < best_plus_diff
                    and current_need > 0
                ):
                    best_plus_diff, best_index_plus = diff, i
            else:
                if diff < best_minus_diff and offered > 0:
                    best_minus_diff, best_index_minus = diff, i

        best_indx = -1
        if (
            best_index_plus >= 0
            and best_plus_diff <= self._threshold
            and len(plist[best_index_plus]) > 0
        ):
            best_indx = best_index_plus
        elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
            best_indx = best_index_minus

        if best_indx >= 0 and current_need > 0:
            partner_ids = plist[best_indx]
            accepted_quantity = sum(
                current_step_offers[p][QUANTITY] for p in partner_ids
            )
            response.update(
                {
                    partner: SAOResponse(
                        ResponseType.ACCEPT_OFFER, current_step_offers[partner]
                    )
                    for partner in partner_ids
                }
            )
            remaining_need = max(0, current_need - accepted_quantity)
            if self._last_level_uses_current_supplier_only_mode():
                counter_partners = [
                    p
                    for p in awi.my_suppliers
                    if p in self.negotiators.keys() and p not in response.keys()
                ]
                self._last_level_add_current_supplier_counters(
                    response,
                    counter_partners,
                    remaining_need,
                )
            else:
                future_partners = [
                    p
                    for p in awi.my_suppliers
                    if p in self.negotiators.keys() and p not in response.keys()
                ]
                for partner, offer in self._last_level_future_supplier_offers(
                    future_partners
                ).items():
                    response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            return response

        remaining_partners = [
            p
            for p in awi.my_suppliers
            if p in self.negotiators.keys() and p not in response.keys()
        ]
        if self._last_level_uses_current_supplier_only_mode():
            self._last_level_add_current_supplier_counters(
                response, remaining_partners, current_need
            )
        else:
            distribution = dict(zip(remaining_partners, repeat(0)))
            if remaining_partners and current_need > 0:
                distribution |= self._last_level_distribute_supplier_needs(
                    remaining_partners, current_need
                )

            future_supplier_partners = []
            for partner, quantity in distribution.items():
                if quantity > 0:
                    response[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (
                            quantity,
                            awi.current_step,
                            self._last_level_supplier_counter_price(partner),
                        ),
                    )
                else:
                    future_supplier_partners.append(partner)

            for partner, offer in self._last_level_future_supplier_offers(
                future_supplier_partners
            ).items():
                response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        return response

    def _last_level_supplier_first_price(self, partner):
        if not self._adjust_last_level_proposal_price:
            return self.best_price(partner)
        return self._last_level_inventory_adjusted_supplier_price(partner)

    def _last_level_supplier_counter_price(self, partner):
        return self.price(partner)

    def _last_level_inventory_adjusted_supplier_price(self, partner):
        min_price, max_price = self._unit_price_bounds(partner)
        target_buffer = self._calc_last_level_inventory_buffer_quantity()
        if target_buffer <= 0:
            return min_price

        current_inventory = int(self._awi_value("current_inventory_input", 0) or 0)
        buffer_ratio = max(0.0, min(1.0, current_inventory / target_buffer))
        shortage_ratio = 1.0 - buffer_ratio
        price_range = max_price - min_price
        return min_price + price_range * shortage_ratio * 0.5

    def _last_level_distribute_supplier_needs(self, partners, needs) -> dict[str, int]:
        partners = list(partners)
        if not self._last_level_uses_current_supplier_only_mode():
            return self.distribute_todays_supplie_consume_needs(partners, needs)
        if not partners or needs <= 0:
            return dict(zip(partners, repeat(0)))
        random.shuffle(partners)
        return dict(zip(partners, distribute(needs, len(partners))))

    def _last_level_uses_current_supplier_only_mode(self) -> bool:
        return self._last_level_negotiation_mode in (2, 3, 4, 5)

    def _last_level_add_current_supplier_counters(
        self, response, partners, needs: int
    ) -> None:
        distribution = self._last_level_distribute_supplier_needs(partners, needs)
        for partner, quantity in distribution.items():
            if quantity > 0:
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (
                        quantity,
                        self.awi.current_step,
                        self._last_level_supplier_counter_price(partner),
                    ),
                )
            else:
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _last_level_future_supplier_offers(self, partners):
        if self._last_level_uses_current_supplier_only_mode():
            distribution = self._last_level_distribute_supplier_needs(
                partners, self._supplier_need_at(self.awi.current_step)
            )
            return {
                partner: (
                    quantity,
                    self.awi.current_step,
                    self._last_level_supplier_first_price(partner),
                )
                for partner, quantity in distribution.items()
                if quantity > 0
            }

        if self._input_inventory_cap_enabled():
            return {}

        partners = list(partners)
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = {}

        step1_partners = partners[: int(len(partners) * 0.5)]
        step2_partners = partners[int(len(partners) * 0.5) : int(len(partners) * 0.8)]
        step3_partners = partners[int(len(partners) * 0.8) :]

        for step, step_partners in [
            (s + 1, step1_partners),
            (s + 2, step2_partners),
            (s + 3, step3_partners),
        ]:
            if step >= n or not step_partners:
                continue
            step_needs = int(self._supplier_need_at(step) / 3)
            if step_needs <= 0:
                continue
            for partner, quantity in zip(
                step_partners, distribute(step_needs, len(step_partners))
            ):
                if quantity > 0:
                    response[partner] = (
                        quantity,
                        step,
                        self._last_level_supplier_first_price(partner),
                    )

        return response

    def _middle_level_expected_counter_all(self, offers, states):
        response = {}
        for edge in ("supplier", "consumer"):
            self._middle_level_update_edge_responses(edge, offers, response)
        return response

    def _middle_level_legacy_update_edge_responses(self, edge, offers, response) -> None:
        awi = self.awi
        is_supplier_edge = edge == "supplier"
        all_partners = awi.my_suppliers if is_supplier_edge else awi.my_consumers
        if not all_partners:
            return

        edge_needs = awi.needed_supplies if is_supplier_edge else awi.needed_sales
        needs = 0
        if is_supplier_edge:
            needs = self._legacy_supplier_need_at(awi.current_step)
            if edge_needs <= 0:
                needs = 0
        elif awi.total_sales_at(awi.current_step) <= awi.n_lines:
            needs = self._legacy_consumer_need_at(awi.current_step)

        partners = {_ for _ in all_partners if _ in offers.keys()}
        current_step_offers = {}
        future_step_offers = {}
        for partner in partners:
            offer = offers[partner]
            if offer is None:
                continue
            if not self.is_valid_price(offer[UNIT_PRICE], partner):
                continue
            if offer[TIME] == awi.current_step:
                current_step_offers[partner] = offer
            else:
                future_step_offers[partner] = offer

        current_step_partners = {
            partner for partner in partners if partner in current_step_offers.keys()
        }

        duplicate_list = [0 for _ in range(awi.n_steps)]
        for partner, offer in future_step_offers.items():
            step = offer[TIME]
            if step <= awi.n_steps:
                if (
                    offer[QUANTITY] + duplicate_list[step - 1]
                    <= self.needs_at(step, partner)
                ):
                    response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    duplicate_list[step - 1] += offer[QUANTITY]

        best_index_plus = -1
        best_plus_diff = float("inf")
        best_index_minus = -1
        best_minus_diff = float("inf")
        plist = list(powerset(current_step_partners))
        best_indx = -1
        for i, partner_ids in enumerate(plist):
            offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
            diff = abs(offered - needs)
            if offered - needs >= 0:
                if diff < best_plus_diff and needs > 0:
                    best_plus_diff, best_index_plus = diff, i
            else:
                if diff < best_minus_diff and offered > 0:
                    best_minus_diff, best_index_minus = diff, i

        has_accept_offer = True
        if (
            best_index_plus >= 0
            and best_plus_diff <= self._threshold
            and len(plist[best_index_plus]) > 0
        ):
            best_indx = best_index_plus
        elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
            best_indx = best_index_minus
        else:
            has_accept_offer = False

        flag = 0
        if has_accept_offer and needs > 0:
            partner_ids = plist[best_indx]
            others = list(current_step_partners.difference(partner_ids))
            others = list(set(others) - set(response.keys()))
            response.update(
                {
                    partner: SAOResponse(
                        ResponseType.ACCEPT_OFFER, current_step_offers[partner]
                    )
                    for partner in partner_ids
                }
            )
            for partner, offer in self._middle_level_legacy_future_edge_offers(
                edge, others
            ).items():
                response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            flag = 1

        if flag != 1:
            other_partners = {
                partner
                for partner in all_partners
                if partner not in response.keys() and partner in self.negotiators.keys()
            }
            distribution = self.distribute_todays_needs(other_partners)
            future_partners = []
            for partner, quantity in distribution.items():
                if quantity > 0:
                    response[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (quantity, awi.current_step, self.price(partner)),
                    )
                else:
                    future_partners.append(partner)
            for partner, offer in self._middle_level_legacy_future_edge_offers(
                edge, future_partners
            ).items():
                response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

    def _middle_level_legacy_future_edge_offers(self, edge, partners):
        if edge == "supplier":
            return self.future_supplie_offer(list(partners))
        return self.future_consume_offer(list(partners))

    def _middle_level_update_edge_responses(self, edge, offers, response, states=None) -> None:
        awi = self.awi
        is_supplier_edge = edge == "supplier"
        all_partners = awi.my_suppliers if is_supplier_edge else awi.my_consumers
        if (
            self._middle_level_mode8_active_sell_phase()
            and not is_supplier_edge
        ):
            self._middle_level_mode8_update_consumer_responses(offers, response, states)
            return
        if (
            self._level_label() == "middle"
            and self._middle_level_mode8_family()
            and not is_supplier_edge
        ):
            phase = self._middle_level_mode5_current_phase()
            if phase != "normal":
                self._middle_level_mode8_update_initial_consumer_responses(
                    offers, response, phase
                )
                return
        mode5_phase = None
        mode5_advance_after_response = False
        if (
            self._middle_level_mode5_family()
            and not is_supplier_edge
        ):
            mode5_phase = self._middle_level_mode5_current_phase()
        edge_partners = [
            partner
            for partner in all_partners
            if partner in self.negotiators.keys() and partner in offers.keys()
        ]

        current_step_offers = {}
        future_step_offers = {}
        for partner in edge_partners:
            offer = offers[partner]
            if offer is None:
                continue
            if not self.is_valid_price(offer[UNIT_PRICE], partner):
                continue
            if offer[TIME] == awi.current_step:
                current_step_offers[partner] = offer
            else:
                future_step_offers[partner] = offer
        if (
            mode5_phase == "initial_sell_probe"
            and (current_step_offers or future_step_offers)
        ):
            mode5_advance_after_response = True

        accepted_future_quantities = Counter()
        for partner, offer in future_step_offers.items():
            if (
                is_supplier_edge
                and (
                    self._middle_level_mode4_buy_cap
                    or self._middle_level_mode5_family()
                )
            ):
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            step = offer[TIME]
            if step < awi.current_step or step >= awi.n_steps:
                continue
            if (
                mode5_phase == "input_fill"
                and (
                    not self._middle_level_mode5_future_sell_enabled()
                    or not self._middle_level_mode5_is_late_consumer_step(
                        step, len(all_partners)
                    )
                )
            ):
                continue
            remaining_need = (
                self._middle_level_edge_need(edge, step)
                - accepted_future_quantities[step]
            )
            if (
                self._middle_level_negotiation_mode == 7
                and not is_supplier_edge
            ):
                remaining_capacity = (
                    self._middle_level_mode7_remaining_sell_capacity(step)
                    - accepted_future_quantities[step]
                )
                remaining_need = min(remaining_need, remaining_capacity)
            if 0 < offer[QUANTITY] <= remaining_need:
                response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                accepted_future_quantities[step] += offer[QUANTITY]

        current_need = self._middle_level_edge_need(edge, awi.current_step)
        if self._middle_level_negotiation_mode == 7 and not is_supplier_edge:
            current_need = min(
                current_need,
                self._middle_level_mode7_remaining_sell_capacity(awi.current_step),
            )
        current_step_partners = {
            partner
            for partner in current_step_offers.keys()
            if partner not in response.keys()
        }
        plist = list(powerset(current_step_partners))
        best_index_plus = -1
        best_plus_diff = float("inf")
        best_index_minus = -1
        best_minus_diff = float("inf")

        for i, partner_ids in enumerate(plist):
            offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
            diff = abs(offered - current_need)
            if offered - current_need >= 0:
                if (
                    (
                        not (
                            is_supplier_edge
                            and (
                                self._middle_level_mode4_buy_cap
                                or self._middle_level_mode5_family()
                            )
                        )
                        and not (
                            self._middle_level_negotiation_mode == 7
                            and not is_supplier_edge
                        )
                        or offered <= current_need
                    )
                    and diff < best_plus_diff
                    and current_need > 0
                ):
                    best_plus_diff, best_index_plus = diff, i
            else:
                if diff < best_minus_diff and offered > 0:
                    best_minus_diff, best_index_minus = diff, i

        best_indx = -1
        if (
            best_index_plus >= 0
            and best_plus_diff <= self._threshold
            and len(plist[best_index_plus]) > 0
        ):
            best_indx = best_index_plus
        elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
            best_indx = best_index_minus

        if best_indx >= 0 and current_need > 0:
            partner_ids = plist[best_indx]
            response.update(
                {
                    partner: SAOResponse(
                        ResponseType.ACCEPT_OFFER, current_step_offers[partner]
                    )
                    for partner in partner_ids
                }
            )
            future_partners = [
                partner
                for partner in all_partners
                if partner in self.negotiators.keys() and partner not in response.keys()
            ]
            if not is_supplier_edge:
                future_partners = self._filter_sell_targets(future_partners)
            for partner, offer in self._middle_level_future_edge_offers(
                edge, future_partners
            ).items():
                response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            if mode5_advance_after_response:
                self._middle_level_mode5_advance_after_sell_response()
            return

        remaining_partners = [
            partner
            for partner in all_partners
            if partner in self.negotiators.keys() and partner not in response.keys()
        ]
        if not is_supplier_edge:
            remaining_partners = self._filter_sell_targets(remaining_partners)
        distribution = dict(zip(remaining_partners, repeat(0)))
        if remaining_partners and current_need > 0:
            distribution |= self._middle_level_distribute_edge_needs(
                edge, remaining_partners, current_need
            )

        future_partners = []
        for partner, quantity in distribution.items():
            if quantity > 0:
                price = self.price(partner)
                if is_supplier_edge and self._middle_level_mode8_family():
                    price = self._middle_level_mode8_buy_price(partner)
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (quantity, awi.current_step, price),
                )
            else:
                future_partners.append(partner)

        for partner, offer in self._middle_level_future_edge_offers(
            edge, future_partners
        ).items():
            response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        if mode5_advance_after_response:
            self._middle_level_mode5_advance_after_sell_response()

    def _middle_level_mode8_update_consumer_responses(
        self, offers, response, states=None
    ) -> None:
        consumers = [
            partner
            for partner in self.awi.my_consumers
            if partner in self.negotiators.keys()
        ]
        planned_quantities = Counter()
        remaining_partners = []
        for partner in consumers:
            offer = offers.get(partner)
            if offer is None:
                if partner in offers:
                    remaining_partners.append(partner)
                continue
            if not self.is_valid_price(offer[UNIT_PRICE], partner):
                remaining_partners.append(partner)
                continue
            if self._middle_level_mode8_accepts_offer(
                partner,
                offer,
                planned_quantities,
                None if states is None else states.get(partner),
            ):
                response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                planned_quantities[int(offer[TIME])] += int(offer[QUANTITY])
            else:
                remaining_partners.append(partner)

        remaining_partners = [
            partner
            for partner in consumers
            if partner not in response.keys()
        ]
        if self._middle_level_mode5_future_sell_enabled():
            for partner, offer in self._middle_level_mode8_sell_offers(
                remaining_partners,
                planned_quantities,
                include_current_step=False,
            ).items():
                response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        for partner in remaining_partners:
            if partner not in response:
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _middle_level_mode8_update_initial_consumer_responses(
        self, offers, response, phase: str
    ) -> None:
        consumers = [
            partner
            for partner in self.awi.my_consumers
            if partner in self.negotiators.keys()
        ]
        planned_quantities = Counter()
        remaining_partners = []
        saw_offer = False
        for partner in consumers:
            offer = offers.get(partner)
            if offer is not None:
                saw_offer = True
            if offer is not None and self._middle_level_mode8_accepts_initial_sell_offer(
                partner, offer, planned_quantities
            ):
                response[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                planned_quantities[int(offer[TIME])] += int(offer[QUANTITY])
            else:
                remaining_partners.append(partner)

        if not self._middle_level_mode5_future_sell_enabled():
            for partner in remaining_partners:
                if partner not in response:
                    response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            if phase == "initial_sell_probe" and saw_offer:
                self._middle_level_mode5_advance_after_sell_response()
            return

        for partner, offer in self._middle_level_mode5_far_consumer_offers(
            remaining_partners, self._middle_level_mode8_fill_sell_quantity
        ).items():
            response[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        for partner in remaining_partners:
            if partner not in response:
                response[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)

        if phase == "initial_sell_probe" and saw_offer:
            self._middle_level_mode5_advance_after_sell_response()

    def _middle_level_edge_need(self, edge, step: int | None = None) -> int:
        if edge == "supplier":
            return self._supplier_need_at(step)
        return self._consumer_need_at(step)

    def _middle_level_edge_mode(self, edge) -> int:
        return self._middle_level_negotiation_mode

    def _middle_level_distribute_edge_needs(self, edge, partners, needs) -> dict[str, int]:
        if edge == "supplier":
            return self._middle_level_distribute_supplier_needs(partners, needs)
        return self._middle_level_distribute_consumer_needs(partners, needs)

    def _middle_level_distribute_supplier_needs(self, partners, needs) -> dict[str, int]:
        partners = list(partners)
        if self._middle_level_mode8_family():
            response = dict(zip(partners, repeat(0)))
            if partners and needs > 0:
                response |= dict(zip(partners, distribute(needs, len(partners))))
            return response
        return self.distribute_todays_supplie_consume_needs(list(partners), needs)

    def _middle_level_distribute_consumer_needs(self, partners, needs) -> dict[str, int]:
        partners = self._filter_sell_targets(partners)
        return self.distribute_todays_supplie_consume_needs(list(partners), needs)

    def _middle_level_future_edge_offers(self, edge, partners):
        if edge == "supplier":
            return self._middle_level_future_supplier_offers(partners)
        return self._middle_level_future_consumer_offers(partners)

    def _middle_level_future_supplier_offers(self, partners):
        if (
            self._middle_level_mode4_buy_cap
            or self._middle_level_mode5_family()
        ):
            return {}

        return self._middle_level_future_offers(
            partners, lambda step: self._middle_level_edge_need("supplier", step)
        )

    def _middle_level_future_consumer_offers(self, partners):
        partners = self._filter_sell_targets(partners)
        if self._middle_level_mode5_family():
            if not self._middle_level_mode5_future_sell_enabled():
                return {}
            phase = self._middle_level_mode5_current_phase()
            if phase == "initial_sell_probe":
                quantity_fn = (
                    self._middle_level_mode8_fill_sell_quantity
                    if self._middle_level_mode8_family()
                    else lambda step: self._middle_level_mode5_initial_sell_quantity()
                )
                return self._middle_level_mode5_far_consumer_offers(
                    partners, quantity_fn
                )
            if self._middle_level_mode8_family() and phase == "input_fill":
                return self._middle_level_mode5_far_consumer_offers(
                    partners, self._middle_level_mode8_fill_sell_quantity
                )
            return self._middle_level_mode5_far_consumer_offers(
                partners, self._middle_level_mode5_future_sell_quantity
            )
        return self._middle_level_future_offers(
            partners, lambda step: self._middle_level_edge_need("consumer", step)
        )

    def _middle_level_future_offers(self, partners, need_fn):
        partners = list(partners)
        awi = self.awi
        s = awi.current_step
        response = {}
        step_partners_list = [
            (s + 1, partners[: int(len(partners) * 0.5)]),
            (s + 2, partners[int(len(partners) * 0.5) : int(len(partners) * 0.8)]),
            (s + 3, partners[int(len(partners) * 0.8) :]),
        ]

        for step, step_partners in step_partners_list:
            if step >= awi.n_steps or not step_partners:
                continue
            step_needs = int(need_fn(step) / 3)
            if step_needs <= 0:
                continue
            for partner, quantity in zip(
                step_partners, distribute(step_needs, len(step_partners))
            ):
                if quantity > 0:
                    response[partner] = (quantity, step, self.best_price(partner))

        return response

    def _legacy_counter_all(self, offers, states):
        response = dict()
        awi = self.awi
                                                      
        for edge_needs, all_partners, issues in [
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
                       
            needs = 0
            if self.is_supplier(all_partners[0]):
                needs = self._legacy_supplier_need_at(awi.current_step)
                if edge_needs <= 0:
                    needs = 0
            elif self.is_consumer(all_partners[0]):
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = self._legacy_consumer_need_at(awi.current_step)

                                  
            partners = {_ for _ in all_partners if _ in offers.keys()}

                                                    
            current_step_offers = dict()                
            future_step_offers = dict()                

            for p in partners:
                if offers[p] is None:
                    continue
                if offers[p][TIME] == self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    current_step_offers[p] = offers[p]
                elif offers[p][TIME] != self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    future_step_offers[p] = offers[p]

                                     
            current_step_partners = {
                _ for _ in partners if _ in current_step_offers.keys()
            }
            {_ for _ in partners if _ in future_step_offers.keys()}

                                                        
            duplicate_list = [0 for _ in range(awi.n_steps)]       
            for p, x in future_step_offers.items():
                step = future_step_offers[p][TIME]
                if step <= awi.n_steps:      
                    if (
                        future_step_offers[p][QUANTITY] + duplicate_list[step - 1]
                        <= self.needs_at(step, p)
                    ):                                         
                        response[p] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, future_step_offers[p]
                        )
                        duplicate_list[step - 1] += future_step_offers[p][
                            QUANTITY
                        ]              

                                                                      
                                                       

            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))
            _best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = current_step_partners.difference(partner_ids)
                offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)

                diff = abs(offered - needs)
                if offered - needs >= 0:
                    if diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = diff, i
                else:
                    if diff < best_minus_diff and offered > 0:
                        best_minus_diff, best_index_minus = diff, i

            has_accept_offer = True
                                                        
            if (
                best_index_plus >= 0
                and best_plus_diff <= self._threshold
                and len(plist[best_index_plus]) > 0
            ):
                best_indx = best_index_plus
            elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
                best_indx = best_index_minus
            else:
                has_accept_offer = False

                                                                                       
                                
            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = plist[best_indx]
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))
                others_s = []
                others_c = []
                response.update(
                    {
                        k: SAOResponse(
                            ResponseType.ACCEPT_OFFER, current_step_offers[k]
                        )
                        for k in partner_ids
                    }
                )
                                     
                for x in others:
                    if self.is_supplier(x):
                        others_s.append(x)
                    if self.is_consumer(x):
                        others_c.append(x)

                others_s_dict = dict()
                others_s_dict |= self.future_supplie_offer(others_s)
                others_c_dict = dict()
                others_c_dict |= self.future_consume_offer(others_c)
                for k, x in others_s_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in others_c_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                flag = 1                  

            if flag != 1:
                                                                                         
                                            
                                                                                
                                               
                other_partners = {
                    _
                    for _ in all_partners
                    if _ not in response.keys() and _ in self.negotiators.keys()
                }                      
                distribution = self.distribute_todays_needs(other_partners)
                future_supplie_partner = []
                future_consume_partner = []

                for k, q in distribution.items():
                    if q > 0:
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, self.price(k)),
                        )
                    elif self.is_supplier(k):
                        future_supplie_partner.append(k)
                    elif self.is_consumer(k):
                        future_consume_partner.append(k)

                future_supplie_offer_dict = dict()
                future_supplie_offer_dict |= self.future_supplie_offer(
                    future_supplie_partner
                )
                future_consume_offer_dict = dict()
                future_consume_offer_dict |= self.future_consume_offer(
                    future_consume_partner
                )
                for k, x in future_supplie_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in future_consume_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

                        
    def _issues_for_partner(self, partner):
        if self.is_supplier(partner):
            return self.awi.current_input_issues
        return self.awi.current_output_issues

    def is_valid_price(self, price, partner):
        issues = self._issues_for_partner(partner)
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return price >= minp
        elif self.is_supplier(partner):
            return price <= maxp
        else:
            False

    def needs_at(self, step, partner):                    
        need = 0
        if self.is_supplier(partner):
            need = self._legacy_supplier_need_at(step)
        elif self.is_consumer(partner):
            need = self._legacy_consumer_need_at(step)

        return need

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
                                                                                  
        if partners is None:
            partners = self.negotiators.keys()

                                           
        response = dict(zip(partners, repeat(0)))

                                       
        supplie_partners = []
        consume_partners = []
        for x in partners:
            if self.is_supplier(x):
                supplie_partners.append(x)
            elif self.is_consumer(x):
                consume_partners.append(x)
        consume_partners = self._filter_sell_targets(consume_partners)

        awi = self.awi
        supplie_needs = self._legacy_supplier_need_at(awi.current_step)
        consume_needs = self._legacy_consumer_need_at(awi.current_step)
        n_supplier = len(supplie_partners)
        n_consumer = len(consume_partners)

        if n_supplier > 0 and supplie_needs > 0:
            response |= self.distribute_todays_supplie_consume_needs(
                supplie_partners, supplie_needs
            )

        if (
            n_consumer > 0
            and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= self.awi.n_lines
        ):                                               
            response |= self.distribute_todays_supplie_consume_needs(
                consume_partners, consume_needs
            )

        return response

    def distribute_todays_supplie_consume_needs(
        self, partners, needs
    ) -> dict[str, int]:
        response = dict(zip(partners, repeat(0)))
        if not partners or needs <= 0:
            return response
        random.shuffle(partners)
        partners = partners[: max(1, int(self._ptoday() * len(partners)))]
        n_partners = len(partners)

                                                            
        if needs < n_partners <= 0:
            partners = random.sample(partners, random.randint(1, needs))
            n_partners = len(partners)

        response |= dict(zip(partners, distribute(needs, n_partners)))

        return response

    def future_supplie_offer(self, list):
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        d1 = dict()
        d2 = dict()
        d3 = dict()
        response = dict()
                             
        step1_list = list[: int(len(list) * 0.5)]
        n_step1 = len(step1_list)
        step2_list = list[int(len(list) * 0.5) : int(len(list) * 0.8)]
        n_step2 = len(step2_list)
        step3_list = list[int(len(list) * 0.8) :]
        n_step3 = len(step3_list)
        divisor = self._first_level_future_offer_divisor_value()
                     
        if s + 1 < n:
            step1_needs = int(self._legacy_supplier_need_at(s + 1) / divisor)
            if step1_needs > 0 and n_step1 > 0:      
                d1 |= dict(zip(step1_list, distribute(step1_needs, n_step1)))
                for k, q in d1.items():
                    if q > 0:
                        if not self._future_sell_step_allowed_by_bankruptcy_forecast(
                            k, s + 1
                        ):
                            continue
                        response[k] = (q, s + 1, self.best_price(k))

                     
        if s + 2 < n:
            step2_needs = int(self._legacy_supplier_need_at(s + 2) / divisor)
            if step2_needs > 0 and n_step2 > 0:
                d2 |= dict(zip(step2_list, distribute(step2_needs, n_step2)))
                for k, q in d2.items():
                    if q > 0:
                        if not self._future_sell_step_allowed_by_bankruptcy_forecast(
                            k, s + 2
                        ):
                            continue
                        response[k] = (q, s + 2, self.best_price(k))

                     
        if s + 3 < n:
            step3_needs = int(self._legacy_supplier_need_at(s + 3) / divisor)
            if step3_needs > 0 and n_step3 > 0:
                d3 |= dict(zip(step3_list, distribute(step3_needs, n_step3)))
                for k, q in d3.items():
                    if q > 0:
                        if not self._future_sell_step_allowed_by_bankruptcy_forecast(
                            k, s + 3
                        ):
                            continue
                        response[k] = (q, s + 3, self.best_price(k))

        return response

    def future_consume_offer(self, list):
        list = self._filter_sell_targets(list)
        response = dict()
        awi = self.awi
        s = awi.current_step           
        n = awi.n_steps          
        d1 = dict()
        d2 = dict()
        d3 = dict()
                             
        step1_list = list[: int(len(list) * 0.5)]
        n_step1 = len(step1_list)
        step2_list = list[int(len(list) * 0.5) : int(len(list) * 0.8)]
        n_step2 = len(step2_list)
        step3_list = list[int(len(list) * 0.8) :]
        n_step3 = len(step3_list)
        divisor = self._first_level_future_offer_divisor_value()
                    
        if (
            s + 1 < n and awi.total_sales_at(s + 1) <= self.awi.n_lines
        ):                                      
            step1_needs = int(self._legacy_consumer_need_at(s + 1) / divisor)
            if step1_needs > 0 and n_step1 > 0:
                d1 |= dict(zip(step1_list, distribute(step1_needs, n_step1)))
                for k, q in d1.items():
                    if q > 0:
                        response[k] = (q, s + 1, self.best_price(k))

                    
        if s + 2 < n and awi.total_sales_at(s + 2) <= self.awi.n_lines:
            step2_needs = int(self._legacy_consumer_need_at(s + 2) / divisor)
            if step2_needs > 0 and n_step2 > 0:
                d2 |= dict(zip(step2_list, distribute(step2_needs, n_step2)))
                for k, q in d2.items():
                    if q > 0:
                        response[k] = (q, s + 2, self.best_price(k))

                    
        if s + 3 < n and awi.total_sales_at(s + 2) <= self.awi.n_lines:
            step3_needs = int(self._legacy_consumer_need_at(s + 3) / divisor)
            if step3_needs > 0 and n_step3 > 0:
                d3 |= dict(zip(step3_list, distribute(step3_needs, n_step3)))
                for k, q in d3.items():
                    if q > 0:
                        response[k] = (q, s + 3, self.best_price(k))

        return response

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        pmin, pmax = self._unit_price_bounds(partner)
        return pmin if self.is_supplier(partner) else pmax

    def _unit_price_bounds(self, partner):
        issues = self._issues_for_partner(partner)
        issue = issues[UNIT_PRICE]
        return issue.min_value, issue.max_value

    def price(self, partner):
                  
                                    
        issues = self._issues_for_partner(partner)
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value

        if self.is_consumer(partner):
            return max(maxp * 0.7, minp)
        else:
            return min(minp * 1.2, maxp)


                                                               
PenguinAgent = HorizonAwareAgent


