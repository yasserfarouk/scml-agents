from __future__ import annotations

from scml import OneShotAWI

"""
LitaAgentCIR — 库存敏感型统一策略（SDK 对接版）
=================================================
"""

import math
import random
from copy import deepcopy

# ------------------ 基础依赖 ------------------
from itertools import combinations as iter_combinations  # Added for combinations
from typing import Dict, List, Optional, Tuple  # Added Optional
from uuid import uuid4

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.std import (
    QUANTITY,
    TIME,
    UNIT_PRICE,
    StdAWI,
    StdSyncAgent,
)

from .inventory_manager_cirs import (
    IMContract,
    IMContractType,
    InventoryManagerCIR,
    MaterialType,
)

# 内部工具 & manager
__all__ = ["LitaAgentCIRS"]

# ------------------ 主代理实现 ------------------
# Main agent implementation


class LitaAgentCIRS(StdSyncAgent):
    """重构后的 LitaAgent CIR。"""

    # ------------------------------------------------------------------
    # 🌟 1. 初始化
    # 1. Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args,
        concession_curve_power: float = 1.5,
        capacity_tight_margin_increase: float = 0.07,
        procurement_cash_flow_limit_percent: float = 0.75,
        p_threshold: float = 0.5,
        q_threshold: float = -0.2,
        # 新增参数用于控制组合评估策略
        # ---
        # New parameters to control combination evaluation strategy
        combo_evaluation_strategy: str = "simulated_annealing",
        # 可选 "k_max", "beam_search", "simulated_annealing", "exhaustive_search" / Options: "k_max", "beam_search", "simulated_annealing", "exhaustive_search" # MODIFIED
        max_combo_size_for_k_max: int = 6,  # 当 strategy == "k_max" 时使用 / Used when strategy == "k_max"
        beam_width_for_beam_search: int = 3,
        # 当 strategy == "beam_search" 时使用 / Used when strategy == "beam_search"
        iterations_for_sa: int = 200,
        # 当 strategy == "simulated_annealing" 时使用 / Used when strategy == "simulated_annealing"
        sa_initial_temp: float = 1.0,  # SA 初始温度 / SA initial temperature
        sa_cooling_rate: float = 0.95,  # SA 冷却速率 / SA cooling rate
        threshold_time_decay_factor: float = 0.98,
        inventory_pressure_threshold_raw: float = 50.0,
        inventory_pressure_threshold_product: float = 50.0,
        price_concession_opponent_factor: float = 0.5,
        price_concession_time_factor: float = 0.02,
        price_concession_round_factor: float = 0.05,
        # 修改：over_procurement_factor 变为 initial_over_procurement_factor
        # MODIFIED: over_procurement_factor becomes initial_over_procurement_factor
        initial_over_procurement_factor: float = 0.1,  # 初始超采购比例 / Initial over-procurement percentage
        # 新增参数用于自动调节超采购因子
        # ---
        # New parameters for automatic adjustment of over-procurement factor
        over_procurement_adjustment_rate: float = 0.02,  # 每次调整的步长 / Adjustment step size
        over_procurement_min_factor: float = 0.0,  # 最小超采购因子 / Minimum over-procurement factor
        over_procurement_max_factor: float = 0.5,
        # 最大超采购因子 (例如，最多超采50%) / Maximum over-procurement factor (e.g., at most 50% over)
        over_procurement_success_lower_bound: float = 0.7,
        # 采购成功率低于此值则增加因子 / Increase factor if procurement success rate is below this
        over_procurement_success_upper_bound: float = 1.2,
        # 采购成功率高于此值则减少因子 / Decrease factor if procurement success rate is above this
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.procurement_cash_flow_limit_percent = procurement_cash_flow_limit_percent
        self.concession_curve_power = concession_curve_power
        self.capacity_tight_margin_increase = capacity_tight_margin_increase
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold

        # 存储组合评估策略相关的参数
        # ---
        # Store parameters related to combination evaluation strategy
        self.combo_evaluation_strategy = combo_evaluation_strategy
        self.max_combo_size_for_k_max = max_combo_size_for_k_max
        self.beam_width = beam_width_for_beam_search
        self.sa_iterations = iterations_for_sa
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate

        self.threshold_time_decay_factor = threshold_time_decay_factor
        self.inventory_pressure_threshold_raw = inventory_pressure_threshold_raw
        self.inventory_pressure_threshold_product = inventory_pressure_threshold_product
        self.price_concession_opponent_factor = price_concession_opponent_factor
        self.price_concession_time_factor = price_concession_time_factor
        self.price_concession_round_factor = price_concession_round_factor

        # 修改：over_procurement_factor 是动态的，从 initial 值开始
        # MODIFIED: over_procurement_factor is dynamic, starting from initial value
        self.over_procurement_factor = initial_over_procurement_factor
        self.over_procurement_adjustment_rate = over_procurement_adjustment_rate
        self.over_procurement_min_factor = over_procurement_min_factor
        self.over_procurement_max_factor = over_procurement_max_factor
        self.over_procurement_success_lower_bound = over_procurement_success_lower_bound
        self.over_procurement_success_upper_bound = over_procurement_success_upper_bound

        # —— 运行时变量 ——
        self.im: Optional[InventoryManagerCIR] = None  # Updated type hint
        self._market_price_avg: float = 0.0
        self._market_material_price_avg: float = 0.0
        self._market_product_price_avg: float = 0.0
        self._recent_material_prices: List[float] = []
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30
        self.model = None
        self.concession_model = None
        self.sales_completed: Dict[int, int] = {}
        self.purchase_completed: Dict[
            int, int
        ] = {}  # 用于记录每日实际完成的采购总量（非超采购目标）

        self.partner_stats: Dict[str, Dict[str, float]] = {}
        self.partner_models: Dict[str, Dict[str, float]] = {}
        self._last_partner_offer: Dict[str, Outcome] = {}

        self._sales_successes_since_margin_update: int = 0
        self._sales_failures_since_margin_update: int = 0

        # 新增：用于跟踪每日采购目标和实际达成量，以调整超采购因子
        # ---
        # NEW: For tracking daily procurement targets and achievements to adjust over-procurement factor
        self.daily_targeted_procurement_for_adjustment: Dict[int, int] = {}
        self.daily_achieved_procurement_for_adjustment: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # 🌟 2. World / 日常回调
    # ------------------------------------------------------------------

    # ... (init, before_step 方法保持不变) ...
    def init(self) -> None:
        """在 World 初始化后调用；此处创建库存管理器。"""
        # Determine processing_cost
        # 反正加工成本都是固定的，scml好像会自动优化这个，就当做0了
        processing_cost = 0.0
        daily_capacity = self.awi.n_lines

        self.im = InventoryManagerCIR(
            raw_storage_cost=self.awi.current_storage_cost,  # same cost for raw and product
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=processing_cost,
            daily_production_capacity=daily_capacity,
            max_simulation_day=self.awi.n_steps,
            current_day=self.awi.current_step,
        )

    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "CustomInventoryManager 尚未初始化!"
        current_day = self.awi.current_step

        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(current_day, 0)
        self.purchase_completed.setdefault(current_day, 0)
        # 新增：初始化当日采购目标和达成量跟踪
        # NEW: Initialize daily procurement target and achievement tracking
        self.daily_targeted_procurement_for_adjustment.setdefault(current_day, 0)
        self.daily_achieved_procurement_for_adjustment.setdefault(current_day, 0)

        # 首先将外生协议写入im (这会调用 plan_production 更新计划)
        # First, write exogenous contracts into the inventory manager (this will call plan_production to update the plan)
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            if exogenous_contract_quantity > 0:
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_supply"
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.SUPPLY,
                    quantity=int(exogenous_contract_quantity),
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day
                    bankruptcy_risk=0,
                    material_type=MaterialType.RAW,
                )
                self.im.add_transaction(exogenous_contract)

        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            if exogenous_contract_quantity > 0:
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_demand"
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.DEMAND,
                    quantity=int(
                        exogenous_contract_quantity
                    ),  # 确保是整数 / Ensure it's an integer
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day
                    bankruptcy_risk=0,
                    material_type=MaterialType.PRODUCT,
                )
                self.im.add_transaction(exogenous_contract)

        # 在外生协议添加并重新规划生产后，再计算不足量
        # After exogenous contracts are added and production is replanned, then calculate the insufficiency
        self.today_insufficient = self.im.get_today_insufficient_raw(current_day)
        self.total_insufficient = self.im.get_total_insufficient_raw(
            current_day, horizon=14
        )  # Default horizon 14 days

    def step(self) -> None:
        """每天结束时调用：执行 IM 的日终操作并刷新市场均价，并调整超采购因子。"""
        assert self.im, "CustomInventoryManager 尚未初始化!"
        current_day = self.awi.current_step

        # --- 调整超采购因子 ---
        # --- Adjust Over-Procurement Factor ---
        targeted_today = self.daily_targeted_procurement_for_adjustment.get(
            current_day, 0
        )
        achieved_today = self.daily_achieved_procurement_for_adjustment.get(
            current_day, 0
        )

        if targeted_today > 0:
            success_ratio = achieved_today / targeted_today
            if (
                success_ratio < self.over_procurement_success_lower_bound
            ):  # 例如，低于70%的目标
                self.over_procurement_factor += self.over_procurement_adjustment_rate
            elif (
                success_ratio > self.over_procurement_success_upper_bound
            ):  # 例如，超过120%的目标
                self.over_procurement_factor -= (
                    self.over_procurement_adjustment_rate * 0.5
                )  # 减少时幅度小一些

            self.over_procurement_factor = max(
                self.over_procurement_min_factor,
                min(self.over_procurement_max_factor, self.over_procurement_factor),
            )

        # 让 IM 完成收货 / 生产 / 交付 / 规划
        # CustomInventoryManager.process_day_end_operations advances its own current_day
        result = self.im.process_day_end_operations(self.awi.current_step)
        # self.im.update_day() # This is no longer needed.
        # —— 更新市场均价估计 ——
        # Ensure lists are not empty before calculating average
        if self._recent_material_prices:
            self._market_material_price_avg = sum(self._recent_material_prices) / len(
                self._recent_material_prices
            )
        if self._recent_product_prices:
            self._market_product_price_avg = sum(self._recent_product_prices) / len(
                self._recent_product_prices
            )

        self._print_daily_status_report(result)

    # ------------------------------------------------------------------
    # 🌟 3. 价格工具
    # Pricing utilities
    # ------------------------------------------------------------------

    def _is_supplier(self, pid: str) -> bool:
        return pid in self.awi.my_suppliers

    def _is_consumer(self, pid: str) -> bool:
        return pid in self.awi.my_consumers

    # ------------------------------------------------------------------
    # 🌟 4. first_proposals — 首轮报价（可简化）
    # ------------------------------------------------------------------
    def first_proposals(self) -> Dict[str, Outcome]:
        """
        Generates initial proposals to partners.
        Prices are set to the agent's optimal based on NMI.
        Needs/opportunities are distributed among available partners.
        Applies over-procurement factor.
        生成向伙伴的初始报价。
        价格根据NMI设置为代理的最优价格。
        需求/机会被分配给可用的伙伴。
        """
        proposals: Dict[str, Outcome] = {}
        current_day = self.awi.current_step
        n_steps = self.awi.n_steps

        if not self.im:
            return proposals

        # --- 1. 采购原材料以满足短缺 (Procure raw materials to meet shortfall) ---
        base_target_procurement_quantity = (
            self.total_insufficient if self.total_insufficient is not None else 0
        )

        adjusted_target_procurement_quantity = 0
        if base_target_procurement_quantity > 0:
            adjusted_target_procurement_quantity = int(
                round(
                    base_target_procurement_quantity
                    * (1 + self.over_procurement_factor)
                )
            )

        # 记录当日的目标采购量（用于后续调整超采购因子）
        # Record today's target procurement quantity (for later adjustment of over-procurement factor)
        self.daily_targeted_procurement_for_adjustment[current_day] = (
            adjusted_target_procurement_quantity
        )

        supplier_negotiators = [
            nid
            for nid in self.negotiators.keys()
            if self._is_supplier(nid)
            and not (self.awi.is_first_level and nid in self.awi.my_suppliers)
        ]

        if adjusted_target_procurement_quantity > 0 and supplier_negotiators:
            # ... (后续的分配逻辑与之前类似, 使用 adjusted_target_procurement_quantity 作为总需求)
            # ... (Subsequent distribution logic is similar to before, using adjusted_target_procurement_quantity as total demand)
            supplier_min_q_map: Dict[str, int] = {}
            for nid in supplier_negotiators:
                nmi_s = self.get_nmi(nid)
                min_q_s = 1  # Default min quantity
                if nmi_s and nmi_s.issues[QUANTITY] is not None:
                    min_q_s = int(round(nmi_s.issues[QUANTITY].min_value))
                supplier_min_q_map[nid] = max(1, min_q_s)  # Ensure min_q is at least 1

            # Sort suppliers by their min_q_nmi to prioritize those with smaller minimums if total need is low
            # This is a simple heuristic. More complex would be to solve a knapsack-like problem.
            sorted_supplier_nids = sorted(
                supplier_negotiators, key=lambda nid: supplier_min_q_map[nid]
            )
            remaining_procurement_need = adjusted_target_procurement_quantity

            for nid in sorted_supplier_nids:
                if remaining_procurement_need <= 0:
                    break  # All needs met

                nmi = self.get_nmi(nid)
                min_q_nmi = supplier_min_q_map[nid]
                max_q_nmi = float("inf")
                min_p_nmi = 0.01
                # max_p_nmi is not strictly needed for proposing our best price (min_p_nmi for buying)
                # but good to have for clamping if we were to use a different pricing strategy.
                max_p_nmi_from_nmi = float("inf")
                min_t_nmi, max_t_nmi = current_day + 1, n_steps - 1

                if nmi:
                    if nmi.issues[QUANTITY] is not None:
                        # min_q_nmi already fetched
                        max_q_nmi = nmi.issues[QUANTITY].max_value
                    if nmi.issues[UNIT_PRICE] is not None:
                        min_p_nmi = nmi.issues[UNIT_PRICE].min_value
                        max_p_nmi_from_nmi = nmi.issues[UNIT_PRICE].max_value
                    if nmi.issues[TIME] is not None:
                        min_t_nmi = max(min_t_nmi, nmi.issues[TIME].min_value)
                        max_t_nmi = min(max_t_nmi, nmi.issues[TIME].max_value)

                # Determine proposal quantity for this supplier
                # Propose up to remaining need, but not less than NMI min, and not more than NMI max.
                propose_q_for_this_supplier = min(remaining_procurement_need, max_q_nmi)
                propose_q_for_this_supplier = max(
                    propose_q_for_this_supplier, min_q_nmi
                )

                if (
                    propose_q_for_this_supplier <= 0
                    or propose_q_for_this_supplier > remaining_procurement_need
                ):
                    # If min_q_nmi is greater than remaining need, we can't propose to this supplier for this need.
                    # Or if calculated quantity is invalid.
                    continue

                propose_q = int(round(propose_q_for_this_supplier))
                propose_t = current_day + 1
                if (
                    self.total_insufficient
                    and self.total_insufficient > self.im.daily_production_capacity * 2
                ):
                    propose_t = max(current_day + 1, min_t_nmi)
                else:
                    propose_t = max(current_day + 2, min_t_nmi)

                propose_t = min(propose_t, max_t_nmi)
                propose_t = max(propose_t, current_day + 1)
                propose_t = min(propose_t, n_steps - 1)
                propose_p = min_p_nmi
                propose_p = min(propose_p, max_p_nmi_from_nmi)

                if propose_q > 0:
                    proposals[nid] = (propose_q, propose_t, propose_p)
                    remaining_procurement_need -= propose_q

        # --- 2. 销售产成品 (Sell finished products) ---
        # ... (销售逻辑保持不变) ...
        if not self.awi.is_last_level:
            sellable_horizon = min(3, n_steps - (current_day + 1))
            estimated_sellable_quantity = 0
            if sellable_horizon > 0:
                current_product_stock = self.im.get_inventory_summary(
                    current_day, MaterialType.PRODUCT
                ).get("current_stock", 0)
                planned_production_in_horizon = 0
                for d_offset in range(sellable_horizon):
                    planned_production_in_horizon += self.im.production_plan.get(
                        current_day + d_offset, 0
                    )
                committed_sales_in_horizon = 0
                for contract_s in self.im.pending_demand_contracts:
                    if (
                        current_day
                        <= contract_s.delivery_time
                        < current_day + sellable_horizon
                    ):
                        committed_sales_in_horizon += contract_s.quantity
                estimated_sellable_quantity = (
                    current_product_stock
                    + planned_production_in_horizon
                    - committed_sales_in_horizon
                )
                estimated_sellable_quantity = max(0, estimated_sellable_quantity)

            consumer_negotiators = [
                nid for nid in self.negotiators.keys() if self._is_consumer(nid)
            ]

            if estimated_sellable_quantity > 0 and consumer_negotiators:
                # Similar distribution logic for selling
                consumer_min_q_map: Dict[str, int] = {}
                for nid_c in consumer_negotiators:  # Renamed to avoid conflict
                    nmi_c = self.get_nmi(nid_c)
                    min_q_c = 1
                    if nmi_c and nmi_c.issues[QUANTITY] is not None:
                        min_q_c = int(round(nmi_c.issues[QUANTITY].min_value))
                    consumer_min_q_map[nid_c] = max(1, min_q_c)

                sorted_consumer_nids = sorted(
                    consumer_negotiators, key=lambda nid: consumer_min_q_map[nid]
                )
                remaining_sellable_quantity = estimated_sellable_quantity

                for nid in sorted_consumer_nids:
                    if remaining_sellable_quantity <= 0:
                        break

                    nmi = self.get_nmi(nid)
                    min_q_nmi = consumer_min_q_map[nid]
                    max_q_nmi = float("inf")
                    # min_p_nmi is not strictly needed for proposing our best price (max_p_nmi for selling)
                    min_p_nmi_from_nmi = 0.01
                    max_p_nmi = float("inf")
                    min_t_nmi, max_t_nmi = current_day + 1, n_steps - 1

                    if nmi:
                        if nmi.issues[QUANTITY] is not None:
                            max_q_nmi = nmi.issues[QUANTITY].max_value
                        if nmi.issues[UNIT_PRICE] is not None:
                            min_p_nmi_from_nmi = nmi.issues[UNIT_PRICE].min_value
                            max_p_nmi = nmi.issues[UNIT_PRICE].max_value
                        if nmi.issues[TIME] is not None:
                            min_t_nmi = max(min_t_nmi, nmi.issues[TIME].min_value)
                            max_t_nmi = min(max_t_nmi, nmi.issues[TIME].max_value)

                    propose_q_for_this_consumer = min(
                        remaining_sellable_quantity, max_q_nmi
                    )
                    propose_q_for_this_consumer = max(
                        propose_q_for_this_consumer, min_q_nmi
                    )

                    if (
                        propose_q_for_this_consumer <= 0
                        or propose_q_for_this_consumer > remaining_sellable_quantity
                    ):
                        continue

                    propose_q = int(round(propose_q_for_this_consumer))
                    propose_t = current_day + 2
                    propose_t = max(propose_t, min_t_nmi)
                    propose_t = min(propose_t, max_t_nmi)
                    propose_t = max(propose_t, current_day + 1)
                    propose_t = min(propose_t, n_steps - 1)
                    propose_p = max_p_nmi
                    propose_p = max(propose_p, min_p_nmi_from_nmi)

                    if propose_q > 0:
                        proposals[nid] = (propose_q, propose_t, propose_p)
                        remaining_sellable_quantity -= propose_q

        return proposals

    # ... (score_offers, normalize_final_score, calculate_inventory_cost_score, _evaluate_offer_combinations*, _calculate_combination_profit_and_normalize, _generate_counter_offer, counter_all, get_partner_id, _print_daily_status_report 保持不变)
    # ... (score_offers, normalize_final_score, calculate_inventory_cost_score, _evaluate_offer_combinations*, _calculate_combination_profit_and_normalize, _generate_counter_offer, counter_all, get_partner_id, _print_daily_status_report remain unchanged)
    # ---
    # The methods listed above are assumed to be correct and complete from previous steps.
    # We are only showing the diff for the new functionality.
    # ---
    def score_offers(
        self,
        offer_combination: Dict[str, Outcome],  # 一个报价组合
        current_im: InventoryManagerCIR,  # 当前的库存管理器状态
        awi: OneShotAWI,  # AWI 实例，用于获取当前日期、总天数等
    ) -> Tuple[float, float]:
        """
        评估一个报价组合的分数。
        分数 = (接受组合前的库存成本) - (接受组合后的库存成本)。
        成本由 calculate_inventory_cost_score 计算，越低越好。
        因此，本方法返回的分数越高，代表该报价组合带来的成本降低越多，越有利。
        ---
        Evaluates the score of an offer combination.
        Score = (inventory cost before accepting combo) - (inventory cost after accepting combo).
        Cost is calculated by calculate_inventory_cost_score; lower is better.
        Thus, a higher score returned by this method means the offer combo leads to a greater cost reduction, which is more favorable.
        Parameter:
            Offer list
            Inventory Manager(im)
            AWI
        Return:
            Tuple[raw_score, norm_score]
        """
        today = awi.current_step
        actual_last_simulation_day = awi.n_steps - 1
        unit_shortfall_penalty = self.awi.current_shortfall_penalty

        # unit_storage_cost
        current_unit_storage_cost = self.awi.current_storage_cost

        # 2. 计算 score_a: 接受报价组合前的总库存成本
        # ---
        # 2. Calculate score_a: total inventory cost before accepting the offer combination
        im_before = deepcopy(current_im)
        im_before.is_deepcopy = True
        score_a = self.calculate_inventory_cost_score(
            im_state=im_before,
            current_day=today,
            last_simulation_day=actual_last_simulation_day,
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost,
        )

        # 3. 计算 score_b: 接受报价组合后的总库存成本
        # ---
        # 3. Calculate score_b: total inventory cost after accepting the offer combination
        im_after = deepcopy(current_im)
        im_after.is_deepcopy = True
        for negotiator_id, offer_outcome in offer_combination.items():
            if not offer_outcome:
                continue
            quantity, time, unit_price = offer_outcome
            is_supply_contract_for_agent = self._is_supplier(negotiator_id)
            contract_type = (
                IMContractType.SUPPLY
                if is_supply_contract_for_agent
                else IMContractType.DEMAND
            )
            material_type = (
                MaterialType.RAW
                if is_supply_contract_for_agent
                else MaterialType.PRODUCT
            )
            sim_contract = IMContract(
                contract_id=str(uuid4()),
                partner_id=negotiator_id,
                type=contract_type,
                quantity=int(quantity),
                price=unit_price,
                delivery_time=time,
                material_type=material_type,
                bankruptcy_risk=0.0,
            )
            im_after.add_transaction(sim_contract)
        score_b = self.calculate_inventory_cost_score(
            im_state=im_after,
            current_day=today,
            last_simulation_day=actual_last_simulation_day,
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost,
        )

        # 4. 确保成本分数 a 和 b 不为负 (成本理论上应 >= 0)
        if score_a < 0:
            score_a = 0.0
        if score_b < 0:
            score_b = 0.0

        # 5. 计算最终分数: score_a - score_b
        #    如果 score_b < score_a (接受组合后成本降低), 则 final_score 为正 (好)
        #    如果 score_b > score_a (接受组合后成本增加), 则 final_score 为负 (差)
        raw_final_score = score_a - score_b
        normalized_final_score = self.normalize_final_score(raw_final_score, score_a)
        return raw_final_score, normalized_final_score

    def normalize_final_score(self, final_score: float, score_a: float) -> float:
        """
        将 final_score (score_a - score_b) 归一化到 [0, 1] 区间。
        score_a 是接受组合前的成本。
        """
        if score_a < 0:  # 理论上 score_a (成本) 不应为负，做个保护
            score_a = 0.0
        if score_a == 0:
            if final_score == 0:
                return 0.5
            else:
                return 0.25
        normalized_value = 0.5 + 0.5 * (final_score / score_a)
        normalized_value = max(0.0, min(1.0, normalized_value))
        return normalized_value

    def calculate_inventory_cost_score(
        self,
        im_state: InventoryManagerCIR,
        current_day: int,
        last_simulation_day: int,
        unit_shortfall_penalty: float,
        unit_storage_cost: float,
    ) -> float:
        total_cost_score = 0.0
        im_state.plan_production(up_to_day=last_simulation_day)
        sim_eval_im_for_shortfall = deepcopy(im_state)
        sim_eval_im_for_shortfall.is_deepcopy = True
        sim_eval_im_for_shortfall.current_day = current_day
        for d in range(current_day + 1, last_simulation_day + 1):
            total_demand_qty_on_d = 0.0
            for contract in sim_eval_im_for_shortfall.pending_demand_contracts:
                if contract.delivery_time == d:
                    total_demand_qty_on_d += contract.quantity
            if total_demand_qty_on_d == 0:
                continue  # 当天无需求，继续下一天 / No demand for this day, continue to the next

            # 获取在 d 天开始时可用于交付的总产品量
            # ---
            # Get total products available for delivery at the start of day 'd'
            # 注意：get_inventory_summary(d, ...) 返回的是 d 天开始时的库存和基于当前计划的预估可用量
            # ---
            # Note: get_inventory_summary(d, ...) returns stock at the start of day d and estimated availability based on current plans
            total_available_to_deliver_on_d = (
                sim_eval_im_for_shortfall.get_inventory_summary(
                    d, MaterialType.PRODUCT
                )["estimated_available"]
            )

            # 3. Calculate shortfall for day 'd'
            if total_demand_qty_on_d > total_available_to_deliver_on_d:
                shortfall_on_d = total_demand_qty_on_d - total_available_to_deliver_on_d
                total_cost_score += shortfall_on_d * unit_shortfall_penalty
            # 为了准确模拟后续天的缺货，需要模拟当天的交付（即使只是估算）
            # 这部分在原代码中缺失，但对于多日缺货计算是重要的。
            # 为简化，我们假设 get_inventory_summary 已经考虑了这一点，或者缺货计算是独立的。
            # 如果要更精确，这里应该更新 sim_eval_im_for_shortfall 的产品批次。
            # ---
            # To accurately simulate shortfall for subsequent days, today's delivery (even if estimated) needs to be simulated.
            # This part was missing in the original code but is important for multi-day shortfall calculation.
            # For simplicity, we assume get_inventory_summary already considers this, or shortfall calculation is independent.
            # For more precision, product batches in sim_eval_im_for_shortfall should be updated here.

        # B. 计算总存储成本
        # ---
        # B. Calculate Total Storage Cost
        # 使用传入的 im_state 进行存储成本计算，因为它代表了假设决策后的状态。
        # 它的 current_day 应该仍然是 current_day (即评估开始的日期)。
        # 我们将在这个副本上模拟每一天的结束操作。
        # ---
        # Use the passed im_state for storage cost calculation as it represents the state after a hypothetical decision.
        # Its current_day should still be current_day (i.e., the start day of the evaluation).
        # We will simulate end-of-day operations on this copy.
        sim_eval_im_for_storage = deepcopy(
            im_state
        )  # 使用一个新的副本来模拟存储成本计算过程 / Use a new copy to simulate the storage cost calculation process
        sim_eval_im_for_storage.is_deepcopy = True
        sim_eval_im_for_storage.current_day = current_day

        # Re-initialize a sim for storage cost calculation based on the *final state* of inventory after all demands met/shortfalled
        # This uses the sim_eval_im which has processed deliveries/productions up to last_simulation_day

        for d in range(
            current_day, last_simulation_day + 1
        ):  # 循环到 last_simulation_day (包含) / Loop up to last_simulation_day (inclusive)
            # 获取 d 天开始时的库存用于计算当天的存储成本
            # ---
            # Get stock at the start of day d to calculate storage cost for that day
            raw_stock_info = sim_eval_im_for_storage.get_inventory_summary(
                d, MaterialType.RAW
            )
            product_stock_info = sim_eval_im_for_storage.get_inventory_summary(
                d, MaterialType.PRODUCT
            )
        for d in range(current_day, last_simulation_day + 1):
            raw_stock_info = im_state.get_inventory_summary(d, MaterialType.RAW)
            product_stock_info = im_state.get_inventory_summary(d, MaterialType.PRODUCT)

            # As per prompt clarification: 'current_stock' is SOD, stored for the entirety of day d.
            daily_storage_cost = (
                raw_stock_info.get("current_stock", 0.0) * unit_storage_cost
                + product_stock_info.get("current_stock", 0.0) * unit_storage_cost
            )
            total_cost_score += daily_storage_cost

            # 推进模拟副本的天数以进行下一天的存储成本计算
            # ---
            # Advance the day of the simulation copy for the next day's storage cost calculation
            sim_eval_im_for_storage.process_day_end_operations(
                d
            )  # 这会将 sim_eval_im_for_storage.current_day 推进到 d + 1 / This will advance sim_eval_im_for_storage.current_day to d + 1

        # C. 计算期末库存处置成本
        # ---
        # C. Calculate excess inventory penalty (disposal cost at the end)
        # 此时，sim_eval_im_for_storage.current_day 应该是 last_simulation_day + 1
        # ---
        # At this point, sim_eval_im_for_storage.current_day should be last_simulation_day + 1
        day_for_disposal_check = last_simulation_day + 1

        # 我们需要的是在 last_simulation_day 结束后，即第 day_for_disposal_check 天开始时的库存
        # ---
        # We need the inventory at the start of day_for_disposal_check, which is after last_simulation_day ends.
        remain_raw = sim_eval_im_for_storage.get_inventory_summary(
            day_for_disposal_check, MaterialType.RAW
        )["current_stock"]
        remain_product = sim_eval_im_for_storage.get_inventory_summary(
            day_for_disposal_check, MaterialType.PRODUCT
        )["current_stock"]
        inventory_penalty = (
            remain_raw + remain_product
        ) * self.awi.current_disposal_cost
        total_cost_score += inventory_penalty
        return total_cost_score

    def _evaluate_offer_combinations_exhaustive(
        self,
        offers: Dict[str, Outcome],
        im: InventoryManagerCIR,
        awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用“全局搜索”（枚举所有非空子集）策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the "exhaustive search" (all non-empty subsets) strategy,
        primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0
        offer_items_list: List[Tuple[str, Outcome]] = list(offers.items())
        num_offers_available = len(offer_items_list)
        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        highest_norm_score: float = -1.0

        for i in range(
            1, num_offers_available + 1
        ):  # Loop for all possible combination sizes
            for combo_as_tuple_of_tuples in iter_combinations(offer_items_list, i):
                current_combination_list_of_tuples = list(combo_as_tuple_of_tuples)
                current_combination_dict = dict(current_combination_list_of_tuples)
                _raw_cost_reduction, current_norm_score = self.score_offers(
                    offer_combination=current_combination_dict, current_im=im, awi=awi
                )
                if current_norm_score > highest_norm_score:
                    highest_norm_score = current_norm_score
                    best_combination_items = current_combination_list_of_tuples
                elif (
                    current_norm_score == highest_norm_score and best_combination_items
                ):
                    # 如果分数相同，优先选择包含 offer 数量较少的组合
                    # ---
                    # If scores are the same, prefer combinations with fewer offers
                    if len(current_combination_list_of_tuples) < len(
                        best_combination_items
                    ):
                        best_combination_items = current_combination_list_of_tuples
        return best_combination_items, highest_norm_score

    def _evaluate_offer_combinations_k_max(
        self,
        offers: Dict[str, Outcome],
        im: InventoryManagerCIR,
        awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用“限制K大小” 策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the "limit K size" strategy, primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0

        # 将字典形式的 offers 转换为 (negotiator_id, Outcome) 元组的列表，方便组合
        offer_items_list: List[Tuple[str, Outcome]] = list(offers.items())
        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        highest_norm_score: float = -1.0
        for i in range(
            1, min(len(offer_items_list), self.max_combo_size_for_k_max) + 1
        ):
            for combo_as_tuple_of_tuples in iter_combinations(offer_items_list, i):
                # combo_as_tuple_of_tuples 保证了组合非空，因为 i 从 1 开始
                # ---
                # combo_as_tuple_of_tuples ensures the combination is non-empty as i starts from 1
                current_combination_list_of_tuples = list(combo_as_tuple_of_tuples)
                current_combination_dict = dict(current_combination_list_of_tuples)

                # 直接调用 score_offers 获取 norm_score
                # ---
                # Directly call score_offers to get norm_score
                _raw_cost_reduction, current_norm_score = self.score_offers(
                    offer_combination=current_combination_dict, current_im=im, awi=awi
                )
                if current_norm_score > highest_norm_score:
                    highest_norm_score = current_norm_score
                    best_combination_items = current_combination_list_of_tuples
                elif (
                    current_norm_score == highest_norm_score and best_combination_items
                ):
                    if len(current_combination_list_of_tuples) < len(
                        best_combination_items
                    ):
                        best_combination_items = current_combination_list_of_tuples
        return best_combination_items, highest_norm_score

    def _evaluate_offer_combinations_beam_search(
        self,
        offers: Dict[str, Outcome],
        im: InventoryManagerCIR,
        awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用 Beam Search 策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the Beam Search strategy, primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0
        offer_items_list = list(offers.items())

        # beam 存储 (组合字典, norm_score) 元组
        # 初始束可以包含一个“哨兵”空组合，其分数为极低，以启动流程，
        # 但在选择和扩展时，我们只关心非空组合。
        # ---
        # beam stores (combo_dict, norm_score) tuples
        # Initial beam can contain a "sentinel" empty combo with a very low score to start the process,
        # but we only care about non-empty combinations during selection and expansion.
        beam: List[Tuple[Dict[str, Outcome], float]] = [({}, -float("inf"))]

        # 迭代构建组合
        # ---
        # Iteratively build combinations
        for k_round in range(len(offer_items_list)):  # 最多 M 轮 / At most M rounds
            candidates: List[Tuple[Dict[str, Outcome], float]] = []
            # processed_in_this_round 用于避免在同一轮次对完全相同的组合（基于NID集合）进行多次评估
            # ---
            # processed_in_this_round is used to avoid evaluating the exact same combination (based on NID set) multiple times in the same round
            processed_combo_keys_in_this_round = set()

            for current_combo_dict, _current_norm_score in beam:
                for offer_idx, (nid, outcome) in enumerate(offer_items_list):
                    if (
                        nid not in current_combo_dict
                    ):  # 确保不重复添加同一个伙伴的报价到当前路径
                        # ---
                        # Ensure not adding the same partner's offer repeatedly to the current path
                        new_combo_dict_list = list(current_combo_dict.items())
                        new_combo_dict_list.append((nid, outcome))
                        new_combo_dict_list.sort(
                            key=lambda x: x[0]
                        )  # 排序以确保组合键的唯一性
                        # ---
                        # Sort to ensure uniqueness of the combination key

                        # new_combo_dict_list 现在至少包含一个元素
                        # ---
                        # new_combo_dict_list now contains at least one element
                        new_combo_tuple_key = tuple(
                            item[0] for item in new_combo_dict_list
                        )
                        if new_combo_tuple_key in processed_combo_keys_in_this_round:
                            continue
                        processed_combo_keys_in_this_round.add(new_combo_tuple_key)
                        new_combo_dict_final = dict(new_combo_dict_list)

                        # 只有非空组合才进行评估
                        # ---
                        # Only evaluate non-empty combinations
                        if new_combo_dict_final:
                            _raw, norm_score = self.score_offers(
                                offer_combination=new_combo_dict_final,
                                current_im=im,
                                awi=awi,
                            )
                            candidates.append((new_combo_dict_final, norm_score))

            if not candidates:
                break  # 没有新的有效候选组合可以生成 / No new valid candidates can be generated

            # 将上一轮束中的有效（非空）组合也加入候选，因为它们可能是最终解
            # ---
            # Add valid (non-empty) combinations from the previous beam to candidates, as they might be the final solution
            for prev_combo_dict, prev_norm_score in beam:
                if prev_combo_dict:  # 只添加非空组合 / Only add non-empty combinations
                    # 避免重复添加已在candidates中的组合
                    # ---
                    # Avoid re-adding combinations already in candidates (based on object identity or a proper key)
                    # 为简单起见，这里假设如果它在beam中，并且是有效的，就值得再次考虑
                    # ---
                    # For simplicity, assume if it was in the beam and valid, it's worth considering again
                    # 更健壮的做法是检查是否已在candidates中（基于内容）
                    # ---
                    # A more robust approach would be to check if already in candidates (based on content)
                    candidates.append((prev_combo_dict, prev_norm_score))

            # 去重，因为上一轮的beam可能与新生成的candidates有重合
            # ---
            # Deduplicate, as the previous beam might overlap with newly generated candidates
            unique_candidates_dict: Dict[
                Tuple[str, ...], Tuple[Dict[str, Outcome], float]
            ] = {}
            for cand_dict, cand_score in candidates:
                if not cand_dict:
                    continue  # 忽略空的候选 / Ignore empty candidates
                cand_key = tuple(sorted(cand_dict.keys()))
                if (
                    cand_key not in unique_candidates_dict
                    or cand_score > unique_candidates_dict[cand_key][1]
                ):
                    unique_candidates_dict[cand_key] = (cand_dict, cand_score)

            sorted_candidates = sorted(
                list(unique_candidates_dict.values()), key=lambda x: x[1], reverse=True
            )
            beam = sorted_candidates[: self.beam_width]

            if (
                not beam or not beam[0][0]
            ):  # 如果束为空，或者束中最好的也是空组合（不应发生）
                # ---
                # If beam is empty, or the best in beam is an empty combo (should not happen)
                break
            if beam[0][1] < -0.99:  # 如果最好的候选 norm_score 仍然极差
                # ---
                # If the best candidate's norm_score is still extremely poor
                break

        # 从最终的束中选择适应度最高的非空组合
        # ---
        # Select the non-empty combination with the highest fitness from the final beam
        final_best_combo_dict: Optional[Dict[str, Outcome]] = None
        final_best_norm_score: float = -1.0

        for combo_d, n_score in beam:
            if combo_d:  # 确保组合非空 / Ensure combination is non-empty
                if n_score > final_best_norm_score:
                    final_best_norm_score = n_score
                    final_best_combo_dict = combo_d

        if final_best_combo_dict:
            return list(final_best_combo_dict.items()), final_best_norm_score
        else:
            return None, -1.0

    def _evaluate_offer_combinations_simulated_annealing(
        self,
        offers: Dict[str, Outcome],
        im: InventoryManagerCIR,
        awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用模拟退火策略评估组合，主要基于库存得分。
        确保最终选择的组合至少包含一个offer（如果可能）。
        ---
        Evaluates combinations using the Simulated Annealing strategy, primarily based on inventory score.
        Ensures the finally selected combination contains at least one offer (if possible).
        """
        if not offers:
            return None, -1.0

        offer_items_list = list(offers.items())
        num_offers = len(offer_items_list)

        # 初始解：可以从随机选择一个报价开始，以确保初始解非空
        # ---
        # Initial solution: can start by randomly selecting one offer to ensure the initial solution is non-empty
        if num_offers > 0:
            initial_nid, initial_outcome = random.choice(offer_items_list)
            current_solution_dict: Dict[str, Outcome] = {initial_nid: initial_outcome}
        else:  # 理论上不会到这里，因为上面有 if not offers 判断
            # ---
            # Theoretically won't reach here due to the 'if not offers' check above
            return None, -1.0

        _raw_init, current_norm_score = self.score_offers(
            current_solution_dict, im, awi
        )

        best_solution_dict = deepcopy(current_solution_dict)
        best_norm_score = current_norm_score

        temp = self.sa_initial_temp

        for i in range(self.sa_iterations):
            i + 1
            if temp < 1e-3:
                break

            neighbor_solution_dict = deepcopy(current_solution_dict)
            if num_offers == 0:
                break  # Should not happen due to initial check / 由于初始检查，不应发生

            action_type = random.choice(["add", "remove", "swap"])
            action_successful = False  # 标记邻域操作是否成功生成了一个与当前不同的解
            # ---
            # Flag if neighborhood operation successfully generated a different solution

            if action_type == "add" and len(neighbor_solution_dict) < num_offers:
                available_to_add = [
                    item
                    for item in offer_items_list
                    if item[0] not in neighbor_solution_dict
                ]
                if available_to_add:
                    nid_to_add, outcome_to_add = random.choice(available_to_add)
                    neighbor_solution_dict[nid_to_add] = outcome_to_add
                    action_successful = True
            elif (
                action_type == "remove" and len(neighbor_solution_dict) > 1
            ):  # 确保移除后至少还可能有一个（如果目标是保持非空）
                # 或者允许移除到空，但后续评估要处理
                # ---
                # Ensure at least one might remain after removal (if goal is to keep non-empty)
                # Or allow removal to empty, but subsequent evaluation must handle it
                nid_to_remove = random.choice(list(neighbor_solution_dict.keys()))
                del neighbor_solution_dict[nid_to_remove]
                action_successful = True
            elif (
                action_type == "swap" and neighbor_solution_dict
            ):  # 确保当前解非空才能交换
                # ---
                # Ensure current solution is non-empty to swap
                available_to_add = [
                    item
                    for item in offer_items_list
                    if item[0] not in neighbor_solution_dict
                ]
                if available_to_add:  # 必须有东西可以换入
                    # ---
                    # Must have something to swap in
                    nid_to_remove = random.choice(list(neighbor_solution_dict.keys()))
                    removed_outcome = neighbor_solution_dict.pop(nid_to_remove)

                    possible_to_add_for_swap = [
                        item for item in available_to_add if item[0] != nid_to_remove
                    ]
                    if possible_to_add_for_swap:
                        nid_to_add, outcome_to_add = random.choice(
                            possible_to_add_for_swap
                        )
                        neighbor_solution_dict[nid_to_add] = outcome_to_add
                        action_successful = True
                    else:  # 没有其他可换入的，把移除的加回去
                        # ---
                        # No other to swap in, add the removed one back
                        neighbor_solution_dict[nid_to_remove] = removed_outcome

            if (
                not action_successful or not neighbor_solution_dict
            ):  # 如果邻域操作未改变解，或导致空解，则跳过此次迭代
                # （除非我们允许评估空解，但这里我们要求非空）
                # ---
                # If neighborhood op didn't change solution, or resulted in empty solution, skip iteration
                # (unless we allow evaluating empty solutions, but here we require non-empty)
                if (
                    not neighbor_solution_dict and current_solution_dict
                ):  # 如果邻居变空了，但当前非空，则重新生成邻居
                    continue  # If neighbor became empty but current is not, regenerate neighbor

            # 只有当邻域解非空时才评估
            # ---
            # Only evaluate if the neighbor solution is non-empty
            if not neighbor_solution_dict:
                neighbor_norm_score = -float("inf")  # 给空解一个极差的分数
                # ---
                # Give empty solution a very poor score
            else:
                _raw_neighbor, neighbor_norm_score = self.score_offers(
                    neighbor_solution_dict, im, awi
                )

            if neighbor_norm_score > current_norm_score:
                current_solution_dict = deepcopy(neighbor_solution_dict)
                current_norm_score = neighbor_norm_score
                if (
                    current_norm_score > best_norm_score and current_solution_dict
                ):  # 确保最佳解也非空
                    # ---
                    # Ensure best solution is also non-empty
                    best_solution_dict = deepcopy(current_solution_dict)
                    best_norm_score = current_norm_score
            elif temp > 1e-9:  # 仅当温度足够高时才考虑接受差解
                # ---
                # Only consider accepting worse solutions if temperature is high enough
                delta_fitness = current_norm_score - neighbor_norm_score
                acceptance_probability = math.exp(-delta_fitness / temp)
                if (
                    random.random() < acceptance_probability and neighbor_solution_dict
                ):  # 确保接受的也是非空解
                    # ---
                    # Ensure accepted is also non-empty
                    current_solution_dict = deepcopy(neighbor_solution_dict)
                    current_norm_score = neighbor_norm_score

            temp *= self.sa_cooling_rate

        if not best_solution_dict:
            return None, -1.0

        return list(best_solution_dict.items()), best_norm_score

    def _evaluate_offer_combinations(
        self,
        offers: Dict[str, Outcome],
        im: InventoryManagerCIR,
        awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float, float]:
        """
        评估报价组合，主要基于库存得分 (norm_score)。
        在确定最佳组合后，再为其计算一次利润得分 (norm_profit)。
        确保返回的最佳组合至少包含一个offer（如果输入offers非空）。
        ---
        Evaluates offer combinations, primarily based on inventory score (norm_score).
        Profit score (norm_profit) is calculated once for the determined best combination.
        Ensures the returned best combination contains at least one offer (if input offers is non-empty).
        """
        if not offers:
            return None, -1.0, 0.0

        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        best_norm_score: float = (
            -1.0
        )  # 初始化为无效分数 / Initialize to an invalid score

        if self.combo_evaluation_strategy == "k_max":
            best_combination_items, best_norm_score = (
                self._evaluate_offer_combinations_k_max(offers, im, awi)
            )
        elif self.combo_evaluation_strategy == "exhaustive_search":
            best_combination_items, best_norm_score = (
                self._evaluate_offer_combinations_exhaustive(offers, im, awi)
            )
        elif self.combo_evaluation_strategy == "beam_search":
            best_combination_items, best_norm_score = (
                self._evaluate_offer_combinations_beam_search(offers, im, awi)
            )
        elif self.combo_evaluation_strategy == "simulated_annealing":
            best_combination_items, best_norm_score = (
                self._evaluate_offer_combinations_simulated_annealing(offers, im, awi)
            )
        else:
            best_combination_items, best_norm_score = (
                self._evaluate_offer_combinations_k_max(offers, im, awi)
            )

        if best_combination_items:  # 确保找到了一个非空的最佳组合
            # ---
            # Ensure a non-empty best combination was found
            best_combo_dict = dict(best_combination_items)
            _actual_profit, norm_profit_of_best = (
                self._calculate_combination_profit_and_normalize(
                    offer_combination=best_combo_dict, awi=awi
                )
            )
            return best_combination_items, best_norm_score, norm_profit_of_best
        else:
            return None, -1.0, 0.0

    def _calculate_combination_profit_and_normalize(
        self,
        offer_combination: Dict[str, Outcome],
        awi: OneShotAWI,
        # production_cost_per_unit: float = 0.0 # 生产成本明确为0
    ) -> Tuple[float, float]:
        """
        计算报价组合的直接盈利，并将其归一化到 [-1, 1] 区间。
        盈利 = (销售收入) - (采购支出)。生产成本在此版本中设为0。
        归一化基于从 NMI 获取的估算最大潜在盈利和最大潜在亏损。
        1.0 表示非常好的盈利。
        0.0 表示盈亏平衡。
        -1.0 表示较大的亏损。

        返回:
            Tuple[float, float]: (原始盈利, 归一化后的盈利)
        """
        actual_profit = 0.0
        # Represents the profit (revenue - cost) in the best-case price scenario for the agent
        max_potential_profit_scenario = 0.0
        # Represents the profit (revenue - cost) in the worst-case price scenario for the agent
        min_potential_profit_scenario = (
            0.0  # This will likely be negative, representing max loss
        )

        for negotiator_id, outcome in offer_combination.items():
            if not outcome:
                continue
            quantity, _, unit_price = outcome

            nmi = self.get_nmi(negotiator_id)
            is_selling_to_consumer = not self._is_supplier(negotiator_id)

            min_est_price = nmi.issues[UNIT_PRICE].min_value
            max_est_price = nmi.issues[UNIT_PRICE].max_value

            if is_selling_to_consumer:  # We are selling products
                # Actual profit from this offer
                actual_profit += quantity * unit_price
                # Contribution to max potential profit scenario (sell at highest price)
                max_potential_profit_scenario += quantity * max_est_price
                # Contribution to min potential profit scenario (sell at lowest price)
                min_potential_profit_scenario += quantity * min_est_price
            else:  # We are buying raw materials
                # Actual profit from this offer (it's a cost)
                actual_profit -= quantity * unit_price
                # Contribution to max potential profit scenario (buy at lowest price)
                # Cost is minimized, so profit contribution is - (quantity * min_est_price)
                max_potential_profit_scenario -= quantity * min_est_price
                # Contribution to min potential profit scenario (buy at highest price)
                # Cost is maximized, so profit contribution is - (quantity * max_est_price)
                min_potential_profit_scenario -= quantity * max_est_price

        # Normalize the actual_profit
        normalized_profit = 0.0

        # The range of potential profit is [min_potential_profit_scenario, max_potential_profit_scenario]
        # We want to map this range to [-1, 1]

        profit_range = max_potential_profit_scenario - min_potential_profit_scenario

        if profit_range <= 1e-6:  # Effectively zero or invalid range (e.g. max < min)
            if (
                actual_profit > 1e-6
            ):  # If there's actual profit despite no discernible range
                normalized_profit = 1.0
            elif actual_profit < -1e-6:  # If there's actual loss
                normalized_profit = -1.0
            else:  # actual_profit is also near zero
                normalized_profit = 0.0
        else:
            # Linear mapping: y = (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min
            # Here, x is actual_profit, [x_min, x_max] is [min_potential_profit_scenario, max_potential_profit_scenario]
            # And [y_min, y_max] is [-1, 1]
            normalized_profit = (
                -1.0
                + 2.0 * (actual_profit - min_potential_profit_scenario) / profit_range
            )

        # Clamp the result to [-1, 1] in case actual_profit falls outside the estimated scenario range
        normalized_profit = max(-1.0, min(1.0, normalized_profit))

        return actual_profit, normalized_profit

    def _generate_counter_offer(
        self,
        negotiator_id: str,
        original_offer: Outcome,
        states: Dict[str, SAOState],  # 参数名为 states (复数)
        optimize_for_inventory: bool,
        optimize_for_profit: bool,
        inventory_target_quantity: Optional[int] = None,
        # For Case 1.2, specific need from this partner / 针对情况1.2，来自此伙伴的特定需求
    ) -> Optional[Outcome]:
        """
        Generates a counter-offer based on optimization goals using heuristics.
        It adjusts quantity, time, and price of the original_offer.
        For time adjustments, it simulates the impact on inventory score.
        使用启发式方法，根据优化目标生成还价。
        它会调整原始报价的数量、时间和价格。
        对于时间调整，它会模拟其对库存分数的影响。
        """
        orig_q, orig_t, orig_p = original_offer

        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        min_q_nmi, max_q_nmi = (
            nmi.issues[QUANTITY].min_value,
            nmi.issues[QUANTITY].max_value,
        )
        min_p_nmi, max_p_nmi = (
            nmi.issues[UNIT_PRICE].min_value,
            nmi.issues[UNIT_PRICE].max_value,
        )
        min_t_nmi, max_t_nmi = nmi.issues[TIME].min_value, nmi.issues[TIME].max_value

        # Initialize new_q, new_t, new_p with original values
        # 用原始值初始化 new_q, new_t, new_p
        new_q, new_t, new_p = orig_q, orig_t, orig_p
        is_buying = self._is_supplier(
            negotiator_id
        )  # True if we are buying from this supplier / 如果我们从此供应商处购买，则为 True

        # Heuristic parameters
        # 启发式参数

        # --- Store initial proposed quantity and price before time evaluation ---
        # --- 在时间评估前存储初始提议的数量和价格 ---

        if optimize_for_inventory:
            # Quantity adjustment logic (applied before time evaluation for simplicity in this version)
            # 数量调整逻辑 (在此版本中为简单起见，在时间评估前应用)
            if is_buying:
                current_agent_shortfall = (
                    self.total_insufficient
                    if self.total_insufficient is not None
                    else 0
                )
                effective_need_delta = (
                    inventory_target_quantity
                    if inventory_target_quantity is not None
                    else current_agent_shortfall
                )
                if effective_need_delta > 0:
                    new_q = min(orig_q * 1.2, float(effective_need_delta))
                    new_q = max(new_q, min_q_nmi)
                else:
                    new_q = orig_q * 0.8
                    new_q = max(new_q, min_q_nmi)
            else:
                new_q = orig_q * 1.2
        new_q = int(round(new_q))
        new_q = max(int(round(min_q_nmi)), min(new_q, int(round(max_q_nmi))))
        if new_q <= 0:
            if min_q_nmi > 0:
                new_q = int(round(min_q_nmi))
            else:
                return None
        candidate_times = {orig_t}
        if is_buying and orig_t > min_t_nmi:
            candidate_times.add(max(min_t_nmi, orig_t - 1))
        elif not is_buying and orig_t < max_t_nmi:
            candidate_times.add(min(max_t_nmi, orig_t + 1))
        best_t_for_inventory = orig_t
        highest_simulated_score_for_time = -float("inf")
        price_for_time_eval = orig_p
        for t_candidate in candidate_times:
            offer_to_score = {negotiator_id: (new_q, t_candidate, price_for_time_eval)}
            _, current_sim_score = self.score_offers(offer_to_score, self.im, self.awi)
            if current_sim_score > highest_simulated_score_for_time:
                highest_simulated_score_for_time = current_sim_score
                best_t_for_inventory = t_candidate
        new_t = best_t_for_inventory
        nmi_mid_p = (min_p_nmi + max_p_nmi) / 2.0
        base_target_p = nmi_mid_p
        if optimize_for_profit:
            base_target_p = min_p_nmi if is_buying else max_p_nmi
        elif optimize_for_inventory:
            base_target_p = max_p_nmi if is_buying else min_p_nmi

        relative_time = states.get(negotiator_id).relative_time
        round_concession_factor = relative_time**self.concession_curve_power
        opponent_last_offer_outcome = self._last_partner_offer.get(negotiator_id)
        opponent_last_price = (
            opponent_last_offer_outcome[UNIT_PRICE]
            if opponent_last_offer_outcome
            else orig_p
        )
        price_towards_opponent = (
            1.0 - self.price_concession_opponent_factor
        ) * base_target_p + self.price_concession_opponent_factor * opponent_last_price
        inventory_pressure = 0.0
        if optimize_for_inventory:
            if is_buying:
                raw_stock = self.im.get_inventory_summary(
                    self.awi.current_step, MaterialType.RAW
                ).get("current_stock", 0)
                if raw_stock < self.inventory_pressure_threshold_raw:
                    inventory_pressure = 1.0 - (
                        raw_stock / self.inventory_pressure_threshold_raw
                        if self.inventory_pressure_threshold_raw > 0
                        else 1.0
                    )
                    inventory_pressure = max(0.0, min(1.0, inventory_pressure))
            else:
                product_stock = self.im.get_inventory_summary(
                    self.awi.current_step, MaterialType.PRODUCT
                ).get("current_stock", 0)
                if product_stock > self.inventory_pressure_threshold_product:
                    inventory_pressure = (
                        (product_stock - self.inventory_pressure_threshold_product)
                        / self.inventory_pressure_threshold_product
                        if self.inventory_pressure_threshold_product > 0
                        else 0.0
                    )
                    inventory_pressure = max(0.0, inventory_pressure)
        final_target_p = price_towards_opponent
        concession_magnitude = (
            round_concession_factor + inventory_pressure
        ) * self.price_concession_round_factor
        if is_buying:
            max_possible_concession_price = max_p_nmi
            final_target_p = (
                price_towards_opponent
                + (max_possible_concession_price - price_towards_opponent)
                * concession_magnitude
            )
        else:
            min_possible_concession_price = min_p_nmi
            final_target_p = (
                price_towards_opponent
                - (price_towards_opponent - min_possible_concession_price)
                * concession_magnitude
            )
        if new_t != orig_t:
            if is_buying and new_t < orig_t:
                final_target_p *= 1 + self.price_concession_time_factor
            elif not is_buying and new_t > orig_t:
                final_target_p *= 1 - self.price_concession_time_factor
        new_p = final_target_p
        new_t = int(round(new_t))
        new_t = max(min_t_nmi, min(new_t, max_t_nmi))
        new_p = max(min_p_nmi, min(new_p, max_p_nmi))
        if new_p <= 0:
            if min_p_nmi > 0.001:
                new_p = min_p_nmi
            else:
                new_p = 0.01

        # Avoid countering with an offer identical to the original
        # 避免提出与原始报价相同的还价
        if new_q == orig_q and new_t == orig_t and abs(new_p - orig_p) < 1e-5:
            return None
        return new_q, new_t, new_p

    def counter_all(
        self,
        offers: Dict[
            str, Outcome
        ],  # partner_id -> (q, t, p) / 伙伴ID -> (数量, 时间, 价格)
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}
        if not offers:
            return responses

        if not self.im or not self.awi:
            for nid_key in offers.keys():
                responses[nid_key] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses
        for nid, offer_outcome in offers.items():
            if offer_outcome:
                self._last_partner_offer[nid] = offer_outcome
        for nid in offers.keys():
            responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
        current_day = self.awi.current_step
        time_decay_factor = self.threshold_time_decay_factor**current_day
        dynamic_p_threshold = self.p_threshold * time_decay_factor
        dynamic_q_threshold = self.q_threshold * time_decay_factor
        best_combination_items, norm_score, norm_profit = (
            self._evaluate_offer_combinations(offers, self.im, self.awi)
        )

        # --- 新增调试输出 ---
        # --- Added debug output ---
        # --- 调试输出结束 ---
        # --- End of debug output ---

        if (
            best_combination_items is None
        ):  # No valid combination found / 未找到有效组合
            # 信息 ({self.id} @ {self.awi.current_step}): _evaluate_offer_combinations 未找到最佳组合。所有报价均被拒绝。
            return responses  # All already set to REJECT / 所有均已设置为拒绝

        best_combo_outcomes_dict = dict(best_combination_items)
        best_combo_nids_set = set(best_combo_outcomes_dict.keys())
        if norm_score > dynamic_p_threshold and norm_profit > dynamic_q_threshold:
            for nid, outcome in best_combo_outcomes_dict.items():
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, outcome)

            # 1.2 Counter offers to OTHERS if unmet needs exist (primarily for procurement of raw materials)
            # 1.2 如果存在未满足的需求，则向其他方提出还价 (主要针对原材料采购)
            # Simulate accepted offers in a temporary IM to get a more accurate remaining need.
            # 在临时IM中模拟已接受的报价，以获得更准确的剩余需求。
            temp_im_for_case1_counters = deepcopy(self.im)
            temp_im_for_case1_counters.is_deepcopy = True
            for nid_accepted, outcome_accepted in best_combo_outcomes_dict.items():
                is_supply_contract = self._is_supplier(nid_accepted)
                contract_type = (
                    IMContractType.SUPPLY
                    if is_supply_contract
                    else IMContractType.DEMAND
                )
                material_type = (
                    MaterialType.RAW if is_supply_contract else MaterialType.PRODUCT
                )
                # Create a unique ID for the temporary contract for simulation
                # 为模拟创建临时合约的唯一ID
                temp_contract_id = f"temp_accept_{nid_accepted}_{self.id}_{self.awi.current_step}_{uuid4()}"

                sim_contract = IMContract(
                    contract_id=temp_contract_id,
                    partner_id=nid_accepted,
                    type=contract_type,
                    quantity=int(outcome_accepted[QUANTITY]),
                    price=outcome_accepted[UNIT_PRICE],
                    delivery_time=outcome_accepted[TIME],
                    material_type=material_type,
                    bankruptcy_risk=0.0,
                )
                temp_im_for_case1_counters.add_transaction(
                    sim_contract
                )  # This updates plan in temp_im / 这会更新 temp_im 中的计划

            # Get remaining raw material insufficiency after hypothetically accepting the best combo
            # 在假设接受最佳组合后，获取剩余的原材料不足量
            remaining_need_after_accepts = (
                temp_im_for_case1_counters.get_total_insufficient_raw(
                    self.awi.current_step, horizon=14
                )
            )

            if remaining_need_after_accepts > 0:
                # Identify negotiators not in the best combo, who are suppliers (for raw material needs)
                # 识别不在最佳组合中且为供应商的谈判者 (针对原材料需求)
                negotiators_to_counter_case1 = [
                    nid
                    for nid in offers.keys()
                    if nid not in best_combo_nids_set and self._is_supplier(nid)
                ]
                if negotiators_to_counter_case1:
                    # Distribute the remaining need among these negotiators
                    # 将剩余需求分配给这些谈判者
                    qty_per_negotiator_case1 = math.ceil(
                        remaining_need_after_accepts / len(negotiators_to_counter_case1)
                    )
                    qty_per_negotiator_case1 = max(
                        1, qty_per_negotiator_case1
                    )  # Ensure at least 1 / 确保至少为1

                    for nid_to_counter in negotiators_to_counter_case1:
                        original_offer = offers[nid_to_counter]
                        # Generate counter-offer focusing on inventory (filling the need)
                        # 生成以库存为重点的还价 (填补需求)
                        counter_outcome = self._generate_counter_offer(
                            nid_to_counter,
                            original_offer,
                            states,
                            optimize_for_inventory=True,
                            optimize_for_profit=False,  # Primary focus is filling the need / 主要重点是填补需求
                            inventory_target_quantity=qty_per_negotiator_case1,
                        )
                        if counter_outcome:
                            responses[nid_to_counter] = SAOResponse(
                                ResponseType.REJECT_OFFER, counter_outcome
                            )
        elif norm_score <= dynamic_p_threshold:
            also_optimize_for_profit = norm_profit <= dynamic_q_threshold
            if also_optimize_for_profit:
                # 信息 ({self.id} @ {self.awi.current_step}): 情况2/4 (合并 - 情况4类型): 对所有报价进行库存优化然后利润优化 (分数差, 利润差)。
                pass
            else:  # norm_profit > self.q_threshold (original Case 2) / norm_profit > self.q_threshold (原始情况2)
                # 信息 ({self.id} @ {self.awi.current_step}): 情况2/4 (合并 - 情况2类型): 对所有报价进行库存优化 (分数差, 利润OK)。
                pass

            # Do NOT accept any offers from `best_combination` or any other.
            # Counter all offers based on the determined optimization strategy.
            # 不接受来自 `best_combination` 或任何其他组合的任何报价。
            # 根据确定的优化策略对所有报价进行还价。
            for nid, original_offer in offers.items():
                counter_outcome = self._generate_counter_offer(
                    nid,
                    original_offer,
                    states,
                    optimize_for_inventory=True,
                    optimize_for_profit=also_optimize_for_profit,
                )
                if counter_outcome:
                    responses[nid] = SAOResponse(
                        ResponseType.REJECT_OFFER, counter_outcome
                    )
        elif norm_profit <= dynamic_q_threshold:
            # 信息 ({self.id} @ {self.awi.current_step}): 情况3: 对所有报价进行价格优化 (分数OK, 利润差)。

            # Do NOT accept any offers.
            # Counter all offers to improve profit; inventory score was deemed acceptable.
            # 不接受任何报价。
            # 对所有报价进行还价以提高利润；库存分数被认为是可接受的。
            for nid, original_offer in offers.items():
                counter_outcome = self._generate_counter_offer(
                    nid,
                    original_offer,
                    states,
                    optimize_for_inventory=False,
                    optimize_for_profit=True,
                )
                if counter_outcome:
                    responses[nid] = SAOResponse(
                        ResponseType.REJECT_OFFER, counter_outcome
                    )

        else:
            # This path should ideally not be reached if all conditions are covered.
            # All offers will remain REJECTED by default.
            # 如果所有条件都已覆盖，则理想情况下不应到达此路径。
            # 默认情况下，所有报价都将保持被拒绝状态。
            pass

        return responses

    # ------------------------------------------------------------------
    # 🌟 6. 谈判回调
    # ------------------------------------------------------------------
    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        assert self.im, "CustomInventoryManager 尚未初始化"
        current_day = self.awi.current_step  # 获取当前日期
        partner = self.get_partner_id(contract)
        is_supply = partner in self.awi.my_suppliers

        if not is_supply:
            self._sales_successes_since_margin_update += 1

        im_type = IMContractType.SUPPLY if is_supply else IMContractType.DEMAND
        mat_type = MaterialType.RAW if is_supply else MaterialType.PRODUCT
        agreement = contract.agreement
        if not agreement:
            return

        new_c = IMContract(
            contract_id=contract.id,
            partner_id=partner,
            type=im_type,
            quantity=int(agreement["quantity"]),
            price=agreement["unit_price"],
            delivery_time=agreement["time"],
            bankruptcy_risk=0.0,
            material_type=mat_type,
        )
        added = self.im.add_transaction(new_c)
        assert added, (
            f"❌ ({self.id}) CustomIM.add_transaction 失败! contract={contract.id}"
        )

        self.today_insufficient = self.im.get_today_insufficient_raw(current_day)
        self.total_insufficient = self.im.get_total_insufficient_raw(
            current_day, horizon=14
        )

        # 更新每日采购/销售完成量
        if is_supply:  # 原材料采购合同
            self.purchase_completed[current_day] = (
                self.purchase_completed.get(current_day, 0) + agreement["quantity"]
            )
            # 更新用于调整超采购因子的实际达成量
            # Update actual achieved quantity for over-procurement factor adjustment
            self.daily_achieved_procurement_for_adjustment[current_day] = (
                self.daily_achieved_procurement_for_adjustment.get(current_day, 0)
                + agreement["quantity"]
            )

        elif not is_supply:  # 产成品销售合同
            self.sales_completed[current_day] = (
                self.sales_completed.get(current_day, 0) + agreement["quantity"]
            )

    def get_partner_id(self, contract: Contract) -> str:
        for p in contract.partners:
            if p != self.id:
                return p
        return "unknown_partner"

    def _print_daily_status_report(self, result) -> None:
        """输出每日库存、生产和销售状态报告，包括未来预测"""

        current_day = self.awi.current_step
        horizon_days = min(10, self.awi.n_steps - current_day)
        header = "|   日期    |  原料真库存  |  原料预计库存   | 计划生产  |  剩余产能  |  产品真库存  |  产品预计库存  |  已签署销售量  |  实际产品交付  |"
        "|" + "-" * (len(header) + 24) + "|"

        # 当前日期及未来预测
        for day_offset in range(horizon_days):
            forecast_day = current_day + day_offset

            # 从IM获取数据
            raw_summary = self.im.get_inventory_summary(forecast_day, MaterialType.RAW)
            product_summary = self.im.get_inventory_summary(
                forecast_day, MaterialType.PRODUCT
            )

            raw_summary.get("current_stock", 0)
            raw_summary.get("estimated_available", 0)

            product_summary.get("current_stock", 0)
            product_summary.get("estimated_available", 0)

            # 计划生产量 - CustomIM stores production_plan as Dict[day, qty]
            self.im.production_plan.get(forecast_day, 0)

            # 剩余产能
            self.im.get_available_production_capacity(forecast_day)

            # 已签署的销售合同数量 - CustomIM stores these in self.pending_demand_contracts
            signed_sales = 0
            # Iterate through pending_demand_contracts that are for the forecast_day
            for dem_contract in self.im.pending_demand_contracts:
                if dem_contract.delivery_time == forecast_day:
                    signed_sales += dem_contract.quantity

            # Delivered products might not be directly in result dict from CustomIM.
            # This was from the old IM. Let's assume 0 for now or get from CustomIM if it provides this.
            # For simplicity, let's show 0 if not available in result.
            (
                result.get("delivered_products", 0)
                if isinstance(result, dict) and day_offset == 0
                else 0
            )

            # 格式化并输出


# ----------------------------------------------------
# Inventory Cost Score Calculation Helper
# ----------------------------------------------------


if __name__ == "__main__":
    pass
