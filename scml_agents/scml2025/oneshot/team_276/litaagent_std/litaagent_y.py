#!/usr/bin/env python
"""
LitaAgentY — SCML 2025  Standard 赛道谈判代理（重构版）
===================================================
LitaAgentY — SCML 2025 Standard Track Negotiation Agent (Refactored)
=================================================================

本文件 **完全重写** 了旧版 *litaagent_n.py* 中混乱的出价逻辑，
并修复了与 `InventoryManager` 的接口 BUG。
This file **completely rewrites** the chaotic bidding logic in the old *litaagent_n.py*,
and fixes interface BUGs with `InventoryManager`.

核心改动
--------
Core Changes
--------
1. **采购三分法**：把原料购买划分为 `紧急需求 / 计划性需求 / 可选性采购` 三类，
   对应 `_process_emergency_supply_offers()` / `_process_planned_supply_offers()` /
   `_process_optional_supply_offers()` 三个子模块。
   **Three-Tier Procurement**: Divides raw material purchases into `Emergency Demand / Planned Demand / Optional Procurement` three categories,
   corresponding to the three submodules: `_process_emergency_supply_offers()`, `_process_planned_supply_offers()`, and
   `_process_optional_supply_offers()`.
2. **销售产能约束**：新增 `_process_sales_offers()`，严格保证在交货期内
   不会签约超出总产能的产品合同，且确保售价满足 `min_profit_margin`。
   **Sales Capacity Constraint**: Added `_process_sales_offers()` to strictly ensure that
   product contracts exceeding total production capacity are not signed within the delivery period, and that selling prices meet `min_profit_margin`.
3. **利润策略可调**：`min_profit_margin` 与 `cheap_price_discount` 两个参数
   可在运行时动态调整；并预留接口 `update_profit_strategy()` 供 RL 或
   外部策略模块调用。
   **Adjustable Profit Strategy**: The `min_profit_margin` and `cheap_price_discount` parameters
   can be dynamically adjusted at runtime; an interface `update_profit_strategy()` is reserved for RL or
   external strategy modules.
4. **IM 交互修复**：在 `on_negotiation_success()` 中正确解析对手 ID，
   构造 `IMContract` 并调用 `InventoryManager.add_transaction()`；断言
   添加成功并打印日志。
   **IM Interaction Fix**: Correctly parses opponent ID in `on_negotiation_success()`,
   constructs `IMContract`, and calls `InventoryManager.add_transaction()`; asserts
   successful addition and prints logs.
5. **模块化 `counter_all()`**：顶层逻辑只负责按伙伴角色拆分报价并分发
   到四个子函数，代码层次清晰，可维护性大幅提升。
   **Modular `counter_all()`**: The top-level logic is only responsible for splitting offers by partner role and distributing them
   to four sub-functions, significantly improving code hierarchy clarity and maintainability.
6. **保持 RL 接口**：保留 ObservationManager / ActionManager 等占位，
   不破坏未来集成智能策略的接口。
   **Retain RL Interface**: Keeps placeholders like ObservationManager / ActionManager,
   without breaking the interface for future integration of intelligent strategies.
7. **早期计划采购**：利用 `InventoryManager` 的需求预测，在罚金较低时
   提前锁定原料，减少后期短缺罚金。
   **Early Planned Procurement**: Utilizes `InventoryManager`'s demand forecasting to
   lock in raw materials early when penalties are low, reducing late-stage shortage penalties.
8. **数量敏感的让步**：当短缺风险增大时更倾向接受更大数量，避免多轮议价。
   **Quantity-Sensitive Concession**: More inclined to accept larger quantities when shortage risk increases, avoiding multiple negotiation rounds.
9. **对手建模增强**：记录伙伴的合同成功率与均价，估计其保留价格以调整报价。
   **Enhanced Opponent Modeling**: Records partner's contract success rate and average price, estimates their reservation price to adjust offers.
10. **帕累托意识还价**：反价时综合调整价格、数量与交货期，尝试沿帕累托前沿
    探索互利方案。
    **Pareto-Aware Counter-Offers**: Comprehensively adjusts price, quantity, and delivery time when countering, attempting to explore
    mutually beneficial solutions along the Pareto frontier.
11. **贝叶斯对手建模**：通过在线逻辑回归更新每个伙伴的接受概率，推断其保
    留价格并生成更趋于帕累托最优的报价。
    **Bayesian Opponent Modeling**: Updates each partner's acceptance probability via online logistic regression, infers their reservation
    price, and generates offers closer to Pareto optimality.

使用说明
--------
Usage Instructions
--------
- 关键参数：
  Key Parameters:
    * `min_profit_margin`   —— 最低利润率要求（如 0.10 ⇒ 10%）。
                              Minimum profit margin requirement (e.g., 0.10 ⇒ 10%).
    * `cheap_price_discount`—— 机会性囤货阈值，低于市场均价 *该比例* 视为超低价。
                              Opportunistic stockpiling threshold, prices below market average * this ratio are considered ultra-low.
- 可在外部通过 `agent.update_profit_strategy()` 动态修改。
  Can be dynamically modified externally via `agent.update_profit_strategy()`.
- 如需接入 RL，可在 `decide_with_model()` 中填充模型调用逻辑。
  If RL integration is needed, fill in model call logic in `decide_with_model()`.
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter, defaultdict  # Added defaultdict / 添加了 defaultdict

# ------------------ 基础依赖 ------------------
# Basic Dependencies
# ------------------
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from numpy.random import choice as np_choice  # type: ignore
from scml.std import (
    QUANTITY,
    TIME,
    UNIT_PRICE,
    StdAWI,
    StdSyncAgent,
)

# 内部工具 & manager
# Internal Tools & Manager
from .inventory_manager_n import (
    IMContract,
    IMContractType,
    InventoryManager,
    MaterialType,
)

__all__ = ["LitaAgentY"]

# ------------------ 辅助函数 ------------------
# Helper functions
# ------------------


def _split_partners(partners: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """按 50 % / 30 % / 20 % 三段切分伙伴列表。"""
    """Splits the partner list into three segments: 50% / 30% / 20%."""
    # Split partners into 50%, 30% and 20% groups
    n = len(partners)
    return (
        partners[: int(n * 0.5)],
        partners[int(n * 0.5) : int(n * 0.8)],
        partners[int(n * 0.8) :],
    )


def _distribute(q: int, n: int) -> List[int]:
    """随机将 ``q`` 单位分配到 ``n`` 个桶，保证每桶至少 1（若可行）。"""
    """Randomly distributes ``q`` units into ``n`` buckets, ensuring each bucket gets at least 1 (if feasible)."""
    # Randomly distribute ``q`` units into ``n`` buckets, each getting at least one if possible
    if n <= 0:
        return []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n

    r = Counter(np_choice(n, q - n))
    return [r.get(i, 0) + 1 for i in range(n)]


# ------------------ 主代理实现 ------------------
# Main agent implementation
# ------------------


class LitaAgentY(StdSyncAgent):
    """重构后的 LitaAgent N。支持三类采购策略与产能约束销售。"""

    """Refactored LitaAgent N. Supports three types of procurement strategies and capacity-constrained sales."""

    # ------------------------------------------------------------------
    # 🌟 1. 初始化
    # 🌟 1. Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args,
        min_profit_margin: float = 0.10,
        cheap_price_discount: float = 0.70,
        ptoday: float = 1.00,
        concession_curve_power: float = 1.5,
        capacity_tight_margin_increase: float = 0.07,
        procurement_cash_flow_limit_percent: float = 0.75,  # Added from Step 6 / 从步骤6添加
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        # Parameters
        self.total_insufficient = None
        self.today_insufficient = None
        self.min_profit_margin = min_profit_margin
        self.initial_min_profit_margin = (
            min_profit_margin  # Added from Step 7 / 从步骤7添加
        )
        self.cheap_price_discount = cheap_price_discount
        self.procurement_cash_flow_limit_percent = (
            procurement_cash_flow_limit_percent  # Added from Step 6 / 从步骤6添加
        )
        self.concession_curve_power = (
            concession_curve_power  # Added from Step 9.b / 从步骤9.b添加
        )
        self.capacity_tight_margin_increase = (
            capacity_tight_margin_increase  # Added from Step 9.d / 从步骤9.d添加
        )

        if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
            print(
                f"🤖 LitaAgentY {self.id} initialized with: \n"
                f"  min_profit_margin={self.min_profit_margin:.3f}, \n"
                f"  initial_min_profit_margin={self.initial_min_profit_margin:.3f}, \n"
                f"  cheap_price_discount={self.cheap_price_discount:.2f}, \n"
                f"  procurement_cash_flow_limit_percent={self.procurement_cash_flow_limit_percent:.2f}, \n"
                f"  concession_curve_power={self.concession_curve_power:.2f}, \n"
                f"  capacity_tight_margin_increase={self.capacity_tight_margin_increase:.3f}"
            )

        # —— 运行时变量 ——
        # Runtime Variables
        self.im: InventoryManager | None = None
        self._market_price_avg: float = 0.0
        self._market_material_price_avg: float = 0.0
        self._market_product_price_avg: float = 0.0
        self._recent_material_prices: List[float] = []
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30
        self._ptoday: float = ptoday
        self.model = None
        self.concession_model = None
        self._last_offer_price: Dict[str, float] = {}
        self.sales_completed: Dict[int, int] = {}
        self.purchase_completed: Dict[int, int] = {}

        self.partner_stats: Dict[str, Dict[str, float]] = {}
        self.partner_models: Dict[str, Dict[str, float]] = {}
        self._last_partner_offer: Dict[
            str, float
        ] = {}  # Stores the last price offered by a partner / 存储伙伴的最新报价价格

        # Counters for dynamic profit margin adjustment (Added from Step 7)
        # 用于动态利润率调整的计数器 (从步骤7添加)
        self._sales_successes_since_margin_update: int = 0
        self._sales_failures_since_margin_update: int = 0

    # ------------------------------------------------------------------
    # 🌟 2. World / 日常回调
    # 🌟 2. World / Daily Callbacks
    # ------------------------------------------------------------------

    def init(self) -> None:
        """在 World 初始化后调用；此处创建库存管理器。"""
        """Called after World initialization; InventoryManager is created here."""
        self.im = InventoryManager(
            raw_storage_cost=self.awi.current_storage_cost,
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=0,  # 如果有工艺成本可在此填写 / Fill in processing cost here if any
            daily_production_capacity=self.awi.n_lines,
            max_day=self.awi.n_steps,
        )
        if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
            pass  # print(f"🤖 LitaAgentY {self.id} IM initialized. Daily Capacity: {self.im.daily_production_capacity}")

    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        """Called before each day starts, to synchronize key daily demand information."""
        assert self.im, (
            "InventoryManager 尚未初始化!"
        )  # InventoryManager not initialized!
        current_day = (
            self.awi.current_step
        )  # Use local var for f-string clarity / 使用局部变量以提高f-string清晰度

        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(current_day, 0)
        self.purchase_completed.setdefault(current_day, 0)

        # MODIFIED: 先加入外生协议，再计算需求
        # MODIFIED: Add exogenous contracts first, then calculate demand
        # 首先将外生协议写入im
        # First, write exogenous contracts into the inventory manager
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            if exogenous_contract_quantity > 0:  # Added from Step 11 / 从步骤11添加
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = (
                    "simulator_exogenous_supply"  # More specific name / 更具体的名称
                )
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.SUPPLY,
                    quantity=exogenous_contract_quantity,
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day / 外生协议为当日
                    bankruptcy_risk=0,
                    material_type=MaterialType.RAW,
                )
                self.im.add_transaction(exogenous_contract)
                if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
                    print(
                        f"📥 Day {current_day} ({self.id}): Added exogenous SUPPLY contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}"
                    )

        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            if exogenous_contract_quantity > 0:  # Added from Step 11 / 从步骤11添加
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = (
                    "simulator_exogenous_demand"  # More specific name / 更具体的名称
                )
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.DEMAND,
                    quantity=exogenous_contract_quantity,
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day / 外生协议为当日
                    bankruptcy_risk=0,
                    material_type=MaterialType.PRODUCT,
                )
                self.im.add_transaction(exogenous_contract)
                if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
                    print(
                        f"📤 Day {current_day} ({self.id}): Added exogenous DEMAND contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}"
                    )

        # 在外生协议添加后，再计算需求
        # After exogenous contracts are added, then calculate demand
        self.today_insufficient = self.im.get_today_insufficient(current_day)
        self.total_insufficient = self.im.get_total_insufficient(current_day)
        if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
            print(
                f"🌞 Day {current_day} ({self.id}) starting. Today Insufficient Raw: {self.today_insufficient}, Total Insufficient Raw (horizon): {self.total_insufficient} (calculated AFTER exogenous contracts)"
            )

        # Update dynamic parameters (Added from Step 4 & 7)
        # 更新动态参数 (从步骤4和7添加)
        self._update_dynamic_stockpiling_parameters()
        self._update_dynamic_profit_margin_parameters()

    def step(self) -> None:
        """每天结束时调用：执行 IM 的日终操作并刷新市场均价。"""
        """Called at the end of each day: executes IM's end-of-day operations and refreshes market average prices."""
        assert self.im, (
            "InventoryManager 尚未初始化!"
        )  # InventoryManager not initialized!
        # 让 IM 完成收货 / 生产 / 交付 / 规划
        # Let IM complete receiving / production / delivery / planning
        result = self.im.process_day_operations()
        self.im.update_day()  # This increments self.im.current_day / 这会增加 self.im.current_day
        # —— 更新市场均价估计 ——
        # Update market average price estimates
        # Ensure lists are not empty before calculating average
        # 计算平均值前确保列表不为空
        if self._recent_material_prices:
            self._market_material_price_avg = sum(self._recent_material_prices) / len(
                self._recent_material_prices
            )
        if self._recent_product_prices:
            self._market_product_price_avg = sum(self._recent_product_prices) / len(
                self._recent_product_prices
            )
        if os.path.exists("env.test"):  # Added from Step 11 / 从步骤11添加
            print(
                f"🌙 Day {self.awi.current_step} ({self.id}) ending. Market Material Avg Price: {self._market_material_price_avg:.2f}, Market Product Avg Price: {self._market_product_price_avg:.2f}. IM is now on day {self.im.current_day}."
            )

        # 输出每日状态报告
        # Output daily status report
        self._print_daily_status_report(result)

    # Method from Step 4 (Turn 15), logging improved in Step 11 (Turn 37)
    # 方法来自步骤4 (轮次15), 日志在步骤11 (轮次37) 改进
    def _update_dynamic_stockpiling_parameters(self) -> None:
        """Dynamically adjusts cheap_price_discount based on game state."""
        """根据博弈状态动态调整 cheap_price_discount。"""
        if not self.im:
            return

        current_day = self.awi.current_step
        total_days = self.awi.n_steps

        if total_days == 0:
            return

        current_raw_inventory = self.im.get_inventory_summary(
            current_day, MaterialType.RAW
        )["current_stock"]

        future_total_demand_horizon = min(total_days, current_day + 10)
        future_total_demand = 0
        if self.im:
            for d_iter in range(current_day + 1, future_total_demand_horizon + 1):
                if d_iter >= total_days:
                    break
                future_total_demand += self.im.get_total_insufficient(
                    d_iter
                )  # This is net need / 这是净需求

        market_avg_raw_price = self._market_material_price_avg

        if current_day > total_days * 0.8:  # Late game / 游戏后期
            new_cheap_discount = 0.40
            reason = "Late game"
        elif current_day > total_days * 0.5:  # Mid game / 游戏中期
            new_cheap_discount = 0.60
            reason = "Mid game"
        else:  # Early game / 游戏早期
            new_cheap_discount = 0.70
            reason = "Early game"

        if future_total_demand > 0:  # If there is future demand / 如果未来有需求
            if current_raw_inventory > future_total_demand * 1.5:
                new_cheap_discount = min(new_cheap_discount, 0.50)
                reason += ", High inventory (>150% demand)"
            elif current_raw_inventory > future_total_demand * 1.0:
                new_cheap_discount = min(new_cheap_discount, 0.60)
                reason += ", Sufficient inventory (>100% demand)"
        elif (
            current_day > 5
        ):  # No future demand and not very early game / 没有未来需求且不是非常早期
            pass  # Keep game stage based discount / 保留基于游戏阶段的折扣

        if (
            future_total_demand > 0
            and current_raw_inventory < future_total_demand * 0.5
        ):
            new_cheap_discount = max(
                new_cheap_discount, 0.80
            )  # Be more aggressive if low stock and future demand / 如果库存低且未来有需求，则更积极
            reason += ", Low inventory (<50% demand) & future demand exists"

        if (
            future_total_demand == 0 and current_day > 5
        ):  # No future demand and not very early game / 没有未来需求且不是非常早期
            new_cheap_discount = min(
                new_cheap_discount, 0.30
            )  # Be very conservative if no future demand / 如果没有未来需求，则非常保守
            reason = "No future demand (override)"

        final_new_cheap_discount = max(
            0.20, min(0.85, new_cheap_discount)
        )  # Clamp the discount / 限制折扣范围

        if (
            abs(self.cheap_price_discount - final_new_cheap_discount) > 1e-3
        ):  # If changed significantly / 如果变化显著
            old_discount = self.cheap_price_discount
            self.update_profit_strategy(cheap_price_discount=final_new_cheap_discount)
            if os.path.exists("env.test"):
                print(
                    f"📈 Day {current_day} ({self.id}): cheap_price_discount changed from {old_discount:.2f} to {self.cheap_price_discount:.2f}. Reason: {reason}. "
                    f"InvRaw: {current_raw_inventory}, FutDemandRaw(10d): {future_total_demand}, MktPriceRaw: {market_avg_raw_price:.2f}"
                )
        elif os.path.exists(
            "env.test"
        ):  # Log even if not changed, for transparency / 即使未更改也记录日志，以提高透明度
            print(
                f"🔎 Day {current_day} ({self.id}): cheap_price_discount maintained at {self.cheap_price_discount:.2f}. Evaluated Reason: {reason}. "
                f"InvRaw: {current_raw_inventory}, FutDemandRaw(10d): {future_total_demand}, MktPriceRaw: {market_avg_raw_price:.2f}"
            )

    # Method from Step 9.b (Turn 30)
    # 方法来自步骤9.b (轮次30)
    def get_avg_raw_cost_fallback(
        self,
        current_day_for_im_summary: int,
        best_price_pid_for_fallback: str | None = None,
    ) -> float:
        """Gets average raw material cost, with fallbacks."""
        """获取平均原材料成本，带回退机制。"""
        avg_raw_cost = 0.0
        if self._market_material_price_avg > 0:
            avg_raw_cost = self._market_material_price_avg
        elif self.im:
            im_avg_raw_cost = self.im.get_inventory_summary(
                current_day_for_im_summary, MaterialType.RAW
            )["average_cost"]
            if im_avg_raw_cost > 0:
                avg_raw_cost = im_avg_raw_cost
            elif self.im.raw_batches:  # Fallback to current batch average if IM summary is zero / 如果IM摘要为零，则回退到当前批次平均值
                total_cost = sum(
                    b.unit_cost * b.remaining
                    for b in self.im.raw_batches
                    if b.remaining > 0
                )
                total_qty_in_batches = sum(
                    b.remaining for b in self.im.raw_batches if b.remaining > 0
                )
                if total_qty_in_batches > 0:
                    avg_raw_cost = total_cost / total_qty_in_batches

        if (
            avg_raw_cost <= 0 and best_price_pid_for_fallback
        ):  # Further fallback using NMI best price / 使用NMI最优价格进一步回退
            avg_raw_cost = (
                self._best_price(best_price_pid_for_fallback) * 0.4
            )  # Heuristic: 40% of NMI best price / 启发式：NMI最优价格的40%
        elif avg_raw_cost <= 0:  # Absolute fallback / 绝对回退
            avg_raw_cost = 10.0  # Arbitrary non-zero value / 任意非零值
        return avg_raw_cost

    # NEW Helper method for inventory health
    # 新增库存健康状况辅助方法
    def _get_raw_inventory_health_status(self, current_day: int) -> str:
        """
        Estimates raw inventory health.
        Returns "low", "medium", or "high".
        估算原材料库存健康状况。
        返回 "low", "medium", 或 "high"。
        """
        if not self.im:
            return "medium"  # Default if IM not available / 如果IM不可用则为默认值

        current_raw_stock = self.im.get_inventory_summary(
            current_day, MaterialType.RAW
        )["current_stock"]

        future_horizon = 5
        estimated_consumption_next_horizon = 0
        for i in range(1, future_horizon + 1):
            check_day = current_day + i
            if check_day >= self.awi.n_steps:
                break
            daily_need = self.im.get_total_insufficient(check_day)
            if daily_need <= 0:
                daily_need_proxy = self.im.daily_production_capacity * 0.5
                estimated_consumption_next_horizon += daily_need_proxy
            else:
                estimated_consumption_next_horizon += daily_need

        if (
            estimated_consumption_next_horizon == 0
            and current_raw_stock > self.im.daily_production_capacity
        ):
            return "high"
        if estimated_consumption_next_horizon == 0:
            return "medium"

        if current_raw_stock < estimated_consumption_next_horizon * 0.4:
            return "low"
        elif current_raw_stock > estimated_consumption_next_horizon * 1.5:
            return "high"
        else:
            return "medium"

    # Method from Step 9.d (Turn 35)
    # 方法来自步骤9.d (轮次35)
    def _is_production_capacity_tight(
        self, day: int, quantity_being_considered: int = 0
    ) -> bool:
        """Checks if production capacity for a given day is tight, considering current commitments and a new potential quantity."""
        """检查给定日期的生产能力是否紧张，考虑当前承诺和新的潜在数量。"""
        if not self.im:
            return False
        signed_sales_for_day = 0
        for contract_detail in self.im.get_pending_contracts(
            is_supply=False, day=day
        ):  # Get sales contracts for that day / 获取当天的销售合同
            if (
                contract_detail.material_type == MaterialType.PRODUCT
            ):  # Ensure it's a product contract / 确保是产品合同
                signed_sales_for_day += contract_detail.quantity
        remaining_capacity = (
            self.im.daily_production_capacity
            - signed_sales_for_day
            - quantity_being_considered
        )
        is_tight = remaining_capacity < (
            self.im.daily_production_capacity * 0.20
        )  # Tight if less than 20% capacity remains / 如果剩余产能少于20%，则视为紧张
        return is_tight

    # ------------------------------------------------------------------
    # 🌟 3. 价格工具
    # 🌟 3. Pricing utilities
    # ------------------------------------------------------------------

    def _is_supplier(self, pid: str) -> bool:
        """Checks if a partner ID is a supplier."""
        """检查伙伴ID是否为供应商。"""
        return pid in self.awi.my_suppliers

    def _is_consumer(self, pid: str) -> bool:
        """Checks if a partner ID is a consumer."""
        """检查伙伴ID是否为消费者。"""
        return pid in self.awi.my_consumers

    def _best_price(self, pid: str) -> float:
        """Gets the best possible price from NMI for a partner (min for buying, max for selling)."""
        """从NMI获取伙伴的最佳可能价格（采购取最小，销售取最大）。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        return issue.min_value if self._is_supplier(pid) else issue.max_value

    def _is_price_too_high(self, pid: str, price: float) -> bool:
        """Checks if a price is outside the NMI acceptable range (too high for buying, too low for selling)."""
        """检查价格是否超出NMI可接受范围（采购价过高，销售价过低）。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        if self._is_supplier(pid):  # Buying from supplier / 从供应商处采购
            return (
                price > issue.max_value
            )  # Price is too high if it's above NMI max / 如果价格高于NMI最大值，则价格过高
        return (
            price < issue.min_value
        )  # Selling to consumer, price is "too high" (bad for us) if below NMI min / 销售给消费者，如果价格低于NMI最小值，则价格“过高”（对我们不利）

    def _clamp_price(self, pid: str, price: float) -> float:
        """Clamps a price within the NMI min/max values for a partner."""
        """将价格限制在伙伴的NMI最小/最大值范围内。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        return max(issue.min_value, min(issue.max_value, price))

    def _expected_price(self, pid: str, default: float) -> float:
        """Estimates the expected price from a partner based on historical data and opponent model."""
        """根据历史数据和对手模型估算伙伴的预期价格。"""
        stats = self.partner_stats.get(pid)
        if stats and stats.get("contracts", 0) > 0:
            mean = stats.get("avg_price", default)
            success_count = stats.get("success", 0)
            contracts_count = stats.get("contracts", 0)
            if (
                success_count > 1 and contracts_count >= success_count
            ):  # Ensure variance is well-defined / 确保方差定义良好
                var = stats.get("price_M2", 0.0) / (success_count - 1)
            else:
                var = 0.0  # No variance if only one success or no successes / 如果只有一个成功或没有成功，则没有方差
            std = var**0.5
            rate = success_count / max(1, contracts_count)  # Success rate / 成功率
            base = (
                mean + std * (1 - rate)
            )  # Adjust mean by std dev scaled by failure rate / 按失败率缩放的标准差调整均值
        else:
            base = default  # No historical data, use default / 没有历史数据，使用默认值

        if self._is_supplier(
            pid
        ):  # If buying, ensure expected price is not too low (e.g., below a fraction of shortfall penalty) / 如果是采购，确保预期价格不会太低（例如，低于一部分短缺罚金）
            base = max(
                base, self.awi.current_shortfall_penalty * 0.8
            )  # Heuristic floor / 启发式下限

        model_price = self._estimate_reservation_price(
            pid, base
        )  # Get model-based reservation price / 获取基于模型的保留价格
        return (
            (base + model_price) / 2
        )  # Blend historical/heuristic base with model price / 将历史/启发式基础与模型价格混合

    # ------------------------------------------------------------------
    # 🌟 3-b. 动态让步策略
    # 🌟 3-b. Dynamic Concession Strategy
    # ------------------------------------------------------------------

    def _calc_opponent_concession(self, pid: str, price: float) -> float:
        """Calculates the opponent's concession rate based on their last offer."""
        """根据对手的最新报价计算其让步率。"""
        last = self._last_offer_price.get(pid)
        self._last_offer_price[pid] = (
            price  # Update last offer price / 更新最新报价价格
        )
        if (
            last is None or last == 0
        ):  # No previous offer or invalid last price / 没有先前的报价或无效的最后价格
            return 0.0
        return abs(price - last) / abs(last)  # Relative concession / 相对让步

    # Modified in Step 9.b (Turn 30)
    # 在步骤9.b (轮次30) 修改
    def _concession_multiplier(self, rel_time: float, opp_rate: float = 0.0) -> float:
        """Calculates a concession multiplier based on relative time and opponent's concession rate."""
        """根据相对时间和对手的让步率计算让步乘数。"""
        if self.concession_model:  # If a custom concession model exists, use it / 如果存在自定义让步模型，则使用它
            return self.concession_model(rel_time, opp_rate)
        # Apply a non-linear curve to relative time, and factor in opponent's rate
        # 对相对时间应用非线性曲线，并考虑对手的让步率
        non_linear_rel_time = rel_time**self.concession_curve_power
        base = (
            non_linear_rel_time * (1 + opp_rate)
        )  # Higher opponent rate can increase our concession / 对手让步率越高，我们的让步可能越大
        return max(0.0, min(1.0, base))  # Clamp between 0 and 1 / 限制在0和1之间

    # Modified in Step 9.b (Turn 30)
    # 在步骤9.b (轮次30) 修改
    def _apply_concession(
        self,
        pid: str,
        target_price: float,  # Agent's ideal target price for this round / 代理本轮的理想目标价格
        state: SAOState | None,
        current_price: float,  # Opponent's current offer price / 对手的当前报价价格
    ) -> float:
        """Applies concession to a target price based on negotiation state and opponent behavior."""
        """根据谈判状态和对手行为对目标价格应用让步。"""
        start_price = self._best_price(
            pid
        )  # Agent's absolute best NMI price / 代理的绝对NMI最优价格
        opp_rate = self._calc_opponent_concession(
            pid, current_price
        )  # Opponent's concession rate / 对手的让步率
        rel_time = (
            state.relative_time if state else 0.0
        )  # Relative time in negotiation / 谈判中的相对时间

        base_mult = self._concession_multiplier(
            rel_time, opp_rate
        )  # Base concession multiplier / 基础让步乘数
        adjusted_mult = base_mult  # Start with base, adjust further based on context / 从基础开始，根据上下文进一步调整

        # Blend agent's target with estimated opponent expectation
        # 将代理的目标与估算的对手期望混合
        current_final_target_price = (
            target_price + self._expected_price(pid, target_price)
        ) / 2
        log_reason_parts = [
            f"BaseTarget: {target_price:.2f}, ExpectedTarget: {current_final_target_price:.2f}"
        ]

        if self._is_consumer(pid):  # Selling to consumer / 销售给消费者
            is_late_stage = rel_time > 0.7
            is_very_late_stage = rel_time > 0.85

            if is_very_late_stage:  # Very late in negotiation, be more aggressive in conceding / 谈判非常后期，更积极地让步
                adjusted_mult = max(
                    base_mult, 0.80
                )  # Ensure at least 80% concession if very late / 如果非常后期，确保至少80%的让步
                log_reason_parts.append(
                    f"SalesVeryLateStage(>{0.85 * 100}%)->MultFloor={adjusted_mult:.2f}"
                )

                if (
                    self.im
                ):  # Calculate absolute minimum profitable price / 计算绝对最低盈利价格
                    abs_min_price = (
                        self.get_avg_raw_cost_fallback(self.awi.current_step, pid)
                        + self.im.processing_cost
                    ) * (1 + self.min_profit_margin)
                    # Move target price closer to absolute minimum if already low / 如果价格已经很低，则将目标价格移近绝对最小值
                    current_final_target_price = max(
                        abs_min_price,
                        current_final_target_price * 0.5 + abs_min_price * 0.5,
                    )
                    log_reason_parts.append(
                        f"AbsMinPrice: {abs_min_price:.2f}, NewFinalTarget: {current_final_target_price:.2f}"
                    )

            elif is_late_stage:  # Late in negotiation / 谈判后期
                adjusted_mult = max(
                    base_mult, 0.60
                )  # Ensure at least 60% concession / 确保至少60%的让步
                log_reason_parts.append(
                    f"SalesLateStage(>{0.7 * 100}%)->MultFloor={adjusted_mult:.2f}"
                )
        else:  # Buying from supplier / 从供应商处采购
            # Increase concession if shortfall penalty is high
            # 如果短缺罚金高，则增加让步
            penalty_factor = min(
                1.0, self.awi.current_shortfall_penalty / 10.0
            )  # Scale penalty effect / 缩放罚金效应
            adjusted_mult = (
                base_mult + penalty_factor
            )  # Add penalty factor to concession / 将罚金因子加到让步中
            if penalty_factor > 0:
                log_reason_parts.append(f"ProcurePenaltyFactor:{penalty_factor:.2f}")

        adjusted_mult = max(
            0.0, min(1.0, adjusted_mult)
        )  # Clamp final multiplier / 限制最终乘数

        if self._is_consumer(
            pid
        ):  # Selling: concede from start_price (NMI max) towards target / 销售：从起始价格（NMI最大值）向目标让步
            conceded_price = (
                start_price - (start_price - current_final_target_price) * adjusted_mult
            )
            conceded_price = max(
                current_final_target_price, conceded_price
            )  # Don't concede beyond target / 不要让步超过目标
        else:  # Buying: concede from start_price (NMI min) towards target / 采购：从起始价格（NMI最小值）向目标让步
            conceded_price = (
                start_price + (current_final_target_price - start_price) * adjusted_mult
            )
            conceded_price = min(
                current_final_target_price, conceded_price
            )  # Don't concede beyond target / 不要让步超过目标

        final_conceded_price = self._clamp_price(
            pid, conceded_price
        )  # Ensure price is within NMI bounds / 确保价格在NMI范围内

        if (
            os.path.exists("env.test")
            and abs(final_conceded_price - current_price) > 1e-3
        ):  # Log if price changed significantly / 如果价格变化显著则记录日志
            log_reason_parts.append(
                f"RelTime:{rel_time:.2f} OppRate:{opp_rate:.2f} BaseMult:{base_mult:.2f} AdjMult:{adjusted_mult:.2f}"
            )
            print(
                f"CONCESSION Day {self.awi.current_step} ({self.id}) for {pid} (RelTime: {rel_time:.2f}): CurrPrice={current_price:.2f}, Target={current_final_target_price:.2f}, Start={start_price:.2f}, Mult={adjusted_mult:.2f} -> NewPrice={final_conceded_price:.2f}. Reasons: {'|'.join(log_reason_parts)}"
            )
        return final_conceded_price

    # ------------------------------------------------------------------
    # 🌟 Opponent utility estimation using logistic regression
    # 🌟 使用逻辑回归估算对手效用
    # ------------------------------------------------------------------

    def _update_acceptance_model(self, pid: str, price: float, accepted: bool) -> None:
        """Updates the logistic regression model for a partner's acceptance probability."""
        """更新伙伴接受概率的逻辑回归模型。"""
        model = self.partner_models.setdefault(
            pid, {"w0": 0.0, "w1": 0.0}
        )  # Initialize model if not exists / 如果模型不存在则初始化
        x = (
            price if self._is_supplier(pid) else -price
        )  # Feature: price (negative for selling) / 特征：价格（销售为负）
        z = model["w0"] + model["w1"] * x  # Linear combination / 线性组合
        try:
            pred = 1.0 / (
                1.0 + math.exp(-z)
            )  # Sigmoid function for probability / Sigmoid函数计算概率
        except OverflowError:  # Handle potential overflow if z is too large/small / 处理z过大/过小导致的溢出
            pred = 1.0 if z > 0 else 0.0
        y = 1.0 if accepted else 0.0  # True label / 真实标签
        err = y - pred  # Prediction error / 预测误差
        lr = 0.05  # Learning rate / 学习率
        # Gradient descent update / 梯度下降更新
        model["w0"] += lr * err
        model["w1"] += lr * err * x

    def _estimate_reservation_price(self, pid: str, default: float) -> float:
        """Estimates a partner's reservation price using the logistic regression model."""
        """使用逻辑回归模型估算伙伴的保留价格。"""
        model = self.partner_models.get(pid)
        if (
            not model or abs(model["w1"]) < 1e-6
        ):  # If no model or weight is too small, use default / 如果没有模型或权重过小，则使用默认值
            return default
        # Reservation price is where probability is 0.5 (z=0) => w0 + w1*x = 0 => x = -w0/w1
        # 保留价格是概率为0.5 (z=0) 的点 => w0 + w1*x = 0 => x = -w0/w1
        reservation_x = -model["w0"] / model["w1"]
        price_sign = (
            1.0 if self._is_supplier(pid) else -1.0
        )  # Adjust sign based on buying/selling / 根据采购/销售调整符号
        return reservation_x * price_sign

    # Modified in Step 9.c (Turn 32) & 9.d (Turn 35)
    # 在步骤9.c (轮次32) 和 9.d (轮次35) 修改
    def _pareto_counter_offer(
        self, pid: str, qty: int, t: int, price: float, state: SAOState | None
    ) -> Outcome:
        """Generates a counter-offer, potentially exploring Pareto improvements."""
        """生成还价，可能会探索帕累托改进。"""
        opp_price_est = self._estimate_reservation_price(
            pid, price
        )  # Opponent's estimated reservation price / 对手的估算保留价格
        best_own_price = self._best_price(
            pid
        )  # Agent's best NMI price / 代理的NMI最优价格
        # Agent's target price after concession
        # 代理让步后的目标价格
        agent_target_conceded_price = self._apply_concession(
            pid, best_own_price, state, price
        )
        # Blend opponent's estimate and agent's target
        # 混合对手的估算和代理的目标
        current_calculated_price = (opp_price_est + agent_target_conceded_price) / 2.0
        current_calculated_price = self._clamp_price(
            pid, current_calculated_price
        )  # Ensure within NMI / 确保在NMI范围内

        proposed_outcome_qty = qty
        proposed_outcome_time = max(
            t, self.awi.current_step
        )  # Ensure time is not in the past / 确保时间不在过去
        proposed_outcome_price = current_calculated_price

        reason_log = [
            f"BaseCalcPrice: {current_calculated_price:.2f} for Q:{qty} T:{t}"
        ]

        if self._is_consumer(pid) and self.im:  # Selling products / 销售产品
            # Absolute minimum profitable price for current qty/time
            # 当前数量/时间的绝对最低盈利价格
            abs_min_price_for_current_qty_time = (
                self.get_avg_raw_cost_fallback(self.awi.current_step, pid)
                + self.im.processing_cost
            ) * (1 + self.min_profit_margin)
            # Check if current calculated price is very close to our walkaway price
            # 检查当前计算价格是否非常接近我们的底价
            is_near_walkaway = abs(
                current_calculated_price - abs_min_price_for_current_qty_time
            ) < (abs_min_price_for_current_qty_time * 0.03)  # Within 3% / 3%以内
            opponent_price_is_lower = (
                price < current_calculated_price
            )  # Opponent's offer is better than our calculated price / 对手的报价优于我们的计算价格

            if (
                is_near_walkaway and opponent_price_is_lower
            ):  # If at our limit and opponent is offering better, explore Pareto / 如果已到我们的极限且对手出价更好，则探索帕累托
                reason_log.append(
                    f"Near walkaway ({abs_min_price_for_current_qty_time:.2f}), opp_price ({price:.2f}) lower. Exploring Pareto."
                )
                qty_issue = self.get_nmi(pid).issues[QUANTITY]
                max_possible_qty_issue = (
                    qty_issue.max_value
                    if isinstance(qty_issue.max_value, int)
                    else proposed_outcome_qty * 2
                )  # NMI max quantity or double / NMI最大数量或两倍

                # Try increasing quantity for a price reduction
                # 尝试增加数量以换取价格降低
                increased_qty = int(
                    proposed_outcome_qty * 1.25
                )  # Increase by 25% / 增加25%
                increased_qty = min(
                    increased_qty, max_possible_qty_issue
                )  # Clamp to NMI max / 限制在NMI最大值
                additional_qty = increased_qty - proposed_outcome_qty

                if additional_qty > 0 and not self._is_production_capacity_tight(
                    proposed_outcome_time, additional_qty
                ):  # If can produce more / 如果可以生产更多
                    price_reduction_for_qty_increase = (
                        0.02  # e.g., 2% price cut / 例如，降价2%
                    )
                    new_price_for_larger_qty = current_calculated_price * (
                        1 - price_reduction_for_qty_increase
                    )
                    if (
                        new_price_for_larger_qty >= abs_min_price_for_current_qty_time
                    ):  # Still profitable / 仍然盈利
                        proposed_outcome_price = new_price_for_larger_qty
                        proposed_outcome_qty = increased_qty
                        reason_log.append(
                            f"ParetoTry: Qty+ ({proposed_outcome_qty}) for Price- ({proposed_outcome_price:.2f})"
                        )
                elif (
                    additional_qty > 0
                ):  # Cannot increase quantity due to capacity / 由于产能无法增加数量
                    reason_log.append(
                        f"ParetoQtyIncSkip: Capacity tight for additional {additional_qty} on day {proposed_outcome_time}"
                    )

                # If quantity increase didn't work, try delaying delivery for a price reduction
                # 如果增加数量无效，尝试延迟交货以换取价格降低
                if not (
                    proposed_outcome_qty > qty
                    and proposed_outcome_price < current_calculated_price
                ):  # If no Pareto improvement via quantity / 如果通过数量没有帕累托改进
                    delayed_time = (
                        proposed_outcome_time + 2
                    )  # Delay by 2 days / 延迟2天
                    time_issue = self.get_nmi(pid).issues[TIME]
                    max_time_issue = (
                        time_issue.max_value
                        if isinstance(time_issue.max_value, int)
                        else self.awi.n_steps - 1
                    )  # NMI max time / NMI最大时间
                    if delayed_time < min(
                        self.awi.n_steps, max_time_issue + 1
                    ):  # If delay is valid / 如果延迟有效
                        price_reduction_for_delay = (
                            0.03  # e.g., 3% price cut / 例如，降价3%
                        )
                        new_price_for_delayed_delivery = current_calculated_price * (
                            1 - price_reduction_for_delay
                        )
                        if (
                            new_price_for_delayed_delivery
                            >= abs_min_price_for_current_qty_time
                        ):  # Still profitable / 仍然盈利
                            proposed_outcome_price = new_price_for_delayed_delivery
                            proposed_outcome_time = delayed_time
                            reason_log.append(
                                f"ParetoTry: Time+ ({proposed_outcome_time}) for Price- ({proposed_outcome_price:.2f})"
                            )

            if (
                os.path.exists("env.test") and len(reason_log) > 1
            ):  # Log if Pareto exploration happened / 如果进行了帕累托探索则记录日志
                pass  # print(f"🔎 Day {self.awi.current_step} ({self.id}) Pareto Sales to {pid}: {' | '.join(reason_log)}")

        elif (
            self._is_supplier(pid) and self.awi.current_shortfall_penalty > 1.0
        ):  # Buying and shortfall penalty is significant / 采购且短缺罚金显著
            # Try to secure more quantity if penalty is high
            # 如果罚金高，尝试获取更多数量
            qty_issue = self.get_nmi(pid).issues[QUANTITY]
            new_qty = int(proposed_outcome_qty * 1.1)  # Increase by 10% / 增加10%
            proposed_outcome_qty = min(
                new_qty,
                qty_issue.max_value
                if isinstance(qty_issue.max_value, int)
                else new_qty,
            )  # Clamp to NMI max / 限制在NMI最大值
            if proposed_outcome_qty > qty:  # If quantity increased / 如果数量增加
                reason_log.append(f"ProcurePenaltyQtyInc: {proposed_outcome_qty}")
                if os.path.exists("env.test"):
                    print(
                        f"📦 Day {self.awi.current_step} ({self.id}) Pareto Buy from {pid}: {' | '.join(reason_log)}"
                    )

        # Final clamping of quantity and time to NMI bounds
        # 最终将数量和时间限制在NMI范围内
        final_qty_issue = self.get_nmi(pid).issues[QUANTITY]
        proposed_outcome_qty = max(
            final_qty_issue.min_value,
            min(
                proposed_outcome_qty,
                final_qty_issue.max_value
                if isinstance(final_qty_issue.max_value, int)
                else proposed_outcome_qty,
            ),
        )
        final_time_issue = self.get_nmi(pid).issues[TIME]
        proposed_outcome_time = max(
            final_time_issue.min_value,
            min(
                proposed_outcome_time,
                final_time_issue.max_value
                if isinstance(final_time_issue.max_value, int)
                else proposed_outcome_time,
            ),
        )
        proposed_outcome_time = max(
            proposed_outcome_time, self.awi.current_step
        )  # Ensure delivery is not in past / 确保交货不在过去
        return (proposed_outcome_qty, proposed_outcome_time, proposed_outcome_price)

    # ------------------------------------------------------------------
    # 🌟 3-a. 需求计算和需求分配
    # 🌟 3-a. Demand Calculation and Distribution
    # ------------------------------------------------------------------

    def _get_sales_demand_first_layer(self) -> int:
        """Calculates sales demand for a first-layer agent (sells raw materials)."""
        """计算第一层代理（销售原材料）的销售需求。"""
        if not self.im:
            return 0
        # Sell available raw materials, up to max producible (which is effectively raw material for first layer)
        # 销售可用原材料，上限为最大可生产量（对于第一层代理，这实际上是原材料）
        today_inventory_material = int(
            min(
                self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)[
                    "estimated_available"
                ],
                self.im.get_max_possible_production(self.awi.current_step),
            )
        )
        return today_inventory_material

    def _get_sales_demand_last_layer(self) -> int:
        """Last layer agent does not sell through negotiation."""
        """最后一层代理不通过谈判销售。"""
        return 0  # Last layer does not sell via negotiation / 最后一层不通过谈判销售

    def _get_sales_demand_middle_layer_today(self) -> int:
        """Calculates today's sales demand for a middle-layer agent (sells products)."""
        """计算中间层代理（销售产品）的当日销售需求。"""
        if not self.im:
            return 0
        # Sell available products today
        # 销售当日可用产品
        today_inventory_product = int(
            self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)[
                "estimated_available"
            ]
        )
        return today_inventory_product

    def _get_sales_demand_middle_layer(self, day: int) -> int:
        """Calculates future sales demand for a middle-layer agent for a specific day."""
        """计算中间层代理在特定日期的未来销售需求。"""
        if not self.im:
            return 0
        # Sell products estimated to be available on a future day
        # 销售预计在未来某天可用的产品
        future_inventory_product = int(
            self.im.get_inventory_summary(day, MaterialType.PRODUCT)[
                "estimated_available"
            ]
        )
        return future_inventory_product

    def _get_supply_demand_middle_last_layer_today(self) -> tuple[int, int, float]:
        """Gets today's supply demand (emergency, planned, optional) for middle/last layer agents."""
        """获取中间/最后一层代理的当日供应需求（紧急、计划、可选）。"""
        if not self.im:
            return 0, 0, 0.0
        return (
            self.im.get_today_insufficient(
                self.awi.current_step
            ),  # Emergency need / 紧急需求
            self.im.get_total_insufficient(
                self.awi.current_step
            ),  # Planned need (total horizon) / 计划需求（总预测期）
            self.im.get_total_insufficient(self.awi.current_step) * 0.2,
        )  # Optional need (20% of total) / 可选需求（总需求的20%）

    def _get_supply_demand_middle_last_layer(self, day: int) -> tuple[int, int, float]:
        """Gets future supply demand for middle/last layer agents for a specific day."""
        """获取中间/最后一层代理在特定日期的未来供应需求。"""
        if not self.im:
            return 0, 0, 0.0
        # For future days, all needs are considered "planned" or "optional" from today's perspective
        # 从今天的角度来看，未来几天的所有需求都被视为“计划性”或“可选性”
        return (
            self.im.get_total_insufficient(
                day
            ),  # Treat as planned for that day / 当天视为计划性
            self.im.get_total_insufficient(day),  # Total for that day / 当天总计
            self.im.get_total_insufficient(day)
            * 0.2,  # Optional portion for that day / 当天的可选部分
        )

    def _get_supply_demand_first_layer(self) -> Tuple[int, int, int]:
        """First layer agent does not procure raw materials through negotiation."""
        """第一层代理不通过谈判采购原材料。"""
        return (
            0,
            0,
            0,
        )  # First layer does not buy raw materials via negotiation / 第一层不通过谈判购买原材料

    def _distribute_todays_needs(
        self, partners: Iterable[str] | None = None
    ) -> Dict[str, int]:
        """Distributes today's total needs (buy/sell) among specified partners."""
        """将当日的总需求（采购/销售）分配给指定的伙伴。"""
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)
        response: Dict[str, int] = {
            p: 0 for p in partners
        }  # Initialize with 0 for all / 所有伙伴初始化为0
        if not self.im:
            return response

        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]
        buy_need_emergency, buy_need_planned, buy_need_optional = 0, 0, 0
        sell_need = 0

        # Determine needs based on agent's layer in the supply chain
        # 根据代理在供应链中的层级确定需求
        if self.awi.is_first_level:  # Sells raw materials, buys nothing via negotiation / 销售原材料，不通过谈判采购
            _, _, buy_need_optional_float = (
                self._get_supply_demand_first_layer()
            )  # Should be 0 / 应为0
            buy_need_optional = int(buy_need_optional_float)
            sell_need = self._get_sales_demand_first_layer()
        elif self.awi.is_last_level:  # Buys raw materials, sells nothing via negotiation / 采购原材料，不通过谈判销售
            buy_need_emergency, buy_need_planned, buy_need_optional_float = (
                self._get_supply_demand_middle_last_layer_today()
            )
            buy_need_optional = int(buy_need_optional_float)
            sell_need = self._get_sales_demand_last_layer()  # Should be 0 / 应为0
        else:  # Middle layer: buys raw, sells products / 中间层：采购原材料，销售产品
            buy_need_emergency, buy_need_planned, buy_need_optional_float = (
                self._get_supply_demand_middle_last_layer_today()
            )
            buy_need_optional = int(buy_need_optional_float)
            sell_need = self._get_sales_demand_middle_layer_today()

        total_buy_need = buy_need_emergency + buy_need_planned + buy_need_optional
        if suppliers and total_buy_need > 0:
            response.update(self._distribute_to_partners(suppliers, total_buy_need))
        if consumers and sell_need > 0:
            response.update(self._distribute_to_partners(consumers, sell_need))
        return response

    def _distribute_to_partners(
        self, partners: List[str], needs: int
    ) -> Dict[str, int]:
        """Distributes a specific quantity of `needs` among a list of `partners`."""
        """将特定数量的 `needs` 分配给 `partners` 列表中的伙伴。"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}
        needs = int(needs)  # Ensure needs is an integer / 确保needs是整数

        # MODIFIED: Removed sorting by success rate to promote wider, less biased initial distribution.
        # 修改：移除了按成功率排序，以促进更广泛、偏差更小的初始分配。
        # partners.sort(
        #     key=lambda p: self.partner_stats.get(p, {}).get("success", 0)
        #     / max(1, self.partner_stats.get(p, {}).get("contracts", 0)),
        #     reverse=True,
        # )

        # _ptoday (default 1.0) determines the fraction of partners to consider.
        # _ptoday (默认为1.0) 决定了要考虑的伙伴比例。
        k = max(
            1, int(len(partners) * self._ptoday)
        )  # Number of partners to distribute among / 要分配的伙伴数量
        k = min(
            k, len(partners)
        )  # Ensure k is not more than available partners / 确保k不超过可用伙伴数量

        chosen_partners_for_distribution = partners[
            :k
        ]  # If _ptoday=1.0, this is all partners / 如果_ptoday=1.0，则为所有伙伴
        if not chosen_partners_for_distribution:  # Should not happen if partners is not empty and k>=1 / 如果partners非空且k>=1，则不应发生
            return {p: 0 for p in partners}

        # The _distribute function handles cases where needs < len(chosen_partners_for_distribution)
        # by giving 1 to 'needs' partners and 0 to others.
        # _distribute 函数处理 needs < len(chosen_partners_for_distribution) 的情况，
        # 给 'needs' 个伙伴分配1，其他伙伴分配0。
        quantities = _distribute(needs, len(chosen_partners_for_distribution))

        distribution = dict(zip(chosen_partners_for_distribution, quantities))

        # Ensure all original partners are in the response, with 0 if not chosen or got 0.
        # 确保所有原始伙伴都在响应中，如果未被选中或分配为0，则值为0。
        final_distribution = {p: 0 for p in partners}
        final_distribution.update(distribution)

        if os.path.exists("env.test"):
            distributed_to = {p: q for p, q in final_distribution.items() if q > 0}
            if distributed_to:
                print(
                    f"ဖြန့်ဝေ ({self.id}): Distributing {needs} to {len(chosen_partners_for_distribution)} partners (k={k} from _ptoday={self._ptoday}). Actual distribution: {distributed_to}"
                )
            else:
                print(
                    f"ဖြန့်ဝေ ({self.id}): Needs {needs} but no distribution made to {len(chosen_partners_for_distribution)} partners (k={k})."
                )

        return final_distribution

    # ------------------------------------------------------------------
    # 🌟 4. first_proposals — 首轮报价（可简化）
    # 🌟 4. first_proposals — Initial Offers (Can be simplified)
    # ------------------------------------------------------------------
    # Modified in Step 9.a (Turn 28) & 9.d (Turn 35)
    # 在步骤9.a (轮次28) 和 9.d (轮次35) 修改
    def first_proposals(self) -> Dict[str, Outcome]:
        """Generates initial proposals to partners based on distributed needs."""
        """根据分配的需求向伙伴生成初始报价。"""
        partners = list(self.negotiators.keys())
        if not partners:
            return {}  # No negotiators / 没有谈判者
        # Filter out partners based on agent's layer (e.g., first layer doesn't buy)
        # 根据代理的层级筛选伙伴（例如，第一层不采购）
        filtered: List[str] = []
        for pid in partners:
            if self._is_supplier(pid) and self.awi.is_first_level:
                continue  # First layer doesn't buy / 第一层不采购
            if self._is_consumer(pid) and self.awi.is_last_level:
                continue  # Last layer doesn't sell / 最后一层不销售
            filtered.append(pid)
        if not filtered:
            return {}  # No valid partners to propose to / 没有有效的伙伴可以提议

        distribution = self._distribute_todays_needs(
            filtered
        )  # Get needs distribution / 获取需求分配
        today = self.awi.current_step
        proposals: Dict[str, Outcome] = {}

        for pid, qty in distribution.items():
            if qty <= 0:
                continue  # No need for this partner / 此伙伴无需求
            nmi = self.get_nmi(pid)
            if nmi is None:
                continue  # No NMI available for this partner
            time_issue = nmi.issues[TIME]
            # Propose delivery time within NMI, not before today
            # 在NMI范围内提议交货时间，不早于今天
            delivery_time = max(today, time_issue.min_value)
            delivery_time = min(delivery_time, time_issue.max_value)
            qty_issue = nmi.issues[QUANTITY]
            # Propose quantity within NMI bounds
            # 在NMI范围内提议数量
            final_qty = min(qty, qty_issue.max_value)
            final_qty = max(final_qty, qty_issue.min_value)
            if final_qty <= 0:
                continue  # Invalid quantity after clamping / 限制后数量无效

            price_for_proposal: float
            reason_log_parts: List[
                str
            ] = []  # For logging decision process / 用于记录决策过程

            if self._is_consumer(pid):  # Selling to consumer / 销售给消费者
                avg_raw_cost = self.get_avg_raw_cost_fallback(
                    today, pid
                )  # Estimated raw material cost / 估算的原材料成本
                reason_log_parts.append(f"AvgRawCostEst: {avg_raw_cost:.2f}")
                unit_cost_estimate = avg_raw_cost + (
                    self.im.processing_cost if self.im else 0
                )  # Total unit cost / 总单位成本
                reason_log_parts.append(
                    f"UnitCostEst (raw+proc): {unit_cost_estimate:.2f}"
                )
                target_margin = (
                    self.min_profit_margin
                )  # Base target profit margin / 基础目标利润率

                # Increase margin if capacity is tight for the proposed delivery day and quantity
                # 如果提议的交货日期和数量导致产能紧张，则提高利润率
                if self._is_production_capacity_tight(delivery_time, final_qty):
                    target_margin += self.capacity_tight_margin_increase
                    reason_log_parts.append(
                        f"CapacityTight! Margin adj by +{self.capacity_tight_margin_increase:.3f} -> {target_margin:.3f}"
                    )
                else:
                    reason_log_parts.append(f"BaseTargetMargin: {target_margin:.3f}")

                # Adjust margin based on historical performance with this partner
                # 根据与此伙伴的历史表现调整利润率
                partner_data = self.partner_stats.get(pid)
                if (
                    partner_data
                    and partner_data.get("contracts", 0) >= 3
                    and unit_cost_estimate > 0
                ):  # Sufficient history / 足够的历史数据
                    historical_avg_price = partner_data["avg_price"]
                    historical_profit_margin = (
                        historical_avg_price - unit_cost_estimate
                    ) / unit_cost_estimate
                    if (
                        historical_profit_margin > target_margin + 0.05
                    ):  # If historically achieved much higher margin / 如果历史上实现了更高的利润率
                        adjustment_factor = (
                            0.5  # Blend towards historical / 向历史利润率靠拢
                        )
                        target_margin = (
                            target_margin
                            + (historical_profit_margin - target_margin)
                            * adjustment_factor
                        )
                        reason_log_parts.append(
                            f"HistMargin ({historical_profit_margin:.3f}) -> AdjustedTargetMargin: {target_margin:.3f}"
                        )
                    success_rate = partner_data["success"] / partner_data["contracts"]
                    if success_rate > 0.8:  # Reliable partner bonus / 可靠伙伴奖励
                        bonus_for_reliability = 0.02
                        target_margin += bonus_for_reliability
                        reason_log_parts.append(
                            f"ReliablePartner (SR {success_rate:.2f}) -> Bonus {bonus_for_reliability:.3f} -> NewTargetMargin: {target_margin:.3f}"
                        )

                initial_price = unit_cost_estimate * (
                    1 + target_margin
                )  # Calculate initial proposal price / 计算初始提议价格
                # Ensure price is at least minimally profitable
                # 确保价格至少有最低利润
                absolute_min_profitable_price = unit_cost_estimate * (
                    1 + self.min_profit_margin
                )
                price_for_proposal = max(initial_price, absolute_min_profitable_price)
                price_for_proposal = self._clamp_price(
                    pid, price_for_proposal
                )  # Clamp to NMI / 限制在NMI范围内
                # Final check to ensure it's still above absolute minimum after clamping
                # 最终检查以确保在限制后仍高于绝对最小值
                price_for_proposal = max(
                    price_for_proposal, absolute_min_profitable_price
                )
                reason_log_parts.append(
                    f"CalcPrice: {initial_price:.2f} -> Clamped/MinEnsured: {price_for_proposal:.2f} (AbsMinProfitable: {absolute_min_profitable_price:.2f})"
                )
                if os.path.exists("env.test"):
                    print(
                        f"📈 Day {today} ({self.id}) Proposal to Consumer {pid}: Qty={final_qty}, Price={price_for_proposal:.2f}. Reasons: {' | '.join(reason_log_parts)}"
                    )
            else:  # Buying from supplier: propose our best NMI price / 从供应商处采购：提议我们的NMI最优价格
                price_for_proposal = self._best_price(
                    pid
                )  # NMI min price for buying / 采购的NMI最低价格
                if os.path.exists("env.test"):
                    print(
                        f"📦 Day {today} ({self.id}) Proposal to Supplier {pid}: Qty={final_qty}, Price={price_for_proposal:.2f} (Best price for buying)"
                    )
            proposals[pid] = (final_qty, delivery_time, price_for_proposal)
        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # 🌟 5. counter_all — Core Negotiation Logic (Dispatches to submodules)
    # ------------------------------------------------------------------

    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """Handles all incoming offers by dispatching them to specialized processing methods."""
        """通过将所有传入报价分派给专门的处理方法来处理它们。"""
        responses: Dict[str, SAOResponse] = {}
        if not self.im:  # InventoryManager not initialized / 库存管理器未初始化
            for pid in offers.keys():
                responses[pid] = SAOResponse(
                    ResponseType.END_NEGOTIATION, None
                )  # End negotiation if IM is missing / 如果缺少IM则结束谈判
            return responses
        # Separate offers for consumers (sales) and suppliers (procurement)
        # 分离消费者（销售）和供应商（采购）的报价
        demand_offers = {p: o for p, o in offers.items() if self._is_consumer(p)}
        demand_states = {p: states[p] for p in demand_offers}
        # Sum of quantities from today's supply offers (used in sales capacity check)
        # 今日供应报价的总数量（用于销售能力检查）
        sum_qty_supply_offer_today = sum(
            offer[QUANTITY] for pid, offer in offers.items() if self._is_supplier(pid)
        )
        responses.update(
            self._process_sales_offers(
                demand_offers, demand_states, sum_qty_supply_offer_today
            )
        )

        supply_offers = {p: o for p, o in offers.items() if self._is_supplier(p)}
        supply_states = {p: states[p] for p in supply_offers}
        responses.update(self._process_supply_offers(supply_offers, supply_states))
        return responses

    # ------------------------------------------------------------------
    # 🌟 5‑1. 供应报价拆分三类
    # 🌟 5‑1. Splitting Supply Offers into Three Categories
    # ------------------------------------------------------------------

    def _process_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """Processes supply offers by categorizing them into emergency, planned, or optional."""
        """通过将供应报价分类为紧急、计划或可选来处理它们。"""
        res: Dict[str, SAOResponse] = {}
        if (
            not offers or not self.im
        ):  # No offers or IM not initialized / 没有报价或IM未初始化
            return res
        today = self.awi.current_step
        emergency_offers = {}  # For today's immediate needs / 满足今日紧急需求
        planned_offers = {}  # For future planned needs / 满足未来计划需求
        optional_offers = {}  # For opportunistic buying / 机会性采购
        current_today_insufficient = self.im.get_today_insufficient(
            today
        )  # Raw materials needed today / 今日所需原材料
        # Sort offers by price (cheapest first), then by quantity (largest first)
        # 按价格（最低优先）排序报价，然后按数量（最大优先）排序
        sorted_offers = sorted(
            offers.items(), key=lambda item: (item[1][UNIT_PRICE], -item[1][QUANTITY])
        )

        # Categorize offers
        # 对报价进行分类
        for pid, offer in sorted_offers:
            self._last_partner_offer[pid] = offer[
                UNIT_PRICE
            ]  # Record opponent's last price / 记录对手的最新价格
            offer_time = offer[TIME]
            if (
                offer_time == today and current_today_insufficient > 0
            ):  # If delivery is today and still need materials today / 如果交货是今天且今天仍需要材料
                emergency_offers[pid] = offer
            elif (
                offer_time > today and self.im.get_total_insufficient(offer_time) > 0
            ):  # If delivery is future and have future needs / 如果交货是未来且未来有需求
                planned_offers[pid] = offer
            else:  # All other supply offers are optional / 所有其他供应报价都是可选的
                optional_offers[pid] = offer

        # Process emergency offers first
        # 首先处理紧急报价
        if emergency_offers:
            em_res = self._process_emergency_supply_offers(
                emergency_offers, {p: states[p] for p in emergency_offers}
            )
            res.update(em_res)
            # Update remaining today's insufficiency after accepting emergency offers
            # 接受紧急报价后更新当日剩余不足量
            for resp in em_res.values():
                if (
                    resp.response == ResponseType.ACCEPT_OFFER
                    and resp.outcome
                    and resp.outcome[TIME] == today
                ):
                    current_today_insufficient -= resp.outcome[QUANTITY]
            current_today_insufficient = max(
                0, current_today_insufficient
            )  # Ensure not negative / 确保不为负

        # If still need materials for today, try to adapt planned/optional offers for today's delivery
        # 如果今天仍需要材料，尝试调整计划/可选报价以满足今天的交货
        if current_today_insufficient > 0:
            adaptable_offers = list(planned_offers.items()) + list(
                optional_offers.items()
            )  # Combine planned and optional / 合并计划和可选
            adaptable_offers.sort(
                key=lambda item: item[1][UNIT_PRICE]
            )  # Sort by price / 按价格排序
            for pid, offer_to_adapt in adaptable_offers:
                if current_today_insufficient <= 0:
                    break  # Stop if today's needs are met / 如果满足了今天的需求则停止
                original_qty, _, original_price = offer_to_adapt
                state = states.get(pid)
                qty_to_request_today = min(
                    original_qty, current_today_insufficient
                )  # Request needed quantity for today / 请求今天所需的数量
                # Counter with delivery for today
                # 以今天的交货时间还价
                countered_outcome_today = self._pareto_counter_offer(
                    pid, qty_to_request_today, today, original_price, state
                )
                res[pid] = SAOResponse(
                    ResponseType.REJECT_OFFER, countered_outcome_today
                )
                # Remove from original categories as it's now handled for today
                # 从原始类别中移除，因为它现在已为今天处理
                if pid in planned_offers:
                    planned_offers.pop(pid)
                if pid in optional_offers:
                    optional_offers.pop(pid)

        # Process remaining planned and optional offers for their original delivery times
        # 处理剩余的计划和可选报价（针对其原始交货时间）
        if planned_offers:
            plan_res = self._process_planned_supply_offers(
                planned_offers, {p: states[p] for p in planned_offers}
            )
            res.update(plan_res)
        if optional_offers:
            optional_res = self._process_optional_supply_offers(
                optional_offers, {p: states[p] for p in optional_offers}
            )
            res.update(optional_res)
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑a 紧急需求处理
    # 🌟 5‑1‑a Emergency Demand Processing
    # ------------------------------------------------------------------
    def _process_emergency_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """Processes supply offers for immediate, emergency needs for today."""
        """处理当日紧急需求的供应报价。"""
        res: Dict[str, SAOResponse] = {}
        if not offers or not self.im:
            return res  # No offers or IM not initialized / 没有报价或IM未初始化
        remain_needed = self.im.get_today_insufficient(
            self.awi.current_step
        )  # Materials needed today / 今日所需材料
        if remain_needed <= 0:
            return res  # No emergency need / 没有紧急需求
        ordered_offers = sorted(
            offers.items(), key=lambda x: x[1][UNIT_PRICE]
        )  # Sort by price (cheapest first) / 按价格排序（最低优先）
        penalty = (
            self.awi.current_shortfall_penalty
        )  # Current shortfall penalty / 当前短缺罚金

        for pid, offer in ordered_offers:
            if remain_needed <= 0:
                break  # Stop if needs are met / 如果需求已满足则停止
            qty, time, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            state = states.get(pid)
            self._recent_material_prices.append(
                price
            )  # Update market price tracking / 更新市场价格跟踪
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            # Accept if price is not much higher than penalty, or very late in negotiation
            # 如果价格不比罚金高太多，或者谈判已到非常后期，则接受
            if (
                price <= penalty * 1.1
            ):  # Price is at most 10% above penalty / 价格最多比罚金高10%
                accept_qty = min(
                    qty, remain_needed
                )  # Accept needed quantity / 接受所需数量
                if (
                    qty <= remain_needed and price <= penalty
                ):  # Offer fully covers part of need and price is good / 报价完全覆盖部分需求且价格良好
                    res[pid] = SAOResponse(
                        ResponseType.ACCEPT_OFFER, (accept_qty, time, price)
                    )
                    remain_needed -= accept_qty
                elif (
                    qty > remain_needed and price <= penalty
                ):  # Offer is larger than need but price is good / 报价大于需求但价格良好
                    res[pid] = SAOResponse(
                        ResponseType.ACCEPT_OFFER, (accept_qty, time, price)
                    )  # Accept only what's needed / 只接受所需数量
                    remain_needed -= accept_qty
                elif (
                    price > penalty
                    and price <= penalty * 1.1
                    and state
                    and state.relative_time > 0.8
                ):  # Price slightly above penalty, but very late / 价格略高于罚金，但非常后期
                    res[pid] = SAOResponse(
                        ResponseType.ACCEPT_OFFER, (accept_qty, time, price)
                    )
                    remain_needed -= accept_qty
                else:  # Price acceptable but conditions not met for full accept, or quantity mismatch / 价格可接受但条件不满足完全接受，或数量不匹配
                    counter_offer = self._pareto_counter_offer(
                        pid, accept_qty, time, price, state
                    )
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            else:  # Price is too high, try to negotiate down / 价格过高，尝试谈判降低
                target_price_for_counter = min(
                    price, penalty * 0.95
                )  # Target slightly below penalty / 目标略低于罚金
                conceded_price = self._apply_concession(
                    pid, target_price_for_counter, state, price
                )  # Apply concession logic / 应用让步逻辑
                counter_offer = self._pareto_counter_offer(
                    pid, min(qty, remain_needed), time, conceded_price, state
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑b 计划性需求处理
    # 🌟 5‑1‑b Planned Demand Processing
    # ------------------------------------------------------------------
    # Modified in Step 3 (Turn 13), Step 5 (Turn 15/verified pre-existing), Step 11 (logging)
    # 在步骤3 (轮次13), 步骤5 (轮次15/验证已存在), 步骤11 (日志) 修改
    def _process_planned_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """Processes supply offers for future, planned needs based on profitability and inventory headroom."""
        """根据盈利能力和库存余量处理未来计划需求的供应报价。"""
        res: Dict[str, SAOResponse] = {}
        if not self.im:
            return res  # IM not initialized / IM未初始化
        accepted_quantities_for_planned_this_call = defaultdict(
            int
        )  # Track accepted quantities in this call / 跟踪此调用中接受的数量
        sorted_offers = sorted(
            offers.items(), key=lambda item: item[1][UNIT_PRICE]
        )  # Sort by price / 按价格排序

        for pid, offer in sorted_offers:
            qty_original, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            qty = float(qty_original)  # Use float for calculations / 计算时使用浮点数
            self._last_partner_offer[pid] = (
                price  # Record opponent's price / 记录对手价格
            )
            state = states.get(pid)
            self._recent_material_prices.append(
                price
            )  # Update market price tracking / 更新市场价格跟踪
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            # Estimate profitability: max affordable raw price based on estimated product selling price and margin
            # 估算盈利能力：基于预估产品售价和利润率的最大可承受原材料价格
            est_sell_price = (
                self._market_product_price_avg
                if self._market_product_price_avg > 0
                else price * 2.0
            )  # Fallback if no market avg / 如果没有市场平均价则回退
            min_profit_for_product = est_sell_price * self.min_profit_margin
            max_affordable_raw_price_jit = (
                est_sell_price - self.im.processing_cost - min_profit_for_product
            )  # JIT = Just-In-Time (no storage cost) / JIT = 准时制（无存储成本）
            # Estimate storage cost for holding the material until delivery time 't'
            # 估算将材料保存至交货时间 't' 的存储成本
            days_held_estimate = max(
                0, t - (self.awi.current_step + 1)
            )  # Number of days material will be stored / 材料将被存储的天数
            estimated_storage_cost_per_unit = (
                self.im.raw_storage_cost * days_held_estimate
            )
            effective_price = (
                price + estimated_storage_cost_per_unit
            )  # Price including storage / 包括存储的有效价格
            price_is_acceptable = (
                effective_price <= max_affordable_raw_price_jit
            )  # Is it profitable considering storage? / 考虑存储是否盈利？

            # Calculate procurement limit and remaining headroom for the delivery date 't'
            # 计算交货日期 't' 的采购限额和剩余空间
            current_total_needed_for_date_t = float(
                self.im.get_total_insufficient(t)
            )  # Total raw needed for day t / 第t天所需的原材料总量

            # MODIFIED: Apply inventory health adjustment to procurement limit
            # 修改：对采购限额应用库存健康状况调整
            inventory_health = self._get_raw_inventory_health_status(
                self.awi.current_step
            )
            procurement_aggressiveness_factor = 1.0
            if inventory_health == "low":
                procurement_aggressiveness_factor = (
                    1.15  # Be more aggressive if inventory is low / 如果库存低则更积极
                )
            elif inventory_health == "high":
                procurement_aggressiveness_factor = 0.90  # Be more conservative if inventory is high / 如果库存高则更保守

            procurement_limit_for_date_t = (
                current_total_needed_for_date_t * procurement_aggressiveness_factor
            )
            # Ensure it's at least the need, but add a small buffer if not high inventory
            # 确保至少满足需求，但如果库存不高则添加少量缓冲
            procurement_limit_for_date_t = max(
                current_total_needed_for_date_t, procurement_limit_for_date_t
            )
            if (
                inventory_health != "high"
            ):  # Add small buffer if not high inventory / 如果库存不高则添加少量缓冲
                procurement_limit_for_date_t = max(
                    procurement_limit_for_date_t, current_total_needed_for_date_t * 1.02
                )

            inventory_summary_for_t = self.im.get_inventory_summary(t, MaterialType.RAW)
            inventory_already_secured_for_t = float(
                inventory_summary_for_t.get("estimated_available", 0.0)
            )  # Already have or committed / 已有或已承诺
            newly_accepted_for_t_this_call = float(
                accepted_quantities_for_planned_this_call.get(t, 0.0)
            )  # Accepted in this processing loop / 此处理循环中已接受
            total_committed_so_far_for_t = (
                inventory_already_secured_for_t + newly_accepted_for_t_this_call
            )
            remaining_headroom_for_t = max(
                0.0, procurement_limit_for_date_t - total_committed_so_far_for_t
            )  # Space left to procure / 剩余采购空间
            accept_qty = min(
                qty, remaining_headroom_for_t
            )  # Quantity to accept from this offer / 从此报价中接受的数量
            accept_qty_int = int(round(accept_qty))

            log_prefix = f"🏭 Day {self.awi.current_step} ({self.id}) PlannedSupply Offer from {pid} (Q:{qty_original} P:{price:.2f} T:{t}): InvHealth={inventory_health}, AggroFactor={procurement_aggressiveness_factor:.2f} "
            log_details = f"EffPrice={effective_price:.2f} (StoreCost={estimated_storage_cost_per_unit:.2f}), JITLimit={max_affordable_raw_price_jit:.2f}. Headroom={remaining_headroom_for_t:.1f}, AcceptableQty={accept_qty_int}."

            if (
                accept_qty_int > 0 and price_is_acceptable
            ):  # If can accept some quantity and price is good / 如果可以接受一些数量且价格良好
                outcome_tuple = (accept_qty_int, t, price)
                if accept_qty_int == qty_original:
                    res[pid] = SAOResponse(
                        ResponseType.ACCEPT_OFFER, offer
                    )  # Accept original if full qty taken / 如果全部数量都被接受则接受原始报价
                else:
                    res[pid] = SAOResponse(
                        ResponseType.REJECT_OFFER, outcome_tuple
                    )  # Counter with partial quantity / 以部分数量还价
                accepted_quantities_for_planned_this_call[t] += (
                    accept_qty_int  # Update accepted for this call / 更新此调用中已接受的数量
                )
                if os.path.exists("env.test"):
                    print(log_prefix + f"Accepted Qty {accept_qty_int}. " + log_details)
            else:  # Cannot accept or price is not good / 无法接受或价格不好
                rejection_reason = ""
                if accept_qty_int <= 0:
                    rejection_reason += "NoHeadroomOrZeroAcceptQty;"
                if not price_is_acceptable:
                    rejection_reason += "PriceUnacceptable(Effective);"

                if not price_is_acceptable:  # If price is the issue, try to negotiate down / 如果是价格问题，尝试谈判降低
                    target_quoted_price_for_negotiation = (
                        max_affordable_raw_price_jit - estimated_storage_cost_per_unit
                    )  # Target price before storage / 存储前的目标价格
                    conceded_actual_price_to_offer = self._apply_concession(
                        pid, target_quoted_price_for_negotiation, state, price
                    )  # Apply concession / 应用让步
                    # Determine quantity for counter: original if some headroom, or min of original/headroom
                    # 确定还价数量：如果有空间则为原始数量，否则为原始数量/空间的最小值
                    qty_for_counter = (
                        qty_original
                        if accept_qty_int > 0
                        else min(qty_original, int(round(remaining_headroom_for_t)))
                    )
                    if qty_for_counter <= 0 and qty_original > 0:
                        qty_for_counter = 1  # Ensure at least 1 if original was >0 / 如果原始数量>0，则确保至少为1
                    elif (
                        qty_for_counter <= 0 and qty_original <= 0
                    ):  # Cannot make a valid counter / 无法做出有效还价
                        res[pid] = SAOResponse(
                            ResponseType.REJECT_OFFER, None
                        )  # Reject outright / 直接拒绝
                        if os.path.exists("env.test"):
                            print(
                                log_prefix
                                + f"Rejected ({rejection_reason}). No valid counter. "
                                + log_details
                            )
                        continue
                    counter_offer_tuple = self._pareto_counter_offer(
                        pid, qty_for_counter, t, conceded_actual_price_to_offer, state
                    )
                    res[pid] = SAOResponse(
                        ResponseType.REJECT_OFFER, counter_offer_tuple
                    )
                    if os.path.exists("env.test"):
                        print(
                            log_prefix
                            + f"Rejected ({rejection_reason}). Countering. "
                            + log_details
                            + f" Counter: Q:{counter_offer_tuple[0]} P:{counter_offer_tuple[2]:.2f} T:{counter_offer_tuple[1]}"
                        )
                else:  # No headroom, reject outright / 没有空间，直接拒绝
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    if os.path.exists("env.test"):
                        print(
                            log_prefix
                            + f"Rejected ({rejection_reason}). No counter. "
                            + log_details
                        )
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑c 机会性采购处理
    # 🌟 5‑1‑c Optional Procurement Processing
    # ------------------------------------------------------------------
    # Modified in Step 3 (Turn 13), Step 11 (logging)
    # 在步骤3 (轮次13), 步骤11 (日志) 修改
    def _process_optional_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """Processes supply offers for optional, opportunistic procurement if prices are very cheap."""
        """如果价格非常便宜，则处理可选的、机会性的采购供应报价。"""
        res: Dict[str, SAOResponse] = {}
        if not self.im:
            return res  # IM not initialized / IM未初始化
        accepted_quantities_for_optional_this_call = defaultdict(
            int
        )  # Track accepted quantities / 跟踪接受的数量
        sorted_offers = sorted(
            offers.items(), key=lambda item: item[1][UNIT_PRICE]
        )  # Sort by price / 按价格排序

        for pid, offer in sorted_offers:
            qty_original, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            qty = float(qty_original)  # Use float for calculations / 计算时使用浮点数
            self._last_partner_offer[pid] = (
                price  # Record opponent's price / 记录对手价格
            )
            state = states.get(pid)
            self._recent_material_prices.append(
                price
            )  # Update market price tracking / 更新市场价格跟踪
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            # Determine if price is "cheap" based on market average and discount factor
            # 根据市场平均价和折扣因子确定价格是否“便宜”
            cheap_threshold = (
                (
                    self._market_material_price_avg
                    if self._market_material_price_avg > 0
                    else price * 1.5
                )
                * self.cheap_price_discount
            )  # Fallback if no market avg / 如果没有市场平均价则回退
            price_is_cheap = price <= cheap_threshold

            # Calculate procurement limit and headroom, adjusted by inventory health
            # 计算采购限额和空间，根据库存健康状况进行调整
            demand_at_t = float(
                self.im.get_total_insufficient(t)
            )  # Demand for raw materials on day t / 第t天原材料需求

            # MODIFIED: Apply inventory health adjustment to optional procurement limit
            # 修改：对可选采购限额应用库存健康状况调整
            inventory_health = self._get_raw_inventory_health_status(
                self.awi.current_step
            )
            optional_procurement_allowance_factor = 1.0
            if inventory_health == "high":
                optional_procurement_allowance_factor = (
                    0.1  # Drastically reduce if high stock / 如果库存高则大幅减少
                )
            elif (
                inventory_health == "low"
            ):  # If inventory is low, be slightly more open to good optional deals / 如果库存低，则对好的可选交易稍微开放一些
                optional_procurement_allowance_factor = 1.1

            # Base limit: small fraction of capacity if no demand, or 120% of demand
            # 基础限额：如果没有需求则是产能的一小部分，或者需求的120%
            base_optional_limit = (
                float(self.im.daily_production_capacity * 0.2)
                if demand_at_t == 0
                else demand_at_t * 1.2
            )

            procurement_limit_for_date_t = (
                base_optional_limit * optional_procurement_allowance_factor
            )
            # Ensure optional procurement doesn't vastly exceed specific future demand if it exists
            # 确保可选采购不会大幅超过特定的未来需求（如果存在）
            if demand_at_t > 0:
                procurement_limit_for_date_t = min(
                    procurement_limit_for_date_t, demand_at_t * 1.2
                )

            inventory_summary_for_t = self.im.get_inventory_summary(t, MaterialType.RAW)
            inventory_already_secured_for_t = float(
                inventory_summary_for_t.get("estimated_available", 0.0)
            )
            newly_accepted_for_t_this_call = float(
                accepted_quantities_for_optional_this_call.get(t, 0.0)
            )
            total_committed_so_far_for_t = (
                inventory_already_secured_for_t + newly_accepted_for_t_this_call
            )
            remaining_headroom_for_t = max(
                0.0, procurement_limit_for_date_t - total_committed_so_far_for_t
            )
            accept_qty = min(qty, remaining_headroom_for_t)
            accept_qty_int = int(round(accept_qty))

            log_prefix = f"🏭 Day {self.awi.current_step} ({self.id}) OptionalSupply Offer from {pid} (Q:{qty_original} P:{price:.2f} T:{t}): InvHealth={inventory_health}, AllowanceFactor={optional_procurement_allowance_factor:.2f} "
            log_details = f"PriceIsCheap={price_is_cheap} (Threshold={cheap_threshold:.2f}). Headroom={remaining_headroom_for_t:.1f}, AcceptableQty={accept_qty_int}."

            if (
                accept_qty_int > 0 and price_is_cheap
            ):  # If can accept some and price is cheap / 如果可以接受一些且价格便宜
                res[pid] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offer
                )  # Accept original offer / 接受原始报价
                accepted_quantities_for_optional_this_call[t] += (
                    accept_qty_int  # Update accepted for this call / 更新此调用中已接受的数量
                )
                if os.path.exists("env.test"):
                    print(log_prefix + f"Accepted Qty {accept_qty_int}. " + log_details)
            else:  # Cannot accept or price not cheap enough / 无法接受或价格不够便宜
                rejection_reason = ""
                if accept_qty_int <= 0:
                    rejection_reason += "NoHeadroomOrZeroAcceptQty;"
                if not price_is_cheap:
                    rejection_reason += "PriceNotCheap;"

                if not price_is_cheap:  # If price is not cheap, try to negotiate to cheap threshold / 如果价格不便宜，尝试谈判至便宜阈值
                    conceded_price = self._apply_concession(
                        pid, cheap_threshold, state, price
                    )  # Apply concession / 应用让步
                    counter_offer_tuple = self._pareto_counter_offer(
                        pid, qty_original, t, conceded_price, state
                    )
                    res[pid] = SAOResponse(
                        ResponseType.REJECT_OFFER, counter_offer_tuple
                    )
                    if os.path.exists("env.test"):
                        print(
                            log_prefix
                            + f"Rejected ({rejection_reason}). Countering. "
                            + log_details
                            + f" Counter: Q:{counter_offer_tuple[0]} P:{counter_offer_tuple[2]:.2f} T:{counter_offer_tuple[1]}"
                        )
                else:  # No headroom, reject outright / 没有空间，直接拒绝
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    if os.path.exists("env.test"):
                        print(
                            log_prefix
                            + f"Rejected ({rejection_reason}). No counter. "
                            + log_details
                        )
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑2. 销售报价处理
    # 🌟 5‑2. Sales Offer Processing
    # ------------------------------------------------------------------
    # Modified in Step 9.d (Turn 35), Step 11 (logging)
    # 在步骤9.d (轮次35), 步骤11 (日志) 修改
    def _process_sales_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
        sum_qty_supply_offer_today: int,  # Sum of quantities from today's supply offers / 今日供应报价的总数量
    ) -> Dict[str, SAOResponse]:
        """Processes sales offers, ensuring capacity and material availability, and profitability."""
        """处理销售报价，确保产能和材料可用性以及盈利能力。"""
        res: Dict[str, SAOResponse] = {}
        if not self.im:
            return res  # IM not initialized / IM未初始化

        for pid, offer in offers.items():
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            state = states.get(pid)
            self._recent_product_prices.append(
                price
            )  # Update market price tracking / 更新市场价格跟踪
            if len(self._recent_product_prices) > self._avg_window:
                self._recent_product_prices.pop(0)

            # MODIFIED: Capacity and material availability check using IM's estimated_available
            # 修改：使用IM的estimated_available检查产能和材料可用性

            # Available production capacity on day 't' for *new* production
            # (IM's get_production_plan(t) reflects what's already committed for day t)
            # 第 't' 天用于*新*生产的可用生产能力
            # (IM的get_production_plan(t)反映了已为第t天承诺的内容)
            available_capacity_on_day_t = (
                self.im.daily_production_capacity - self.im.get_production_plan(t)
            )

            # Estimated raw materials available at the start of day 't' (for production on day 't')
            # 第 't' 天开始时可用的预估原材料（用于第 't' 天的生产）
            materials_available_at_start_of_day_t = self.im.get_inventory_summary(
                t, MaterialType.RAW
            ).get("estimated_available", 0)

            # Maximum quantity we can produce for THIS offer on day 't'
            # considering both capacity and materials available specifically for production on day 't'.
            # 我们可以在第 't' 天为这个特定报价生产的最大数量，
            # 同时考虑专门用于第 't' 天生产的产能和可用材料。
            max_producible_for_offer_on_day_t = min(
                available_capacity_on_day_t, materials_available_at_start_of_day_t
            )

            if (
                qty > max_producible_for_offer_on_day_t
            ):  # If offered quantity exceeds what we can produce / 如果报价数量超过我们的生产能力
                available_qty_for_offer = int(max_producible_for_offer_on_day_t)
                # Check if we can counter with a valid NMI quantity
                # 检查是否可以用有效的NMI数量还价
                if (
                    available_qty_for_offer > 0
                    and available_qty_for_offer
                    >= self.get_nmi(pid).issues[QUANTITY].min_value
                ):
                    # Try to counter with the maximum possible producible quantity
                    # 尝试以最大可生产数量还价
                    counter_offer_outcome = self._pareto_counter_offer(
                        pid, available_qty_for_offer, t, price, state
                    )
                    res[pid] = SAOResponse(
                        ResponseType.REJECT_OFFER, counter_offer_outcome
                    )
                    if os.path.exists("env.test"):
                        print(
                            f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: Over capacity/material for day {t} (max_prod={max_producible_for_offer_on_day_t:.0f}). Countering with Qty {available_qty_for_offer}."
                        )
                else:
                    # If cannot even produce a minimal valid quantity, reject.
                    # 如果连最小有效数量都无法生产，则拒绝。
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    if os.path.exists("env.test"):
                        print(
                            f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: No/Insufficient capacity/material for day {t} (max_prod={max_producible_for_offer_on_day_t:.0f}, NMI_min={self.get_nmi(pid).issues[QUANTITY].min_value}). Rejecting."
                        )
                continue  # Move to next offer / 处理下一个报价

            # If here, qty is producible on day t. Proceed with profitability check.
            # 如果到这里，数量在第t天是可生产的。继续进行盈利能力检查。
            avg_raw_cost = self.get_avg_raw_cost_fallback(
                self.awi.current_step, pid
            )  # Estimated raw material cost / 估算的原材料成本
            unit_cost = (
                avg_raw_cost + self.im.processing_cost
            )  # Total unit cost / 总单位成本
            current_min_margin_for_calc = (
                self.min_profit_margin
            )  # Base profit margin / 基础利润率
            reason_log_parts = [f"BaseMinMargin: {current_min_margin_for_calc:.3f}"]

            # Increase margin if capacity is tight for the delivery day and this quantity
            # 如果交货日期的产能因这个数量而紧张，则提高利润率
            if self._is_production_capacity_tight(
                t, qty
            ):  # Pass qty of current offer / 传递当前报价的数量
                current_min_margin_for_calc += self.capacity_tight_margin_increase
                reason_log_parts.append(
                    f"CapacityTight! AdjustedMinMargin: {current_min_margin_for_calc:.3f}"
                )
                if os.path.exists("env.test"):
                    print(
                        f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: Capacity tight for day {t}, using increased margin {current_min_margin_for_calc:.3f}."
                    )

            min_sell_price = unit_cost * (
                1 + current_min_margin_for_calc
            )  # Minimum acceptable selling price / 最低可接受售价

            if (
                price >= min_sell_price
            ):  # If offer price is profitable / 如果报价价格有利可图
                res[pid] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offer
                )  # Accept the offer / 接受报价
                if os.path.exists("env.test"):
                    print(
                        f"✅ Day {self.awi.current_step} ({self.id}) Sales Offer from {pid} (Q:{qty} P:{price:.2f} T:{t}): Accepted. MinSellPrice={min_sell_price:.2f}. Reasons: {'|'.join(reason_log_parts)}"
                    )
            else:  # Price not profitable, try to counter / 价格无利可图，尝试还价
                target_price_for_counter = min_sell_price  # Counter with our minimum acceptable price / 以我们的最低可接受价格还价
                conceded_price = self._apply_concession(
                    pid, target_price_for_counter, state, price
                )  # Apply concession logic / 应用让步逻辑
                counter_offer = self._pareto_counter_offer(
                    pid, qty, t, conceded_price, state
                )  # Generate Pareto-aware counter / 生成帕累托意识还价
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                if os.path.exists("env.test"):
                    print(
                        f"❌ Day {self.awi.current_step} ({self.id}) Sales Offer from {pid} (Q:{qty} P:{price:.2f} T:{t}): Rejected (Price < MinSellPrice {min_sell_price:.2f}). Countering with P:{counter_offer[UNIT_PRICE]:.2f}. Reasons: {'|'.join(reason_log_parts)}"
                    )
        return res

    # ------------------------------------------------------------------
    # 🌟 6. 谈判回调
    # 🌟 6. Negotiation Callbacks
    # ------------------------------------------------------------------

    def get_partner_id(self, contract: Contract) -> str:
        """Extracts the partner's ID from a contract object."""
        """从合同对象中提取伙伴ID。"""
        for p in contract.partners:
            if p != self.id:
                return p
        if os.path.exists("env.test"):
            print(
                f"⚠️ ({self.id}) Could not determine partner ID for contract {contract.id}, partners: {contract.partners}, my ID: {self.id}"
            )
        return "unknown_partner"  # Should ideally not happen / 理想情况下不应发生

    # Modified in Step 7 (Turn 20)
    # 在步骤7 (轮次20) 修改
    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation fails."""
        """在谈判失败时调用。"""
        for pid in partners:
            if pid == self.id:  # Skip self / 跳过自己
                continue
            if self._is_consumer(pid):  # It's a sales negotiation / 这是销售谈判
                self._sales_failures_since_margin_update += 1  # Increment failure counter for dynamic margin adjustment / 增加失败计数器以进行动态利润率调整

            # Update partner statistics for failed negotiation
            # 更新伙伴的失败谈判统计数据
            stats = self.partner_stats.setdefault(
                pid,
                {"avg_price": 0.0, "price_M2": 0.0, "contracts": 0, "success": 0},
            )
            stats["contracts"] += 1  # Increment total negotiations / 增加总谈判次数
            last_price = self._last_partner_offer.get(
                pid
            )  # Get opponent's last offered price / 获取对手的最新报价价格
            if last_price is not None:
                self._update_acceptance_model(
                    pid, last_price, False
                )  # Update opponent model with rejection / 用拒绝更新对手模型

    # Modified in Step 7 (Turn 20)
    # 在步骤7 (轮次20) 修改
    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        """Called when a negotiation succeeds and a contract is formed."""
        """在谈判成功并形成合同时调用。"""
        assert self.im, (
            "InventoryManager 尚未初始化"
        )  # InventoryManager not initialized
        partner = self.get_partner_id(contract)  # Get partner ID / 获取伙伴ID
        if partner == "unknown_partner":  # Safety check / 安全检查
            if os.path.exists("env.test"):
                print(
                    f"Error ({self.id}): Could not identify partner for contract {contract.id}. Skipping IM update."
                )
            return

        is_supply = (
            partner in self.awi.my_suppliers
        )  # Check if it's a supply contract / 检查是否为供应合同
        if not is_supply:  # If it's a sales contract / 如果是销售合同
            self._sales_successes_since_margin_update += 1  # Increment success counter for dynamic margin / 增加成功计数器以进行动态利润率调整

        im_type = IMContractType.SUPPLY if is_supply else IMContractType.DEMAND
        mat_type = MaterialType.RAW if is_supply else MaterialType.PRODUCT
        agreement = (
            contract.agreement
        )  # Get contract agreement details / 获取合同协议详情
        if (
            not agreement
        ):  # Should always have an agreement on success / 成功时应始终有协议
            if os.path.exists("env.test"):
                print(
                    f"Error ({self.id}): Contract {contract.id} has no agreement. Skipping IM update."
                )
            return

        # Create IMContract object and add to InventoryManager
        # 创建IMContract对象并添加到库存管理器
        new_c = IMContract(
            contract_id=contract.id,
            partner_id=partner,
            type=im_type,
            quantity=agreement["quantity"],
            price=agreement["unit_price"],
            delivery_time=agreement["time"],
            bankruptcy_risk=0.0,  # Assume no bankruptcy risk for now / 目前假设无破产风险
            material_type=mat_type,
        )
        added = self.im.add_transaction(new_c)  # Add to IM / 添加到IM
        assert added, (
            f"❌ ({self.id}) IM.add_transaction 失败! contract={contract.id}"
        )  # IM.add_transaction failed!

        # Update partner statistics for successful negotiation
        # 更新伙伴的成功谈判统计数据
        stats = self.partner_stats.setdefault(
            partner, {"avg_price": 0.0, "price_M2": 0.0, "contracts": 0, "success": 0}
        )
        stats["contracts"] += 1  # Increment total negotiations / 增加总谈判次数
        stats["success"] += 1  # Increment successful negotiations / 增加成功谈判次数
        price = agreement["unit_price"]
        n_success = stats[
            "success"
        ]  # Number of successful contracts for this partner / 此伙伴的成功合同数量
        # Update running average and variance (M2 for Welford's algorithm)
        # 更新运行平均值和方差（用于Welford算法的M2）
        delta = price - stats["avg_price"]
        stats["avg_price"] += delta / n_success
        if n_success > 1:
            stats["price_M2"] += delta * (price - stats["avg_price"])
        self._update_acceptance_model(
            partner, price, True
        )  # Update opponent model with acceptance / 用接受更新对手模型

        # Re-calculate insufficiency after new contract
        # 新合同签订后重新计算不足量
        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)

        # Update daily completed quantities
        # 更新每日完成数量
        if (
            is_supply and agreement["time"] == self.awi.current_step
        ):  # If supply delivered today / 如果供应今日交付
            self.purchase_completed[self.awi.current_step] += agreement["quantity"]
        elif (
            not is_supply and agreement["time"] == self.awi.current_step
        ):  # If sale delivered today / 如果销售今日交付
            self.sales_completed[self.awi.current_step] += agreement["quantity"]

        if os.path.exists("env.test"):
            pass  # print(f"✅ [{self.awi.current_step}] ({self.id}) Contract {contract.id} added to IM: {new_c}")

    # ------------------------------------------------------------------
    # 🌟 7. 动态策略调节接口
    # 🌟 7. Dynamic strategy adjustment API
    # ------------------------------------------------------------------
    def _update_dynamic_profit_margin_parameters(self) -> None:
        """Dynamically adjusts `min_profit_margin` based on inventory, demand, game stage, and sales conversion."""
        """根据库存、需求、游戏阶段和销售转化率动态调整 `min_profit_margin`。"""
        if not self.im:
            return  # IM not available / IM不可用
        current_day = self.awi.current_step
        total_days = self.awi.n_steps
        if total_days == 0:
            return  # Avoid division by zero / 避免除零

        new_min_profit_margin = (
            self.initial_min_profit_margin
        )  # Start with initial base margin / 从初始基础利润率开始
        reason_parts = [
            f"Base: {new_min_profit_margin:.3f}"
        ]  # For logging reasons for change / 用于记录更改原因

        # Get current product inventory and estimate future product demand
        # 获取当前产品库存并估算未来产品需求
        current_product_inventory = self.im.get_inventory_summary(
            current_day, MaterialType.PRODUCT
        )["current_stock"]
        future_total_product_demand_horizon = min(
            total_days, current_day + 5
        )  # Look ahead 5 days / 向前看5天
        future_total_product_demand = 0
        for d_offset in range(1, future_total_product_demand_horizon - current_day + 1):
            check_day = current_day + d_offset
            if check_day >= total_days:
                break  # Don't go beyond simulation end / 不要超出模拟结束时间
            for contract_detail in self.im.get_pending_contracts(
                is_supply=False, day=check_day
            ):  # Sum pending sales / 汇总待定销售
                future_total_product_demand += contract_detail.quantity
        reason_parts.append(
            f"InvProd: {current_product_inventory}, FutProdDemand(5d): {future_total_product_demand}"
        )

        # Rule A: Adjust margin based on inventory vs demand
        # 规则A：根据库存与需求调整利润率
        rule_a_applied = False
        if (
            current_product_inventory == 0 and future_total_product_demand == 0
        ):  # No stock, no demand / 没有库存，没有需求
            reason_parts.append("RuleA: No stock & no immediate demand, using base.")
            rule_a_applied = True
        elif (
            future_total_product_demand > 0
            and current_product_inventory > future_total_product_demand * 2.0
        ):  # Very high inventory / 库存非常高
            new_min_profit_margin = (
                0.05  # Lower margin to sell more / 降低利润率以销售更多
            )
            reason_parts.append(
                f"RuleA: High Inv vs Demand (>2x) -> set to {new_min_profit_margin:.3f}"
            )
            rule_a_applied = True
        elif (
            future_total_product_demand > 0
            and current_product_inventory > future_total_product_demand * 1.0
        ):  # Moderately high inventory / 库存中等偏高
            new_min_profit_margin = 0.07
            reason_parts.append(
                f"RuleA: Med Inv vs Demand (>1x) -> set to {new_min_profit_margin:.3f}"
            )
            rule_a_applied = True
        elif (
            future_total_product_demand == 0
            and current_product_inventory > self.im.daily_production_capacity * 1.0
        ):  # No demand, but more than 1 day's production in stock / 没有需求，但库存超过1天的产量
            new_min_profit_margin = 0.06
            reason_parts.append(
                f"RuleA: No Demand & Inv > 1 day prod -> set to {new_min_profit_margin:.3f}"
            )
            rule_a_applied = True
        if (
            not rule_a_applied
        ):  # Default if no specific condition met / 如果没有满足特定条件则为默认
            reason_parts.append(
                "RuleA: Defaulted (no specific high/low inv condition met)."
            )

        initial_margin_after_rule_a = (
            new_min_profit_margin  # Margin after Rule A / 规则A后的利润率
        )

        # Rule B: Further adjust if inventory is low relative to demand or capacity
        # 规则B：如果库存相对于需求或产能较低，则进一步调整
        rule_b_applied = False
        if (
            future_total_product_demand > 0
            and current_product_inventory < future_total_product_demand * 0.5
        ):  # Low inventory vs demand / 库存相对于需求较低
            new_min_profit_margin = max(
                initial_margin_after_rule_a, 0.15
            )  # Increase margin if stock is scarce / 如果库存稀缺则提高利润率
            if abs(new_min_profit_margin - initial_margin_after_rule_a) > 1e-5:
                reason_parts.append(
                    f"RuleB: Low Inv vs Demand (<0.5x) -> max with 0.15 -> {new_min_profit_margin:.3f}"
                )
            rule_b_applied = True
        elif (
            current_product_inventory < self.im.daily_production_capacity * 0.5
        ):  # Low inventory vs production capacity / 库存相对于生产能力较低
            new_min_profit_margin = max(initial_margin_after_rule_a, 0.12)
            if abs(new_min_profit_margin - initial_margin_after_rule_a) > 1e-5:
                reason_parts.append(
                    f"RuleB: Low Inv vs Capacity (<0.5 day prod) -> max with 0.12 -> {new_min_profit_margin:.3f}"
                )
            rule_b_applied = True
        if (
            not rule_b_applied
            and abs(initial_margin_after_rule_a - new_min_profit_margin) < 1e-5
        ):  # If Rule B didn't change anything / 如果规则B没有改变任何东西
            pass  # No specific log needed if no change / 如果没有变化则不需要特定日志

        initial_margin_after_rule_b = (
            new_min_profit_margin  # Margin after Rule B / 规则B后的利润率
        )

        # Rule C: Adjust margin based on game stage (late game, be more aggressive to sell)
        # 规则C：根据游戏阶段调整利润率（游戏后期，更积极地销售）
        rule_c_applied = False
        if current_day > total_days * 0.85:  # Last 15% of game / 游戏的最后15%
            new_min_profit_margin = min(
                initial_margin_after_rule_b, 0.03
            )  # Very low margin / 非常低的利润率
            if abs(new_min_profit_margin - initial_margin_after_rule_b) > 1e-5:
                reason_parts.append(
                    f"RuleC: End Game (Last 15%) -> min with 0.03 -> {new_min_profit_margin:.3f}"
                )
            rule_c_applied = True
        elif current_day > total_days * 0.6:  # Late mid-game (>60%) / 游戏中后期 (>60%)
            new_min_profit_margin = min(initial_margin_after_rule_b, 0.08)
            if abs(new_min_profit_margin - initial_margin_after_rule_b) > 1e-5:
                reason_parts.append(
                    f"RuleC: Late Mid-Game (>60%) -> min with 0.08 -> {new_min_profit_margin:.3f}"
                )
            rule_c_applied = True
        if (
            not rule_c_applied
            and abs(initial_margin_after_rule_b - new_min_profit_margin) < 1e-5
        ):  # If Rule C didn't change anything / 如果规则C没有改变任何东西
            pass  # No specific log needed if no change / 如果没有变化则不需要特定日志

        # Rule D: Adaptive adjustment based on recent sales conversion rate
        # 规则D：根据最近的销售转化率进行自适应调整
        # Adjust margin slightly up for successes, down for failures (every 5 successes/each failure)
        # 成功则略微提高利润率，失败则降低（每5次成功/每次失败）
        margin_adjustment_from_conversion = (
            self._sales_successes_since_margin_update // 5
        ) * 0.005 - self._sales_failures_since_margin_update * 0.005
        if margin_adjustment_from_conversion != 0:
            current_margin_before_adaptive = new_min_profit_margin
            new_min_profit_margin += margin_adjustment_from_conversion
            reason_parts.append(
                f"RuleD: Adaptive adj: {margin_adjustment_from_conversion:.4f} (S:{self._sales_successes_since_margin_update},F:{self._sales_failures_since_margin_update}) Cur->New: {current_margin_before_adaptive:.3f}->{new_min_profit_margin:.3f}"
            )
        self._sales_successes_since_margin_update = 0  # Reset counters / 重置计数器
        self._sales_failures_since_margin_update = 0

        # Final clamping of the profit margin
        # 最终限制利润率范围
        final_new_min_profit_margin = max(
            0.02, min(0.25, new_min_profit_margin)
        )  # Clamp between 2% and 25% / 限制在2%和25%之间
        if (
            abs(final_new_min_profit_margin - new_min_profit_margin) > 1e-5
        ):  # Log if clamped / 如果被限制则记录日志
            reason_parts.append(
                f"Clamped from {new_min_profit_margin:.3f} to {final_new_min_profit_margin:.3f}"
            )

        # Apply the new margin if it has changed significantly
        # 如果利润率变化显著则应用新的利润率
        if (
            abs(self.min_profit_margin - final_new_min_profit_margin) > 1e-4
        ):  # Threshold for change / 变化阈值
            old_margin = self.min_profit_margin
            self.update_profit_strategy(min_profit_margin=final_new_min_profit_margin)
            if os.path.exists("env.test"):
                print(
                    f"📈 Day {current_day} ({self.id}): min_profit_margin changed from {old_margin:.3f} to {self.min_profit_margin:.3f}. Reasons: {' | '.join(reason_parts)}"
                )
        elif os.path.exists(
            "env.test"
        ):  # Log even if not changed, for transparency / 即使未更改也记录日志，以提高透明度
            print(
                f"🔎 Day {current_day} ({self.id}): min_profit_margin maintained at {self.min_profit_margin:.3f}. Evaluated Reasons: {' | '.join(reason_parts)}"
            )

    def update_profit_strategy(
        self,
        *,
        min_profit_margin: float | None = None,
        cheap_price_discount: float | None = None,
    ) -> None:
        """Allows external update of profit strategy parameters."""
        """允许外部更新利润策略参数。"""
        if min_profit_margin is not None:
            self.min_profit_margin = min_profit_margin
        if cheap_price_discount is not None:
            self.cheap_price_discount = cheap_price_discount

    def decide_with_model(
        self, obs: Any
    ) -> Any:  # Placeholder for RL model integration / RL模型集成占位符
        """Placeholder for decision making using an RL model."""
        """使用RL模型进行决策的占位符。"""
        return None

    def _print_daily_status_report(self, result) -> None:
        """输出每日库存、生产和销售状态报告，包括未来预测"""
        """Outputs a daily status report of inventory, production, and sales, including future forecasts."""
        if (
            not self.im or not os.path.exists("env.test")
        ):  # Only print if IM exists and in test environment / 仅当IM存在且在测试环境中时打印
            return

        current_day = self.awi.current_step
        horizon_days = min(
            10, self.awi.n_steps - current_day
        )  # Forecast for next 10 days or remaining days / 预测未来10天或剩余天数

        # Table header / 表头
        header = "|   日期    |  原料真库存  |  原料预计库存   | 计划生产  |  剩余产能  |  产品真库存  |  产品预计库存  |  已签署销售量  |  实际产品交付  |"
        # Date | Raw True Inv | Raw Est Inv | Planned Prod | Remain Cap | Prod True Inv | Prod Est Inv | Signed Sales | Actual Prod Deliv
        (
            "|" + "-" * (len(header) + 24) + "|"
        )  # Adjust separator length based on content / 根据内容调整分隔符长度

        pass  # print("\n📊 每日状态报告")  # Daily Status Report
        pass  # print(separator)
        pass  # print(header)
        pass  # print(separator)

        # Current day and future forecast rows / 当前日期和未来预测行
        for day_offset in range(horizon_days):
            forecast_day = current_day + day_offset

            # Get data from InventoryManager / 从库存管理器获取数据
            raw_summary = self.im.get_inventory_summary(forecast_day, MaterialType.RAW)
            product_summary = self.im.get_inventory_summary(
                forecast_day, MaterialType.PRODUCT
            )

            raw_current_stock = int(raw_summary["current_stock"])
            raw_estimated = int(raw_summary["estimated_available"])

            product_current_stock = int(product_summary["current_stock"])
            product_estimated = int(product_summary["estimated_available"])

            # Planned production quantity / 计划生产量
            planned_production = int(self.im.get_production_plan(forecast_day))

            # Remaining production capacity / 剩余生产能力
            remaining_capacity = int(
                self.im.get_available_production_capacity(forecast_day)
            )

            # Quantity of signed sales contracts for the forecast day
            # 预测日已签署的销售合同数量
            signed_sales = 0
            for contract in self.im.get_pending_contracts(
                is_supply=False, day=forecast_day
            ):  # Iterate through pending sales / 遍历待定销售
                if contract.material_type == MaterialType.PRODUCT:
                    signed_sales += contract.quantity

            # Format and print row / 格式化并打印行
            day_str = (
                f"{forecast_day} (T+{day_offset})"
                if day_offset == 0
                else f"{forecast_day} (T+{day_offset})"
            )
            # Actual delivered products are from 'result' of process_day_operations, only for current day (offset 0)
            # 实际交付的产品来自 process_day_operations 的 'result'，仅适用于当前日期 (偏移量0)
            print(
                f"| {day_str:^6} | {raw_current_stock:^10} | {raw_estimated:^12} | {planned_production:^8} | {remaining_capacity:^8} | {product_current_stock:^10} | {product_estimated:^12} | {signed_sales:^12} | {(result.get('delivered_products', 0) if day_offset == 0 and isinstance(result, dict) else 0):^12} |"
            )

        pass  # print(separator)
        pass  # print()  # Extra newline for readability / 额外换行以提高可读性


if __name__ == "__main__":
    if os.path.exists("env.test"):
        pass  # print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
        # Module loaded successfully, LitaAgentY can be used in the competition framework.
