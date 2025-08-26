# inventory_manager_n.py
# 这是给Agent使用的库存管理器，用来管理代理的协议、生产、和库存。
# This is the Inventory Manager for Agents to manage the agent's protocols, production, and inventory.

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


class IMContractType(Enum):
    SUPPLY = auto()  # 采购／上游供给 Upstream Supply
    DEMAND = auto()  # 销售／下游需求 Downstream Demand


class MaterialType(Enum):
    RAW = auto()  # 原材料 Raw material
    PRODUCT = auto()  # 产品 product


@dataclass
class IMContract:
    contract_id: str
    partner_id: str
    type: IMContractType
    quantity: float
    price: float
    delivery_time: int  # 交割日（天数）Day for delivery
    bankruptcy_risk: float
    material_type: MaterialType


@dataclass
class Batch:
    batch_id: str
    remaining: float  # 当前剩余量（FIFO 扣减时会减少） Current remaining quantity
    unit_cost: float  # 单位成本（入库价，不含存储） Unit cost (storage not included)
    production_time: int  # 入库／生产日期 Production date for products, delivery date for raw materials


class InventoryManager:
    def __init__(
            self,
            raw_storage_cost: float,
            product_storage_cost: float,
            processing_cost: float,
            # 可选：日生产能力，若不指定则视为无限 Unlimited if not specified
            daily_production_capacity: Optional[float] = None,
            # 可选：最大仿真天数，默认100天 Should be initialized while the instance is created
            max_day: int = 100,
    ):
        # 当前仿真天
        # Current simulation day
        self.max_day = max_day  # 默认100天，可以在初始化时传入 Default is 100 days and can be provided at init
        self.current_day: int = 0

        # 成本参数 Cose parameters
        self.raw_storage_cost = raw_storage_cost
        self.product_storage_cost = product_storage_cost
        self.processing_cost = processing_cost
        self.daily_production_capacity = (
            daily_production_capacity if daily_production_capacity is not None else float("inf")
        )

        # 库存批次 Inventory batches
        self.raw_batches: List[Batch] = []
        self.product_batches: List[Batch] = []

        # 未交割的合同 Pending contracts
        self._pending_supply: List[IMContract] = []
        self._pending_demand: List[IMContract] = []

        # 生产计划  Production plan
        self.production_plan: Dict[int, float] = {}

        # 不足原料记录，结构为 {day: {"daily": 当日需要的原材料, "total": 总共需要的原材料}}
        # daily 表示必须在当天获取的原材料量，否则会违约
        # total 表示从当前到该日期总共还需要的原材料量
        # Insufficient raw material record, structure is {day: {"daily": raw material needed on that day, "total": total necessary raw material}}
        # 'daily' indicates the amount of raw material that must be obtained on that day, otherwise deliver will be fail
        # 'total' indicates the total amount of raw material needed from now to that date
        self.insufficient_raw: Dict[int, Dict[str, float]] = {}

        # 可扩展：最大可能生产
        # Maximum possible production, this is not related to the inventory
        self._max_possible_prod_cache: Dict[int, float] = {}

    def add_transaction(self, contract: IMContract) -> bool:
        """签订一份采购（上游）或销售（下游）合同。
        Sign a supply (upstream) or demand (downstream) contract."""
        # Sign a supply (upstream) or demand (downstream) contract.

        if contract.delivery_time < self.current_day:
            return False
        if contract.type == IMContractType.SUPPLY:
            self._pending_supply.append(contract)
        else:
            self._pending_demand.append(contract)

        # 每次添加新合同后，重新规划生产计划
        # Re-plan the production schedule whenever a new contract is added
        # 并计算不足库存量，确保代理可以及时应对变化
        # Calculate inventory shortfalls to ensure the agent can react in time
        self.plan_production(self.max_day)  # Use self.max_day for full horizon planning

        return True

    def void_negotiated_contract(self, contract_id: str) -> bool:
        """
        Removes a contract from pending lists if it was cancelled at signing.
        This is called when a contract previously added via on_negotiation_success
        is ultimately not signed (e.g., due to sign_all_contracts logic or opponent cancellation).
        """
        found_and_removed = False
        # Search in pending supply contracts
        for contract in self._pending_supply:
            if contract.contract_id == contract_id:
                self._pending_supply.remove(contract)
                found_and_removed = True
                break

        if not found_and_removed:
            # Search in pending demand contracts
            for contract in self._pending_demand:
                if contract.contract_id == contract_id:
                    self._pending_demand.remove(contract)
                    found_and_removed = True
                    break

        if found_and_removed:
            # Re-plan production as voiding a contract can change needs
            self.plan_production(self.max_day)  # Use self.max_day for full horizon planning
        return found_and_removed

    def receive_materials(self) -> List[str]:
        """
        将当天到货的上游合同转成原材料入库批次，FIFO 管理；
        返回本日完成交割的合同 ID 列表。
        Convert upstream contracts that arrive on the same day into raw material inbound batches, FIFO management;
        Returns a list of contract IDs that have completed delivery on the current day.
        """
        arrived = [c for c in self._pending_supply if c.delivery_time == self.current_day]
        for c in arrived:
            batch = Batch(
                batch_id=c.contract_id,
                remaining=c.quantity,
                unit_cost=c.price,
                production_time=self.current_day
            )
            self.raw_batches.append(batch)
            self._pending_supply.remove(c)
        return [c.contract_id for c in arrived]

    def _compute_summary(
            self,
            day: int,
            batches: List[Batch],
            storage_cost_per_unit: float,
            pending: List[IMContract],
            mtype: MaterialType
    ):
        """通用：计算指定 day 的真实库存 & 预期库存摘要。
        General utility to compute the real and expected inventory summary for the given day."""
        # 真实库存 Real inventory
        total_qty = 0.0
        total_cost = 0.0  # 含存储累积 Accumulated including storage costs
        total_store_cost = 0.0
        for b in batches:
            if b.remaining <= 0:
                continue
            days_stored = max(0, day - b.production_time)
            c0 = b.unit_cost * b.remaining
            cs = storage_cost_per_unit * days_stored * b.remaining
            total_qty += b.remaining
            total_cost += c0 + cs
            total_store_cost += cs

        avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
        total_est = 0.0
        # 预期可用：指的是从今天开始，按照计划生产和交付的情况下，到指定日期的可用数量（指定日期之后的不计算）
        # [可用于生产的原材料]当日真实库存 + 当日至指定日期到货 - 当日至指定日期前一天的生产计划Estimated available inventory, = real + future(contracted)until day
        if mtype == MaterialType.RAW:
            future_in_inv = sum(c.quantity for c in pending if self.current_day <= c.delivery_time <= day)
            future_prod_plan = sum(self.get_production_plan(prod_day) for prod_day in range(self.current_day, day))

            total_est = total_qty + future_in_inv - future_prod_plan
        # [可用于交割的产品]当日真实库存 - 当日至指定日期前一天的交割 + 当日至指定日期的生产计划 Estimated available inventory, = real - future(contracted) until day + production_plan until day
        elif mtype == MaterialType.PRODUCT:
            future_out_inv = sum(c.quantity for c in pending if self.current_day <= c.delivery_time < day)
            future_prod_plan = sum(self.get_production_plan(prod_day) for prod_day in range(self.current_day, day + 1))
            total_est = total_qty - future_out_inv + future_prod_plan
        # 预期成本：真实（含存储）+ 未来（仅入库价，不含后续存储） Estimated cost = real (including storage) + future (only inbound price, excluding subsequent storage)
        future_cost = sum(c.price * c.quantity for c in pending if c.delivery_time >= day)
        est_avg_cost = (total_cost + future_cost) / total_est if total_est > 0 else 0.0

        return {
            "current_stock": total_qty,
            "average_cost": avg_cost,
            "storage_cost": total_store_cost,
            "estimated_available": total_est,
            "estimated_average_cost": est_avg_cost
        }

    def get_inventory_summary(self, day: int, mtype: MaterialType) -> Dict[str, float]:
        """
        查询指定 day 的库存摘要。
        day 可大于 current_day，用于查看未来的预期（storage_cost 仍按 day 计算现有批次存储）。
        Queries the inventory summary for the specified day.
        day can be greater than current_day to see what to expect in the future (storage_cost still calculates existing batch storage by day).
        """
        if mtype == MaterialType.RAW:
            return self._compute_summary(
                day, self.raw_batches, self.raw_storage_cost, self._pending_supply, MaterialType.RAW
            )
        else:
            return self._compute_summary(
                day, self.product_batches, self.product_storage_cost, self._pending_demand, MaterialType.PRODUCT
            )

    def get_today_insufficient(self, day: int) -> int:
        """
        获取当天的不足原料量。
        Returns the amount of raw material shortage for the day.
        """
        if day in self.insufficient_raw:
            return int(self.insufficient_raw[day]["daily"])
        else:
            return 0

    def get_total_insufficient(self, day: int) -> int:
        """
        获取到指定日期的总不足原料量。
        Returns the total amount of raw material shortage to the specified date.
        """
        if day in self.insufficient_raw:
            return int(self.insufficient_raw[day]["total"])
        else:
            return 0

    def update_day(self):
        """
        将当前日期推进到下一天。
        Advances the current date to the next day.
        """
        # self.process_day_operations() # process_day_operations is called by agent's step() now.
        self.current_day += 1

    # ----------------------------------------------------------------
    # 以下为后续功能（框架预留，供后续扩展）：
    # The following section contains hooks for future extensions
    # ----------------------------------------------------------------

    def jit_production_plan_abs(
            self,
            start_day: int,
            horizon: int,
            capacity: int,
            inv_raw: int,
            inv_prod: int,
            future_raw_deliver: Dict[int, int],
            future_prod_deliver: Dict[int, int],
    ) -> Dict[str, Dict[int, int]]:
        """
        JIT 最晚化生产计划（绝对日历版）

        所有 dict 的 key 都是绝对日 (int)。
        仅处理 start_day … start_day + horizon 这一区间；其他键会被忽略。
        """
        t, H, cap = start_day, horizon, capacity
        size = H + 1  # 天数

        # ---------- 0. 预先把稀疏 dict 投影到稠密 list ----------
        raw_in: List[int] = [0] * size
        dem: List[int] = [0] * size

        for day, qty in future_raw_deliver.items():
            d = day - t
            if 0 <= d <= H:
                raw_in[d] = qty

        for day, qty in future_prod_deliver.items():
            d = day - t
            if 0 <= d <= H:
                dem[d] = qty

        # ---------- 1. 累计销量 C_k 与 M_d ----------
        cum_dem = [0] * size
        s = 0
        for i in range(size):
            s += dem[i]
            cum_dem[i] = s

        M_d = [0] * size
        M = float("-inf")
        for i in range(H, -1, -1):
            M = max(M, cum_dem[i] - cap * i)
            M_d[i] = M

        # ---------- 2. 正向滚动生产 ----------
        prod_plan = {}
        rem_cap = {}
        emer_raw = {}
        raw_after = [0] * size  # 仅内部用

        raw_stock, prod_stock, cum_prod = inv_raw, inv_prod, 0

        for d in range(size):
            day = t + d  # 绝对日

            raw_stock += raw_in[d]

            min_cum_prod = max(0, M_d[d] + cap * d - inv_prod)
            need_today = max(0, min_cum_prod - cum_prod)

            # 紧急采购
            if need_today > raw_stock:
                emer_qty = need_today - raw_stock
                emer_raw[day] = emer_qty
                raw_stock += emer_qty
            else:
                emer_raw[day] = 0

            # 生产
            raw_stock -= need_today
            prod_stock += need_today
            cum_prod += need_today

            # 发货
            prod_stock -= dem[d]  # 保证 ≥0

            # 记录
            prod_plan[day] = need_today
            rem_cap[day] = cap - need_today
            raw_after[d] = raw_stock

        # ---------- 3. 规划性原料需求 ----------
        suf_prod = 0
        suf_raw_in = 0
        planned_raw = {}

        for d in range(H, -1, -1):
            day = t + d
            suf_prod += prod_plan[day]
            suf_raw_in += raw_in[d]
            future_need = suf_prod
            future_sup = raw_after[d] + (suf_raw_in - raw_in[d])
            planned_raw[day] = max(0, future_need - future_sup)

        return {
            "prod_plan": prod_plan,
            "remaining_capacity": rem_cap,
            "emergency_raw_demand": emer_raw,
            "planned_raw_demand": planned_raw,
        }

    def plan_production(self, up_to_day: int):
        """
        根据已签合同和库存，生成从 current_day 到 up_to_day 的总体生产计划，
        同时计算每日所需原材料不足量，包括当日必须购买的原材料和未来所需总原材料。

        采用贪心策略：尽量延迟生产，但确保能满足所有需求。
        Generate an overall production schedule from current_day to up_to_day based on signed contracts and inventory.
        It also calculates the daily shortfall of raw materials required, including the raw materials that must be purchased for the day and the total raw materials required for the future.

        Use a greedy strategy: try to delay production as much as possible, but make sure that all demands are satisfied.
        """
        # Plan Horizon 规划的范围
        horizon = up_to_day - self.current_day

        # 清空当前的生产计划（重新规划）和不足原料记录 Empty the current production schedule (reprogramming) and records of insufficient raw materials
        self.production_plan = {day: 0.0 for day in range(self.current_day, up_to_day + 1)}
        self.insufficient_raw = {day: {"daily": 0.0, "total": 0.0} for day in range(self.current_day, up_to_day + 1)}

        # 获取当前原材料库存 Get current raw material inventory
        inv_raw = sum(batch.remaining for batch in self.raw_batches)

        # 获取当前产品库存 Get current product inventory
        inv_prod = sum(batch.remaining for batch in self.product_batches)

        # Capacity of a day 一天的最大产能
        capacity = self.daily_production_capacity

        # 按交割日期排序的需求合同 Sales Contracts sorted by delivery date
        demand_contracts = sorted(
            [c for c in self._pending_demand if c.delivery_time <= up_to_day],
            key=lambda c: c.delivery_time
        )

        # 按交割日期排序的供应合同 Supply Contracts sorted by delivery date
        supply_contracts = sorted(
            [c for c in self._pending_supply if c.delivery_time <= up_to_day],
            key=lambda c: c.delivery_time
        )

        # 计算每天的需求量（产品交割） Daily demand for product delivery
        daily_demand = {day: 0 for day in range(self.current_day, up_to_day + 1)}
        for contract in demand_contracts:
            day = contract.delivery_time
            if day in daily_demand:
                daily_demand[day] += int(contract.quantity)

        # 计算每天的供应量（原材料到货） Daily supply of raw materials
        daily_supply = {day: 0 for day in range(self.current_day, up_to_day + 1)}
        for contract in supply_contracts:
            day = contract.delivery_time
            if day in daily_supply:
                daily_supply[day] += int(contract.quantity)

        result = self.jit_production_plan_abs(
            start_day=self.current_day,
            horizon=horizon,
            capacity=int(capacity),
            inv_raw=inv_raw,
            inv_prod=inv_prod,
            future_raw_deliver=daily_supply,
            future_prod_deliver=daily_demand,
        )

        # 需要计算的值
        self.production_plan = result["prod_plan"]
        for day in range(self.current_day, up_to_day + 1):
            if day not in self.insufficient_raw:
                self.insufficient_raw[day] = {"daily": 0.0, "total": 0.0}
            self.insufficient_raw[day]["daily"] = result["emergency_raw_demand"].get(day, 0)
            self.insufficient_raw[day]["total"] = result["planned_raw_demand"].get(day, 0)

        return self.production_plan

    def deliver_products(self) -> int:
        """
        当天向下游交割产品，FIFO 方式扣减 product_batches。
        返回本日完成交割的合同 ID 列表。
        如果当天产品不足，将交付所有可用产品，并重新规划生产计划。
        Delivered product downstream on the same day. product_batches are deducted in FIFO mode.
        Returns a list of contract IDs that have completed delivery on the current day.
        If there are not enough products for the day, all available products will be delivered and the production schedule will be re-planned.
        """
        # 查找当天需要交割的销售合同
        # Find the sales contracts that need to be delivered on the same day
        to_deliver = [c for c in self._pending_demand if c.delivery_time == self.current_day]

        # 计算需交付总量
        # Calculate the total amount to be delivered
        total_demand = sum(c.quantity for c in to_deliver)

        # 检查可用产品是否足够
        # Check if available products are sufficient
        total_available = sum(batch.remaining for batch in self.product_batches)

        if total_available < total_demand:
            # 记录不足数量（交付时发现的不足）
            # Insufficient quantity (discovered during delivery)
            shortfall = total_demand - total_available

            # 更新不足记录
            # UPDATE the shortage record
            if self.current_day not in self.insufficient_raw:
                self.insufficient_raw[self.current_day] = {"daily": shortfall, "total": shortfall}
            else:
                # 在实际交付时发现的不足立即成为当日不足
                self.insufficient_raw[self.current_day]["daily"] = max(
                    self.insufficient_raw[self.current_day]["daily"],
                    shortfall
                )
                # 确保总不足量至少等于当日不足量
                self.insufficient_raw[self.current_day]["total"] = max(
                    self.insufficient_raw[self.current_day]["total"],
                    shortfall
                )

            # 交付所有可用产品
            # Deliver all available products
            delivered_contracts = []
            remaining_available = total_available

            # 按合同交付，直到用完可用产品
            # Deliver according to the contract until all available products are used up
            for contract in to_deliver:
                if remaining_available <= 0:
                    break

                # 确定可以交付的数量
                deliverable = min(contract.quantity, remaining_available)

                # 按FIFO原则减少产品批次
                self._reduce_product_inventory(deliverable)

                # 更新可用量
                remaining_available -= deliverable

                # 如果完全满足了合同需求，则标记为已交付
                if deliverable == contract.quantity:
                    delivered_contracts.append(contract.contract_id)
                    self._pending_demand.remove(contract)
                else:
                    # 部分满足，更新合同需求量
                    # Partially fulfilled; update the contract quantity
                    # 因为当前实现没有直接更新合同的方式，我们记录不足并保留合同
                    # Because the current implementation cannot update contracts directly, we record the deficit and keep the contract
                    # 后续可以考虑添加更新合同需求量的功能
                    # Future work: allow updating the contract quantity
                    pass

            # 重新规划生产计划，考虑新的不足量
            self.plan_production(self.max_day)  # Use self.max_day for full horizon

            return total_available
        else:
            # 可以完全满足需求
            # Demands can be fully met
            # 按FIFO原则减少产品批次
            # Reduce product batches according to FIFO principle
            self._reduce_product_inventory(total_demand)

            # 更新合同状态
            delivered_ids = [c.contract_id for c in to_deliver]
            for contract in to_deliver:
                self._pending_demand.remove(contract)

            return total_available

    def _reduce_product_inventory(self, quantity: float):
        """
        使用FIFO原则从产品库存中减少指定数量。
        Use the FIFO principle to reduce a specified quantity from product inventory.
        """
        remaining_to_reduce = quantity

        # 临时复制一个批次列表，避免在迭代时修改
        product_batches_copy = self.product_batches.copy()

        for i, batch in enumerate(product_batches_copy):
            if batch.remaining <= 0:
                continue

            if batch.remaining >= remaining_to_reduce:
                # 该批次足够完成剩余需求
                # This batch is enough to meet the remaining demand
                self.product_batches[i].remaining -= remaining_to_reduce
                remaining_to_reduce = 0
                break
            else:
                # 该批次不足，全部使用
                # This batch is insufficient, use it all
                remaining_to_reduce -= batch.remaining
                self.product_batches[i].remaining = 0

        # 清理剩余量为0的批次
        # Clean up batches with remaining quantity of 0
        self.product_batches = [b for b in self.product_batches if b.remaining > 0]

    def get_udpp(self, current_day: int, n_steps: int) -> dict[int, int]:
        """
        Calculates the unsatisfied daily production plan (UDPP) for all future steps.
        This method simulates inventory flow day-by-day to determine the actual shortfall.

        为所有未来时间点计算未满足的日生产计划 (UDPP)。
        此方法通过逐日模拟库存流来确定实际的物料短缺量。

        Args:
            current_day: The current simulation day. / 当前的模拟日期。
            n_steps: The total number of steps in the simulation. / 模拟的总天数。

        Returns:
            A dictionary mapping each future day to its unsatisfied raw material need.
            一个将未来每一天映射到其未满足的原材料需求的字典。
        """
        # 1. Ensure the production plan is up-to-date.
        # 1. 确保生产计划是基于最新合同的。
        self.plan_production(self.max_day)

        # 2. Initialization
        # 2. 初始化
        udpp = {}
        inventory_on_hand = self.get_inventory_summary(day=current_day, mtype=MaterialType.RAW)["current_stock"]

        # Extract future deliveries from contracts.
        # 从合同中提取未来的交货计划。
        deliveries = {}
        for contract in self.get_pending_contracts():
            if contract.type == IMContractType.SUPPLY:
                day = contract.delivery_time
                if day >= current_day:
                    deliveries[day] = deliveries.get(day, 0) + contract.quantity

        # 3. Simulate inventory flow day-by-day.
        # 3. 逐日模拟库存流。
        for day in range(current_day, n_steps):
            # a. Materials are delivered at the beginning of the day.
            # a. 物料在当天开始时入库。
            inventory_on_hand += deliveries.get(day, 0)

            # b. Get production demand for the day.
            # b. 获取当日的生产需求。
            demand_today = self.production_plan.get(day, 0)

            if demand_today == 0:
                udpp[day] = 0
                continue

            # c. Calculate the shortfall for the day (the UDPP).
            # c. 计算当日的短缺量（即UDPP）。
            if inventory_on_hand >= demand_today:
                udpp[day] = 0
            else:
                udpp[day] = demand_today - inventory_on_hand

            # d. Update simulated inventory after production.
            # d. 更新生产后的模拟库存。
            inventory_on_hand = max(0, inventory_on_hand - demand_today)

        return udpp

    def get_production_plan_all(self) -> Dict[int, float]:
        """返回已经排定的生产计划：{day: quantity, ...}"""
        return self.production_plan

    def get_production_plan(self, day: int | None = None):
        """返回指定日期的生产计划量，或全部计划。"""
        if day is None:
            return self.get_production_plan_all()
        return self.get_production_plan_all().get(day, 0)

    def get_total_future_production_plan(self) -> float:
        """
        返回未来的生产计划之和（当前日期之后的计划）。
        Returns the future production plan (the plan after the current date).
        """
        return sum(
            qty for day, qty in self.production_plan.items() if day > self.current_day
        )

    def get_max_possible_production(self, day: int) -> float:
        """
        计算从 current_day 到 day 的最大可能生产（不考虑库存限制，
        仅受日生产能力约束）。
        Calculate the maximum possible production from current_day to day (regardless of inventory constraints.
        constrained only by the daily production capacity).
        """
        if day < self.current_day:
            return 0.0
        days = day - self.current_day + 1
        return days * self.daily_production_capacity

    def get_available_production_capacity(self, day: int) -> float:
        """
        从当前日期到 ``day`` 的剩余可用生产能力。

        该值在 ``get_max_possible_production`` 的上限基础上，扣除了
        已排定 ``production_plan`` 中的产量，以反映考虑既有合同后的真实
        可用产能。
        """
        if day < self.current_day:
            return 0.0
        if self.daily_production_capacity == float("inf"):
            return float("inf")

        max_prod = self.get_max_possible_production(day)
        planned = sum(
            self.production_plan.get(d, 0) for d in range(self.current_day, day + 1)
        )
        remaining = max_prod - planned
        return max(0.0, remaining)

    def get_insufficient_raw(self) -> dict[int, dict[str, float]]:
        """
        返回因原料不足未能完成的生产/交割需求情况。
        Returns production/delivery requirements that could not be fulfilled due to lack of feedstock.
        """
        return self.insufficient_raw

    def simulate_future_inventory(self, up_to_day: int) -> Dict[int, Dict]:
        """
        模拟从当前日期到指定日期的库存变化，
        包括接收原料、生产和交付产品的影响。
        返回每天的库存预测。
        Simulates the change in inventory from the current date to a specified date that
        Includes the impact of receiving raw materials, production, and delivering products.
        Returns a daily inventory forecast.
        """
        # 创建结果字典
        # Create a dictionary for results
        result = {}

        # 复制当前状态以进行模拟
        # Copy the current state for simulation
        raw_inventory = sum(batch.remaining for batch in self.raw_batches)
        product_inventory = sum(batch.remaining for batch in self.product_batches)

        # 按交割日期排序的需求合同
        # Demand contracts sorted by delivery date
        demand_contracts = sorted(
            [c for c in self._pending_demand if c.delivery_time <= up_to_day],
            key=lambda c: c.delivery_time
        )

        # 按交割日期排序的供应合同
        # Supply contracts sorted by delivery date
        supply_contracts = sorted(
            [c for c in self._pending_supply if c.delivery_time <= up_to_day],
            key=lambda c: c.delivery_time
        )

        # 模拟每天的变化
        # Simulate day-by-day changes
        for day in range(self.current_day, up_to_day + 1):
            # 添加当天的原料供应
            day_supply = sum(c.quantity for c in supply_contracts if c.delivery_time == day)
            raw_inventory += day_supply

            # 使用当天的生产计划
            day_production = self.production_plan.get(day, 0)
            if 0 < day_production <= raw_inventory:
                raw_inventory -= day_production
                product_inventory += day_production

            # 减去当天的产品需求
            day_demand = sum(c.quantity for c in demand_contracts if c.delivery_time == day)
            product_inventory = max(0, product_inventory - day_demand)

            # 记录当天的库存情况
            result[day] = {
                'raw_inventory': raw_inventory,
                'product_inventory': product_inventory,
                'day_supply': day_supply,
                'day_production': day_production,
                'day_demand': day_demand
            }

        return result

    def process_day_operations(self):
        """
        执行当天所有标准操作流程：
        1. 接收今日到货的材料
        2. 执行今日的生产计划
        3. 交付今日需要交付的产品
        4. 更新未来的生产计划（默认30天）

        这是一个便捷方法，允许代理在一天结束时一次性执行所有操作。
        代理也可以选择分别调用各个操作方法以获得更精细的控制。

        返回一个包含各步骤执行结果的字典。
        Perform all standard operating procedures for the day:
        1. Receive materials arriving today
        2. Execute today's production plan
        3. Deliver the products that need to be delivered today
        4. update the production plan for the future (default 30 days)

        This is a convenient method that allows the agent to perform all operations at once at the end of the day.
        The agent can also choose to call each operation method separately for finer control.

        A dictionary containing the results of each step is returned.
        注意：并不是非要显式地调用这个方法，代理可以选择分别调用各个操作方法以获得更精细的控制。
        Note: It is not necessary to call this method explicitly; the agent may choose to call each action method separately for finer control.
        """
        # 执行每日操作
        received_materials = self.receive_materials()

        # 执行生产
        self.execute_production(self.current_day)

        # 交付产品
        delivered_products = self.deliver_products()

        # 重新计划生产
        self.plan_production(self.max_day)  # Use self.max_day for full horizon

        return {
            "received_materials": received_materials,
            "delivered_products": delivered_products
        }

    def get_pending_contracts(self, is_supply: bool = None, day: int = None) -> List[IMContract]:
        """
        获取所有未交割的合同列表。

        参数:
            is_supply: 如果为True，只返回采购合同；如果为False，只返回销售合同；
                      如果为None，返回所有合同
            day: 如果指定，只返回在该日期交割的合同；如果为None，返回所有未交割合同

        返回:
            满足条件的合同列表
        Gets a list of all undelivered contracts.

        Parameters.
            is_supply: if True, returns only purchase contracts; if False, returns only sales contracts;
                      If None, return all contracts
            day: if specified, only contracts delivered on this date are returned; if None, all open contracts are returned

        Returns.
            List of contracts that meet the condition
        """
        contracts = []

        if is_supply is None or is_supply is True:
            if day is None:
                contracts.extend(self._pending_supply)
            else:
                contracts.extend([c for c in self._pending_supply if c.delivery_time == day])

        if is_supply is None or is_supply is False:
            if day is None:
                contracts.extend(self._pending_demand)
            else:
                contracts.extend([c for c in self._pending_demand if c.delivery_time == day])

        return contracts

    def get_batch_details(self, day: int, mtype: MaterialType) -> List[Dict]:
        """
        返回指定日期指定类型库存的批次详情，包括每批次的数量、单价、
        存储成本和存储天数。
        """
        batches = self.raw_batches if mtype == MaterialType.RAW else self.product_batches
        storage_cost = self.raw_storage_cost if mtype == MaterialType.RAW else self.product_storage_cost

        details = []
        for batch in batches:
            if batch.remaining <= 0:
                continue

            days_stored = max(0, day - batch.production_time)
            storage_cost_total = storage_cost * days_stored * batch.remaining

            details.append({
                'batch_id': batch.batch_id,
                'quantity': batch.remaining,
                'base_unit_cost': batch.unit_cost,
                'current_unit_cost': batch.unit_cost + storage_cost * days_stored,
                'storage_days': days_stored,
                'storage_cost': storage_cost_total,
                'production_time': batch.production_time
            })

        # 按生产时间排序（FIFO顺序）
        return sorted(details, key=lambda x: x['production_time'])

    def execute_production(im, day):
        """模拟执行生产过程"""
        # Simulate the production process
        production_plan = im.get_production_plan_all()
        production_qty = production_plan.get(day, 0)

        if production_qty <= 0:
            return

        # 检查原材料库存是否足够
        raw_summary = im.get_inventory_summary(day, MaterialType.RAW)

        if raw_summary['current_stock'] >= production_qty:
            # 模拟生产批次创建
            # Simulate creation of a production batch
            batch_id = f"PROD_{day}"
            unit_cost = raw_summary['average_cost'] + im.processing_cost

            # 创建产品批次
            # Create a product batch
            batch = Batch(
                batch_id=batch_id,
                remaining=production_qty,
                unit_cost=unit_cost,
                production_time=day
            )

            # 减少原材料库存（模拟FIFO减少）
            # Reduce raw material inventory in FIFO order
            remaining_to_reduce = production_qty
            raw_batches_copy = im.raw_batches.copy()

            for i, batch_item in enumerate(raw_batches_copy):  # Renamed batch to batch_item
                if batch_item.remaining <= 0:
                    continue

                if batch_item.remaining >= remaining_to_reduce:
                    im.raw_batches[i].remaining -= remaining_to_reduce
                    remaining_to_reduce = 0
                    break
                else:
                    remaining_to_reduce -= batch_item.remaining
                    im.raw_batches[i].remaining = 0

            # 清理剩余量为0的批次
            # Remove batches with zero remaining quantity
            im.raw_batches = [b for b in im.raw_batches if b.remaining > 0]

            # 添加产品批次
            # Add the product batch to inventory
            im.product_batches.append(batch)
        else:
            # 记录原材料不足情况
            # Record the shortage of raw materials
            if day not in im.insufficient_raw:
                im.insufficient_raw[day] = {"daily": production_qty, "total": production_qty}
            else:
                im.insufficient_raw[day]["daily"] += production_qty
                im.insufficient_raw[day]["total"] += production_qty
