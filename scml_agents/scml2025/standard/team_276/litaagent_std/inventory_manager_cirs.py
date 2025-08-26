import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import uuid # For contract IDs
import copy # For deepcopy

if TYPE_CHECKING:
    # This is to avoid circular dependencies if CustomInventoryManager methods type hint themselves
    # or if other modules need to type hint CustomInventoryManager without full import.
    pass

class MaterialType(Enum):
    RAW = auto()
    PRODUCT = auto()

class IMContractType(Enum):
    SUPPLY = auto()  # Buying from supplier (supplies raw to agent)
    DEMAND = auto()  # Selling to consumer (agent demands product from its stock)

@dataclass
class IMContract:
    partner_id: str
    contract_id: str
    type: IMContractType
    quantity: int
    price: float
    delivery_time: int
    material_type: MaterialType
    bankruptcy_risk: float = 0.0

@dataclass
class Batch: # For tracking inventory
    batch_id: str # Added default factory for batch_id
    original_quantity: int
    remaining_quantity: int
    unit_cost: float # Cost at which it was acquired (raw) or produced (product)
    arrival_or_production_time: int
    material_type: MaterialType

class InventoryManagerCIR:
    def __init__(self,
                 raw_storage_cost: float,
                 product_storage_cost: float,
                 processing_cost: float,
                 daily_production_capacity: int, # Was float, now int (except for float('inf'))
                 max_simulation_day: int,
                 current_day: int = 0):

        self.current_day: int = current_day
        self.max_simulation_day: int = max_simulation_day

        # Costs
        self.raw_storage_cost_per_unit_per_day: float = raw_storage_cost
        self.product_storage_cost_per_unit_per_day: float = product_storage_cost
        self.processing_cost_per_unit: float = processing_cost # Cost to convert 1 raw to 1 product

        self.daily_production_capacity: float = daily_production_capacity

        # Core Data Structures
        self.raw_material_batches: List[Batch] = []
        self.product_batches: List[Batch] = []

        # Pending contracts (future commitments)
        # These store IMContract objects
        self.pending_supply_contracts: List[IMContract] = [] # Raw materials agent will receive
        self.pending_demand_contracts: List[IMContract] = [] # Products agent must deliver

        # Production plans: Dict[day, quantity_to_produce]
        self.production_plan: Dict[int, float] = {}

        # Statistics / Metrics (optional, can be expanded later)
        self.total_raw_material_acquired: float = 0.0
        self.total_products_produced: float = 0.0
        self.total_products_sold: float = 0.0
        # ... any other metrics useful for the agent's strategy

        self.is_deepcopy = False
    # --- Core Logic Methods ---

    def add_transaction(self, contract: IMContract) -> bool:
        if contract.delivery_time < self.current_day:
            return False

        if contract.type == IMContractType.SUPPLY:
            if contract.material_type != MaterialType.RAW:
                return False
            self.pending_supply_contracts.append(contract)
            self.pending_supply_contracts.sort(key=lambda c: c.delivery_time)
        elif contract.type == IMContractType.DEMAND:
            if contract.material_type != MaterialType.PRODUCT:
                return False
            self.pending_demand_contracts.append(contract)
            self.pending_demand_contracts.sort(key=lambda c: c.delivery_time)
        else:
            return False

        # In a full system, adding a transaction, especially a demand, might trigger re-planning.
        self.plan_production()
        return True

    def get_inventory_summary(self, day: int, mtype: MaterialType) -> Dict[str, int]:
        """预期可用库存是*包括*了当天入库、生产或者交付的，现在库存是*不包括*还没生产的东西的，即使是未来的日期也会返回*实际上已经生产并入库的*产品库存量"""
        batches_to_consider = self.raw_material_batches if mtype == MaterialType.RAW else self.product_batches
        storage_cost_per_unit = self.raw_storage_cost_per_unit_per_day if mtype == MaterialType.RAW else self.product_storage_cost_per_unit_per_day

        current_stock_qty = 0
        total_cost_of_current_stock = 0.0

        for batch in batches_to_consider:
            if batch.arrival_or_production_time < day:
                current_stock_qty += batch.remaining_quantity
                storage_duration = max(0, day - batch.arrival_or_production_time)
                cost_for_batch = batch.unit_cost * batch.remaining_quantity + \
                                 storage_cost_per_unit * storage_duration * batch.remaining_quantity
                total_cost_of_current_stock += cost_for_batch

        average_cost_of_current_stock = (total_cost_of_current_stock / current_stock_qty) if current_stock_qty > 0 else 0.0

        estimated_available_qty = current_stock_qty

        # For estimated cost, start with the cost of current stock
        estimated_total_cost = total_cost_of_current_stock
        # Keep track of quantities added for weighted average of future items
        future_qty_added_for_estimation = 0
        cost_of_future_qty_added = 0.0

        if mtype == MaterialType.RAW:
            for contract in self.pending_supply_contracts:
                # Only consider future deliveries up to 'day' that haven't logically arrived yet based on 'self.current_day'
                if self.current_day <= contract.delivery_time <= day:
                    estimated_available_qty += contract.quantity
                    # For estimated cost, add cost of these future items (excluding storage for now)
                    cost_of_future_qty_added += contract.price * contract.quantity
                    future_qty_added_for_estimation += contract.quantity
            # Raw materials are consumed by production.
            # This assumes production plan is accurate for future consumption.
            for prod_day in range(self.current_day, day + 1):
                planned_prod = self.production_plan.get(prod_day, 0)
                estimated_available_qty -= planned_prod
                # Cost impact of consumption is complex for estimation; for now, focus on quantity.

        elif mtype == MaterialType.PRODUCT:
            # Products are added by production.
            for prod_day in range(self.current_day, day + 1):
                planned_prod = self.production_plan.get(prod_day, 0)
                estimated_available_qty += planned_prod
                # For estimated cost, if current stock is zero, we need to estimate cost of future production
                if current_stock_qty == 0: # Simplified: only estimate if no current stock
                    # This is a rough estimate. A better one would be avg raw cost at prod_day + processing.
                    # For now, if production_plan is empty, this won't add much.
                    # Using current avg raw cost as a proxy if available, else just processing cost.
                    # This part remains very simplified due to plan_production being a placeholder.
                    # Let's assume for simplicity: if plan_production is not implemented,
                    # this cost contribution is primarily from processing_cost_per_unit for future items.
                    # A more robust way is needed when plan_production is active.

                    # Simplified: Use a proxy for cost of future products if no current stock
                    # This needs more refinement when plan_production is detailed.
                    # For now, let's say it's based on processing_cost and a hypothetical raw cost.
                    # If self.raw_material_batches is not empty, use its latest cost.
                    proxy_raw_cost = 0.0
                    if self.raw_material_batches: # A very rough proxy
                        proxy_raw_cost = self.raw_material_batches[-1].unit_cost

                    cost_of_future_qty_added += planned_prod * (proxy_raw_cost + self.processing_cost_per_unit)
                    future_qty_added_for_estimation += planned_prod


            for contract in self.pending_demand_contracts:
                if self.current_day <= contract.delivery_time <= day:
                    estimated_available_qty -= contract.quantity

        estimated_average_cost_val = 0.0
        if estimated_available_qty > 0:
            if current_stock_qty > 0 : # If there's current stock, its avg cost is a good base
                 # Refined: weighted average of current stock cost and future items cost
                total_estimated_qty_for_cost = current_stock_qty + future_qty_added_for_estimation
                if total_estimated_qty_for_cost > 0:
                     estimated_average_cost_val = (total_cost_of_current_stock + cost_of_future_qty_added) / total_estimated_qty_for_cost
                else: # Should not happen if estimated_available_qty > 0
                    estimated_average_cost_val = 0.0
            elif future_qty_added_for_estimation > 0: # No current stock, but future items
                estimated_average_cost_val = cost_of_future_qty_added / future_qty_added_for_estimation
        else: # estimated_available_qty <= 0
            estimated_average_cost_val = 0.0


        return {
            "current_stock": current_stock_qty,
            "average_cost": average_cost_of_current_stock,
            "estimated_available": estimated_available_qty,
            "estimated_average_cost": estimated_average_cost_val
        }

    def _receive_materials(self, day_being_processed: int):
        processed_contracts_indices = []
        for i, contract in enumerate(self.pending_supply_contracts):
            if contract.delivery_time == day_being_processed:
                if contract.material_type != MaterialType.RAW:
                    continue

                new_batch = Batch(
                    # batch_id=contract.contract_id, # Using contract_id might lead to non-unique batch_ids if contracts are renegotiated/split. Default UUID is safer.
                    batch_id=str(uuid.uuid4()),
                    original_quantity=contract.quantity,
                    remaining_quantity=contract.quantity,
                    unit_cost=contract.price,
                    arrival_or_production_time=day_being_processed,
                    material_type=MaterialType.RAW
                )
                self.raw_material_batches.append(new_batch)
                self.total_raw_material_acquired += contract.quantity
                processed_contracts_indices.append(i)

        # Remove processed contracts (iterate in reverse to avoid index issues)
        for i in sorted(processed_contracts_indices, reverse=True):
            del self.pending_supply_contracts[i]

        # Sort raw material batches by arrival time for strict FIFO if not already guaranteed
        self.raw_material_batches.sort(key=lambda b: b.arrival_or_production_time)


    def _execute_production(self, day_being_processed: int):
        planned_qty_to_produce = self.production_plan.get(day_being_processed, 0)
        if planned_qty_to_produce <= 0:
            return

        # Sort raw material batches to ensure FIFO
        self.raw_material_batches.sort(key=lambda b: b.arrival_or_production_time)

        available_raw_material = sum(b.remaining_quantity for b in self.raw_material_batches)

        actual_produced_qty = min(planned_qty_to_produce, available_raw_material, self.daily_production_capacity)

        if actual_produced_qty < planned_qty_to_produce:
            # Store shortfall if needed: self.production_shortfalls[(day_being_processed, "reason")] = planned_qty_to_produce - actual_produced_qty
            pass

        if actual_produced_qty <= 0:
            return

        consumed_raw_material_cost = 0.0
        qty_to_consume_for_production = actual_produced_qty # 1 unit of raw makes 1 unit of product

        temp_batches_to_remove_indices = []
        for i, batch in enumerate(self.raw_material_batches):
            if qty_to_consume_for_production <= 0:
                break

            consume_from_this_batch = min(qty_to_consume_for_production, batch.remaining_quantity)
            consumed_raw_material_cost += consume_from_this_batch * batch.unit_cost
            batch.remaining_quantity -= consume_from_this_batch
            qty_to_consume_for_production -= consume_from_this_batch

            if batch.remaining_quantity <= 0:
                temp_batches_to_remove_indices.append(i)

        for i in sorted(temp_batches_to_remove_indices, reverse=True):
            del self.raw_material_batches[i]

        production_cost_per_unit = (consumed_raw_material_cost / actual_produced_qty) + self.processing_cost_per_unit if actual_produced_qty > 0 else self.processing_cost_per_unit

        product_batch = Batch(
            batch_id=str(uuid.uuid4()),
            original_quantity=actual_produced_qty,
            remaining_quantity=actual_produced_qty,
            unit_cost=production_cost_per_unit,
            arrival_or_production_time=day_being_processed,
            material_type=MaterialType.PRODUCT
        )
        self.product_batches.append(product_batch)
        self.total_products_produced += actual_produced_qty

        # Sort product batches by arrival time for strict FIFO if not already guaranteed
        self.product_batches.sort(key=lambda b: b.arrival_or_production_time)


    def _deliver_products(self, day_being_processed: int):
        processed_contracts_indices = []

        # Sort product batches to ensure FIFO
        self.product_batches.sort(key=lambda b: b.arrival_or_production_time)

        for i, contract in enumerate(self.pending_demand_contracts):
            if contract.delivery_time == day_being_processed:
                if contract.material_type != MaterialType.PRODUCT:
                    continue

                qty_to_deliver = contract.quantity
                available_product_stock = sum(b.remaining_quantity for b in self.product_batches)

                actual_delivered_qty = min(qty_to_deliver, available_product_stock)

                if actual_delivered_qty < qty_to_deliver:
                    # Store shortfall: self.delivery_shortfalls[(day_being_processed, contract.contract_id)] = qty_to_deliver - actual_delivered_qty
                    pass

                if actual_delivered_qty <= 0:
                    processed_contracts_indices.append(i) # Still mark as processed for the day
                    continue

                temp_batches_to_remove_indices = []
                qty_fulfilled_for_contract = 0
                for batch_idx, batch in enumerate(self.product_batches):
                    if qty_fulfilled_for_contract >= actual_delivered_qty:
                        break

                    deliver_from_this_batch = min(actual_delivered_qty - qty_fulfilled_for_contract, batch.remaining_quantity)
                    batch.remaining_quantity -= deliver_from_this_batch
                    qty_fulfilled_for_contract += deliver_from_this_batch

                    if batch.remaining_quantity <= 0:
                        temp_batches_to_remove_indices.append(batch_idx)

                for batch_idx in sorted(temp_batches_to_remove_indices, reverse=True):
                    del self.product_batches[batch_idx]

                self.total_products_sold += actual_delivered_qty
                processed_contracts_indices.append(i)

        for i in sorted(processed_contracts_indices, reverse=True):
            del self.pending_demand_contracts[i]


    def process_day_end_operations(self, day_being_processed: int):
        if day_being_processed != self.current_day:
            # Optionally, align current_day or raise error: self.current_day = day_being_processed
            pass


        # 1. Receive materials scheduled for today
        self._receive_materials(day_being_processed)

        # 2. Execute production planned for today
        self._execute_production(day_being_processed)

        # 3. Deliver products scheduled for today
        self._deliver_products(day_being_processed)

        # 4. Advance current day
        self.current_day = day_being_processed + 1

        # 5. Re-plan production for future (still a placeholder call)
        # This should ideally use the new self.current_day as its starting point for planning
        self.plan_production(up_to_day=self.max_simulation_day)

        # Return a summary or status if needed
        return {
            "raw_batches_count": len(self.raw_material_batches),
            "product_batches_count": len(self.product_batches),
            "pending_supply_count": len(self.pending_supply_contracts),
            "pending_demand_count": len(self.pending_demand_contracts),
        }


    def get_today_insufficient_raw(self, day: int) -> int:
        # Placeholder - Detailed implementation in Part 3 (with plan_production)
        # This would typically be: planned_production_for_day - current_raw_stock_for_day
        # For now, if no production plan, it's 0.
        planned_production_today = self.production_plan.get(day, 0)
        if planned_production_today <= 0:
            return 0

        # A simplified check on current raw stock at the start of 'day'
        raw_stock_today = self.get_inventory_summary(day, MaterialType.RAW)["current_stock"]
        return max(0, planned_production_today - raw_stock_today)

    # This method has been removed as it was a duplicate of the method below.
    # The implementation below is more complete and is the one that should be used.

    def get_total_insufficient_raw(self, target_day: int, horizon: int) -> int:
        total_shortfall = 0

        # Initial simulated raw stock: sum of batches arrived before target_day
        # AND pending contracts delivering before target_day but on/after current_day.
        simulated_raw_stock = 0
        for r_batch in self.raw_material_batches: # Already received stock
            if r_batch.arrival_or_production_time < target_day:
                simulated_raw_stock += r_batch.remaining_quantity

        for s_contract in self.pending_supply_contracts: # Future stock arriving before target_day
            if self.current_day <= s_contract.delivery_time < target_day:
                simulated_raw_stock += s_contract.quantity

        for i in range(horizon):
            d = target_day + i # Current day in the horizon we are evaluating

            # Add raw materials arriving *on* day d
            # These are from pending_supply_contracts delivering on day 'd'
            for s_contract in self.pending_supply_contracts:
                if s_contract.delivery_time == d: # Arriving exactly on this simulated day 'd'
                    simulated_raw_stock += s_contract.quantity

            production_needed = self.production_plan.get(d, 0)

            if production_needed > 0:
                if simulated_raw_stock >= production_needed:
                    simulated_raw_stock -= production_needed
                else: # simulated_raw_stock < production_needed
                    current_day_shortfall = production_needed - simulated_raw_stock
                    total_shortfall += current_day_shortfall
                    simulated_raw_stock = 0 # All stock used up for this day's production

        return total_shortfall

    def plan_production(self, up_to_day: Optional[int] = None):
        # 修改点：将 up_to_day 的默认值设为实际的最后一天索引
        # ---
        # Modified: Set the default value of up_to_day to the actual last day index
        if up_to_day is None:
            up_to_day = self.max_simulation_day - 1 # max_simulation_day 是总步数 (e.g., 50), 所以最后一天索引是 max_simulation_day - 1 / max_simulation_day is the total number of steps (e.g., 50), so the last day index is max_simulation_day - 1

        # 清空当前的生产计划（重新规划）
        # ---
        # Clear the current production plan (re-planning)
        # 确保只清除从当前天到规划截止日期的计划
        # ---
        # Ensure only plans from current_day to up_to_day are cleared
        for day_to_clear in range(self.current_day, up_to_day + 1): # up_to_day is inclusive
            if day_to_clear in self.production_plan:
                del self.production_plan[day_to_clear]

        demands_by_day: Dict[int, int] = {}
        for contract in self.pending_demand_contracts:
            # 只考虑在规划窗口内的需求 [self.current_day, up_to_day]
            # ---
            # Only consider demands within the planning window [self.current_day, up_to_day]
            if self.current_day <= contract.delivery_time <= up_to_day: # up_to_day is inclusive
                demands_by_day[contract.delivery_time] = demands_by_day.get(contract.delivery_time, 0) + contract.quantity

        # Simulate raw material availability for each day in the planning window
        # This will be updated as we plan production
        simulated_raw_stock_by_day = {}

        # Initialize with current raw stock
        initial_raw_stock = 0
        for r_batch in self.raw_material_batches:
            if r_batch.arrival_or_production_time < self.current_day:
                initial_raw_stock += r_batch.remaining_quantity

        # Set initial stock for current day
        simulated_raw_stock_by_day[self.current_day] = initial_raw_stock

        # Add future deliveries to the appropriate days
        for s_contract in self.pending_supply_contracts:
            if self.current_day <= s_contract.delivery_time <= up_to_day:
                delivery_day = s_contract.delivery_time
                simulated_raw_stock_by_day[delivery_day] = simulated_raw_stock_by_day.get(delivery_day, 0) + s_contract.quantity

        # Propagate stock to future days (without considering production yet)
        for day in range(self.current_day + 1, up_to_day + 1):
            if day not in simulated_raw_stock_by_day:
                simulated_raw_stock_by_day[day] = 0
            # Add previous day's remaining stock
            simulated_raw_stock_by_day[day] += simulated_raw_stock_by_day.get(day - 1, 0)

        # Iterate backwards for demand days to implement JIT (schedule later demands first)
        sorted_demand_delivery_days = sorted(demands_by_day.keys(), reverse=True)

        for demand_delivery_day in sorted_demand_delivery_days:
            needed_for_demand_at_delivery_day = demands_by_day[demand_delivery_day]
            remaining_to_plan_for_this_demand = needed_for_demand_at_delivery_day

            # --- BEGIN MODIFICATION FOR DETAILED LOGGING ---
            bottleneck_details_for_this_demand: List[str] = []
            # --- END MODIFICATION ---

            # Try to schedule production as late as possible for this demand
            # Production for demand_delivery_day can happen on or before demand_delivery_day
            for prod_day in range(demand_delivery_day, self.current_day - 1, -1):
                if remaining_to_plan_for_this_demand <= 0:
                    break # All planned for this specific demand

                # Capacity available on prod_day, considering already planned items for other demands
                capacity_before_this_slice = self.daily_production_capacity - self.production_plan.get(prod_day, 0)

                # Raw material available on prod_day
                raw_before_this_slice = simulated_raw_stock_by_day.get(prod_day, 0)

                # Potential to plan based on current remaining demand for *this* specific demand
                potential_to_plan = remaining_to_plan_for_this_demand

                # Can only plan what we have capacity and raw materials for
                can_plan_on_prod_day = min(potential_to_plan, capacity_before_this_slice, raw_before_this_slice)

                if can_plan_on_prod_day < potential_to_plan and potential_to_plan > 0:
                    # Determine the bottleneck(s) for this specific prod_day and potential_to_plan
                    is_capacity_bottleneck = (capacity_before_this_slice <= raw_before_this_slice and capacity_before_this_slice < potential_to_plan)
                    is_raw_bottleneck = (raw_before_this_slice < capacity_before_this_slice and raw_before_this_slice < potential_to_plan)
                    # This covers the case where both are equal and limiting
                    is_both_equally_limiting = (raw_before_this_slice == capacity_before_this_slice and raw_before_this_slice < potential_to_plan)

                    if is_both_equally_limiting:
                        bottleneck_details_for_this_demand.append(f"Day {prod_day}:Cap&Raw_Limit({raw_before_this_slice:.0f})")
                    elif is_capacity_bottleneck:
                        bottleneck_details_for_this_demand.append(f"Day {prod_day}:Cap_Limit({capacity_before_this_slice:.0f})")
                    elif is_raw_bottleneck:
                        bottleneck_details_for_this_demand.append(f"Day {prod_day}:Raw_Limit({raw_before_this_slice:.0f})")
                    # If can_plan_on_prod_day is 0 and potential_to_plan > 0, it implies either capacity_before_this_slice or raw_before_this_slice (or both) was 0.
                    # The above conditions should capture this. For example, if cap=0, raw=10, potential=5 -> cap_limit. if cap=10, raw=0, potential=5 -> raw_limit. if cap=0, raw=0, potential=5 -> cap&raw_limit.

                if can_plan_on_prod_day > 0:
                    # Update production plan
                    self.production_plan[prod_day] = self.production_plan.get(prod_day, 0) + can_plan_on_prod_day
                    remaining_to_plan_for_this_demand -= can_plan_on_prod_day

                    # Update simulated raw stock (consume raw materials)
                    simulated_raw_stock_by_day[prod_day] -= can_plan_on_prod_day

                    # Update future days' raw stock to reflect this consumption
                    for future_day in range(prod_day + 1, up_to_day + 1):
                        if future_day in simulated_raw_stock_by_day:
                            simulated_raw_stock_by_day[future_day] -= can_plan_on_prod_day





    def get_available_production_capacity(self, day: int) -> int:
        if day < self.current_day: # Cannot produce in the past
            return 0
        # Considers self.daily_production_capacity and self.production_plan for that day
        planned_production_for_day = self.production_plan.get(day, 0)
        return max(0, self.daily_production_capacity - planned_production_for_day)

# Example usage (optional, for testing during development)
if __name__ == '__main__':

    # Initialize CustomInventoryManager
    cim = InventoryManagerCIR(
        raw_storage_cost=0.01,
        product_storage_cost=0.02,
        processing_cost=2.0,
        daily_production_capacity=30.0,
        max_simulation_day=10,
        current_day=0
    )

    # --- Test plan_production ---
    # No demands initially, plan should be empty
    cim.plan_production() # Call explicitly for controlled test

    # Add demands. add_transaction calls plan_production() internally.
    cim.add_transaction(IMContract("C1", IMContractType.DEMAND, 20, 10, 2, MaterialType.PRODUCT))
    cim.add_transaction(IMContract("C2", IMContractType.DEMAND, 15, 11, 2, MaterialType.PRODUCT))
    cim.add_transaction(IMContract("C3", IMContractType.DEMAND, 25, 12, 1, MaterialType.PRODUCT))
    cim.add_transaction(IMContract("C4", IMContractType.DEMAND, 40, 13, 3, MaterialType.PRODUCT))

    # Expected JIT (Demands: D1:25, D2:35, D3:40. Capacity:30):
    # Plan for D3=40: Plan[3]=30, Plan[2]=10 (spillover)
    # Plan for D2=35 (add to existing Plan[2]=10): Plan[2] becomes 10+min(35, 30-10=20)=30. Spillover=15 to Day 1. Plan[1]=15
    # Plan for D1=25 (add to existing Plan[1]=15): Plan[1] becomes 15+min(25, 30-15=15)=30. Spillover=10 to Day 0. Plan[0]=10
    # Expected: {0: 10.0, 1: 30.0, 2: 30.0, 3: 30.0}

    # --- Test Shortage Calculations ---
    # Setup: current day = 0. Production plan is as above.
    # Add raw material supplies. These will also call plan_production, but it should be idempotent for demands.
    cim.add_transaction(IMContract("S1_new", IMContractType.SUPPLY, 15, 5, 0, MaterialType.RAW)) # 15 on Day 0

    # Process day 0 to receive S1_new and produce based on Plan[0]=10
    cim.process_day_end_operations(0) # current_day becomes 1. S1_new (15) received. 10 produced. Raw left: 5.

    cim.add_transaction(IMContract("S2", IMContractType.SUPPLY, 5, 5.5, 1, MaterialType.RAW)) # 5 on Day 1. Plan re-calculated.


    # Test get_today_insufficient_raw for Day 1 (current_day is 1)
    # Production planned for Day 1 (today): Plan[1] = 30
    # Available raw by start of Day 1 (strictly before day 1): Batch (5 from day 0). S2 arrives *on* Day 1, not counted for "already arrived".
    # Shortfall for Day 1 = max(0, 30 - 5) = 25.
    insufficient_day1 = cim.get_today_insufficient_raw(1)

    # Test get_total_insufficient_raw for Day 1, horizon 3 (Days 1, 2, 3)
    # Current Day = 1.
    # Initial Simulated Raw Stock (before Day 1): 5 (from batch S1_new)
    # Day 1 (d=1):
    #   Incoming on Day 1: S2 (5). Simulated stock = 5 (initial) + 5 (S2) = 10.
    #   Production needed on Day 1: Plan[1] = 30.
    #   Shortfall D1: 30 - 10 = 20. Total Shortfall = 20. Simulated stock becomes 0.
    # Day 2 (d=2):
    #   Incoming on Day 2: None. Simulated stock = 0.
    #   Production needed on Day 2: Plan[2] = 30.
    #   Shortfall D2: 30 - 0 = 30. Total Shortfall = 20 + 30 = 50. Simulated stock = 0.
    # Day 3 (d=3):
    #   Incoming on Day 3: None. Simulated stock = 0.
    #   Production needed on Day 3: Plan[3] = 30.
    #   Shortfall D3: 30 - 0 = 30. Total Shortfall = 50 + 30 = 80. Simulated stock = 0.
    total_insufficient_d1_h3 = cim.get_total_insufficient_raw(target_day=1, horizon=3)

    # Test with more supply
    cim.add_transaction(IMContract("S3", IMContractType.SUPPLY, 100, 6, 2, MaterialType.RAW)) # 100 on Day 2

    # Recalculate total_insufficient_raw for Day 1, horizon 3 (Days 1, 2, 3) with S3
    # Current Day = 1.
    # Initial Simulated Raw Stock (before Day 1): 5 (from batch S1_new)
    # Day 1 (d=1):
    #   Incoming on Day 1: S2 (5). Simulated stock = 5 (initial) + 5 (S2) = 10.
    #   Production needed: Plan[1] = 30.
    #   Shortfall D1: 30 - 10 = 20. Total Shortfall = 20. Simulated stock = 0.
    # Day 2 (d=2):
    #   Incoming on Day 2: S3 (100). Simulated stock = 0 + 100 = 100.
    #   Production needed: Plan[2] = 30.
    #   Simulated stock becomes 100 - 30 = 70. No shortfall for Day 2. Total Shortfall = 20.
    # Day 3 (d=3):
    #   Incoming on Day 3: None. Simulated stock = 70.
    #   Production needed: Plan[3] = 30.
    #   Simulated stock becomes 70 - 30 = 40. No shortfall for Day 3. Total Shortfall = 20.
    total_insufficient_d1_h3_with_S3 = cim.get_total_insufficient_raw(target_day=1, horizon=3)
