import unittest
from copy import deepcopy

# 假设 test_custom_inventory_manager.py 与 inventory_manager_cir.py 在同一目录下
# 或者 inventory_manager_cir.py 所在的包 litaagent_std 在 PYTHONPATH 中
# 如果在 tests 子目录，可能需要 from ..custom_inventory_manager import ...
try:
    # 尝试相对导入（如果测试文件在 litaagent_std 包内的一个子目录，例如 tests）
    from ..custom_inventory_manager import (
        CustomInventoryManager,
        IMContract,
        IMContractType,
        MaterialType,
        InventoryBatch,
    )
except ImportError:
    # 尝试直接导入（如果测试文件与 inventory_manager_cir.py 在同一目录，
    # 或者 litaagent_std 包的父目录在 PYTHONPATH 中，并且测试以模块形式运行）
    from inventory_manager_cir import (
        InventoryManagerCIR,
        IMContract,
        IMContractType,
        MaterialType,
        InventoryBatch,
    )


class TestCustomInventoryManager(unittest.TestCase):

    def setUp(self):
        """在每个测试方法运行前设置环境"""
        self.raw_storage_cost = 0.1
        self.product_storage_cost = 0.2
        self.processing_cost = 10.0
        self.daily_capacity = 50
        self.max_sim_day = 100
        self.initial_day = 0

        self.im = InventoryManagerCIR(
            raw_storage_cost=self.raw_storage_cost,
            product_storage_cost=self.product_storage_cost,
            processing_cost=self.processing_cost,
            daily_production_capacity=self.daily_capacity,
            max_simulation_day=self.max_sim_day,
            current_day=self.initial_day
        )

    def test_initialization(self):
        """测试库存管理器的基本初始化"""
        self.assertEqual(self.im.current_day, self.initial_day)
        self.assertEqual(self.im.processing_cost_per_unit, self.processing_cost)
        self.assertEqual(self.im.daily_production_capacity, self.daily_capacity)
        self.assertEqual(self.im.raw_storage_cost_per_unit_per_day, self.raw_storage_cost)
        self.assertEqual(self.im.product_storage_cost_per_unit_per_day, self.product_storage_cost)
        self.assertEqual(len(self.im.raw_material_batches), 0)
        self.assertEqual(len(self.im.product_batches), 0)
        self.assertEqual(len(self.im.pending_supply_contracts), 0)
        self.assertEqual(len(self.im.pending_demand_contracts), 0)
        self.assertEqual(len(self.im.production_plan), 0)

    def test_add_supply_contract(self):
        """测试添加供应合同"""
        contract = IMContract("supply1", "partnerA", IMContractType.SUPPLY, 100, 5.0, 5, MaterialType.RAW)
        self.assertTrue(self.im.add_transaction(contract))
        self.assertIn(contract, self.im.pending_supply_contracts)

        summary_day5_raw = self.im.get_inventory_summary(5, MaterialType.RAW)
        self.assertEqual(summary_day5_raw["estimated_incoming"], 100)

    def test_add_demand_contract(self):
        """测试添加需求合同并触发生产计划"""
        contract = IMContract("demand1", "partnerB", IMContractType.DEMAND, 30, 20.0, 3, MaterialType.PRODUCT)
        self.assertTrue(self.im.add_transaction(contract))
        self.assertIn(contract, self.im.pending_demand_contracts)

        summary_day3_prod = self.im.get_inventory_summary(3, MaterialType.PRODUCT)
        self.assertEqual(summary_day3_prod["estimated_outgoing"], 30)

        # 检查生产计划是否已更新 (add_transaction 会调用 plan_production)
        # 假设生产会安排在交货日或之前
        planned_qty_for_demand = 0
        for day in range(contract.delivery_time + 1):  # 从第0天到交货日
            planned_qty_for_demand += self.im.production_plan.get(day, 0)
        self.assertGreaterEqual(planned_qty_for_demand, 30, "Production plan should cover the demand")

    def test_add_contract_past_date(self):
        """测试添加交货日期已过的合同"""
        self.im.current_day = 2
        past_contract = IMContract("past_supply", "partnerC", IMContractType.SUPPLY, 10, 1.0, 1, MaterialType.RAW)
        self.assertFalse(self.im.add_transaction(past_contract), "Should not add contract with past delivery date")
        self.assertNotIn(past_contract, self.im.pending_supply_contracts)

    def test_receive_materials(self):
        """测试原材料接收"""
        day_to_receive = 2
        supply_contract = IMContract("supply_rcv", "s1", IMContractType.SUPPLY, 50, 2.0, day_to_receive,
                                     MaterialType.RAW)
        self.im.add_transaction(supply_contract)

        self.im.current_day = day_to_receive  # 模拟时间到达接收日
        self.im._receive_materials(day_to_receive)

        self.assertNotIn(supply_contract, self.im.pending_supply_contracts)
        self.assertEqual(len(self.im.raw_material_batches), 1)
        self.assertEqual(self.im.raw_material_batches[0].quantity, 50)
        self.assertEqual(self.im.raw_material_batches[0].unit_cost, 2.0)

        summary = self.im.get_inventory_summary(day_to_receive, MaterialType.RAW)
        self.assertEqual(summary["current_stock"], 50)

    def test_plan_production_capacity_constrained(self):
        """测试产能受限时的生产计划"""
        self.im.daily_production_capacity = 10  # 限制产能
        demand_contract = IMContract("demand_cap", "c1", IMContractType.DEMAND, 25, 30.0, 2, MaterialType.PRODUCT)
        self.im.add_transaction(demand_contract)  # 会自动调用 plan_production

        # 需求25，日产能10，交货日为第2天
        # 预期：第0天产10，第1天产10，第2天产5 (或类似分布，总计25)
        self.assertEqual(self.im.production_plan.get(0, 0), 10)
        self.assertEqual(self.im.production_plan.get(1, 0), 10)
        self.assertEqual(self.im.production_plan.get(2, 0), 5)

        total_planned = sum(self.im.production_plan.values())
        self.assertEqual(total_planned, 25)

    def test_execute_production_sufficient_raw(self):
        """测试原材料充足时的生产执行"""
        day_to_produce = 1
        # 添加原材料
        self.im.raw_material_batches.append(InventoryBatch(20, 1.0, day_to_produce - 1))  # 假设前一天就有
        self.im.production_plan[day_to_produce] = 15  # 计划生产15

        self.im.current_day = day_to_produce
        produced_qty = self.im._execute_production(day_to_produce)

        self.assertEqual(produced_qty, 15)
        self.assertEqual(self.im.raw_material_batches[0].quantity, 5)  # 20 - 15
        self.assertEqual(len(self.im.product_batches), 1)
        self.assertEqual(self.im.product_batches[0].quantity, 15)
        # 成本 = 原材料成本 + 加工成本
        expected_product_cost = 1.0 + self.processing_cost
        self.assertAlmostEqual(self.im.product_batches[0].unit_cost, expected_product_cost)

    def test_execute_production_insufficient_raw(self):
        """测试原材料不足时的生产执行"""
        day_to_produce = 1
        self.im.raw_material_batches.append(InventoryBatch(5, 1.0, day_to_produce - 1))
        self.im.production_plan[day_to_produce] = 10  # 计划生产10，但只有5单位原料

        self.im.current_day = day_to_produce
        produced_qty = self.im._execute_production(day_to_produce)

        self.assertEqual(produced_qty, 5)  # 只能生产5单位
        self.assertEqual(len(self.im.raw_material_batches), 0)  # 原料耗尽
        self.assertEqual(len(self.im.product_batches), 1)
        self.assertEqual(self.im.product_batches[0].quantity, 5)

    def test_deliver_products_sufficient_stock(self):
        """测试库存充足时的产品交付"""
        day_to_deliver = 3
        # 添加产品库存
        self.im.product_batches.append(InventoryBatch(20, 15.0, day_to_deliver - 1))
        demand_contract = IMContract("demand_deliv", "c2", IMContractType.DEMAND, 10, 25.0, day_to_deliver,
                                     MaterialType.PRODUCT)
        self.im.add_transaction(demand_contract)  # 不会再触发plan_production，因为库存已模拟存在

        self.im.current_day = day_to_deliver
        delivered_summary = self.im._deliver_products(day_to_deliver)

        self.assertNotIn(demand_contract, self.im.pending_demand_contracts)
        self.assertEqual(self.im.product_batches[0].quantity, 10)  # 20 - 10
        self.assertEqual(delivered_summary["total_delivered_quantity"], 10)
        self.assertAlmostEqual(delivered_summary["total_revenue"], 10 * 25.0)

    def test_deliver_products_insufficient_stock(self):
        """测试库存不足时的产品交付（部分交付）"""
        day_to_deliver = 3
        self.im.product_batches.append(InventoryBatch(5, 15.0, day_to_deliver - 1))  # 只有5单位库存
        demand_contract = IMContract("demand_part_deliv", "c3", IMContractType.DEMAND, 10, 25.0, day_to_deliver,
                                     MaterialType.PRODUCT)
        self.im.add_transaction(demand_contract)

        self.im.current_day = day_to_deliver
        delivered_summary = self.im._deliver_products(day_to_deliver)

        # 当前 CustomInventoryManager 的 _deliver_products 会移除合同，即使是部分交付
        self.assertNotIn(demand_contract, self.im.pending_demand_contracts,
                         "Contract should be removed even if partially fulfilled")
        self.assertEqual(len(self.im.product_batches), 0)  # 库存耗尽
        self.assertEqual(delivered_summary["total_delivered_quantity"], 5)  # 只能交付5
        self.assertAlmostEqual(delivered_summary["total_revenue"], 5 * 25.0)

    def test_process_day_end_operations_full_cycle(self):
        """测试完整的日终操作流程"""
        # Day 0: 添加供应和需求
        supply_day1 = IMContract("s_eod", "sup_eod", IMContractType.SUPPLY, 20, 2.0, 1, MaterialType.RAW)
        demand_day1 = IMContract("d_eod", "cus_eod", IMContractType.DEMAND, 10, 20.0, 1, MaterialType.PRODUCT)
        self.im.add_transaction(supply_day1)
        self.im.add_transaction(demand_day1)  # 这会触发plan_production, 计划在第1天生产10

        self.assertEqual(self.im.production_plan.get(1, 0), 10, "Should plan to produce 10 on day 1")

        # Day 1: 执行日终操作
        self.im.current_day = 1  # 手动设置IM内部的当天，因为process_day_end_operations的参数是“即将结束的这一天”
        results_day1 = self.im.process_day_end_operations(simulation_current_day=1)

        # 检查状态
        self.assertEqual(self.im.current_day, 2, "IM current_day should advance to 2")

        # 1. 原材料接收
        self.assertEqual(len(self.im.pending_supply_contracts), 0)
        raw_summary_after_d1 = self.im.get_inventory_summary(1, MaterialType.RAW)  # 查看第1天结束时的状态
        # 20单位原料进来，生产用掉10单位
        self.assertEqual(raw_summary_after_d1["current_stock"], 10, "Raw stock should be 10 after production")

        # 2. 生产执行
        prod_summary_after_d1 = self.im.get_inventory_summary(1, MaterialType.PRODUCT)
        # 生产10单位，交付10单位
        self.assertEqual(prod_summary_after_d1["current_stock"], 0, "Product stock should be 0 after delivery")
        self.assertEqual(results_day1["produced_today"], 10)

        # 3. 产品交付
        self.assertEqual(len(self.im.pending_demand_contracts), 0)
        self.assertEqual(results_day1["delivered_products"], 10)
        self.assertAlmostEqual(results_day1["revenue_today"], 10 * 20.0)

        # 4. 重新规划 (plan_production 被调用，但此时可能没有新的未满足需求)
        # 检查第2天的计划，应该没有，因为之前的需求已满足
        self.assertEqual(self.im.production_plan.get(2, 0), 0,
                         "No production should be planned for day 2 if demand met")

    def test_get_inventory_summary_costs_and_values(self):
        """测试库存摘要中的成本和价值估算"""
        # Day 0: 接收原材料
        self.im.raw_material_batches.append(InventoryBatch(10, 2.0, 0))  # 10单位 @ 2元
        self.im.raw_material_batches.append(InventoryBatch(10, 3.0, 0))  # 10单位 @ 3元

        raw_summary0 = self.im.get_inventory_summary(0, MaterialType.RAW)
        self.assertEqual(raw_summary0["current_stock"], 20)
        self.assertAlmostEqual(raw_summary0["estimated_average_cost"], 2.5)  # (10*2 + 10*3) / 20

        # Day 0: 计划并生产产品
        self.im.production_plan[0] = 5  # 生产5单位
        self.im._execute_production(0)  # 使用平均成本2.5的原料

        prod_summary0 = self.im.get_inventory_summary(0, MaterialType.PRODUCT)
        self.assertEqual(prod_summary0["current_stock"], 5)
        expected_prod_cost = 2.5 + self.processing_cost  # 原料平均成本 + 加工费
        self.assertAlmostEqual(prod_summary0["estimated_average_cost"], expected_prod_cost)

        # Day 1: 检查库存价值 (storage cost)
        # 原料剩余15单位 @ 2.5元 (FIFO，但这里是平均成本) -> 实际上是 (5*2 + 10*3)/15 = 2.666
        # _execute_production 应该按FIFO消耗批次
        # 重新模拟_execute_production的批次消耗
        self.setUp()  # 重置im
        self.im.raw_material_batches.append(InventoryBatch(10, 2.0, 0))
        self.im.raw_material_batches.append(InventoryBatch(10, 3.0, 0))
        self.im.production_plan[0] = 12  # 生产12单位
        self.im._execute_production(0)  # 消耗第一批10单位，第二批2单位

        raw_summary_day1_start = self.im.get_inventory_summary(1, MaterialType.RAW)
        self.assertEqual(raw_summary_day1_start["current_stock"], 8)  # 剩余第二批的8单位
        self.assertAlmostEqual(raw_summary_day1_start["estimated_average_cost"], 3.0)  # 剩余都是3元成本的

        prod_summary_day1_start = self.im.get_inventory_summary(1, MaterialType.PRODUCT)
        self.assertEqual(prod_summary_day1_start["current_stock"], 12)
        # 10单位产品成本 = 2.0(原料) + 10(加工) = 12
        # 2单位产品成本  = 3.0(原料) + 10(加工) = 13
        # 平均产品成本 = (10*12 + 2*13) / 12 = (120 + 26) / 12 = 146 / 12 = 12.1666...
        self.assertAlmostEqual(prod_summary_day1_start["estimated_average_cost"],
                               (10 * (2.0 + self.processing_cost) + 2 * (3.0 + self.processing_cost)) / 12.0)

    def test_get_insufficient_raw(self):
        """测试原材料短缺量的计算"""
        self.im.production_plan[0] = 30
        self.im.production_plan[1] = 20

        self.assertEqual(self.im.get_today_insufficient_raw(0), 30)
        self.assertEqual(self.im.get_total_insufficient_raw(0, horizon=1), 30)  # horizon 1 = today only
        self.assertEqual(self.im.get_total_insufficient_raw(0, horizon=2), 50)  # today + next day

        # 添加一些原材料
        self.im.raw_material_batches.append(InventoryBatch(10, 1.0, 0))
        self.assertEqual(self.im.get_today_insufficient_raw(0), 20)  # 30 - 10
        self.assertEqual(self.im.get_total_insufficient_raw(0, horizon=2), 40)  # (30-10) for day 0 + 20 for day 1

    def test_get_available_production_capacity(self):
        """测试可用生产能力的计算"""
        self.assertEqual(self.im.get_available_production_capacity(0), self.daily_capacity)
        self.im.production_plan[0] = 10
        self.assertEqual(self.im.get_available_production_capacity(0), self.daily_capacity - 10)
        self.im.production_plan[0] = self.daily_capacity + 5  # 超额计划 (plan_production应该避免，但测试_get_available...)
        self.assertEqual(self.im.get_available_production_capacity(0), 0)  # 不应为负

    def test_deepcopy_inventory_manager(self):
        """测试库存管理器的深拷贝功能"""
        self.im.add_transaction(IMContract("s1", "p1", IMContractType.SUPPLY, 10, 1, 1, MaterialType.RAW))
        self.im.raw_material_batches.append(InventoryBatch(5, 1, 0))

        im_copy = self.im.deepcopy()

        # 验证是不同的对象
        self.assertIsNot(self.im, im_copy)
        self.assertIsNot(self.im.pending_supply_contracts, im_copy.pending_supply_contracts)
        self.assertIsNot(self.im.raw_material_batches, im_copy.raw_material_batches)
        self.assertIsNot(self.im.production_plan, im_copy.production_plan)

        # 验证内容相同
        self.assertEqual(len(self.im.pending_supply_contracts), len(im_copy.pending_supply_contracts))
        self.assertEqual(self.im.pending_supply_contracts[0].id, im_copy.pending_supply_contracts[0].id)
        self.assertEqual(len(self.im.raw_material_batches), len(im_copy.raw_material_batches))
        self.assertEqual(self.im.raw_material_batches[0].quantity, im_copy.raw_material_batches[0].quantity)

        # 修改原对象，拷贝不应受影响
        self.im.add_transaction(IMContract("s2", "p2", IMContractType.SUPPLY, 20, 2, 2, MaterialType.RAW))
        self.im.raw_material_batches[0].quantity = 99

        self.assertEqual(len(im_copy.pending_supply_contracts), 1)  # 拷贝仍为1个合同
        self.assertEqual(im_copy.raw_material_batches[0].quantity, 5)  # 拷贝的批次数量不变


if __name__ == '__main__':
    unittest.main()
