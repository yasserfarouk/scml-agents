from .inventory_manager_n import InventoryManager, IMContract, IMContractType, MaterialType

def test_raw_material_inflow_detailed(days=10):
    # 初始化 InventoryManager，只关注原材料入库，不做生产和出库
    im = InventoryManager(
        raw_storage_cost=1.0,
        product_storage_cost=1.0,
        processing_cost=0.0,
        daily_production_capacity=0
    )

    for day in range(0, days):
        qty = day
        price_per_unit = float(day)  # 单价设为 day
        pass # print(f"现在是第 {day} 天:")
        # 新建一个当天到货的原材料供应合同
        contract = IMContract(
            contract_id=f"c{day}",
            partner_id="supplier",
            type=IMContractType.SUPPLY,
            quantity=qty,
            price=price_per_unit,
            delivery_time=day,       # 第 day 天投递
            bankruptcy_risk=0.0,
            material_type=MaterialType.RAW
        )
        im.add_transaction(contract)
        pass # print(f"添加原材料合同: {contract.contract_id}, 交付日期：{contract.delivery_time}, 数量：{contract.quantity}, 单价：{contract.price:.2f}")
        im.receive_materials()


        # 从 InventoryManager 获取库存汇总
        for inv_day in range(day, days):
            summary = im.get_inventory_summary(inv_day, MaterialType.RAW)
            pass # print(f"查看第{inv_day}天库存, 现有库存: {summary['current_stock']}, 平均单价: {summary['average_cost']:.2f}, 存储成本: {summary['storage_cost']:.2f}, 预估可用库存: {summary['estimated_available']}, 预估平均单价: {summary['estimated_average_cost']:.2f}")
        # 获取各批次详细
        batch_details = im.get_batch_details(day, MaterialType.RAW)
        pass # print(f"第 {day} 天的批次详细信息:")
        for batch in batch_details:
            pass # print(f"批次 ID: {batch['batch_id']}, 数量: {batch['quantity']}, 单价: {batch['base_unit_cost']:.2f}, 现单价: {batch['current_unit_cost']},存储天数: {batch['storage_days']}, 存储成本: {batch['storage_cost']:.2f}, 生产时间: {batch['production_time']}")
        im.update_day()  # 推进一天

if __name__ == "__main__":
    test_raw_material_inflow_detailed()