# test_inventory_manager.py
from .inventory_manager_n import InventoryManager, IMContract, IMContractType, MaterialType, Batch
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def test_inventory_manager_10_days():
    """
    对InventoryManager进行全面测试，模拟10天运行。
    每天：
    1. 生成随机的供货协议（原材料）
    2. 生成随机的销售协议（产品）
    3. 接收材料
    4. 生成生产计划
    5. 交付产品
    6. 记录日报
    7. 推进到下一天
    """
    # 初始化InventoryManager，设置合理的成本参数
    im = InventoryManager(
        raw_storage_cost=0.5,  # 原材料存储成本
        product_storage_cost=1.0,  # 产品存储成本
        processing_cost=5.0,  # 加工成本
        daily_production_capacity=10  # 每日生产能力
    )

    # 用于存储每日报告的数据结构
    daily_reports = {
        'day': [],
        'raw_inventory': [],
        'product_inventory': [],
        'raw_avg_cost': [],
        'product_avg_cost': [],
        'contracts_added': [],
        'materials_received': [],
        'products_delivered': [],
        'production_planned': []
    }

    # 签订的合同ID
    supply_contracts = []
    demand_contracts = []

    # 模拟10天
    for day in range(1, 11):
        pass # print(f"\n======== 第 {day} 天 ========")
        # 0. 输出当天的不足原材料情况
        insufficient = im.get_insufficient_raw()
        pass # print("不足的原材料信息：", insufficient)

        # 1. 生成供货协议（原材料）- 数量和价格有波动
        num_supply_contracts = random.randint(1, 3)  # 每天1-3个供货合同
        for i in range(num_supply_contracts):
            # 随机生成原材料数量和价格
            quantity = random.randint(5, 15)  # 5-15个单位
            price = 8.0 + random.uniform(-1.0, 3.0)  # 价格在7-11之间波动

            # 生成合同ID
            contract_id = f"S{day}_{i}"

            # 随机决定交货时间（当天或未来1-3天内）
            delivery_time = day + random.randint(0, 3)

            # 创建合同
            contract = IMContract(
                contract_id=contract_id,
                partner_id=f"Supplier_{i}",
                type=IMContractType.SUPPLY,
                quantity=quantity,
                price=price,
                delivery_time=delivery_time,
                bankruptcy_risk=0.05,
                material_type=MaterialType.RAW
            )

            # 添加交易
            if im.add_transaction(contract):
                supply_contracts.append(contract_id)
                pass # print(f"添加供货协议: ID={contract_id}, 数量={quantity}, 价格={price:.2f}, 交货时间={delivery_time}")
            else:
                pass # print(f"供货协议添加失败: ID={contract_id}")
        # 如果当日的供货协议没有满足当日不足的原材料需求，则补充一个协议补齐需求
        if insufficient and day in insufficient:
            for vals in insufficient[day].items():
                pass # print(vals)
                if vals[1] > 0:
                    # 补充协议
                    quantity = vals['daily']
                    price = 8.0 + random.uniform(-1.0, 3.0)
                    contract_id = f"SUPPLY_{day}_backup"
                    delivery_time = day
                    contract = IMContract(
                        contract_id=contract_id,
                        partner_id=f"Supplier_backup",
                        type=IMContractType.SUPPLY,
                        quantity=quantity,
                        price=price,
                        delivery_time=delivery_time,
                        bankruptcy_risk=0.05,
                        material_type=MaterialType.RAW
                    )
                    if im.add_transaction(contract):
                        supply_contracts.append(contract_id)
                        print(
                            f"添加供货协议: ID={contract_id}, 数量={quantity}, 价格={price:.2f}, 交货时间={delivery_time}")
                    else:
                        pass # print(f"供货协议添加失败: ID={contract_id}")
        # 2. 生成销售协议（产品）- 价格高于原材料
        num_demand_contracts = random.randint(0, 2)  # 每天0-2个销售合同
        for i in range(num_demand_contracts):
            # 随机生成产品数量和价格
            quantity = random.randint(3, 8)  # 3-8个单位
            price = 18.0 + random.uniform(-1.0, 4.0)  # 价格在17-22之间波动

            # 生成合同ID
            contract_id = f"D{day}_{i}"

            # 随机决定交货时间（通常比当前日期晚1-5天）
            delivery_time = day + random.randint(1, 5)

            # 创建合同
            contract = IMContract(
                contract_id=contract_id,
                partner_id=f"Customer_{i}",
                type=IMContractType.DEMAND,
                quantity=quantity,
                price=price,
                delivery_time=delivery_time,
                bankruptcy_risk=0.1,
                material_type=MaterialType.PRODUCT
            )

            # 添加交易
            if im.add_transaction(contract):
                demand_contracts.append(contract_id)
                pass # print(f"添加销售协议: ID={contract_id}, 数量={quantity}, 价格={price:.2f}, 交货时间={delivery_time}")
            else:
                pass # print(f"销售协议添加失败: ID={contract_id}")

        # 3. 接收今天到货的材料
        received_materials = im.receive_materials()
        pass # print(f"接收材料: {received_materials}")

        # 4. 重新规划未来30天的生产计划
        production_plan = im.plan_production(day + 30)

        # 打印未来几天的生产计划
        planned_days = sorted([d for d in production_plan.keys() if d >= day])[:5]  # 只显示未来5天
        planned_str = ", ".join([f"第{d}天: {production_plan[d]}" for d in planned_days])
        pass # print(f"生产计划: {planned_str if planned_str else '无计划生产'}")

        # 打印不足原料情况
        insufficient = im.get_insufficient_raw()
        if insufficient:
            pass # print("不足原料:")
            for d, vals in insufficient.items():
                if d >= day and d <= day + 5:  # 只显示近期的不足
                    pass # print(f"  第{d}天 - 当日必须: {vals['daily']}, 总计需要: {vals['total']}")

        # 5. 模拟执行生产 - 从生产计划中提取当天的生产量
        execute_production(im, day)

        # 6. 交付今天需要交付的产品
        delivered_products = im.deliver_products()
        pass # print(f"交付产品: {delivered_products}")

        # 7. 收集每日报告数据
        raw_summary = im.get_inventory_summary(day, MaterialType.RAW)
        product_summary = im.get_inventory_summary(day, MaterialType.PRODUCT)

        daily_reports['day'].append(day)
        daily_reports['raw_inventory'].append(raw_summary['current_stock'])
        daily_reports['product_inventory'].append(product_summary['current_stock'])
        daily_reports['raw_avg_cost'].append(raw_summary['average_cost'])
        daily_reports['product_avg_cost'].append(product_summary['average_cost'])
        daily_reports['contracts_added'].append(num_supply_contracts + num_demand_contracts)
        daily_reports['materials_received'].append(len(received_materials))
        daily_reports['products_delivered'].append(len(delivered_products))

        # 计算未来5天的计划生产总量
        future_production = sum([v for k, v in production_plan.items() if k >= day and k <= day + 5])
        daily_reports['production_planned'].append(future_production)

        # 8. 更新到下一天
        im.update_day()

    # 打印最终摘要
    pass # print("\n======== 10天模拟总结 ========")
    pass # print(f"供货协议总数: {len(supply_contracts)}")
    pass # print(f"销售协议总数: {len(demand_contracts)}")

    # 获取最终库存汇总
    final_day = 10
    raw_summary = im.get_inventory_summary(final_day, MaterialType.RAW)
    product_summary = im.get_inventory_summary(final_day, MaterialType.PRODUCT)

    pass # print(f"最终原材料库存: {raw_summary['current_stock']} 单位，平均成本: {raw_summary['average_cost']:.2f}")
    pass # print(f"最终产品库存: {product_summary['current_stock']} 单位，平均成本: {product_summary['average_cost']:.2f}")

    # 计算未交付订单
    pending_supply = [c for c in im._pending_supply if c.delivery_time > final_day]
    pending_demand = [c for c in im._pending_demand if c.delivery_time > final_day]

    pass # print(f"未交付原材料订单: {len(pending_supply)} 个")
    pass # print(f"未交付产品订单: {len(pending_demand)} 个")

    # 返回每日报告数据，用于可视化
    return im, daily_reports


def execute_production(im, day):
    """模拟执行生产过程"""
    production_plan = im.get_production_plan()
    production_qty = production_plan.get(day, 0)

    if production_qty <= 0:
        pass # print("今天没有计划生产")
        return

    # 检查原材料库存是否足够
    raw_summary = im.get_inventory_summary(day, MaterialType.RAW)

    if raw_summary['current_stock'] >= production_qty:
        # 模拟生产批次创建
        batch_id = f"PROD_{day}"
        unit_cost = raw_summary['average_cost'] + im.processing_cost

        # 创建产品批次
        batch = Batch(
            batch_id=batch_id,
            remaining=production_qty,
            unit_cost=unit_cost,
            production_time=day
        )

        # 减少原材料库存（模拟FIFO减少）
        remaining_to_reduce = production_qty
        raw_batches_copy = im.raw_batches.copy()

        for i, batch in enumerate(raw_batches_copy):
            if batch.remaining <= 0:
                continue

            if batch.remaining >= remaining_to_reduce:
                im.raw_batches[i].remaining -= remaining_to_reduce
                remaining_to_reduce = 0
                break
            else:
                remaining_to_reduce -= batch.remaining
                im.raw_batches[i].remaining = 0

        # 清理剩余量为0的批次
        im.raw_batches = [b for b in im.raw_batches if b.remaining > 0]

        # 添加产品批次
        im.product_batches.append(batch)

        pass # print(f"执行生产: 生产了 {production_qty} 单位产品")
    else:
        pass # print(f"原材料不足，无法执行计划生产量 {production_qty}")
        # 记录原材料不足情况
        if day not in im.insufficient_raw:
            im.insufficient_raw[day] = {"daily": production_qty, "total": production_qty}
        else:
            im.insufficient_raw[day]["daily"] += production_qty
            im.insufficient_raw[day]["total"] += production_qty


def visualize_results(reports):
    """可视化测试结果"""
    # 转换为DataFrame以便于分析和可视化
    df = pd.DataFrame(reports)

    # 绘制库存变化图表
    plt.figure(figsize=(16, 12))

    # 库存水平变化
    plt.subplot(2, 2, 1)
    plt.plot(df['day'], df['raw_inventory'], 'b-o', label='原材料库存')
    plt.plot(df['day'], df['product_inventory'], 'r-o', label='产品库存')
    plt.title('库存水平随时间变化')
    plt.xlabel('天数')
    plt.ylabel('库存量')
    plt.legend()
    plt.grid(True)

    # 平均成本变化
    plt.subplot(2, 2, 2)
    plt.plot(df['day'], df['raw_avg_cost'], 'b-o', label='原材料平均成本')
    plt.plot(df['day'], df['product_avg_cost'], 'r-o', label='产品平均成本')
    plt.title('平均成本随时间变化')
    plt.xlabel('天数')
    plt.ylabel('成本')
    plt.legend()
    plt.grid(True)

    # 合同和交付情况
    plt.subplot(2, 2, 3)
    plt.bar(df['day'] - 0.2, df['contracts_added'], width=0.4, label='新增合同')
    plt.bar(df['day'] + 0.2, df['materials_received'] + df['products_delivered'], width=0.4, label='完成交付')
    plt.title('合同和交付情况')
    plt.xlabel('天数')
    plt.ylabel('数量')
    plt.legend()
    plt.grid(True)

    # 生产计划
    plt.subplot(2, 2, 4)
    plt.bar(df['day'], df['production_planned'], width=0.6, label='未来5天计划生产量')
    plt.title('生产计划')
    plt.xlabel('天数')
    plt.ylabel('数量')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('inventory_manager_test_results.png')
    plt.show()

    # 显示每日详细数据
    pass # print("\n每日详细报告:")
    pass # print(df)

    return df


if __name__ == "__main__":
    im, reports = test_inventory_manager_10_days()
    df = visualize_results(reports)

    # 打印批次详情
    final_day = 10
    pass # print(f"\n======== 第 {final_day} 天库存批次详情 ========")

    raw_batches = im.get_batch_details(final_day, MaterialType.RAW)
    pass # print("\n原材料批次:")
    if raw_batches:
        for batch in raw_batches:
            print(f"批次ID: {batch['batch_id']}, 数量: {batch['quantity']}, " +
                  f"基本单价: {batch['base_unit_cost']:.2f}, 当前单价: {batch['current_unit_cost']:.2f}, " +
                  f"存储天数: {batch['storage_days']}, 存储成本: {batch['storage_cost']:.2f}")
    else:
        pass # print("没有原材料批次")

    product_batches = im.get_batch_details(final_day, MaterialType.PRODUCT)
    pass # print("\n产品批次:")
    if product_batches:
        for batch in product_batches:
            print(f"批次ID: {batch['batch_id']}, 数量: {batch['quantity']}, " +
                  f"基本单价: {batch['base_unit_cost']:.2f}, 当前单价: {batch['current_unit_cost']:.2f}, " +
                  f"存储天数: {batch['storage_days']}, 存储成本: {batch['storage_cost']:.2f}")
    else:
        pass # print("没有产品批次")