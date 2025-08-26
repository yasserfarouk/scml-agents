# LitaAgentYR Strategy 文档 / Strategy Documentation

## 总体策略 Overview
LitaAgentYR 继承自 StdSyncAgent，通过 InventoryManager 动态评估原料、产能与市场价。
代理采用三阶段采购与产能约束销售策略，并在谈判过程中根据时间和对手让步率调整报价。
所有启发式数值集中在 `HeuristicSettings` 中，方便调节。

The agent extends StdSyncAgent and relies on an InventoryManager to evaluate
raw materials, capacity and market prices. It negotiates using a three tier
procurement policy and a capacity-aware sales policy. Concessions depend on time
progress and opponent concession rate. All heuristic values are stored in the
`HeuristicSettings` dataclass for easy tuning.

## 关键变量 Key Variables
- `heuristics`：当前使用的启发式参数集。Current heuristic configuration.
- `min_profit_ratio`：最低利润率要求。Minimum profit ratio.
- `bargain_threshold`：机会性囤货阈值。Threshold for cheap purchases.
- `cash_flow_limit_ratio`：采购现金流比例上限。Cash flow limit for buying.
- `distribution_ratio_today`：当日需求在伙伴间的分配比例。Ratio of partners to consider for today's demand.

## 主要方法 Main Methods
- `counter_all(offers, states)`：按伙伴角色拆分报价，分别调用销售或采购逻辑。
- `_process_supply_offers`：根据紧急、计划、可选三类需求处理供应报价。
- `_process_sales_offers`：在产能约束下处理销售报价。
- `_calc_conceded_price`：依据谈判进度与对手表现计算让步后的价格。
- `_update_dynamic_stockpiling_parameters`：根据库存和剩余天数调整囤货折扣。
- `update_profit_strategy`：外部接口，可动态修改利润率或囤货阈值。

## 调用关系 Call Flow
1. `counter_all` 收到当前所有报价，根据伙伴是供应商还是消费者分别调用
   `_process_supply_offers` 或 `_process_sales_offers`。
2. 这些处理函数会根据库存、产能及 `heuristics` 设置决定是否接受或还价，
   需要计算价格时均调用 `_calc_conceded_price`。
3. `_update_dynamic_stockpiling_parameters` 在每天结束时触发，调整
   `bargain_threshold` 等值影响后续采购决策。
4. 成交的合同通过 `on_negotiation_success` 写入 InventoryManager 以更新库存。

## 启发式参数说明 Heuristic Parameters
`HeuristicSettings` 定义了所有启发式值，例如让步曲线指数、产能紧张判定阈值、
帕累托探索增量和折扣比例等。创建代理时可以传入自定义实例以改变行为。

