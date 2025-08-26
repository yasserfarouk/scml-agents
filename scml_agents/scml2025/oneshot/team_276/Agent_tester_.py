import pandas as pd
import matplotlib.pyplot as plt
import importlib
import argparse

# Import SCML standard world (2025 or fallback to 2024)
try:
    from scml.std import SCML2025StdWorld
except ImportError:
    from scml.std import SCML2024StdWorld as SCML2025StdWorld

# Import agent classes
# from scml_agents.scml2024 import PenguinAgent # PenguinAgent cannot be used due to version
from scml.oneshot.agents import GreedyOneShotAgent
from scml.std.agents import RandomStdAgent, SyncRandomStdAgent, GreedyStdAgent
from .litaagent_std.litaagent_y import LitaAgentY
from .litaagent_std.litaagent_n import LitaAgentN

# Available agent mapping
AVAILABLE_AGENTS = {
    "LitaAgentY": LitaAgentY,
    "LitaAgentN": LitaAgentN,
    # "PenguinAgent": PenguinAgent,
    "GreedyOneShotAgent": GreedyOneShotAgent,
    "RandomStdAgent": RandomStdAgent,
    "SyncRandomStdAgent": SyncRandomStdAgent,
    "GreedyStdAgent": GreedyStdAgent,
}

# Parse command-line arguments for agent selection
parser = argparse.ArgumentParser(
    description="Run SCML2025 simulation with specified agents."
)
parser.add_argument(
    "--agents",
    nargs="+",
    default=["LitaAgentY", "SyncRandomStdAgent"],
    help="List of agent classes to include (names or Module.Class paths).",
)
args = parser.parse_args()

# Resolve agent classes
agent_classes = []
for name in args.agents:
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    else:
        cls = AVAILABLE_AGENTS.get(name)
        if cls is None:
            available_agent_names = ", ".join(list(AVAILABLE_AGENTS.keys()))
            raise ValueError(
                f"未知代理类型: {name}\n"
                f"可用的代理类型: {available_agent_names}\n"
                f"您也可以通过模块路径导入自定义代理，例如: 'my_module.MyAgent'"
            )
    agent_classes.append(cls)

if not agent_classes:
    agent_classes = [LitaAgentY, SyncRandomStdAgent]

# Create SCML world with specified agents
world_params = SCML2025StdWorld.generate(agent_types=agent_classes, n_steps=50)
world = SCML2025StdWorld(**world_params, construct_graphs=True)

# Run the simulation with the progress bar
world.run_with_progress()

# Analyze results
contracts_df = pd.DataFrame.from_records(world.saved_contracts)
contracts_df.to_csv("contracts.csv", index=False)

# Visualization 1: Bid sequences for each negotiation
# (Attempt to retrieve negotiation traces)
try:
    negotiation_traces = []
    for contract in world.saved_contracts:
        nid = contract.get("negotiation_id", None)
        if not nid:
            continue
        negotiation = None
        if hasattr(world, "get_negotiation"):
            negotiation = world.get_negotiation(nid)
        elif hasattr(world, "negotiations"):
            negotiation = world.negotiations.get(nid)
        if negotiation:
            try:
                trace = negotiation.extended_trace() or negotiation.full_trace()
            except:
                trace = None
            if trace:
                negotiation_traces.append(
                    (contract["buyer_name"], contract["seller_name"], trace)
                )
except Exception:
    negotiation_traces = []
    pass  # print(f"Warning: could not retrieve negotiation traces: {e}")

for buyer, seller, trace in negotiation_traces:
    offers_data = []
    for idx, (step, negotiator, offer) in enumerate(trace):
        # extract price and quantity from offer outcome
        if isinstance(offer, dict):
            price = offer.get("unit_price", None)
            quantity = offer.get("quantity", None)
        else:
            # assuming tuple ordering as (price, quantity, maybe time)
            price = offer[0] if len(offer) > 0 else None
            quantity = offer[1] if len(offer) > 1 else None
        party = (
            "Buyer"
            if (isinstance(negotiator, str) and buyer in negotiator)
            else "Seller"
        )
        offers_data.append((idx, price, quantity, party))
    df_offers = pd.DataFrame(
        offers_data, columns=["round", "price", "quantity", "party"]
    )
    plt.figure()
    for party, grp in df_offers.groupby("party"):
        plt.scatter(
            grp["round"],
            grp["price"],
            s=(grp["quantity"] if grp["quantity"].notnull().all() else 20) * 5,
            label=f"{party} offer",
            alpha=0.7,
        )
        plt.plot(grp["round"], grp["price"], "--", alpha=0.5)
    plt.title(f"Negotiation: {seller} vs {buyer}")
    plt.xlabel("Offer Round")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f"bids_{seller}_{buyer}.png")
    plt.close()

# Visualization 2: Final agreements and Pareto frontier
points = []
point_labels = []  # 存储每个点的买方和卖方信息
agent_contracts = []  # 存储所有涉及代理的合约
lita_agent_y_contracts = []  # 存储所有与LitaAgentY相关的合约

# 首先打印所有合约的基本信息，用于调试
pass  # print("合约总数:", len(contracts_df))
pass  # print("合约详情 (买方, 卖方, 价格, 数量):")
for _, ct in contracts_df.iterrows():
    b = ct["buyer_name"]
    s = ct["seller_name"]
    p = ct["unit_price"]
    q = ct["quantity"]
    pass  # print(f"  {b} → {s}: 价格={p}, 数量={q}")
    # 如果至少一方是代理(非BUYER/SELLER)，则添加到代理合约列表
    if b != "BUYER" or s != "SELLER":
        agent_contracts.append(ct)
    # 如果至少一方是LitaAgentY，则添加到LitaAgentY合约列表
    if "Li" in b or "Li" in s:
        lita_agent_y_contracts.append(ct)

pass  # print(f"\n代理参与的合约数: {len(agent_contracts)}")
pass  # print(f"LitaAgentY参与的合约数: {len(lita_agent_y_contracts)}")

# 只分析与LitaAgentY相关的合约
for ct in lita_agent_y_contracts:
    buyer = ct["buyer_name"]
    seller = ct["seller_name"]
    price = ct["unit_price"]
    qty = ct["quantity"]

    # 计算卖方利润 (简化计算 - 收入)
    seller_profit = price * qty

    # 计算买方利润 (使用环境价格，如果有的话)
    env_price = world_params.get(
        "unit_price", 1.5 * price
    )  # 如果没有环境价格，假设为合约价格的1.5倍
    buyer_profit = env_price * qty - price * qty

    # 对于外部方，设置一个默认利润
    if buyer == "BUYER":
        buyer_profit = 0  # 外部买家利润未知，设为0
    if seller == "SELLER":
        seller_profit = 0  # 外部卖家利润未知，设为0

    # 添加所有与LitaAgentY相关的点
    points.append((buyer_profit, seller_profit))
    point_labels.append(f"{buyer} → {seller}")

# 确定帕累托最优解
pareto_mask = [True] * len(points)
for i, (bp, sp) in enumerate(points):
    for j, (bp2, sp2) in enumerate(points):
        if i != j and bp2 >= bp and sp2 >= sp and (bp2 > bp or sp2 > sp):
            pareto_mask[i] = False
            break
pareto_points = [pts for pts, m in zip(points, pareto_mask) if m]
pareto_labels = [label for label, m in zip(point_labels, pareto_mask) if m]

# 检查是否有足够的数据点来画图
if not points:
    pass  # print("警告: 没有找到LitaAgentY的协议数据，无法绘制帕累托图")
    # 创建一个空的图表，显示警告信息
    plt.figure(figsize=(10, 6))
    plt.text(
        0.5,
        0.5,
        "No LitaAgentY contract data available",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
        color="red",
    )
    plt.savefig("agreements_pareto.png")
    plt.close()
else:
    pass  # print(f"绘制LitaAgentY帕累托图: 共有 {len(points)} 个协议点")

    plt.figure(figsize=(12, 10))  # 增大图表尺寸，为标签腾出更多空间

    # 绘制所有协议点
    bp_vals = [p[0] for p in points]
    sp_vals = [p[1] for p in points]
    plt.scatter(
        bp_vals, sp_vals, color="blue", alpha=0.6, label="LitaAgentY Agreements", s=80
    )

    # 为每个数据点添加标签
    for i, (bp, sp) in enumerate(points):
        plt.annotate(
            point_labels[i],
            (bp, sp),
            xytext=(10, 5),  # 标签偏移量
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", alpha=0.4),
        )

    # 如果有帕累托最优点，则绘制
    if pareto_points:
        bp_p = [p[0] for p in pareto_points]
        sp_p = [p[1] for p in pareto_points]
        plt.scatter(
            bp_p,
            sp_p,
            color="red",
            edgecolors="black",
            marker="D",
            s=100,
            label="Pareto optimal Point",
        )

        # 为帕累托最优点添加突出标签
        for i, (bp, sp) in enumerate(pareto_points):
            plt.annotate(
                pareto_labels[i],
                (bp, sp),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=11,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", lw=1.5),
            )

    # 绘制帕累托前沿线 (如果有多个帕累托点)
    if len(pareto_points) > 1:
        # 按买方利润排序，确保连线正确
        sorted_pareto = sorted(pareto_points, key=lambda x: x[0])

        # 使用插值方法绘制更平滑的帕累托前沿
        bp_sorted = [p[0] for p in sorted_pareto]
        sp_sorted = [p[1] for p in sorted_pareto]

        # 绘制帕累托前沿
        plt.plot(bp_sorted, sp_sorted, "r--", lw=2, alpha=0.7, label="Paretro frontier")

    plt.xlabel("Buyer Profit", fontsize=12)
    plt.ylabel("Seller Profit", fontsize=12)
    plt.title("Contracts and Pareto frontier", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加坐标轴
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    plt.tight_layout()  # 自动调整布局，防止标签被裁剪
    plt.show()
    plt.savefig("agreements_pareto.png", dpi=200)  # 增加分辨率
    plt.close()

    # 打印帕累托最优协议的详细信息
    if pareto_points:
        pass  # print("\nLitaAgentY的帕累托最优协议:")
        for i, (bp, sp) in enumerate(pareto_points):
            pass  # print(f"  {pareto_labels[i]}: 买方利润={bp:.2f}, 卖方利润={sp:.2f}")

# Visualization 3: Agent total profit, bankruptcy, and contracts
agents = sorted(
    {
        a
        for a in (
            contracts_df["buyer_name"].tolist() + contracts_df["seller_name"].tolist()
        )
        if a not in ["BUYER", "SELLER"]
    }
)
profit = {a: 0 for a in agents}
contract_count = {a: 0 for a in agents}
for _, ct in contracts_df.iterrows():
    b = ct["buyer_name"]
    s = ct["seller_name"]
    qty = ct["quantity"]
    price = ct["unit_price"]
    if b in profit:
        profit[b] -= price * qty
        contract_count[b] += 1
    if s in profit:
        profit[s] += price * qty
        contract_count[s] += 1
bankrupt = {a: (profit[a] < 0) for a in agents}

# Print score summary to console
pass  # print("Agent, TotalProfit, Contracts, Bankrupt")
for a in agents:
    pass  # print(f"{a}, {profit[a]:.2f}, {contract_count[a]}, {'YES' if bankrupt[a] else 'NO'}")

# Save summary CSV
summary_df = pd.DataFrame(
    {
        "agent": agents,
        "profit": [profit[a] for a in agents],
        "contracts": [contract_count[a] for a in agents],
        "bankrupt": [bankrupt[a] for a in agents],
    }
)
summary_df.to_csv("score_summary.csv", index=False)

# Plot profit and contracts chart
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
colors = ["tomato" if bankrupt[a] else "skyblue" for a in agents]
bars = ax1.bar(agents, [profit[a] for a in agents], color=colors)
ax2.plot(
    agents,
    [contract_count[a] for a in agents],
    color="green",
    marker="o",
    linestyle="--",
)
ax1.set_xlabel("Agent")
ax1.set_ylabel("Total Profit")
ax2.set_ylabel("Contracts Signed")
ax1.set_title("Agents' Profit, Bankruptcy Status, and Contracts")
# Mark bankrupt agents
for bar, agent in zip(bars, agents):
    if bankrupt[agent]:
        bar.set_hatch("//")
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            "Bankrupt",
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
        )
plt.show()
plt.savefig("agent_stats.png")
plt.close()

world.plot_stats()
plt.show()
world.plot_stats("n_negotiations", ylegend=1.25)
plt.show()
world.plot_stats("bankrupt", ylegend=1.25)
plt.show()
world.plot_stats("n_contracts")
plt.show()
world.plot_stats("score")
plt.show()

# Plot negotiations between every pair of agents with Pareto frontier
from collections import defaultdict

pair_negotiations = defaultdict(list)
for neg in world.saved_negotiations:
    buyer = neg.get("buyer")
    seller = neg.get("seller")
    if not buyer or not seller:
        continue
    pair_negotiations[(buyer, seller)].append(neg)


def pareto_frontier(points):
    mask = [True] * len(points)
    for i, (bp, sp) in enumerate(points):
        for j, (bp2, sp2) in enumerate(points):
            if i != j and bp2 >= bp and sp2 >= sp and (bp2 > bp or sp2 > sp):
                mask[i] = False
                break
    return [p for p, m in zip(points, mask) if m]


for (buyer, seller), negotiations in pair_negotiations.items():
    all_points = []
    plt.figure(figsize=(8, 6))
    for idx, neg in enumerate(negotiations):
        product = neg.get("product", 0)
        env_price = (
            float(world.trading_prices[product])
            if hasattr(world, "trading_prices")
            else None
        )
        trace_points = []
        for step in neg.get("history", []):
            for agent_id, offer in step.get("new_offers", []):
                if not offer:
                    continue
                qty, _time, price = offer
                seller_profit = price * qty
                if env_price is None:
                    env_price = 1.5 * price
                buyer_profit = env_price * qty - price * qty
                trace_points.append((buyer_profit, seller_profit))
                all_points.append((buyer_profit, seller_profit))
        if trace_points:
            tp = list(zip(*trace_points))
            plt.plot(tp[0], tp[1], marker="o", label=f"Negotiation {idx + 1}")

    if all_points:
        pareto = pareto_frontier(all_points)
        pareto.sort(key=lambda x: x[0])
        pp = list(zip(*pareto))
        plt.scatter(pp[0], pp[1], color="red", marker="D", label="Pareto frontier")
        plt.plot(pp[0], pp[1], "r--", alpha=0.7)

    plt.xlabel("Buyer Profit")
    plt.ylabel("Seller Profit")
    plt.title(f"{seller} vs {buyer}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"negotiation_{seller}_{buyer}.png")
    plt.show()
