"""Agent_tester_1.py
This script runs a SCML standard world with user-selected agents and
analyzes the negotiation results using each agent's utility function.
It also visualizes negotiation traces and the Pareto frontier.

该脚本在SCML标准世界中运行用户指定的代理，并利用每个代理的效用函数分析谈判结果。
同时绘制谈判轨迹和帕累托前沿。
"""

import argparse
import importlib
import matplotlib.pyplot as plt
import pandas as pd

# Import SCML world. Fall back to 2024 if 2025 is not available
try:
    from scml.std import SCML2025StdWorld
except ImportError:  # pragma: no cover - fallback for older library
    from scml.std import SCML2024StdWorld as SCML2025StdWorld

# Import commonly used agents
from scml.oneshot.agents import GreedyOneShotAgent
from scml.std.agents import RandomStdAgent, SyncRandomStdAgent, GreedyStdAgent
from .litaagent_std.team_miyajima_oneshot.cautious import CautiousOneShotAgent

# Import local agents defined in this repository
from .litaagent_std.litaagent_y import LitaAgentY
from .litaagent_std.litaagent_n import LitaAgentN

# Map simple names to classes so they can be provided on the command line
AVAILABLE_AGENTS = {
    "LitaAgentY": LitaAgentY,
    "LitaAgentN": LitaAgentN,
    "GreedyOneShotAgent": GreedyOneShotAgent,
    "RandomStdAgent": RandomStdAgent,
    "SyncRandomStdAgent": SyncRandomStdAgent,
    "GreedyStdAgent": GreedyStdAgent,
    "CautiousOneShotAgent": CautiousOneShotAgent,
}

# ---------------------------------------------------------------------------
# Argument parsing / 解析命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Run SCML simulation with chosen agents.\n使用指定的代理运行SCML模拟。"
    )
)
parser.add_argument(
    "--agents",
    nargs="+",
    default=["LitaAgentY", "SyncRandomStdAgent"],
    help=(
        "Agent classes to include (name or module.Class).\n"
        "参与模拟的代理名称或 module.Class 路径。"
    ),
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="Number of days to simulate / 模拟的天数",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve agent class objects / 将代理名称解析为实际的类
# ---------------------------------------------------------------------------
agent_classes = []
for name in args.agents:
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    else:
        cls = AVAILABLE_AGENTS.get(name)
        if cls is None:
            available = ", ".join(AVAILABLE_AGENTS)
            raise ValueError(
                f"Unknown agent {name}. Available: {available}\n"
                f"未知代理 {name}，可选: {available}"
            )
    agent_classes.append(cls)

if not agent_classes:
    raise ValueError("No agent classes resolved / 未解析到任何代理类")

# ---------------------------------------------------------------------------
# World creation / 创建世界
# ---------------------------------------------------------------------------
world_params = SCML2025StdWorld.generate(
    agent_types=agent_classes,
    n_steps=args.steps,
)
# construct_graphs=True ensures ufun is available for every agent
world = SCML2025StdWorld(**world_params, construct_graphs=True)

# ---------------------------------------------------------------------------
# Run the simulation / 运行模拟
# ---------------------------------------------------------------------------
world.run_with_progress()

# ---------------------------------------------------------------------------
# Collect contract data / 收集合约数据
# ---------------------------------------------------------------------------
contracts_df = pd.DataFrame.from_records(world.saved_contracts)
contracts_df.to_csv("contracts.csv", index=False)

# ---------------------------------------------------------------------------
# Evaluate contract utilities using each agent's ufun
# 使用每个代理的效用函数计算合约效用
# ---------------------------------------------------------------------------
points = []
for contract in world.saved_contracts:
    buyer = contract.get("buyer_name")
    seller = contract.get("seller_name")
    qty = contract.get("quantity")
    price = contract.get("unit_price")
    dtime = contract.get("delivery_time")

    offer = (qty, dtime, price)

    buyer_agent = world.agents.get(buyer)
    seller_agent = world.agents.get(seller)

    # Compute utility from the perspective of each party
    # 根据买卖双方各自的效用函数计算效用
    if not buyer_agent:
        print(f"[DEBUG] Buyer agent '{buyer}' not found when evaluating offer {offer}")
        buyer_u = 0
    elif not hasattr(buyer_agent, "ufun") or buyer_agent.ufun is None:
        print(
            f"[DEBUG] Buyer agent '{buyer}' has no ufun when evaluating offer {offer}"
        )
        buyer_u = 0
    else:
        buyer_u = buyer_agent.ufun.from_offers((offer,), (False,))

    if not seller_agent:
        print(
            f"[DEBUG] Seller agent '{seller}' not found when evaluating offer {offer}"
        )
        seller_u = 0
    elif not hasattr(seller_agent, "ufun") or seller_agent.ufun is None:
        print(
            f"[DEBUG] Seller agent '{seller}' has no ufun when evaluating offer {offer}"
        )
        seller_u = 0
    else:
        seller_u = seller_agent.ufun.from_offers((offer,), (True,))

    points.append((buyer_u, seller_u))


# ---------------------------------------------------------------------------
# Compute Pareto frontier / 计算帕累托前沿
# ---------------------------------------------------------------------------
def pareto_frontier(data):
    """Return the Pareto-optimal subset of `data`.

    返回`data`中的帕累托最优点列表。
    """
    result = []
    for i, (bu_i, su_i) in enumerate(data):
        dominated = False
        for j, (bu_j, su_j) in enumerate(data):
            if (
                j != i
                and bu_j >= bu_i
                and su_j >= su_i
                and (bu_j > bu_i or su_j > su_i)
            ):
                dominated = True
                break
        if not dominated:
            result.append((bu_i, su_i))
    return result


pareto = pareto_frontier(points)
pareto.sort(key=lambda x: x[0])

# ---------------------------------------------------------------------------
# Plot agreements and Pareto frontier / 绘制协议点及帕累托前沿
# ---------------------------------------------------------------------------
if points:
    bp, sp = zip(*points)
    plt.scatter(bp, sp, color="blue", label="Agreements 协议")

    if pareto:
        pb, ps = zip(*pareto)
        plt.plot(pb, ps, "r--", label="Pareto front 帕累托前沿")
        plt.scatter(pb, ps, color="red")

    plt.xlabel("Buyer utility / 买方效用")
    plt.ylabel("Seller utility / 卖方效用")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("agreements_pareto.png")
    plt.show()
else:
    pass  # print("No contracts found / 未找到合约")

# ---------------------------------------------------------------------------
# Plot negotiation traces / 绘制谈判轨迹
# ---------------------------------------------------------------------------
for neg in world.saved_negotiations:
    buyer = neg.get("buyer")
    seller = neg.get("seller")
    trace = []
    for step in neg.get("history", []):
        for agent_id, offer in step.get("new_offers", []):
            if not offer:
                continue
            qty, dtime, price = offer
            offer_t = (qty, dtime, price)
            if buyer not in world.agents:
                print(
                    f"[DEBUG] Buyer '{buyer}' missing when evaluating step offer {offer_t}"
                )
                bu = 0
            elif (
                not hasattr(world.agents[buyer], "ufun")
                or world.agents[buyer].ufun is None
            ):
                print(f"[DEBUG] Buyer '{buyer}' has no ufun for offer {offer_t}")
                bu = 0
            else:
                bu = world.agents[buyer].ufun.from_offers((offer_t,), (False,))

            if seller not in world.agents:
                print(
                    f"[DEBUG] Seller '{seller}' missing when evaluating step offer {offer_t}"
                )
                su = 0
            elif (
                not hasattr(world.agents[seller], "ufun")
                or world.agents[seller].ufun is None
            ):
                print(f"[DEBUG] Seller '{seller}' has no ufun for offer {offer_t}")
                su = 0
            else:
                su = world.agents[seller].ufun.from_offers((offer_t,), (True,))
            trace.append((bu, su))

    if not trace:
        continue

    tb, ts = zip(*trace)
    plt.figure()
    plt.plot(tb, ts, marker="o")
    plt.xlabel("Buyer utility / 买方效用")
    plt.ylabel("Seller utility / 卖方效用")
    plt.title(f"{seller} vs {buyer}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"neg_trace_{seller}_{buyer}.png")
    plt.show()
