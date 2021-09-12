from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scml.scml2020 import *

agent_types = [
    DecentralizingAgent,
    BuyCheapSellExpensiveAgent,
    IndDecentralizingAgent,
    MovingRangeAgent,
]

world = SCML2020World(
    **SCML2020World.generate(agent_types=agent_types, n_steps=10),
    construct_graphs=True,
)

_, _ = world.draw()

world.run_with_progress()

plt.plot(world.stats["n_negotiations"])
plt.xlabel("Simulation Step")
plt.ylabel("N. Negotiations")
plt.show()


plt.bar(range(world.n_steps), world.stats_df["step_time"])
plt.xlabel("Simulation Step")
plt.ylabel("Time (s)")
plt.show()

# printing winner's stats
winner = world.winners[0]
pprint(list(_ for _ in world.stats.keys() if winner.name in _))
pprint({k: (v[0], v[-1]) for k, v in world.stats.items() if winner.name in k})
stats = pd.DataFrame(data=world.stats)
fig, axs = plt.subplots(2, 3)
for ax, key in zip(
    axs.flatten().tolist(),
    [
        "score",
        "balance",
        "assets",
        "productivity",
        "spot_market_quantity",
        "spot_market_loss",
    ],
):
    ax.plot(stats[f"{key}_{winner}"])
    ax.set(ylabel=key)
fig.show()

# how much money was transferred
plt.plot(world.stats["activity_level"])
plt.xlabel("Simulation Step")
plt.ylabel("Activitiy Level ($)\nTotal Money Transferred")
plt.show()

plt.plot(world.stats["n_contracts_concluded"], label="Concluded Contracts")
plt.plot(world.stats["n_contracts_cancelled"], label="Cancelled Contracts")
plt.plot(world.stats["n_contracts_signed"], label="Signed Contracts")
plt.plot(world.stats["n_contracts_executed"], label="Executed Contracts")
plt.legend()
plt.xlabel("Simulation Step")
plt.ylabel("N. Contracts")
plt.show()


plt.plot(world.stats["breach_level"])
plt.xlabel("Simulation Step")
plt.ylabel("Total Breach Level")
plt.show()

#  there can be multiple winners
winner_profits = [100 * world.scores()[_.id] for _ in world.winners]
winner_types = [_.short_type_name for _ in world.winners]
print(f"{world.winners} of type {winner_types} won at {winner_profits}%")

# find the keys in stats for the input and output inventory
in_key = [_ for _ in world.stats.keys() if _.startswith(f"inventory_{winner}_input")][0]
out_key = [_ for _ in world.stats.keys() if _.startswith(f"inventory_{winner}_output")][
    0
]

# find input and output product indices
# see what happened to winner's inventory
input_product, output_product = (
    winner.awi.my_input_product,
    winner.awi.my_output_product,
)
# draw
fig, (quantity, value) = plt.subplots(1, 2)
quantity.plot(world.stats[in_key], label=f"Input Product")
quantity.plot(world.stats[out_key], label=f"Output Product")
quantity.set(xlabel="Simulation Step", ylabel="Winner's Total Storage (item)")
quantity.legend()
value.plot(
    np.array(world.stats[in_key])
    * np.array(world.stats[f"trading_price_{input_product}"]),
    label=f"Input Product",
)
value.plot(
    np.array(world.stats[out_key])
    * np.array(world.stats[f"trading_price_{output_product}"]),
    label=f"Output Product",
)
value.set(xlabel="Simulation Step", ylabel="Winner's Inventory Value ($)")
value.legend()
fig.show()


# see what happened to all competitors

fig, (profit, score) = plt.subplots(1, 2)
snames = sorted(world.non_system_agent_names)
for name in snames:
    profit.plot(
        100.0
        * (
            np.asarray(world.stats[f"balance_{name}"])
            / world.stats[f"balance_{name}"][0]
            - 1.0
        ),
        label=name,
    )
    score.plot(100 * np.asarray(world.stats[f"score_{name}"]), label=name)
profit.set(xlabel="Simulation Step", ylabel="Player Profit Ignoring Inventory (%)")
profit.legend(loc="lower left")
score.set(xlabel="Simulation Step", ylabel="Player Score (%)")
fig.show()


fig, (profit, score) = plt.subplots(1, 2)
snames = sorted(world.non_system_agent_names)
for name in snames:
    profit.plot((np.asarray(world.stats[f"balance_{name}"])), label=name)
    score.plot(
        np.asarray(world.stats[f"score_{name}"]) * (world.stats[f"balance_{name}"][0]),
        label=name,
    )
profit.set(xlabel="Simulation Step", ylabel="Player Balance ($)")
profit.legend(loc="lower left")
score.set(xlabel="Simulation Step", ylabel="Player Score Unnormalized ($)")
fig.show()


# end of the game score
fig, (score, profit) = plt.subplots(1, 2)
final_scores = [
    100 * world.stats[f"score_{_}"][-1] for _ in world.non_system_agent_names
]
final_profits = [
    100 * world.stats[f"balance_{_}"][-1] / world.stats[f"balance_{_}"][0] - 100
    for _ in world.non_system_agent_names
]
plt.setp(score.xaxis.get_majorticklabels(), rotation=45)
plt.setp(profit.xaxis.get_majorticklabels(), rotation=45)
score.bar(world.non_system_agent_names, final_scores)
profit.bar(world.non_system_agent_names, final_profits)
score.set(ylabel="Final Score (%)")
profit.set(ylabel="Final Profit (%)")

fig.show()


fig, (score, profit) = plt.subplots(1, 2)
final_scores = [
    world.stats[f"score_{_}"][-1] * (world.stats[f"balance_{_}"][0])
    for _ in world.non_system_agent_names
]
final_profits = [world.stats[f"balance_{_}"][-1] for _ in world.non_system_agent_names]
plt.setp(score.xaxis.get_majorticklabels(), rotation=45)
plt.setp(profit.xaxis.get_majorticklabels(), rotation=45)
score.bar(world.non_system_agent_names, final_scores)
profit.bar(world.non_system_agent_names, final_profits)
score.set(ylabel="Final Unnormalized Score ($)")
profit.set(ylabel="Final Balance  ($)")

fig.show()


fig, axs = plt.subplots(2, 2)
for ax, key in zip(
    axs.flatten().tolist(), ["trading_price", "sold_quantity", "unit_price"]
):
    for p in range(world.n_products):
        ax.plot(world.stats[f"{key}_{p}"], marker="x", label=f"Product {p}")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.legend().set_visible(False)
axs[-1, 0].legend(bbox_to_anchor=(1, -0.5), ncol=3)
fig.show()
