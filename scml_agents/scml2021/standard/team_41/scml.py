import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

fig = plt.figure()

import sys
from pprint import pprint

sys.path.append("/Users/opt/anaconda3/lib/python3.8")
import scml
from scml.oneshot import *
from scml.scml2020 import *

agent_types = [
    DecentralizingAgent,
    MarketAwareDecentralizingAgent,
    SyncRandomOneShotAgent,
]

if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
    )

    _, _ = world.draw()
    plt.show()
    world.run_with_progress()
    plt.plot(world.stats["n_negotiations"])
    plt.xlabel("Simulation Step")
    plt.ylabel("N. Negotiations")
    plt.show()

    pprint(list(_ for _ in world.stats.keys() if "@" not in _))
    [
        "n_contracts_nullified_now",
        "n_bankrupt",
        "trading_price_0",
        "sold_quantity_0",
        "unit_price_0",
        "trading_price_1",
        "sold_quantity_1",
        "unit_price_1",
        "trading_price_2",
        "sold_quantity_2",
        "unit_price_2",
        "trading_price_3",
        "sold_quantity_3",
        "unit_price_3",
        "trading_price_4",
        "sold_quantity_4",
        "unit_price_4",
        "productivity",
        "market_size",
        "production_failures",
        "bankruptcy",
        "n_registered_negotiations_before",
        "n_contracts_executed",
        "n_contracts_erred",
        "n_contracts_nullified",
        "n_contracts_cancelled",
        "n_contracts_dropped",
        "n_breaches",
        "breach_level",
        "n_contracts_signed",
        "n_contracts_concluded",
        "n_negotiations",
        "n_negotiation_rounds_successful",
        "n_negotiation_rounds_failed",
        "n_negotiation_successful",
        "n_negotiation_failed",
        "n_registered_negotiations_after",
        "activity_level",
        "step_time",
        "total_time",
    ]

    plt.bar(range(world.n_steps), world.stats_df["step_time"])
    plt.xlabel("Simulation Step")
    plt.ylabel("Time (s)")
    plt.show()

    winner = world.winners[0]
    pprint(list(_ for _ in world.stats.keys() if winner.name in _))

    [
        "spot_market_quantity_09MAD@3",
        "spot_market_loss_09MAD@3",
        "balance_09MAD@3",
        "inventory_09MAD@3_input",
        "inventory_09MAD@3_output",
        "productivity_09MAD@3",
        "assets_09MAD@3",
        "bankrupt_09MAD@3",
        "score_09MAD@3",
    ]

    # show the first and last value of each of the agent statistics
    pprint({k: (v[0], v[-1]) for k, v in world.stats.items() if winner.name in k})

    {
        "assets_09MAD@3": (0.0, 4233.000000000001),
        "balance_09MAD@3": (47255, 49492),
        "bankrupt_09MAD@3": (False, False),
        "inventory_09MAD@3_input": (0, 0),
        "inventory_09MAD@3_output": (0, 83),
        "productivity_09MAD@3": (0.0, 0.0),
        "score_09MAD@3": (1.0, 1.081855893230937),
        "spot_market_loss_09MAD@3": (0.3, 0.516),
        "spot_market_quantity_09MAD@3": (0, 0),
    }

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
    plt.show()

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

    winner_profits = [100 * world.scores()[_.id] for _ in world.winners]
    winner_types = [_.short_type_name for _ in world.winners]
    print(f"{world.winners} of type {winner_types} won at {winner_profits}%")

    # こんな感じのやつが上から出力される#
    # [09MAD@3] of type ['marketawaredecentralizing'] won at [108.1855893230937]%

    # find the keys in stats for the input and output inventory
    in_key = [
        _ for _ in world.stats.keys() if _.startswith(f"inventory_{winner}_input")
    ][0]
    out_key = [
        _ for _ in world.stats.keys() if _.startswith(f"inventory_{winner}_output")
    ][0]

    # find input and output product indices
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
    plt.show()

    from scml.scml2020.world import is_system_agent

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
    plt.show()

    from scml.scml2020.world import is_system_agent

    fig, (profit, score) = plt.subplots(1, 2)
    snames = sorted(world.non_system_agent_names)
    for name in snames:
        profit.plot((np.asarray(world.stats[f"balance_{name}"])), label=name)
        score.plot(
            np.asarray(world.stats[f"score_{name}"])
            * (world.stats[f"balance_{name}"][0]),
            label=name,
        )
    profit.set(xlabel="Simulation Step", ylabel="Player Balance ($)")
    profit.legend(loc="lower left")
    score.set(xlabel="Simulation Step", ylabel="Player Score Unnormalized ($)")
    plt.show()

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
    plt.show()

    fig, (score, profit) = plt.subplots(1, 2)
    final_scores = [
        world.stats[f"score_{_}"][-1] * (world.stats[f"balance_{_}"][0])
        for _ in world.non_system_agent_names
    ]
    final_profits = [
        world.stats[f"balance_{_}"][-1] for _ in world.non_system_agent_names
    ]
    plt.setp(score.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(profit.xaxis.get_majorticklabels(), rotation=45)
    score.bar(world.non_system_agent_names, final_scores)
    profit.bar(world.non_system_agent_names, final_profits)
    score.set(ylabel="Final Unnormalized Score ($)")
    profit.set(ylabel="Final Balance  ($)")
    plt.show()

    fig, axs = plt.subplots(2, 2)
    for ax, key in zip(
        axs.flatten().tolist(), ["trading_price", "sold_quantity", "unit_price"]
    ):
        for p in range(world.n_products):
            ax.plot(world.stats[f"{key}_{p}"], marker="x", label=f"Product {p}")
            ax.set_ylabel(key.replace("_", " ").title())
            ax.legend().set_visible(False)
    axs[-1, 0].legend(bbox_to_anchor=(1, -0.5), ncol=3)
    plt.show()

    fig, axs = plt.subplots(1, 2)
    for ax, key in zip(
        axs.flatten().tolist(), ["spot_market_quantity", "spot_market_loss"]
    ):
        for a in world.non_system_agent_names:
            ax.plot(world.stats[f"{key}_{a}"], marker="x", label=f"{a}")
            ax.set_ylabel(key.replace("_", " ").title())
            ax.legend().set_visible(False)
    axs[0].legend(bbox_to_anchor=(1, -0.2), ncol=4)
    plt.show()

    # この辺から表が出ない#
    # create a view with only signed contracts
    contracts = world.contracts_df
    signed = contracts.loc[contracts.signed_at >= 0, :]

    fields = [
        "seller_name",
        "buyer_name",
        "delivery_time",
        "quantity",
        "unit_price",
        "signed_at",
        "executed",
        "breached",
        "nullified",
        "erred",
    ]
    signed[fields].sort_values(["quantity", "unit_price"], ascending=False).head(10)

    signed.loc[signed.executed, fields].sort_values(
        ["quantity", "unit_price"], ascending=False
    ).head(10)

    signed.loc[signed.breached, fields[:-4] + ["breaches"]].sort_values(
        ["quantity", "unit_price"], ascending=False
    ).head(10)

    # 表が出ないここまで

    fig, ax = plt.subplots(1, 3)
    for i, x in enumerate(["delivery_time", "quantity", "product_index"]):
        ax[i].scatter(signed.signed_at, signed[x])
        ax[i].set(ylabel=x.replace("_", " ").title(), xlabel="Signing Day")
    plt.show()

    fig, ax = plt.subplots(1, 3)
    for i, x in enumerate(["delivery_time", "unit_price", "product_index"]):
        ax[i].scatter(signed.quantity, signed[x])
        ax[i].set(ylabel=x.replace("_", " ").title(), xlabel="Quantity")
    plt.show()

    bankruptcy = {
        a: np.nonzero(stats[f"bankrupt_{a}"].values)[0]
        for a in world.non_system_agent_names
    }
    pprint({k: "No" if len(v) < 1 else f"at: {v[0]}" for k, v in bankruptcy.items()})

    {
        "00SRO@0": "No",
        "01Dec@0": "No",
        "02Dec@0": "No",
        "03SRO@1": "No",
        "04Dec@1": "No",
        "05SRO@1": "No",
        "06Dec@2": "No",
        "07SRO@2": "at: 22",
        "08MAD@2": "No",
        "09MAD@3": "No",
        "10SRO@3": "No",
        "11MAD@3": "No",
    }

    from negmas import show_edge_colors

    show_edge_colors()

    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()

    world.save_gif("run.gif")

    []

    from IPython.display import HTML

    HTML('<img src="run.gif">')
