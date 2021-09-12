from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scml.scml2020 import *

###
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

# _, _ = world.draw()
# plt.show()
world.run_with_progress()

# plt.plot(world.stats['n_negotiations'])
# plt.xlabel('Simulation Step')
# plt.ylabel('N. Negotiations')
# plt.show()

###
# pprint(list(_ for _ in world.stats.keys() if "@" not in _  ))

# plt.bar(range(world.n_steps), world.stats_df['step_time'])
# plt.xlabel('Simulation Step')
# plt.ylabel('Time (s)')
# plt.show()


###
winner = world.winners[0]
# pprint(list(_ for _ in world.stats.keys() if winner.name in _ ))

###
# #show the first and last value of each of the agent statistics
# pprint({k:(v[0], v[-1]) for k, v in world.stats.items() if winner.name in k })

stats = pd.DataFrame(data=world.stats)
# fig, axs = plt.subplots(2, 3)
# for ax, key in zip(axs.flatten().tolist(), ["score", "balance", "assets", "productivity",
#                          "spot_market_quantity", "spot_market_loss"]):
#     ax.plot(stats[f"{key}_{winner}"])
#     ax.set(ylabel=key)
# fig.show()
# plt.show()

###
# plt.plot(world.stats['activity_level'])
# plt.xlabel('Simulation Step')
# plt.ylabel('Activitiy Level ($)\nTotal Money Transferred')
# plt.show()

###
# plt.plot(world.stats['n_contracts_concluded'], label='Concluded Contracts')
# plt.plot(world.stats['n_contracts_cancelled'], label='Cancelled Contracts')
# plt.plot(world.stats['n_contracts_signed'], label='Signed Contracts')
# plt.plot(world.stats['n_contracts_executed'], label='Executed Contracts')
# plt.legend()
# plt.xlabel('Simulation Step')
# plt.ylabel('N. Contracts')
# plt.show()

###
# plt.plot(world.stats['breach_level'])
# plt.xlabel('Simulation Step')
# plt.ylabel('Total Breach Level')
# plt.show()

###
# winner_profits = [100 * world.scores()[_.id] for _ in world.winners]
# winner_types = [_.short_type_name for _ in world.winners]
# print(f"{world.winners} of type {winner_types} won at {winner_profits}%")

###
# # find the keys in stats for the input and output inventory
# in_key = [_ for _ in world.stats.keys() if _.startswith(f'inventory_{winner}_input')][0]
# out_key = [_ for _ in world.stats.keys() if _.startswith(f'inventory_{winner}_output')][0]

# # find input and output product indices
# input_product, output_product = winner.awi.my_input_product, winner.awi.my_output_product
# # draw
# fig, (quantity, value) = plt.subplots(1, 2)
# quantity.plot(world.stats[in_key], label=f"Input Product")
# quantity.plot(world.stats[out_key], label=f"Output Product")
# quantity.set(xlabel='Simulation Step', ylabel='Winner\'s Total Storage (item)')
# quantity.legend()
# value.plot(np.array(world.stats[in_key]) * np.array(world.stats[f"trading_price_{input_product}"])
#               , label=f"Input Product")
# value.plot(np.array(world.stats[out_key]) * np.array(world.stats[f"trading_price_{output_product}"])
#               , label=f"Output Product")
# value.set(xlabel='Simulation Step', ylabel='Winner\'s Inventory Value ($)')
# value.legend()
# # fig.show()
# plt.show()

###
# # from scml.scml2020.world import is_system_agent
# fig, (profit, score) = plt.subplots(1, 2)
# snames = sorted(world.non_system_agent_names)
# for name in snames:
#     profit.plot(100.0 * (np.asarray(world.stats[f'balance_{name}'])/world.stats[f'balance_{name}'][0] - 1.0), label=name)  # 最初と比較してどれくらい増えたか
#     score.plot(100 * np.asarray(world.stats[f'score_{name}']), label=name)
# profit.set(xlabel='Simulation Step', ylabel='Player Profit Ignoring Inventory (%)')
# profit.legend(loc='lower left')
# score.set(xlabel='Simulation Step', ylabel='Player Score (%)')
# # fig.show()
# plt.show()

###
# # from scml.scml2020.world import is_system_agent
# fig, (profit, score) = plt.subplots(1, 2)
# snames = sorted(world.non_system_agent_names)
# for name in snames:
#     profit.plot((np.asarray(world.stats[f'balance_{name}'])), label=name)
#     score.plot(np.asarray(world.stats[f'score_{name}'])*(world.stats[f'balance_{name}'][0]), label=name)
# profit.set(xlabel='Simulation Step', ylabel='Player Balance ($)')
# profit.legend(loc='lower left')
# score.set(xlabel='Simulation Step', ylabel='Player Score Unnormalized ($)')
# # fig.show()
# plt.show()

###
# contracts = world.contracts_df
# signed = contracts.loc[contracts.signed_at>=0, :]

# fields = ["seller_name", "buyer_name", "delivery_time", "quantity", "unit_price",
#           "signed_at", "executed", "breached", "nullified", "erred"]
# print(signed[fields].sort_values(["quantity", "unit_price"], ascending=False).head(10))

###
# print(world.non_system_agent_names)
# bankruptcy = {a: np.nonzero(stats[f"bankrupt_{a}"].values)[0]
#         for a in world.non_system_agent_names}
# print(bankruptcy)
# pprint({k: "No" if len(v)<1 else f"at: {v[0]}" for k, v in bankruptcy.items()})

###
# from negmas import show_edge_colors
# # show_edge_colors()

# world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
# # plt.show()

# world.save_gif("run.gif")

# from IPython.display import HTML
# HTML('<img src="run.gif">')


###
from scml.scml2020.components import TradePredictionStrategy


class MyPredictor(TradePredictionStrategy):
    def trade_prediction_init(self):
        self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
        self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)


class MyDecentralizingAgent(MyPredictor, DecentralizingAgent):
    pass


def L(cls):  # クラスの継承関係どうなってんやろ
    return [k.__name__ for k in cls.__mro__]


print(L(MyDecentralizingAgent))
print(L(DecentralizingAgent))
