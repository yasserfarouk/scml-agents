import matplotlib.pyplot as plt
import numpy as np
from scml.scml2020 import *
from scml.utils import anac2020_std

# to run collusion tournament from scml.utils import anac2020_collusion


agent_types = [
    DecentralizingAgent,
    BuyCheapSellExpensiveAgent,
    IndDecentralizingAgent,
    MovingRangeAgent,
]
tournament_types = agent_types + [RandomAgent]

results = anac2020_std(
    competitors=tournament_types,
    n_configs=12,  # number of different configurations to generate
    n_runs_per_world=1,  # number of times to repeat every simulation (with agent assignment)
    n_steps=10,  # number of days (simulation steps) per simulation
    print_exceptions=True,
)
# the winner
print(results.winners)

# num of simulations
len(results.scores.run_id.unique())

print(results.score_stats)
print(results.total_scores)

# activity levels per simulation
plt.errorbar(
    range(len(results.agg_stats)),
    results.agg_stats.activity_level_mean,
    np.sqrt(results.agg_stats.activity_level_var),
)
plt.xlabel("Simulation Number")
plt.ylabel("Activity Level")
plt.show()

# We can even get the scores of every agent belonging to every agent type at every simulation
results.scores.loc[:, ["agent_name", "agent_type", "score"]].head()
