# %% Standard imports
from matplotlib import pyplot as plt


from scml.std import *
from scml.runner import WorldRunner

from .litaagent_std.litaagent_cir import LitaAgentCIR
from .litaagent_std.litaagent_p import LitaAgentP
from .litaagent_std.litaagent_y import LitaAgentY

# create a runner that encapsulates a number of configs to evaluate agents
# in the same conditions every time
CONFIGS, REPS, STEPS = 10, 5, 25
context = ANACStdContext(  # what are the rounds here, number of trials, process (needs processes or not), etc. and align with the actual parameters of the live competitoin.
    n_steps=STEPS, n_processes=3, world_params=dict(construct_graphs=True)
)
pass  # print("----------------------------------Now Start with CIR-------------------------------")
single_agent_runner0 = WorldRunner(
    context, n_configs=CONFIGS, n_repetitions=REPS, save_worlds=True
)
full_market_runner0 = WorldRunner.from_runner(
    single_agent_runner0, control_all_agents=True
)

# %% create a world with a single agent and run it
single_agent_runner0(LitaAgentCIR)
single_agent_runner0.draw_worlds_of(LitaAgentCIR)

# %% plot the results
single_agent_runner0.plot_stats(agg=False)
single_agent_runner0.score_summary()
plt.show()
pass  # print("----------------------------------Now Start with Y-------------------------------")
single_agent_runner1 = WorldRunner(
    context, n_configs=CONFIGS, n_repetitions=REPS, save_worlds=True
)
full_market_runner1 = WorldRunner.from_runner(
    single_agent_runner1, control_all_agents=True
)

# %% create a world with a single agent and run it
single_agent_runner1(LitaAgentY)
single_agent_runner1.draw_worlds_of(LitaAgentY)

# %% plot the results
single_agent_runner1.plot_stats(agg=False)
single_agent_runner1.score_summary()
plt.show()
pass  # print("----------------------------------Now Start with P-------------------------------")
single_agent_runner2 = WorldRunner(
    context, n_configs=CONFIGS, n_repetitions=REPS, save_worlds=True
)
full_market_runner2 = WorldRunner.from_runner(
    single_agent_runner2, control_all_agents=True
)

# %% create a world with a single agent and run it
single_agent_runner2(LitaAgentP)
single_agent_runner2.draw_worlds_of(LitaAgentP)

# %% plot the results
single_agent_runner2.plot_stats(agg=False)
single_agent_runner2.score_summary()
plt.show()

"""
#%% create a world with a number of agents and run it
full_market_runner(LitaAgentN)
full_market_runner.draw_worlds_of(LitaAgentN)

#%% plot the results
full_market_runner.plot_stats(agg=False)
plt.show()
pass # print("Plotting stats")
"""
