import matplotlib.pyplot as plt
import pandas as pd
from agents import BetterAgent, BondAgent, SimpleAgent
from scml.oneshot import *
from scml.scml2020 import *
from scml.utils import anac2021_oneshot


def shorten_names(results):
    # just make agent types more readable
    results.score_stats.agent_type = results.score_stats.agent_type.str.split(".").str[
        -1
    ]
    results.kstest.a = results.kstest.a.str.split(".").str[-1]
    results.kstest.b = results.kstest.b.str.split(".").str[-1]
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    results.scores.agent_type = results.scores.agent_type.str.split(".").str[-1]
    results.winners = [_.split(".")[-1] for _ in results.winners]
    return results


def run_simple_example():
    agent_types = [
        BetterAgent,
        BondAgent,
        DecentralizingAgent,
        MarketAwareDecentralizingAgent,
        SyncRandomOneShotAgent,
    ]
    # may take a long time
    world = SCML2021World(
        **SCML2021World.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
    )
    _, _ = world.draw()
    plt.show()

    world.run_with_progress()  # may take few minutes
    plt.plot(world.stats["n_negotiations"])
    plt.xlabel("Simulation Step")
    plt.ylabel("N. Negotiations")
    plt.show()

    winner = world.winners[0]
    print(winner.name)


def run_tournament():
    pd.options.display.float_format = "{:,.2f}".format
    tournament_types = [
        BetterAgent,
        BondAgent,
        RandomOneShotAgent,
        SyncRandomOneShotAgent,
        GreedyOneShotAgent,
        GreedySingleAgreementAgent,
    ]
    # may take a long time
    results = anac2021_oneshot(
        competitors=tournament_types,
        n_configs=3,  # number of different configurations to generate
        n_runs_per_world=1,  # number of times to repeat every simulation (with agent assignment)
        n_steps=10,  # number of days (simulation steps) per simulation
        print_exceptions=True,
        verbose=True,
    )
    results = shorten_names(results)
    return results


if __name__ == "__main__":
    results = run_tournament()
    print(results.score_stats)
