# here we run tournaments to collect statistics which will be collected next to build our agent


# required for running tournament
import time
from pathlib import Path

import pandas as pd
from agent import MyLearnNegotiationAgent, NegotiatorAgent, UnicornAgent
from negmas.helpers import humanize_time
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    IndDecentralizingAgent,
    RandomAgent,
)
from scml.utils import anac2020_collusion, anac2020_std
from tabulate import tabulate


def run(
    competition="std",
    reveal_names=True,
    n_steps=20,
    n_configs=2,
    max_n_worlds_per_config=None,
    n_runs_per_world=2,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std,
                     collusion).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    competitors = [UnicornAgent, DecentralizingAgent]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    return results


if __name__ == "__main__":
    results = run()

    # path = Path(results.params['tournament_path'])
    # names = results.world_stats['name']
    #
    #
    # # get contracts data...
    # all_contracts = []
    # cancelled_contracts = []
    # signed_contracts = []
    # for name in names:
    #     all_contracts.append(pd.read_csv(path / name / 'all_contracts.csv'))
    #     cancelled_contracts.append(pd.read_csv(path / name / 'cancelled_contracts.csv'))
    #     signed_contracts.append((pd.read_csv(path / name / 'signed_contracts.csv')))
