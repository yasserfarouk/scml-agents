import time

from negmas.helpers import humanize_time
from rich import print
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from helpers.BetterSyncAgent import BetterSyncAgent
from scml.std.agents import SyncRandomStdAgent
from scml.utils import anac2024_oneshot, anac2024_std
from tabulate import tabulate


def run(
    competitors=tuple(),
    competition="oneshot",
    reveal_names=True,
    n_steps=10,  # 変更済もともとは10
    n_configs=2,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """

    if competition == "oneshot":
        competitors = list(competitors) + [RandomOneShotAgent, SyncRandomOneShotAgent, BetterSyncAgent]
    else:
        competitors = list(competitors) + [SyncRandomStdAgent, RandomOneShotAgent]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2024_std
    else:
        runner = anac2024_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[
        -1
    ]
    # display results
    pass # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
