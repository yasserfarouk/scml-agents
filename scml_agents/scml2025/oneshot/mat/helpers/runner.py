import time

from negmas.helpers import humanize_time
from rich import print
from scml.utils import (
    anac2024_oneshot,
    anac2024_std,
    DefaultAgentsStd2024,
    DefaultAgentsOneShot2024,
)
from tabulate import tabulate


def run(
    competitors=tuple(),
    competition="oneshot",
    reveal_types=True,
    n_steps=20,
    n_configs=2,
    debug=True,
    serial=False,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles, production graphs etc
        reveal_types: If given, agent names will reveal their type (kind of) and position
        debug: If given, a debug run is used.
        serial: If given, a serial run will be used.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value
        - To use breakpoints in your code under pdb, pass both debug=True and serial=True

    """

    if competition == "oneshot":
        competitors = list(competitors) + list(DefaultAgentsOneShot2024)
    else:
        competitors = list(competitors) + list(DefaultAgentsStd2024)

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
        debug=debug,
        parallelism="serial" if serial else "parallel",
        agent_name_reveals_position=reveal_types,
        agent_name_reveals_type=reveal_types,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[-1]
    # display results
    pass # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import typer

    typer.run(run)
