#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition. This version will use subcomponents. Please refer to the
[game description](http://www.yasserm.com/scml/scml2021.pdf) for all the
callbacks and subcomponents available.

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu


To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/myagent.py

1. Install a venv (recommended)
>> python3 -m venv .venv

2. Activate the venv (required if you installed a venv)
On Linux/Mac:
    >> source .venv/bin/activate
On Windows:
    >> \\.venv\\Scripts\activate.bat

3. Update pip just in case (recommended)

>> pip install -U pip wheel

4. Install SCML

>> pip install scml

5. [Optional] Install last year's agents for STD/COLLUSION tracks only

>> pip install scml-agents

6. Run the script with no parameters (assuming you are )

>> python /{path-to-this-file}/myagent.py

You should see a short tournament running and results reported.
"""


# required for development
# required for running the test tournament
import time

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from myagent2 import StagHunterV2
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
)
from negmas.helpers import humanize_time
from scml.scml2020 import Failure, SCML2020Agent
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

import scml_agents


def run(
    competition="std",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
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

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """

    # competitors = [scml_agents.scml2020.team_may.mmm.MMM,scml_agents.scml2020.team_15.steady_mgr.SteadyMgr,scml_agents.scml2020.team_20.crescent_agent.CrescentAgent,scml_agents.scml2020.team_25.team_agent.Agent30] + [StagHunterV2]

    competitors = list(scml_agents.get_agents(2020, track="std", winners_only=True)) + [
        StagHunterV2
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2021_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "collusion":
        results = anac2021_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "oneshot":
        # Standard agents can run in the OneShot environment but cannot win
        # the OneShot track!!
        from scml.oneshot.agents import GreedyOneShotAgent, RandomOneShotAgent

        competitors = [
            MyAgent,
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        results = anac2021_oneshot(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    # just make agent types shorter in the results
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # show results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    # will run a short tournament against two built-in agents. Default is "std"
    # You can change this from the command line by running something like:
    # >> python3 myagent.py collusion
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "std")
