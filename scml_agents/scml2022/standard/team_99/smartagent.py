#!/usr/bin/env python
"""
**Submitted to ANAC 2022 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2022 SCML.

This module implements a factory manager for the SCM 2022 league of ANAC 2022
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
    >> \.venv\Scripts\activate.bat

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


# required for typing
from typing import Any, Dict, List, Optional, Iterable, Union, Tuple

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    SAONegotiator,
    AspirationNegotiator,
    make_issue,
    NegotiatorMechanismInterface,
    UtilityFunction,
    LinearUtilityFunction,
)
from negmas.helpers import humanize_time, get_class, instantiate
from scml.scml2020 import Failure

# required for development
# required for running the test tournament
import time
from tabulate import tabulate
from scml.utils import anac2022_collusion, anac2022_std, anac2022_oneshot
from scml.scml2020 import (
    SCML2020Agent,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
    DemandDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
    ReactiveTradingStrategy,
    PredictionBasedTradingStrategy,
    TradingStrategy,
    StepNegotiationManager,
    IndependentNegotiationsManager,
    MovingRangeNegotiationManager,
    TradePredictionStrategy,
    AWI,
)
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)
from scml.scml2020.common import ANY_LINE, is_system_agent, NO_COMMAND
from scml.scml2020.components import SignAllPossible
from scml.scml2020.components.prediction import FixedTradePredictionStrategy
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.prediction import MeanERPStrategy
from abc import abstractmethod
from pprint import pformat

import os, sys

sys.path.append(os.path.dirname(__file__))
from mynegotiations import MyNegotiationsManager, MyIndependentNegotiationsManager
from mytrading import MyTradePredictionStrategy, MyTradingStrategy
from myproduction import MyProductionStrategy

__all__ = ["SmartAgent"]


class SmartAgent(
    MyProductionStrategy,
    MyTradingStrategy,
    MyIndependentNegotiationsManager,
    SCML2020Agent,
):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    def init(self):
        super().init()


def run(
    competition="std",
    reveal_names=True,
    n_steps=50,
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
    competitors = [
        SmartAgent,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2022_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
    elif competition == "collusion":
        results = anac2022_collusion(
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
            SmartAgent,
            RandomOneShotAgent,
            GreedyOneShotAgent,
        ]
        results = anac2022_oneshot(
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
    pass  # print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    pass  # print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    # will run a short tournament against two built-in agents. Default is "std"
    # You can change this from the command line by running something like:
    # >> python3 myagent.py collusion
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "std")
