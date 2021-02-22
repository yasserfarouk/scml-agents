"""
**Submitted to ANAC 2020 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to 
the authors and the ANAC 2020 SCML. 

This module implements a factory manager for the SCM 2020 league of ANAC 2019 
competition. This version will use subcomponents. Please refer to the 
[game description](http://www.yasserm.com/scml/scml2020.pdf) for all the 
callbacks and subcomponents available.

Your agent can learn about the state of the world and itself by accessing 
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps  
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your agent can act in the world by calling methods in the AWI it has. 
For example:

- *self.awi.request_negotiation(...)*  # requests a negotiation with one partner
- *self.awi.request_negotiations(...)* # requests a set of negotiations

 
You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list 
  [here](http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html)

"""

# required for development
from scml.scml2020.agents import DoNothingAgent

# required for running the test tournament
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent
from negmas.helpers import humanize_time

# required for typing
from typing import List, Optional, Dict, Any
import numpy as np
from negmas import (
    Issue,
    AgentMechanismInterface,
    Contract,
    Negotiator,
    MechanismState,
    Breach,
)
from scml.scml2020 import Failure
from scml.scml2020 import SCML2020Agent
from scml.scml2020 import PredictionBasedTradingStrategy
from scml.scml2020 import MovingRangeNegotiationManager
from scml.scml2020 import TradeDrivenProductionStrategy


class MyComponentsBasedAgent(
    TradeDrivenProductionStrategy,
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
):
    """
    This is the only class you *need* to implement. You can create the agent
    by combining the following strategies:
    
    1. A trading strategy that decides the quantities to sell and buy
    2. A negotiation manager that decides which negotiations to engage in and 
       uses some controller(s) to control the behavior of all negotiators
    3. A production strategy that decides what to produce

    """


def run(
    competition="std",
    reveal_names=True,
    n_steps=20,
    n_configs=2,
    max_n_worlds_per_config=None,
    n_runs_per_world=1,
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
    competitors = [
        MyComponentsBasedAgent,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
    ]
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


if __name__ == "__main__":
    run()
