"""
**Submitted to ANAC 2020 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to 
the authors and the ANAC 2020 SCML. 

This module implements a factory manager for the SCM 2020 league of ANAC 2019 
competition. This version will not use subcomponents. Please refer to the 
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
  [here](https://scml.readthedocs.io/en/latest/api/scml.scml2020.AWI.html)

"""

# required for running the test tournament
import time

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
)
from negmas.helpers import humanize_time
from scml.scml2020 import SCML2020Agent

# required for development
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)
from scml.scml2020.utils import anac2020_collusion, anac2020_std
from scml.scml2020 import Failure
from tabulate import tabulate


class Ashgent(SCML2020Agent):
    """
    This is the only class you *need* to implement. The current skeleton has a 
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by 
    calling methods in the agent-world-interface instantiated as `self.awi` 
    in your agent. See the documentation for more details

    """

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""
        pass

    def step(self):
        """Called at every production step by the world"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """Called whenever an agent requests a negotiation with you. 
        Return either a negotiator to accept or None (default) to reject it"""

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without 
        agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with 
        agreement"""

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in 
        one step (day)"""
        return [self.id] * len(contracts)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in 
        a step (day)"""

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution 
        (i.e. resoluion is None)"""

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the 
        agent to change what is to be produced in its factory on that step.        
        """
        return commands

    def on_failures(self, failures: List[Failure]) -> None:
        """Called when production fails. If you are careful in 
        what you order in `confirm_production`, you should never see that."""

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""


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
    competitors = [Ashgent, DecentralizingAgent, BuyCheapSellExpensiveAgent]
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
