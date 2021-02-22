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
from typing import Tuple

from negmas import LinearUtilityFunction
from negmas import Contract
from scml.scml2020.common import NO_COMMAND

from scml.scml2020.components import FixedERPStrategy
from scml.scml2020.components import (
    SupplyDrivenProductionStrategy,
    ProductionStrategy,
    StepNegotiationManager,
    IndependentNegotiationsManager,
    DemandDrivenProductionStrategy,
)
from scml.scml2020.components.trading import PredictionBasedTradingStrategy

from scml.scml2020 import SCML2020Agent

__all__ = ["SavingAgent"]


class _NegotiationCallbacks:
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class My_ProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        # inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands_threshold = (self.awi.profile.n_lines / 4) * 3
        commands_sort = np.argsort(self.awi.profile.costs[:, self.awi.my_input_product])
        now_productionschedule = self.awi.state.commands[self.awi.current_step, :]
        schedule_amount = np.count_nonzero(now_productionschedule == NO_COMMAND)
        production_amount = max(
            schedule_amount,
            min(
                self.awi.state.inventory[self.awi.my_input_product], commands_threshold
            ),
        )
        for i in range(self.awi.profile.n_lines):
            if commands_sort[i] < production_amount:
                commands[i] = self.awi.my_input_product
            else:
                commands[i] = NO_COMMAND
        self.awi.set_commands(commands)

    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(earliest_production, step - 1),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )


class SavingAgent(
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    My_ProductionStrategy,
    SCML2020Agent,
):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    pass


def run(
    competition="std",
    reveal_names=True,
    n_steps=50,
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
    competitors = [Saving_Agent, DecentralizingAgent, BuyCheapSellExpensiveAgent]
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
