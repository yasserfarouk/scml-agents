# Elisha Gerson
# 328612304

import sys
from scml.scml2020.components.production import (
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
    DemandDrivenProductionStrategy,
)
from scml.scml2020.components.production import NO_COMMAND
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from scml.scml2020 import SCML2020Agent, SCML2020World, RandomAgent, DecentralizingAgent
from scml.scml2020 import MovingRangeNegotiationManager
from .nego_strategy import SyncController, MyNegotiationManager

# from new_neg import StepBuyBestSellNegManager, _NegotiationCallbacks
from .mixed_neg import StepBuyBestSellNegManager, _NegotiationCallbacks
import matplotlib.pyplot as plt
import numpy as np


# class for production strategy that will integrate the best of supply and best of demand driven production strategies
# one strategy, Production has cost so over production may not be a good idea. On the other hand, the inventory is
# valued in SCML 2020 at half the trading price which means that it may be a good idea to convert inputs to outputs
#  (even if you do not sell that output) if the difference in trading prices at the end of simulation offsets your production costs.

# the idea of this production startegy will be that until a certain point we act as supplyDrivenstratgey - meaning
# we produce every input we get into output - (because it is safe to assume that we will need those outputs on future steps....)
# However after reaching a certain step we will start acting like demand driven and only produce for signed contracts.
# Another option is to also at a certain step instead of acting like demand driven - to start checking if it is worth producing extra
# knowing we will have extra at the end and will get half the trading price

CHANGE_BEHAVIOR_STEP = 5
CHANGE_BEHAVIOR_DIV_ELEM = 1.2


class myProductionStratgey(ProductionStrategy):
    def is_supply_drived(self):
        product_in_price = self.awi.catalog_prices[self.awi.my_input_product]
        product_out_price = self.awi.catalog_prices[self.awi.my_output_product]
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if product_out_price / 2 > product_in_price / 2 + production_cost:
            return True
        else:
            if self.awi.current_step <= int(
                self.awi.n_steps / CHANGE_BEHAVIOR_DIV_ELEM
            ):
                return True
            else:
                return False

    def on_contracts_finalized(self, signed, cancelled, rejectors):
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            step = contract.agreement["time"]
            earliest_production = self.awi.current_step
            if self.is_supply_drived():
                # Act according to supply driven strategy
                if is_seller:
                    continue
                latest = self.awi.n_steps - 2
                # find the earliest time I can do anything about this contract
                if step > latest + 1 or step < earliest_production:
                    continue
                # if I am a seller, I will schedule production
                input_product = contract.annotation["product"]
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(step, latest),
                    line=-1,
                    partial_ok=True,
                )
                self.schedule_range[contract.id] = (
                    min(steps) if len(steps) > 0 else -1,
                    max(steps) if len(steps) > 0 else -1,
                    is_seller,
                )
            else:
                # Act according to  demand driven  strategy
                # do nothing if this is not a sell contract
                if not is_seller:
                    continue
                if step > self.awi.n_steps - 1 or step < earliest_production:
                    continue
                    # Schedule production before the delivery time
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(earliest_production, step - 1),
                    line=-1,
                    partial_ok=True,
                )
                # set the schedule_range which is provided for other components
                self.schedule_range[contract.id] = (
                    min(steps) if len(steps) > 0 else -1,
                    max(steps) if len(steps) > 0 else -1,
                    is_seller,
                )

    def step(self):
        super().step()
        if self.awi.current_step < int(self.awi.n_steps / CHANGE_BEHAVIOR_DIV_ELEM):
            commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
            inputs = min(
                self.awi.state.inventory[self.awi.my_input_product], len(commands)
            )
            commands[:inputs] = self.awi.my_input_product
            commands[inputs:] = NO_COMMAND
            self.awi.set_commands(commands)


class DoNothingAgent(SCML2020Agent):
    """Agent that does nothing"""


class MyAgent3(
    MyNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    """My agent"""

    pass


class MyAgent4(
    MyNegotiationManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    """My agent"""

    pass


class MyAgent5(
    _NegotiationCallbacks,
    StepBuyBestSellNegManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def trade_prediction_init(self):
        super().trade_prediction_init()
        self.expected_inputs = 0 * self.expected_inputs


class MyAgent6(
    _NegotiationCallbacks,
    StepBuyBestSellNegManager,
    PredictionBasedTradingStrategy,
    myProductionStratgey,
    SCML2020Agent,
):
    def trade_prediction_init(self):
        super().trade_prediction_init()
        self.expected_inputs = 0 * self.expected_inputs


if __name__ == "__main__":
    from collections import defaultdict

    def show_agent_scores(world):
        scores = defaultdict(list)
        for aid, score in world.scores().items():
            scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
        scores = {k: sum(v) / len(v) for k, v in scores.items()}
        plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
        plt.show()

    world = SCML2020World(
        **SCML2020World.generate(
            [DecentralizingAgent, MyAgent5, MyAgent6],
            n_steps=10,
            n_processes=3,
            n_agents_per_process=2,
        ),
        construct_graphs=True
    )
    world.run_with_progress()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()
    show_agent_scores(world)
