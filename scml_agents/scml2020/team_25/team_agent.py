import logging

from scml.scml2020 import SCML2020Agent, SCML2020World, RandomAgent, DecentralizingAgent
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from .mixed_neg import StepBuyBestSellNegManager
from .prod_strategy import myProductionStratgey
from .trade_strategy import NewPredictionBasedTradingStrategy
from scml.scml2020.components.trading import (
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
)
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
)
import numpy as np

__all__ = ["Agent30"]


class Agent30(
    _NegotiationCallbacks,
    StepBuyBestSellNegManager,
    NewPredictionBasedTradingStrategy,
    # SupplyDrivenProductionStrategy,
    myProductionStratgey,
    SCML2020Agent,
):
    def get_avg_p(self):
        return self.avg_i_p

    def acceptable_unit_price(self, step: int, sell: bool):
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        expected_inventory = sum(self.inputs_secured[0:step]) - sum(
            self.outputs_secured[0:step]
        )
        self.awi.n_lines
        alpha = expected_inventory / self.awi.n_lines
        if sell:
            if alpha < 0:
                beta = 1.2
            elif alpha <= 4:
                beta = 1
            elif alpha <= 8:
                beta = 1 - alpha * (0.1 / 8)
            else:
                beta = 0.9
            return (production_cost + self.input_cost[step]) * beta

        if alpha > 2 and alpha < 10:
            beta = (6 - alpha / 2) / 5
        elif alpha > 10:
            beta = 0.1
        else:
            beta = 1
        return (self.output_price[step] - production_cost) * beta


