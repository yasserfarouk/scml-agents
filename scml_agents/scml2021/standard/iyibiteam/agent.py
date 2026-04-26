import numpy as np
from scml.scml2020 import SCML2020Agent
from scml.scml2020.components.signing import KeepOnlyGoodPrices

from .components.negotiation_manager import MyNegotiationManager
from .components.production_strategy import MyProductionStrategy
from .components.trade_prediction_strategy import MyTradePredictionStrategy
from .components.trading_strategy import MyTradingStrategy

__all__ = [
    "IYIBIAgent",
]


class _NegotiationCallbacks:
    def acceptable_unit_price(self, step: int, sell: bool):
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def target_quantity(self, step: int, sell: bool):
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps, sell):
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class IYIBIAgent(
    MyTradePredictionStrategy,
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    MyProductionStrategy,
    MyNegotiationManager,
    MyTradingStrategy,
    SCML2020Agent,
):
    pass
