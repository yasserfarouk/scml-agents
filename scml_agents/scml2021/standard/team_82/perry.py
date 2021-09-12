import pickle

from scml import SCML2020Agent
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks

from .MachineLearningBasedTradingStrategy import MachineLearningBasedTradingStrategy
from .MarketAwareProductionStrategy import MarketAwareProductionStrategy
from .ModifiedStepNegotiationManager import ModifiedStepNegotiationManager

__all__ = [
    "PerryTheAgent",
]


class PerryTheAgent(
    _NegotiationCallbacks,
    ModifiedStepNegotiationManager,
    MachineLearningBasedTradingStrategy,
    MarketAwareProductionStrategy,
    SCML2020Agent,
):
    def __init__(self, *args, prediction_model=None, **kwargs):
        self.prediction_model = prediction_model
        super().__init__(*args, **kwargs)

    def step(self):
        super().step()
        self.input_cost[self.awi.current_step] = self.awi.trading_prices[
            self.awi.my_input_product
        ]
        self.output_price[self.awi.current_step] = self.awi.trading_prices[
            self.awi.my_output_product
        ]

    def _urange(self, step, is_seller, time_range):
        prices = (
            self.awi.catalog_prices
            if not self._use_trading
            or not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        time_frame = time_range[1] - time_range[0]
        if time_frame < 0:
            return
        if is_seller:
            price = prices[self.awi.my_output_product]
            if time_frame < self.awi.n_steps / 10:
                return (
                    int(price * self._min_margin + 0.2),
                    int(self._max_margin * price + 0.5),
                )
            else:
                return (
                    int(price * self._min_margin),
                    int(self._max_margin * price + 0.5),
                )
        else:
            price = prices[self.awi.my_input_product]
            if time_frame > self.awi.n_steps / 5:
                return (
                    int(price * self._min_margin - 0.2),
                    int(self._max_margin * price + 0.5),
                )
            else:
                return (
                    int(price * self._min_margin),
                    int(self._max_margin * price + 0.5),
                )
