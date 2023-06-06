import numpy as np
from scml.scml2020.components import MarketAwareTradePredictionStrategy


class MyPredictor(MarketAwareTradePredictionStrategy):
    def trade_prediction_init(self):
        inp = self.awi.my_input_product
        self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
        self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)