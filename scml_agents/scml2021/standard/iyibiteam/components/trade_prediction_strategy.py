from typing import Iterable, List, Optional, Union

import numpy as np
from negmas import Contract
from numpy.core.fromnumeric import take
from scml.scml2020 import MarketAwareTradePredictionStrategy
from scml.scml2020.common import ANY_LINE, is_system_agent
from sklearn.linear_model import LinearRegression, LogisticRegression


class PrintLogger:
    def __init__(self):
        ...
        # self.log_file = open("log_out.txt", "w")

    def log(self, data):
        ...
        # self.log_file.write(data + "\n")


logger = PrintLogger()


class MyTradePredictionStrategy(MarketAwareTradePredictionStrategy):
    """
    Placeholder
    """

    def upgradeSVC(self, SklearnPredictor):
        class UpgradedPredictor:
            def __init__(self, *args, **kwargs):
                self._single_class_label = None
                self.__predictor = SklearnPredictor(*args, **kwargs)

            @staticmethod
            def _has_only_one_class(y):
                return len(np.unique(y)) == 1

            def _fitted_on_single_class(self):
                return self._single_class_label is not None

            def fit(self, X, y=None):
                if self._has_only_one_class(y):
                    self._single_class_label = y[0]
                else:
                    self.__predictor.fit(X, y)
                return self

            def predict(self, X):
                if self._fitted_on_single_class():
                    return np.full(X.shape[0], self._single_class_label)
                else:
                    return self.__predictor.predict(X)

        return UpgradedPredictor

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, self.awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

    def trade_prediction_step(self):
        from sklearn.svm import LinearSVC

        exogenous = self.awi.exogenous_contract_summary
        horizon = self.awi.settings.get("horizon", 1)
        a, b = self.awi.current_step, self.awi.current_step + horizon

        if a > 2:
            inputs = exogenous[self.awi.my_input_product, :a, 0]
            outputs = exogenous[self.awi.my_output_product, :a, 0]

            X = np.array(range(a))
            LinearSVC = self.upgradeSVC(LinearSVC)

            inputs_model = LinearSVC().fit(X.reshape(-1, 1), inputs.ravel())
            predicted_input = inputs_model.predict(np.array(b).reshape(-1, 1))

            outputs_model = LinearSVC().fit(X.reshape(-1, 1), outputs.ravel())
            predicted_output = outputs_model.predict(np.array(b).reshape(-1, 1))

            self.expected_inputs[a:b] = predicted_input
            self.expected_outputs[a:b] = predicted_output
        else:
            self.expected_inputs[a:b] = exogenous[self.awi.my_input_product, a:b, 0]
            self.expected_outputs[a:b] = exogenous[self.awi.my_output_product, a:b, 0]

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
            }
        )
        return state
