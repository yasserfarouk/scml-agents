#!/usr/bin/env python

from typing import List, Tuple

import numpy as np
from scipy.stats import randint as sp_randint
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold

from .other_agents.agent_team86 import AgentOneOneTwo

# QUANTITY = 0
# TIME = 1
# UNIT_PRICE = 2

__all__ = ["ForestAgent"]

MIN_DATA_POINTS_FOR_MODEL = 5
SELLING_TIME_DECAY_FACTOR = 0.85
BUYING_TIME_DECAY_FACTOR = 1.25
N_ESTIMATORS = 5


class ForestAgent(AgentOneOneTwo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize regressors and indices
        self.selling_regressor = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, warm_start=True
        )
        self.buying_regressor = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, warm_start=True
        )
        self.last_selling_index = 0
        self.last_buying_index = 0

        # Initialize alpha parameter for EWMA smoothing
        self.current_weight = 1

    def _update_regressor(
        self,
        regressor: RandomForestRegressor,
        data: List[float],
        last_index: int,
        is_selling: bool,
    ) -> Tuple[RandomForestRegressor, int]:
        # Prepare input and output data for regression
        x = np.array([i for i in range(last_index, len(data))]).reshape(-1, 1)
        y = np.array(data[last_index:]).flatten()

        # Fit regressor if there are new data points
        if len(data) > last_index:
            regressor.fit(x, y, sample_weight=self.current_weight)
            last_index = len(data)

        self.current_weight += 1
        return regressor, last_index

    def _price_range(self, nmi: int) -> Tuple[float, float]:
        # Get the original price range from the parent class
        mn, mx = super()._price_range(nmi)

        # Update the selling regressor and get the new minimum price if the agent is selling
        if self._is_selling(nmi):
            self.selling_regressor, self.last_selling_index = self._update_regressor(
                regressor=self.selling_regressor,
                data=self._best_selling,
                last_index=self.last_selling_index,
                is_selling=True,
            )

            # Predict the price only if the regressor is fitted
            if hasattr(self.selling_regressor, "estimators_"):
                mn = max(
                    mn,
                    self.selling_regressor.predict(
                        np.array([len(self._best_selling)]).reshape(-1, 1)
                    ).item(),
                )

        # Update the buying regressor and get the new maximum price if the agent is buying
        else:
            self.buying_regressor, self.last_buying_index = self._update_regressor(
                regressor=self.buying_regressor,
                data=self._best_selling,
                last_index=self.last_selling_index,
                is_selling=False,
            )

            # Predict the price only if the regressor is fitted
            if hasattr(self.buying_regressor, "estimators_"):
                mx = min(
                    mx,
                    self.buying_regressor.predict(
                        np.array([len(self._best_buying)]).reshape(-1, 1)
                    ).item(),
                )

        return mn, mx
