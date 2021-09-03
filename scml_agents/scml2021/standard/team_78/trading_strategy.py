# required for development
from scml.scml2020.agents import DoNothingAgent

# required for running the test tournament
import time
import sklearn
import pickle as pkl
from tabulate import tabulate
import pathlib

from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent
from negmas.helpers import humanize_time

# required for typing
from typing import List, Optional, Dict, Any
import numpy as np
from negmas import (
    Issue, AgentMechanismInterface, Contract, Negotiator,
    MechanismState, Breach,
)
from scml.scml2020.world import Failure
from scml.scml2020 import SCML2020Agent
from scml.scml2020 import PredictionBasedTradingStrategy
from scml.scml2020 import MovingRangeNegotiationManager
from scml.scml2020 import TradeDrivenProductionStrategy
from scml.scml2020 import TradePredictionStrategy

models_dir = pathlib.Path(__file__).parent / 'models'

class SklearnTradePredictionStrategy(TradePredictionStrategy): 
    def trade_prediction_init(self):
        inp = self.awi.my_input_product
        with open(models_dir / f'sold_quantity_{min(inp, 5)}_predictor_init.pkl', 'rb') as file:
            self.input_quantity_model = pkl.load(file)
        with open(models_dir / f'sold_quantity_{min(inp+1, 5)}_predictor_init.pkl', 'rb') as file:
            self.output_quantity_model = pkl.load(file)

        frac_time_steps = np.linspace(0, 1, self.awi.n_steps, endpoint=False)
        X = frac_time_steps[:, None]
        predicted_input = self.input_quantity_model.predict(X) / 2
        predicted_output = self.output_quantity_model.predict(X) / 2
        self.expected_input = predicted_input
        self.expected_output = predicted_output

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            predicted = np.clip(x, a_min=1, a_max=self.awi.n_lines).astype(np.int32)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

    
    def trade_prediction_step(self):
        pass


class SklearnPredictionBasedTradingStrategy(
    SklearnTradePredictionStrategy, PredictionBasedTradingStrategy
):
    pass
