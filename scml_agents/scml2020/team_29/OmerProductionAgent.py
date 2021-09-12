import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from negmas import Breach, Contract
from scml import SCML2020World
from scml.scml2020 import (
    NO_COMMAND,
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    Factory,
    IndDecentralizingAgent,
    MovingRangeAgent,
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
)
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
)
from scml.scml2020.components.trading import FixedTradePredictionStrategy


class OmerProductionStrategyAgent(ProductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_omer_production_args(self):
        if not hasattr(self, "_production_classifier"):
            self._production_classifier, self._vectorizer = pickle.load(
                open("production_binary_classifier.pkl", "rb")
            )
        if not hasattr(self, "omer_input"):
            self.omer_input = 0
        if not hasattr(self, "omer_output"):
            self.omer_output = 0
        if not hasattr(self, "_features"):
            self._features = [
                {
                    "n_agents": len(
                        "|".join(["|".join(x) for x in self.awi.all_consumers]).split(
                            "|"
                        )
                    )
                    - 1,
                    "step": -1,
                    "steps_to_end": self.awi.state.n_steps - self.awi.current_step,
                    "relative_step": 0,
                    "input_product": self.awi.my_input_product,
                    "output_product": self.awi.my_input_product + 1,
                    "inventory_input": self.awi.state.inventory[
                        self.awi.my_input_product
                    ],
                    "inventory_output": self.awi.state.inventory[
                        self.awi.my_input_product + 1
                    ],
                    "price_in": self.awi.catalog_prices[self.awi.my_input_product],
                    "price_out": self.awi.catalog_prices[self.awi.my_input_product + 1],
                    "n_buy": 0,
                    "n_buy_price_mean": 0,
                    "n_buy_price_std": 0,
                    "n_buy_quantity_mean": 0,
                    "n_buy_quantity_std": 0,
                    "n_buy_time_mean": 0,
                    "n_sell": 0,
                    "n_sell_price_mean": 0,
                    "n_sell_price_std": 0,
                    "n_sell_quantity_mean": 0,
                    "n_sell_quantity_std": 0,
                    "n_sell_time_mean": 0,
                }
            ]

    def _mean(self, l):
        return np.mean(l) if len(l) > 0 else 0

    def _std(self, l):
        return np.std(l) if len(l) > 0 else 0

    def _save_feature_state(self, signed):
        print(self.awi.current_step)
        signed_sell = [x for x in signed if x.annotation["seller"] == self.id]
        signed_buy = [x for x in signed if x.annotation["seller"] != self.id]

        vec = {
            "n_agents": len(
                "|".join(["|".join(x) for x in self.awi.all_consumers]).split("|")
            )
            - 1,
            "step": self.awi.current_step,
            "steps_to_end": self.awi.state.n_steps - self.awi.current_step,
            "relative_step": str(self.awi.relative_time),
            "input_product": self.awi.my_input_product,
            "output_product": self.awi.my_input_product + 1,
            "inventory_input": self.awi.state.inventory[self.awi.my_input_product],
            "inventory_output": self.awi.state.inventory[self.awi.my_input_product + 1],
            "price_in": self.awi.catalog_prices[self.awi.my_input_product],
            "price_out": self.awi.catalog_prices[self.awi.my_input_product + 1],
            "n_buy": len(signed_buy),
            "n_buy_price_mean": self._mean(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_buy
                    if contract
                ]
            ),
            "n_buy_price_std": self._std(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_buy
                    if contract
                ]
            ),
            "n_buy_quantity_mean": self._mean(
                [contract.agreement["quantity"] for contract in signed_buy if contract]
            ),
            "n_buy_quantity_std": self._std(
                [contract.agreement["quantity"] for contract in signed_buy if contract]
            ),
            "n_buy_time_mean": self._mean(
                [
                    contract.agreement["time"] - self.awi.current_step
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell": len(signed_sell),
            "n_sell_price_mean": self._mean(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell_price_std": self._std(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell_quantity_mean": self._mean(
                [contract.agreement["quantity"] for contract in signed_sell if contract]
            ),
            "n_sell_quantity_std": self._std(
                [contract.agreement["quantity"] for contract in signed_sell if contract]
            ),
            "n_sell_time_mean": self._mean(
                [
                    contract.agreement["time"] - self.awi.current_step
                    for contract in signed_sell
                    if contract
                ]
            ),
        }
        self._features.append(vec)

    def _produce_by_prediction(self):
        instance = {}
        for (prev_k, prev_v), (next_k, next_v) in zip(
            self._features[-2].items(), self._features[-1].items()
        ):
            instance["prev_" + prev_k] = prev_v
            instance["next_" + next_k] = next_v
        to_prodeuce = self._production_classifier.predict(
            self._vectorizer.transform(instance)
        ).item()

        if to_prodeuce:
            steps, _ = self.awi.schedule_production(
                process=self.awi.my_input_product,
                repeats=1,
                step=(self.awi.current_step + 1, self.awi.current_step + 1),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range["prediction" + str(self.awi.current_step)] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                False,
            )

    def on_contract_executed(self, contract: Contract) -> None:
        self._load_omer_production_args()
        super().on_contract_executed(contract)
        if self.id == contract.annotation["seller"]:
            self.omer_output += contract.agreement["quantity"]
        else:
            self.omer_input += contract.agreement["quantity"]

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        self._load_omer_production_args()
        super().on_contract_breached(contract, breaches, resolution)
        if self.id == contract.annotation["seller"]:
            self.omer_output += int(
                contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches))
            )
        else:
            self.omer_input += int(
                contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches))
            )

    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        self._load_omer_production_args()
        self._save_feature_state(signed)
        self._produce_by_prediction()
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
