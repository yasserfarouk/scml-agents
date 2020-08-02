# required for typing
from abc import abstractmethod
from typing import Union, Iterable, List, Optional
import numpy as np

# my need
from scml.scml2020.components.trading import *
from scml.scml2020 import *
from negmas import *
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import seaborn as sns


class MyTradePredictor(TradePredictionStrategy):
    #     # 継承して利用する際は最初の引数にしないと反映されない(MRO的に)
    #     #PredictionBasedTradingStrategyとReactiveAgentでしか使われてない
    #     """
    #     TradePredictionStrategy
    #     *FixedTradePredictionStrategy
    #     """
    #     # PredictionBasedTradingStrategyで使う．expectedからneededを決める．
    #     def trade_prediction_init(self):
    #         self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
    #         self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, self.awi.n_lines // 4)  # 元は // 2
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
        if self.awi.current_step < self.awi.n_steps - 1:
            self.output_price[self.awi.current_step + 1] = self.output_price[
                self.awi.current_step
            ]
            self.input_cost[self.awi.current_step + 1] = self.input_cost[
                self.awi.current_step
            ]
        # print(self.id, self.input_cost[self.awi.current_step - 3:self.awi.current_step + 1])
        # print(self.awi.current_step, self.id, self.input_cost)

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

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if not self._add_trade:
            return

        def check_breach(contract, is_seller):
            k_pre = 0
            latest_breach_level = 0.0
            if is_seller:
                d = self.awi.reports_of_agent(contract.annotation["buyer"])
            else:
                d = self.awi.reports_of_agent(contract.annotation["seller"])
            if d is not None:
                for k, v in d.items():
                    if k > k_pre:
                        latest_breach_level = v.breach_level
            return latest_breach_level

        for contract in signed:
            t, q, p = (
                contract.agreement["time"],
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
            )
            # print(self.id, t, q, p)
            step = self.awi.current_step
            contract_weight = 0.5  # 契約による影響度 0.5以下でどれくらいがいいかな？
            # breach_levelを参照して予測値を調整，ステップ数が増えると効いてくる（多分）
            if contract.annotation["seller"] == self.id:
                penalty = pow(check_breach(contract, True), 1)  # 何乗かするより1乗のままが一番性能良かった
                self.expected_outputs[t] += q * (1 - penalty)  # 数の予測
                # 価格も予測してみる
                if step >= 3:
                    predict = self.output_price[step - 1]
                    for i in range(q):
                        predict = (1 - contract_weight) * predict + contract_weight * p
                    predict = round(
                        (
                            penalty * self.output_price[step - 1]
                            + (1 - penalty) * predict
                        )
                    )  # 整数のみだから四捨五入
                    self.output_price[step - 1] = round(
                        (
                            self.output_price[step - 3]
                            + self.output_price[step - 2]
                            + predict
                        )
                        / 3
                    )  # 3step前まで反映
            else:
                penalty = pow(check_breach(contract, False), 1)
                self.expected_inputs[t] += q * (1 - penalty)  # 数の予測
                if step >= 3:
                    # 価格も予測してみる
                    predict = self.input_cost[step - 1]
                    for i in range(q):
                        predict = (1 - contract_weight) * predict + contract_weight * p
                    predict = round(
                        (penalty * self.input_cost[step - 1] + (1 - penalty) * predict)
                    )  # 整数のみだから四捨五入
                    self.input_cost[step - 1] = round(
                        (
                            self.input_cost[step - 3]
                            + self.input_cost[step - 2]
                            + predict
                        )
                        / 3
                    )  # 3step前まで反映


class MyERPredictor(ExecutionRatePredictionStrategy):
    # 継承して利用する際は最初の引数にしないと反映されない(MRO的に)
    # PredictionBasedTradingStrategyとStepNegotiationManagerでしか使われてない
    # PredictionBasedTradingStrategyはon_contract_executedとon_contract_breachedだけ使ってる？でもTradingStrategyには必要ないよな．．．？
    """
    ExecutionRatePredictionStrategy
    FixedERPStrategy
    *MeanERPStrategy
    """

    def __init__(
        self, *args, execution_fraction=0.8, **kwargs
    ):  # execution_fractionの初期値何がいい？
        super().__init__(*args, **kwargs)
        self._execution_fraction = execution_fraction
        self._total_quantity = None

    def predict_quantity(
        self, contract: Contract
    ):  # 不要．本来は_start_negotiationsのexpected_quantityを求めるのに使われるはずだった？
        return contract.agreement["quantity"] * self._execution_fraction

    def init(self):
        super().init()
        self._total_quantity = max(1, self.awi.n_steps * self.awi.n_lines // 10)

    @property
    def internal_state(self):
        state = super().internal_state
        state.update({"execution_fraction": self._execution_fraction})
        return state

    def on_contract_executed(self, contract: Contract) -> None:
        super().on_contract_executed(contract)
        old_total = self._total_quantity
        q = contract.agreement["quantity"]
        self._total_quantity += q
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
        # print(self._execution_fraction)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        old_total = self._total_quantity
        q = int(contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches)))
        self._total_quantity += contract.agreement["quantity"]
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
