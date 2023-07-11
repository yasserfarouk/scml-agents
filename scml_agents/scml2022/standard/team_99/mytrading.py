from typing import Any, Dict, List, Optional, Iterable, Union, Tuple

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    SAONegotiator,
    AspirationNegotiator,
    make_issue,
    NegotiatorMechanismInterface,
    UtilityFunction,
    LinearUtilityFunction,
)
from negmas.helpers import humanize_time, get_class, instantiate
from scml.scml2020 import Failure

# required for development
# required for running the test tournament
import time
from tabulate import tabulate
from scml.utils import anac2022_collusion, anac2022_std, anac2022_oneshot
from scml.scml2020 import (
    SCML2020Agent,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
    DemandDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
    ReactiveTradingStrategy,
    PredictionBasedTradingStrategy,
    TradingStrategy,
    StepNegotiationManager,
    IndependentNegotiationsManager,
    MovingRangeNegotiationManager,
    TradePredictionStrategy,
    AWI,
)
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)
from scml.scml2020.common import ANY_LINE, is_system_agent, NO_COMMAND, FinancialReport
from scml.scml2020.components import SignAllPossible
from scml.scml2020.components.prediction import FixedTradePredictionStrategy
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.prediction import MeanERPStrategy
from abc import abstractmethod
from pprint import pformat


class MyTradePredictionStrategy(TradePredictionStrategy):
    def __init__(self, *args, add_trade=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_trade = add_trade

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, int(self.awi.n_lines))
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
        # ここは常にFalseになるのでfor文へ続く
        if not self._add_trade:
            return
        for contract in signed:
            # ignore contracts I asked for because they are already covered in estimates
            if contract.annotation["caller"] == self.id:
                continue
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            if contract.annotation["seller"] == self.id:
                self.expected_outputs[t] += q
            else:
                self.expected_inputs[t] += q


class MyTradingStrategy(
    MyTradePredictionStrategy,
    MeanERPStrategy,
    TradingStrategy,
):
    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]
        # 売った回数、買った回数を記憶
        self.number_of_sold = [0] * self.awi.n_products
        self.number_of_sold_copied = [0] * self.awi.n_products
        self.number_of_buy = [0] * self.awi.n_products
        self.number_of_buy_copied = [0] * self.awi.n_products

    def _update_needs(self):
        s = self.awi.current_step
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[s:-1] = self.expected_outputs[s + 1 :]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[s + 1 :] = self.expected_inputs[s:-1]

    def before_step(self):
        super().before_step()
        number_of_buy = self.number_of_buy[self.awi.my_input_product]
        number_of_sold = self.number_of_sold[self.awi.my_output_product]

        # 予測売値を更新
        # そのステップで一つでも売れたら値上げする
        if not self.awi.is_last_level:
            if (
                number_of_sold - self.number_of_sold_copied[self.awi.my_output_product]
                > 0
            ):
                self.output_price[self.awi.current_step :] += 1
                self.number_of_sold_copied[self.awi.my_output_product] = number_of_sold
            else:
                # 売れなかったら値下げ
                self.output_price[self.awi.current_step :] -= 1
                self.number_of_sold_copied[self.awi.my_output_product] = number_of_sold
        # 予測コストを更新
        # 買えたら次は少し安く買えるのではと予測
        if not self.awi.is_first_level:
            if number_of_buy - self.number_of_buy_copied[self.awi.my_input_product] > 0:
                self.input_cost[self.awi.current_step :] -= 1
                self.number_of_buy_copied[self.awi.my_input_product] = number_of_buy
            # 買えなかったら次は高く買うと予想
            else:
                self.input_cost[self.awi.current_step :] += 1
                self.number_of_buy_copied[self.awi.my_input_product] = number_of_buy
        # 半分のステップから売価80%に一度値下げ
        if self.awi.current_step == (self.awi.n_steps // 2):
            cut = self.output_price[self.awi.current_step] // 5
            self.output_price[self.awi.current_step :] -= cut
        self._update_needs()

    def step(self):
        super().step()
        self._update_needs()

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        # keeps track of the procution slots consumed by signed contracts processed
        consumed = 0
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            q, t = (
                contract.agreement["quantity"],
                contract.agreement["time"],
            )
            if is_seller:
                self.outputs_secured[t] += q
                self.number_of_sold[self.awi.my_output_product] += 1
            else:
                self.inputs_secured[t] += q
                self.number_of_buy[self.awi.my_input_product] += 1
            # If I intiated the negotiation for this contract, ignore it.
            if contract.annotation["caller"] == self.id:
                continue
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                # If I need to produce, do production
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, _ = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    # register the number of production slots consumed for this contract
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    # this is a sell contract that I did not expect yet. Update needs accordingly
                    # I must buy all my needs one day earlier at most
                    self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            # register that I secured the given outputs
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                # this is a buy contract that I did not expect yet. Update needs accordingly
                # I must sell these inputs after production one day later at least
                self.outputs_needed[t + 1] += max(1, q)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        signatures = [None] * len(contracts)
        # sort contracts by goodness of price, time and then put system contracts first within each time-step
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["time"],
                (
                    x[0].agreement["unit_price"]
                    - self.output_price[x[0].agreement["time"]]
                )
                if x[0].annotation["seller"] == self.id
                else (
                    self.input_cost[x[0].agreement["time"]]
                    - x[0].agreement["unit_price"]
                ),
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
            ),
        )
        sold, bought = 0, 0
        s = self.awi.current_step
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # print("debug/contract", self._format(contract))
            # check that the contract is executable in principle. The second
            # condition checkes that the contract is negotiated and not exogenous
            if t < s and len(contract.issues) == 3:
                continue
            catalog_buy = self.input_cost[t]
            catalog_sell = self.output_price[t]
            # check that the gontract has a good price
            if (is_seller and u < catalog_sell) or (not is_seller and u > catalog_buy):
                continue
            if is_seller:
                trange = (s, t - 1)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought
                # 契約で確保しているinput_productの合計数
                in_secure = 0
                # 残りのステップ数で製造できる限界数
                in_limit_production = (
                    self.awi.n_steps - self.awi.current_step - 1
                ) * self.awi.n_lines
                for x in self.inputs_secured[self.awi.current_step :]:
                    in_secure += x
                # 製造できないinput_productは買わない
                if (
                    self.awi.current_inventory[self.awi.my_input_product]
                    + in_secure
                    + q
                ) > in_limit_production:
                    continue

            # check that I can produce the required quantities even in principle
            steps, _ = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            # print("debug/steps", steps)
            if len(steps) - taken < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                # サインしたら買った回数、売った回数を更新
                if is_seller:
                    sold += q
                else:
                    bought += q
        return signatures

    def _format(self, c: Contract):
        return (
            f"{f'>' if c.annotation['seller'] == self.id else '<'}"
            f"{c.annotation['buyer'] if c.annotation['seller'] == self.id else c.annotation['seller']}: "
            f"{c.agreement['quantity']} of {c.annotation['product']} @ {c.agreement['unit_price']} on {c.agreement['time']}"
        )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            s = self.awi.current_step
            if t < self.awi.current_step:
                continue
            # distribute the missing quantity over time
            if contract.annotation["seller"] == self.id:
                # self.outputs_secured[t] -= missing
                if t > s:
                    for tau in range(t - 1, s - 1, -1):
                        if self.inputs_needed[tau] <= 0:
                            continue
                        if self.inputs_needed[tau] >= missing:
                            self.inputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.inputs_needed[tau] = 0
                        missing -= self.inputs_needed[tau]
                        if missing <= 0:
                            break
                if missing > 0:
                    if t < self.awi.n_steps - 1:
                        for tau in range(t + 1, self.awi.n_steps):
                            if self.outputs_secured[tau] <= 0:
                                continue
                            if self.outputs_secured[tau] >= missing:
                                self.outputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.outputs_secured[tau] = 0
                            missing -= self.outputs_secured[tau]
                            if missing <= 0:
                                break

            else:
                if t < self.awi.n_steps - 1:
                    for tau in range(t + 1, self.awi.n_steps):
                        if self.outputs_needed[tau] <= 0:
                            continue
                        if self.outputs_needed[tau] >= missing:
                            self.outputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.outputs_needed[tau] = 0
                        missing -= self.outputs_needed[tau]
                        if missing <= 0:
                            break
                if missing > 0:
                    if t > s:
                        for tau in range(t - 1, s - 1, -1):
                            if self.inputs_secured[tau] <= 0:
                                continue
                            if self.inputs_secured[tau] >= missing:
                                self.inputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.inputs_secured[tau] = 0
                            missing -= self.inputs_secured[tau]
                            if missing <= 0:
                                break
