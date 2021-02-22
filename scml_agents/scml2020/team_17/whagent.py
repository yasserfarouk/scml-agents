from scml.scml2020 import SCML2020Agent, SCML2020World, RandomAgent, DecentralizingAgent
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import seaborn as sns
from scml.scml2020.agents import DoNothingAgent
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020.agents import (
    DecentralizingAgent,
    BuyCheapSellExpensiveAgent,
    IndDecentralizingAgent,
)
from negmas.helpers import humanize_time
from typing import List, Optional, Dict, Any
import numpy as np
from negmas import (
    Issue,
    AgentMechanismInterface,
    Contract,
    Negotiator,
    MechanismState,
    Breach,
)
from scml.scml2020 import Failure
from scml.scml2020 import SCML2020Agent
from scml.scml2020 import PredictionBasedTradingStrategy
from scml.scml2020 import MovingRangeNegotiationManager
from scml.scml2020 import TradeDrivenProductionStrategy
from negmas import LinearUtilityFunction
from scml.scml2020.components.production import DemandDrivenProductionStrategy
from scml.scml2020.components.trading import TradingStrategy
from scml.scml2020.components.negotiation import IndependentNegotiationsManager
from scml.scml2020.components.negotiation import NegotiationManager
from scml.scml2020.components.negotiation import StepNegotiationManager
from negmas import LinearUtilityFunction
from scml.scml2020 import AWI
from scml.scml2020.components.prediction import MeanERPStrategy
from scml.scml2020.components import FixedTradePredictionStrategy, SignAllPossible
from scml.scml2020.components.production import ProductionStrategy
from scml.scml2020.components.production import SupplyDrivenProductionStrategy
from scml.scml2020.common import NO_COMMAND
from negmas import Contract
from typing import List, Dict, Tuple
from scml.scml2020.components import TradePredictionStrategy
from scml.scml2020.common import is_system_agent
from scml.scml2020.common import ANY_LINE
import functools
import math
from abc import abstractmethod
from dataclasses import dataclass
from pprint import pformat
from typing import Tuple, List, Union, Any, Optional, Dict
import numpy as np
from negmas import (
    SAONegotiator,
    AspirationNegotiator,
    Issue,
    AgentMechanismInterface,
    Negotiator,
    UtilityFunction,
    Contract,
)
from negmas.helpers import get_class, instantiate
from scml.scml2020 import AWI
from scml.scml2020.components.prediction import MeanERPStrategy
from scml.scml2020.services.controllers import StepController, SyncController
from scml.scml2020.common import TIME

__all__ = ["WhAgent"]


class AllmakeProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)

    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            step = contract.agreement["time"]
            if step > latest + 1 or step < earliest_production:
                continue
            input_product = contract.annotation["product"]
            if self.awi.current_step == (self.awi.n_steps - 1):
                if self.awi.profile.costs[0, self.awi.my_input_product] > (
                    self.awi.catalog_prices[self.awi.my_input_product] / 2
                ):
                    continue

            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                # step=(step, latest),
                step=-1,
                line=-1,
                method="earliest",
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )


class AvoidOverproductionTradingStrategy(
    FixedTradePredictionStrategy, MeanERPStrategy, TradingStrategy
):
    def init(self):
        super().init()
        self.inputs_needed = np.zeros(self.awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)
        self.inputs_secured = np.zeros(self.awi.n_steps, dtype=int)
        self.outputs_secured = np.zeros(self.awi.n_steps, dtype=int)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] += q
            else:
                self.inputs_secured[t] += q

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        super().sign_all_contracts(contracts)
        signatures = [None] * len(contracts)
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
                # x[0].agreement["time"],
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
            if t < s and len(contract.issues) == 3:
                continue

            if self.awi.my_suppliers == ["SELLER"]:
                if q > self.awi.n_lines:
                    continue

                if is_seller:
                    if t > self.awi.n_steps - 2:
                        continue

                    zaiko = 0
                    for zzaiko in self.outputs_needed:
                        zaiko += zzaiko
                    if zaiko < 1:
                        if t < s + 3:
                            continue
                    sellprice = max(
                        (
                            self.awi.catalog_prices[self.awi.my_output_product]
                            - self.awi.catalog_prices[self.awi.my_input_product]
                            - self.awi.profile.costs[0, self.awi.my_input_product]
                        )
                        // 2
                        - 1,
                        0,
                    )
                    if (
                        u
                        < self.awi.catalog_prices[self.awi.my_output_product]
                        - sellprice
                    ):
                        continue
                    cansell = zaiko + (self.awi.n_lines - self.inputs_needed[t])
                    if q <= cansell:
                        self.outputs_needed[t] -= q
                        self.inputs_needed[t] += q
                    else:
                        continue
                else:
                    wantbuy = 0
                    needtime = -1
                    for step in range(self.awi.n_steps):
                        wantbuy += self.inputs_needed[step]
                        if self.inputs_needed[step] > 0 and needtime == -1:
                            needtime = step

                    if wantbuy > 0:
                        self.outputs_needed[t] += q
                        self.inputs_needed[t] -= q
                    else:
                        continue

            elif self.awi.my_consumers == ["BUYER"]:
                if q > self.awi.n_lines:
                    continue

                if is_seller:
                    zaiko = 0
                    for zzaiko in self.outputs_needed:
                        zaiko += zzaiko
                    if zaiko < 1:
                        if t < s + 2:
                            continue
                    cansell = zaiko
                    if q <= cansell:
                        self.outputs_needed[t] -= q
                        self.inputs_needed[t] += q
                    else:
                        continue
                else:
                    if t > s + 5:
                        continue

                    wantbuy = self.awi.n_lines - self.outputs_needed[t]
                    if wantbuy > 0:
                        self.outputs_needed[t] += q
                        self.inputs_needed[t] -= q
                    else:
                        continue
            else:
                if q > self.awi.n_lines:
                    continue

                if is_seller:
                    if t > self.awi.n_steps - 2:
                        continue

                    zaiko = 0
                    for zzaiko in self.outputs_needed:
                        zaiko += zzaiko
                    if zaiko < q:
                        if t < s + 2:
                            continue
                    sellprice = max(
                        (
                            self.awi.catalog_prices[self.awi.my_output_product]
                            - self.awi.catalog_prices[self.awi.my_input_product]
                            - self.awi.profile.costs[0, self.awi.my_input_product]
                        )
                        // 2
                        - 1,
                        0,
                    )
                    if (
                        u
                        < self.awi.catalog_prices[self.awi.my_output_product]
                        - sellprice
                    ):
                        continue
                    cansell = zaiko + (self.awi.n_lines - self.inputs_needed[t])
                    if q <= cansell:
                        self.outputs_needed[t] -= q
                        self.inputs_needed[t] += q
                    else:
                        continue
                else:
                    if t < s:
                        continue

                    havetobuy = 0
                    needtime = s - 1
                    for step in range(self.awi.n_steps):
                        havetobuy += self.inputs_needed[step]
                        if self.inputs_needed[step] > 0 and needtime <= (s - 1):
                            needtime = step

                    if t >= needtime:
                        continue

                    if needtime == s + 1:
                        if u < self.awi.catalog_prices[self.awi.my_input_product]:
                            continue
                    elif needtime < s + 3:
                        buyprice2 = max(
                            (
                                self.awi.catalog_prices[self.awi.my_output_product]
                                - self.awi.catalog_prices[self.awi.my_input_product]
                            )
                            // 2
                            - 1,
                            0,
                        )
                        if (
                            u
                            < self.awi.catalog_prices[self.awi.my_input_product]
                            + buyprice2
                        ):
                            continue
                    else:
                        buyprice = max(
                            (
                                self.awi.catalog_prices[self.awi.my_output_product]
                                - self.awi.catalog_prices[self.awi.my_input_product]
                                - self.awi.profile.costs[0, self.awi.my_input_product]
                            )
                            // 2
                            - 1,
                            0,
                        )
                        if (
                            u
                            < self.awi.catalog_prices[self.awi.my_input_product]
                            + buyprice
                        ):
                            continue

                    if havetobuy > 0:
                        self.outputs_needed[t] += q
                        self.inputs_needed[t] -= q
                    else:
                        continue

            signatures[indx] = self.id
            if is_seller:
                sold += q
            else:
                bought += q
        return signatures

    def _format(self, c: Contract):
        super()._format(c)
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
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                if t > 0:
                    self.inputs_needed[t - 1] -= missing
            else:
                self.inputs_secured[t] += missing
                if t < self.awi.n_steps - 1:
                    self.outputs_needed[t + 1] -= missing


@dataclass
class ControllerInfo:
    controller: StepController
    time_step: int
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    expected: int
    done: bool = False


class PreNegotiationManager(IndependentNegotiationsManager):
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return self.negotiator(annotation["seller"] == self.id, issues=issues)

    def negotiator(self, is_seller: bool, issues=None, outcomes=None) -> SAONegotiator:
        if outcomes is None and (
            issues is None or not Issue.enumerate(issues, astype=tuple)
        ):
            return None
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        return instantiate(self.negotiator_type, **params)

    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        super()._start_negotiations(
            product, sell, step, qvalues, uvalues, tvalues, partners
        )

        issues = [
            Issue(qvalues, name="quantity"),
            Issue(tvalues, name="time"),
            Issue(uvalues, name="uvalues"),
        ]
        sortpartner = {}
        if self.awi.current_step > 4:
            reportstep = ((self.awi.current_step // 5) - 1) * 5
            for k in self.awi.reports_at_step(reportstep).values():
                for ne in partners:
                    if ne == k.agent_id and k.breach_level < 1.0:
                        sortpartner[k.agent_id] = k.breach_level
            if len(sortpartner) != 0:
                sortpartners = sorted(sortpartner.items(), key=lambda x: x[1])
                sortpartners_list = [i[0] for i in sortpartners]
                for partner in sortpartners_list:
                    self.awi.request_negotiation(
                        is_buy=not sell,
                        product=product,
                        quantity=qvalues,
                        unit_price=uvalues,
                        time=tvalues,
                        partner=partner,
                        negotiator=self.negotiator(sell, issues=issues),
                    )
            else:
                for partner in partners:
                    self.awi.request_negotiation(
                        is_buy=not sell,
                        product=product,
                        quantity=qvalues,
                        unit_price=uvalues,
                        time=tvalues,
                        partner=partner,
                        negotiator=self.negotiator(sell, issues=issues),
                    )
        else:
            for partner in partners:
                self.awi.request_negotiation(
                    is_buy=not sell,
                    product=product,
                    quantity=qvalues,
                    unit_price=uvalues,
                    time=tvalues,
                    partner=partner,
                    negotiator=self.negotiator(sell, issues=issues),
                )

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            if sum(self.outputs_secured) < self.awi.n_lines * (
                self.awi.n_steps - self.awi.current_step - 2
            ):
                sellnum = min(self.awi.n_lines, self.outputs_secured[step])
            else:
                sellnum = 0
        else:
            if sum(self.inputs_secured) < self.awi.n_lines * (
                self.awi.n_steps - self.awi.current_step - 2
            ):
                buynum = min(self.awi.n_lines, sum(self.outputs_secured))
            else:
                buynum = 0

        if step == self.awi.current_step - 1:
            return 0 if sell else 0
        return sellnum if sell else buynum

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        return (
            self.awi.catalog_prices[self.awi.my_output_product]
            if sell
            else self.awi.catalog_prices[self.awi.my_input_product]
        )

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1))
        return LinearUtilityFunction((0, -0.5, -0.8))


class WhAgent(
    AvoidOverproductionTradingStrategy,
    PreNegotiationManager,
    AllmakeProductionStrategy,
    SCML2020Agent,
):
    def init(self):
        super().init()

    def step(self):
        super().step()
