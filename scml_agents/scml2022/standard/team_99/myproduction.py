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
from scml.scml2020.common import ANY_LINE, is_system_agent, NO_COMMAND
from scml.scml2020.components import SignAllPossible
from scml.scml2020.components.prediction import FixedTradePredictionStrategy
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.prediction import MeanERPStrategy
from abc import abstractmethod
from pprint import pformat


class MyProductionStrategy(ProductionStrategy):
    def step(self):
        super().step()
        # インベントリ内の原材料を各工場に割り振る
        # 最大で工場の生産ライン数(awi.n_lines)まで割り振れる
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
        super().on_contracts_finalized(signed, cancelled, rejectors)
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            # if is_sellerになってた。
            # 自分が売り手の時にスケジュールを更新するのだからif not is_sellerが正しいのでは？
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            # 契約によって必要となるinput_productのID
            input_product = contract.annotation["product"]
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(step, latest),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )
