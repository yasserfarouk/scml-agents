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


class MyNegotiationsManager:
    def __init__(
        self,
        *args,
        horizon=5,
        negotiate_on_signing=True,
        logdebug=False,
        use_trading_prices=True,
        min_price_margin=0.25,
        max_price_margin=0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._horizon = horizon
        self._negotiate_on_signing = negotiate_on_signing
        self._log = logdebug
        self._use_trading = use_trading_prices
        self._min_margin = 1 - min_price_margin
        self._max_margin = 1 + max_price_margin

    @property
    def use_trading(self):
        return self._use_trading

    @use_trading.setter
    def use_trading(self, v):
        self._use_trading = v

    def init(self):
        # set horizon to exogenous horizon
        if self._horizon is None:
            self._horizon = self.awi.bb_read("settings", "exogenous_horizon")
            if self._horizon is None:
                self._horizon = 5

        # let other component work. We init last here as we do not depend on any other components for this method.
        super().init()

    def start_negotiations(
        self,
        product: int,
        quantity: int,
        unit_price: int,
        step: int,
        partners: List[str] = None,
    ) -> None:
        """
        Starts a set of negotiations to buy/sell the product with the given limits
        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            step: The maximum/minimum time for buy/sell
            partners: A list of partners to negotiate with
        Remarks:
            - This method assumes that product is either my_input_product or my_output_product
        """
        awi: AWI
        awi = self.awi  # type: ignore
        is_seller = product == self.awi.my_output_product
        if quantity < 1 or unit_price < 1 or step < awi.current_step + 1:
            # awi.logdebug_agent(
            #     f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{step})"
            # )
            return
        # choose ranges for the negotiation agenda.
        qvalues = (1, quantity)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(step, is_seller, tvalues)
        if tvalues[0] > tvalues[1]:
            return
        if partners is None:
            if is_seller:
                partners = awi.my_consumers
            else:
                partners = awi.my_suppliers
        self._start_negotiations(
            product, is_seller, step, qvalues, uvalues, tvalues, partners
        )

    def step(self):
        """Generates buy and sell negotiations as needed"""
        super().step()

        if self.awi.is_bankrupt():
            return
        s = self.awi.current_step
        if s == 0:
            # in the first step, generate buy/sell negotiations for horizon steps in the future
            last = min(self.awi.n_steps - 1, self._horizon + 2)
            for step in range(1, last):
                self._generate_negotiations(step, False)
                self._generate_negotiations(step, True)
        else:
            # generate buy and sell negotiations to secure inputs/outputs the step after horizon steps
            nxt = s + self._horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self._generate_negotiations(nxt, False)
            self._generate_negotiations(nxt, True)
        if self._log:
            self.awi.logdebug_agent(f"End step:\n{pformat(self.internal_state)}")

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        if self._negotiate_on_signing:
            steps = (self.awi.current_step + 1, self.awi.n_steps)
            in_pre = self.target_quantities(steps, False)
            out_pre = self.target_quantities(steps, True)
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if (not self._negotiate_on_signing) or (in_pre is None) or (out_pre is None):
            return
        inputs_needed = self.target_quantities(steps, False)
        outputs_needed = self.target_quantities(steps, True)
        if (inputs_needed is None) or (outputs_needed is None):
            return
        inputs_needed -= in_pre
        outputs_needed -= out_pre
        for s in np.nonzero(inputs_needed)[0]:
            if inputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_input_product,
                inputs_needed[s],
                self.acceptable_unit_price(s, False),
                s,
            )
        for s in np.nonzero(outputs_needed)[0]:
            if outputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_output_product,
                outputs_needed[s],
                self.acceptable_unit_price(s, True),
                s,
            )

    def _generate_negotiations(self, step: int, sell: bool) -> None:
        """Generates all the required negotiations for selling/buying for the given step"""
        product = self.awi.my_output_product if sell else self.awi.my_input_product
        quantity = self.target_quantity(step, sell)
        unit_price = self.acceptable_unit_price(step, sell)

        if quantity <= 0 or unit_price <= 0:
            return
        self.start_negotiations(
            product=product,
            step=step,
            quantity=min(self.awi.n_lines * (step - self.awi.current_step), quantity),
            unit_price=unit_price,
        )

    def _urange(self, step, is_seller, time_range):
        prices = (
            self.awi.catalog_prices
            if not self._use_trading
            or not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        if is_seller:
            cprice = prices[self.awi.my_output_product]
            return int(cprice * self._min_margin), int(self._max_margin * cprice + 0.5)

        cprice = prices[self.awi.my_input_product]
        return int(cprice * self._min_margin), int(self._max_margin * cprice + 0.5)

    def _trange(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self._horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """
        Returns the target quantity to negotiate about for each step in the range given (beginning included and ending
        excluded) for buying/selling
        Args:
            steps: Simulation step
            sell: Sell or buy
        """
        steps = (max(steps[0], 0), min(steps[-1], self.awi.n_steps))
        return np.array([self.target_quantity(s, sell) for s in range(*steps)])

    @abstractmethod
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
        """
        Actually start negotiations with the given agenda
        Args:
            product: The product to negotiate about.
            sell: If true, this is a sell negotiation
            step: The step
            qvalues: the range of quantities
            uvalues: the range of unit prices
            tvalues: the range of times
            partners: partners
        """
        pass

    @abstractmethod
    def target_quantity(self, step: int, sell: bool) -> int:
        """
        Returns the target quantity to sell/buy at a given time-step
        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement target_quantity")

    @abstractmethod
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """
        Returns the maximum/minimum acceptable unit price for buying/selling at the given time-step
        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement acceptable_unit_price")

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Optional[Negotiator]:
        raise ValueError("You must implement respond_to_negotiation_request")


class MyIndependentNegotiationsManager(MyNegotiationsManager):
    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

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
        # negotiate with all suppliers of the input product I need to produce
        issues = [
            make_issue((int(qvalues[0]), int(qvalues[1])), name="quantity"),
            make_issue((int(tvalues[0]), int(tvalues[1])), name="time"),
            make_issue((int(uvalues[0]), int(uvalues[1])), name="unit_price"),
        ]
        for partner in partners:
            self.awi.request_negotiation(
                is_buy=not sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=self.negotiator(sell, issues=issues, partner=partner),
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Optional[Negotiator]:
        return self.negotiator(
            annotation["seller"] == self.id, issues=issues, partner=initiator
        )

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        """Creates a utility function"""
        if is_seller:
            return LinearUtilityFunction((1, 1, 10), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((1, -1, -10), issues=issues, outcomes=outcomes)

    def negotiator(
        self, is_seller: bool, issues=None, outcomes=None, partner=None
    ) -> SAONegotiator:
        """Creates a negotiator"""
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        return instantiate(self.negotiator_type, id=partner, **params)

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            # In the final step, return the maximum number
            if step >= self.awi.n_steps:
                return self.outputs_needed[step - 1]
            # Up to twice the number of production lines
            return min(self.outputs_needed[step], self.awi.n_lines * 2)
        else:
            # I'm not buying in the final step
            if step >= self.awi.n_steps:
                return 0
            # Up to twice the number of production lines
            return min(self.inputs_needed[step], self.awi.n_lines * 2)

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        # 予測したコストや売値の値を返す
        if sell:
            return self.output_price[step]
        else:
            return self.input_cost[step]
