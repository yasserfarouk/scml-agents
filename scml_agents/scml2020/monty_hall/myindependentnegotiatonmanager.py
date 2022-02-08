import copy
import functools
import math
import time
from dataclasses import dataclass

# required for typing
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Breach,
    Contract,
    Issue,
    LinearUtilityFunction,
    MechanismState,
    Negotiator,
    SAONegotiator,
)
from negmas.helpers import get_class, humanize_time, instantiate
from negmas.outcomes.base_issue import make_issue
from scml.scml2020 import AWI, SCML2020Agent, SCML2020World
from scml.scml2020.common import TIME
from scml.scml2020.services.controllers import StepController, SyncController

# from mynegotiationmanager import MyNegotiationManager


class MyIndependentNegotiationManager:
    def __init__(
        self,
        *args,
        data,
        plan,
        awi,
        agent,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.data = data
        self.plan = plan
        self.awi = awi
        self.agent = agent

        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        self._horizon = 5  # TODO: Decide this value

    def step(self):  # Negotiate every step
        """Generates buy and sell negotiations as needed"""
        s = self.awi.current_step + 1

        if s <= self.data.last_day:
            self.start_negotiations(s, False)

        if s < self.data.n_steps - 1:
            self.start_negotiations(s, True)

    def start_negotiations(
        self,
        step: int,
        is_seller: bool,
    ) -> None:
        """
        Starts a set of negotiations to by/sell the product with the given limits

        Args:
            step: The maximum/minimum time for buy/sell

        Remarks:

            - This method assumes that product is either my_input_product or my_output_product

        """
        awi: AWI
        awi = self.awi  # type: ignore
        if step < awi.current_step + 1:
            return
        # choose ranges for the negotiation agenda.
        qvalues = self._qrange(step, is_seller)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(is_seller)
        if tvalues[0] > tvalues[1]:
            return

        product = self.data.output_product if is_seller else self.data.input_product
        partners = self.data.consumer_list if is_seller else self.data.supplier_list

        self._start_negotiations(
            product, is_seller, step, qvalues, uvalues, tvalues, partners
        )

    def _qrange(self, step: int, sell: bool) -> Tuple[int, int]:
        if sell:
            upper_bound = self.plan.available_output  # Sell everything
        else:
            upper_bound = self.data.n_lines  # Production capacity

        return 1, upper_bound

    def _trange(self, step, is_seller):
        if is_seller:
            return (
                self.awi.current_step + 1,
                min(step + self._horizon, self.data.n_steps - 2),
            )

        return self.awi.current_step + 1, min(step + self._horizon, self.data.last_day)

    def _urange(self, is_seller):
        if is_seller:
            return self.plan.min_sell_price, self.plan.max_sell_price
        return self.plan.min_buy_price, self.plan.max_buy_price

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
            make_issue((int(qvalues[0]), int(max(qvalues))), name="quantity"),
            make_issue((int(tvalues[0]), int(max(tvalues))), name="time"),
            make_issue((int(uvalues[0]), int(max(uvalues))), name="unit_price"),
        ]
        negotiator = self.create_negotiator(sell, issues=issues)

        for partner in partners:
            self.awi.request_negotiation(
                is_buy=not sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=negotiator,
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return self.create_negotiator(
            annotation["seller"] == self.agent.id, issues=issues
        )

    def create_negotiator(self, is_seller: bool, issues=None, outcomes=None):
        """Creates a negotiator"""

        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        return instantiate(self.negotiator_type, **params)

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        if is_seller:
            return LinearUtilityFunction((1, 1, 10), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((1, -1, -10), issues=issues, outcomes=outcomes)
