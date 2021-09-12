# required for running the test tournament
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
    MechanismState,
    Negotiator,
    SAONegotiator,
)
from negmas.helpers import get_class, humanize_time
from scml.scml2020 import AWI, SCML2020Agent, SCML2020World
from scml.scml2020.common import TIME
from scml.scml2020.services.controllers import StepController, SyncController


@dataclass
class ControllerInfo:
    """Keeps a record of information about one of the controllers used by the agent"""

    controller: StepController
    time_step: int
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    expected: int
    done: bool = False


class MyNegotiationManager:
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

        # save construction parameters
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        # attributes that will be read during init() from the AWI
        # -------------------------------------------------------
        self.buyers = self.sellers = None

        # self._horizon = 4  # TODO: Decide this value
        self._horizon = self.plan.horizon
        self.init()
        """Buyer controllers and seller controllers. Each of them is responsible of covering the
        needs for one step (either buying or selling)."""

    def init(self):
        # ================================
        # Negotiation Components
        # ================================
        # initialize one controller for buying and another for selling for each time-step
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.data.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.data.n_steps)
        ]

    def step(self):  # Negotiate every step
        """Generates buy and sell negotiations as needed"""
        s = self.awi.current_step + 1

        if s <= self.data.last_day:
            self.start_negotiations(s, False)

        if s < self.data.n_steps - 1:
            self.start_negotiations(s, True)

        # check plans are being updated at each step to use the new NVM plan
        # print("------------------------------NEGOTIATION------------------------------")
        # print(f"negotiation buy plan: {self.plan.buy_plan}")
        # print(f"negotiation sell plan: {self.plan.sell_plan}")
        # print("-----------------------------------------------------------------------")

    def start_negotiations(self, step: int, is_seller: bool) -> None:
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

    def _urange(self, is_seller):
        """price range"""
        # if is_seller:
        #     return self.plan.min_sell_price, self.plan.max_sell_price
        # return self.plan.min_buy_price, self.plan.max_buy_price

        # TODO: MAKE A REALLY ABUSIVE URANGE
        if is_seller:
            return 35, 3000
        else:
            return 10, 18

    def _trange(self, step, is_seller):
        """returns (min, max)"""
        #        #print(self.data.n_processes)
        #        #print(self.data.process)
        #        #print(self.data.n_steps)
        #        #print(self.data.last_day)

        # negotatiate in future based on horizon
        if is_seller:
            return (
                self.awi.current_step + 1,
                min(step + self._horizon, self.data.n_steps - 2),
            )  # why -2?

        return self.awi.current_step + 1, min(step + self._horizon, self.data.last_day)

        # only use steps now and do not negotiate in the future
        # if is_seller:
        #     return (
        #         self.awi.current_step + 1,
        #         self.data.n_steps - 2)  # why -2?
        #
        # return self.awi.current_step + 1, min(self.awi.current_step + 2, self.data.last_day)

    def _qrange(self, step: int, sell: bool) -> Tuple[int, int]:
        #        #print(f'current: {self.awi.current_step}')
        #        #print(f'step: {step}')
        #        #print(f'plan: {self.plan.sell_plan}')
        #        #print()
        #        if step >= len(self.plan.sell_plan):
        #            return 1,1
        step -= self.awi.current_step
        ##print("---------STEP: " + str(step))

        # changing all instances of step to 0 for now
        if sell:
            ##print(f"qrange buy plan: {self.plan.buy_plan}")
            ##print(f"qrange sell plan: {self.plan.sell_plan}")
            upper_bound = self.plan.sell_plan[0]  # Sell based on the sell plan
            # upper_bound = self.plan.available_output #Sell all inventory
        else:
            upper_bound = self.plan.buy_plan[0]  # Production capacity
        #        #print(f'upper: {upper_bound}')
        return 1, upper_bound

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
        _execution_fraction = 1
        expected_quantity = int(math.floor(qvalues[1] * _execution_fraction))

        # negotiate with everyone
        controller = self.add_controller(
            sell, qvalues[1], uvalues, expected_quantity, step
        )

        if (
            qvalues[1] < qvalues[0]
            or uvalues[1] < uvalues[0]
            or tvalues[1] < tvalues[0]
            or tvalues[1] < self.awi.current_step
        ):
            return
        self.awi.request_negotiations(
            is_buy=not sell,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,
            controller=controller,
            extra=dict(controller_index=step, is_seller=sell),
        )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        # find negotiation parameters
        is_seller = annotation["seller"] == self.agent.id
        #        tmin, tmax = issues[TIME].min_value, issues[TIME].max_value + 1
        tmin = self.awi.current_step
        tmax = tmin + self._horizon - 1
        #        #print(tmin)
        #        #print(tmax)
        # find the time-step for which this negotiation should be added
        step = max(0, tmin - 1) if is_seller else min(self.awi.n_steps - 1, tmax + 1)
        # find the corresponding controller.
        controller_info: ControllerInfo
        controller_info = self.sellers[step] if is_seller else self.buyers[step]
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        target = self.target_quantities((tmin, tmax + 1), is_seller)
        if target <= 0:
            return None

        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.add_controller(
                is_seller,
                target,
                self._urange(is_seller),
                target,
                step,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator()

    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> StepController:
        if is_seller and self.sellers[step].controller is not None:
            return self.sellers[step].controller
        if not is_seller and self.buyers[step].controller is not None:
            return self.buyers[step].controller
        controller = StepController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            step=step,
            urange=urange,
            product=self.awi.my_output_product
            if is_seller
            else self.awi.my_input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
            horizon=self._horizon,
            negotiations_concluded_callback=functools.partial(
                self.__class__.all_negotiations_concluded, self
            ),
            parent_name=self.agent.name,
            awi=self.awi,
        )
        if is_seller:
            assert self.sellers[step].controller is None
            self.sellers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        else:
            assert self.buyers[step].controller is None
            self.buyers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        return controller

    def _get_controller(self, mechanism) -> StepController:
        neg = self._running_negotiations[mechanism.id]
        return neg.negotiator.parent

    def all_negotiations_concluded(  # Used by add controller
        self, controller_index: int, is_seller: bool
    ) -> None:
        """Called by the `StepController` to affirm that it is done negotiating for some time-step"""
        info = (
            self.sellers[controller_index]
            if is_seller
            else self.buyers[controller_index]
        )
        info.done = True
        c = info.controller
        if c is None:
            return
        quantity = c.secured
        target = c.target
        time_range = info.time_range
        if is_seller:
            controllers = self.sellers
        else:
            controllers = self.buyers

        controllers[controller_index].controller = None
        if quantity <= target:
            self.start_negotiations(step=controller_index, is_seller=is_seller)
            return

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> int:
        target = 0
        for i in range(steps[0], steps[1]):
            target += self._qrange(i, sell)[1]

        return target
