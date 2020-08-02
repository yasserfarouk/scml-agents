# required for typing
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

# my need
from scml.scml2020.components.negotiation import *
from scml.scml2020 import *
from negmas import *
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import seaborn as sns
from .prediction import MyERPredictor


class NewNegotiationManager(MyERPredictor):
    """
    NegotiationManager
    StepNegotiationManager  # StepController
    *IndependentNegotiationsManager  # Controllerを使わない
    MovingRangeNegotiationManager  # SyncController
    """

    def __init__(
        self,
        *args,
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.controllers: Dict[bool, SyncController] = {
            False: SyncController(
                is_seller=False,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
            True: SyncController(
                is_seller=True,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
        }
        self._current_end = -1
        self._current_start = -1

    def step(self):
        super().step()
        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return
        for seller, needed, secured, product in [
            (False, self.inputs_needed, self.inputs_secured, self.awi.my_input_product),
            (
                True,
                self.outputs_needed,
                self.outputs_secured,
                self.awi.my_output_product,
            ),
        ]:
            needs = np.max(
                needed[self._current_start : self._current_end]
                - secured[self._current_start : self._current_end]
            )
            if needs < 1:
                continue

            production_cost = np.max(
                self.awi.profile.costs[0, self.awi.my_input_product]
            )
            if seller:
                min_price = (  # acceptable_unit_price
                    # self.awi.catalog_prices[self.awi.my_input_product]
                    # + self.awi.profile.costs[0, self.awi.my_input_product]
                    production_cost
                    + self.input_cost[
                        step
                    ]  # そのステップにおける仕入れの予測値と，生産コストより良ければ売る（いくらで仕入れたかは考慮してない？）
                )
                price_range = (min_price, 2 * min_price)  # _urange()と同じ
            else:
                price_range = (
                    0,
                    # min(
                    #     self.awi.catalog_prices[self.awi.my_input_product],
                    #     self.awi.catalog_prices[self.awi.my_output_product]
                    #     - self.awi.profile.costs[0, product],
                    # ),
                    self.output_price[step]
                    - production_cost,  # そのステップにおける売却予測値から，生産コストを差し引いてそれより良ければ買う（現ステップでいくらで取引されてるかは考慮してない？）
                )
            if price_range[0] >= price_range[1]:
                continue
            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=self.controllers[seller],
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        controller = self.controllers[not annotation["is_buy"]]
        if controller is None:
            return None
        return controller.create_negotiator()


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


class MyNegotiationManager(MyERPredictor, NegotiationManager):
    """
    NegotiationManager
    *StepNegotiationManager
    IndependentNegotiationsManager
    MovingRangeNegotiationManager
    """

    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # save construction parameters
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        # attributes that will be read during init() from the AWI
        # -------------------------------------------------------
        self.buyers = self.sellers = None
        """Buyer controllers and seller controllers. Each of them is responsible of covering the
        needs for one step (either buying or selling)."""

    def init(self):
        super().init()

        # initialize one controller for buying and another for selling for each time-step
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(
                None, i, False, tuple(), 0, 0, False
            )  # controller, time_step, is_seller, time_range, target, expected, done
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]

        # self.awi.logdebug_agent(f"Initialized\n{pformat(self.internal_state)}")

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
        if sell:
            expected_quantity = int(math.floor(qvalues[1] * self._execution_fraction))
        else:
            expected_quantity = int(math.floor(qvalues[1] * self._execution_fraction))

        # negotiate with everyone
        controller = self.add_controller(
            sell, qvalues[1], uvalues, expected_quantity, step
        )
        # self.awi.loginfo_agent(
        #     f"Requesting {'selling' if sell else 'buying'} negotiation "
        #     f"on u={uvalues}, q={qvalues}, t={tvalues}"
        #     f" with {str(partners)} using {str(controller)}"
        # )
        self.awi.request_negotiations(
            is_buy=not sell,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,  # これのリストを部分的に渡せば，交渉相手を制限できそう？
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
        is_seller = annotation["seller"] == self.id
        tmin, tmax = issues[TIME].min_value, issues[TIME].max_value + 1
        # find the time-step for which this negotiation should be added
        step = max(0, tmin - 1) if is_seller else min(self.awi.n_steps - 1, tmax + 1)
        # find the corresponding controller.
        controller_info: ControllerInfo
        controller_info = self.sellers[step] if is_seller else self.buyers[step]
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        target = self.target_quantities((tmin, tmax + 1), is_seller).sum()
        if target <= 0:
            return None
        # self.awi.loginfo_agent(
        #     f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
        #     f"({Issue.num_outcomes(mechanism.issues)})"
        # )
        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.add_controller(
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                int(target),
                step,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator()

    def all_negotiations_concluded(
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

        # self.awi.logdebug_agent(
        #     f"Killing Controller {str(controllers[controller_index].controller)}"
        # )
        controllers[controller_index].controller = None
        if quantity <= target:
            self._generate_negotiations(step=controller_index, sell=is_seller)
            return

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
            parent_name=self.name,
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
