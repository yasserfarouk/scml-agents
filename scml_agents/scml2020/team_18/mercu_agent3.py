"""
**Submitted to ANAC 2020 SCML**
*Authors* KAKUTANI Kazuto kakutani.kazuto@otsukalab.nitech.ac.jp


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2020 SCML.

This module implements a factory manager for the SCM 2020 league of ANAC 2019
competition. This version will not use subcomponents. Please refer to the
[game description](http://www.yasserm.com/scml/scml2020.pdf) for all the
callbacks and subcomponents available.

Your agent can learn about the state of the world and itself by accessing
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your agent can act in the world by calling methods in the AWI it has.
For example:

# requests a negotiation with one partner
- *self.awi.request_negotiation(...)*
- *self.awi.request_negotiations(...)* # requests a set of negotiations


You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list
  [here](https://scml.readthedocs.io/en/latest/api/scml.scml2020.AWI.html)

"""

# required for running the test tournament
import time
import numpy as np
import math
import functools
import random
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable, Union, Tuple, Callable, Type
from abc import abstractmethod
from dataclasses import dataclass, asdict
from pprint import pformat
from tabulate import tabulate

from scml.scml2020 import SCML2020Agent
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
    IndDecentralizingAgent,
    MovingRangeAgent,
    ReactiveAgent,
)
from scml.scml2020.utils import anac2020_collusion, anac2020_std
from scml.scml2020 import Failure
from scml.scml2020.common import is_system_agent, ANY_LINE, NO_COMMAND, TIME, QUANTITY
from scml.scml2020.components.production import ProductionStrategy
from scml.scml2020.components.negotiation import (
    NegotiationManager,
    StepNegotiationManager,
)
from scml.scml2020.components.trading import TradingStrategy
from scml.scml2020.components.prediction import (
    ExecutionRatePredictionStrategy,
    MeanERPStrategy,
)
from scml.scml2020.services.controllers import StepController, SyncController

from negmas import LinearUtilityFunction
from negmas import (
    AspirationMixin,
    AgentWorldInterface,
    PassThroughNegotiator,
    ResponseType,
    SAONegotiator,
    AspirationNegotiator,
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    UtilityFunction,
)
from negmas.sao import SAOController
from negmas.events import Notifier, Notification
from negmas.helpers import humanize_time, get_class, instantiate

__all__ = ["MercuAgent"]


class MyNegotiator(SAONegotiator):
    """
    A negotiator that acts as an end point to a parent Controller.

    This negotiator simply calls its controler for everything.
    """

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.propose(self.id, state)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.respond(self.id, state, offer)
        return ResponseType.REJECT_OFFER

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.on_negotiation_start(self.id, state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.on_negotiation_end(self.id, state)

    def join(self, ami, state, *, ufun=None, role="agent",) -> bool:
        """
        Joins a negotiation.

        Remarks:
            This method first gets permission from the parent controller by calling `before_join` on it and confirming
            the result is `True`, it then joins the negotiation and calls `after_join` of the controller to inform it
            that joining is completed if joining was successful.
        """
        permission = (
            self._Negotiator__parent is None
            or self._Negotiator__parent.before_join(
                self.id, ami, state, ufun=ufun, role=role
            )
        )
        if not permission:
            return False
        if super().join(ami, state, ufun=ufun, role=role):
            if self._Negotiator__parent:
                self._Negotiator__parent.after_join(
                    self.id, ami, state, ufun=ufun, role=role
                )
            return True
        return False


class MyController(SAOController, AspirationMixin, Notifier):
    def __init__(
        self,
        *args,
        target_quantity: int,
        is_seller: bool,
        agent_confidence: Dict[str, int],
        step: int,
        urange: Tuple[int, int],
        product: int,
        partners: List[str],
        negotiator_type: SAONegotiator,
        horizon: int,
        awi: AgentWorldInterface,
        parent_name: str,
        negotiations_concluded_callback: Callable[[int, bool], None],
        negotiator_params: Dict[str, Any] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_name = parent_name
        self.awi = awi
        self.horizon = horizon
        self.negotiations_concluded_callback = negotiations_concluded_callback
        self.is_seller = is_seller
        self.target = target_quantity
        self.urange = urange
        self.partners = partners
        self.product = product
        negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.secured = 0
        if is_seller:
            self.ufun = LinearUtilityFunction((1, 1, 10))
        else:
            self.ufun = LinearUtilityFunction((1, -1, -10))
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)
        self.completed = defaultdict(bool)
        self.agent_confidence = agent_confidence
        self.step = step
        self.retries: Dict[str, int] = defaultdict(int)
        self.max_retries = max_retries

    def join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        joined = super().join(negotiator_id, ami, state, ufun=ufun, role=role)
        if joined:
            self.completed[negotiator_id] = False
        return joined

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.propose(state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> ResponseType:
        # 必要数量達成済み
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.respond(offer=offer, state=state)

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.awi.current_step}] "
            f"secured {self.secured} of {self.target} for {self.parent_name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[MyNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> MyNegotiator:
        neg = super().create_negotiator(negotiator_type, name, cntxt, **kwargs)
        self.completed[neg.id] = False
        return neg

    def time_range(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.step),
                min(step + self.horizon, self.awi.n_steps - 1),
            )
        return self.step, step - 1

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_end(negotiator_id, state)
        agreement = state.agreement
        # mark this negotiation as completed
        self.completed[negotiator_id] = True
        # if there is an agreement increase the secured amount and check if we are done.
        if agreement is not None:
            self.secured += agreement[QUANTITY]
            if self.secured >= self.target:
                self.awi.loginfo(f"Ending all negotiations on controller {str(self)}")
                # If we are done, end all other negotiations
                for k in self.negotiators.keys():
                    if self.completed[k]:
                        continue
                    self.notify(
                        self.negotiators[k][0], Notification("end_negotiation", None)
                    )
        self.kill_negotiator(negotiator_id, force=True)
        if all(self.completed.values()):
            # If we secured everything, just return control to the agent
            if self.secured >= self.target:
                self.awi.loginfo(f"Secured Everything: {str(self)}")
                self.negotiations_concluded_callback(
                    self.awi.current_step, self.is_seller
                )
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.time_range(self.awi.current_step, self.is_seller)

            if self.awi.current_step < tmax + 1 and tmin <= tmax:
                # get a good partner: one that was not retired too much
                random.shuffle(self.partners)
                for other in self.partners:
                    if self.retries[other] <= self.max_retries:
                        partner = other
                        break
                else:
                    return
                self.retries[partner] += 1
                neg = self.create_negotiator()
                self.completed[neg.id] = False
                self.awi.loginfo(
                    f"{str(self)} negotiating with {partner} on u={self.urange}"
                    f", q=(1,{self.target-self.secured}), u=({tmin}, {tmax})"
                )
                self.awi.request_negotiation(
                    not self.is_seller,
                    product=self.product,
                    quantity=(1, self.target - self.secured),
                    unit_price=self.urange,
                    time=(tmin, tmax),
                    partner=partner,
                    negotiator=neg,
                    extra=dict(
                        controller_index=self.awi.current_step, is_seller=self.is_seller
                    ),
                )

    def update_param(self, target_quantity, urange, agent_confidence):
        self.target = target_quantity
        self.urange = urange
        self.agent_confidence = agent_confidence


@dataclass
class ControllerInfo:
    """Keeps a record of information about one of the controllers used by the agent"""

    controller: MyController
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    agent_confidence: Dict[str, int]
    step: int
    done: bool = False


class MyNegotiationManager(NegotiationManager):
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
        self.buyer = self.seller = None

    def init(self):
        super().init()

        # initialize one controller for buying and another for selling
        self.buyer = ControllerInfo(None, False, tuple(), 0, dict(), 0, False)
        self.seller = ControllerInfo(None, True, tuple(), 0, dict(), 0, False)

        self.awi.logdebug_agent(f"Initialized\n{pformat(self.internal_state)}")

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
        if sell:
            expected_quantity = int(math.floor(
                qvalues[1] * self._agent_confidence))
        else:
            expected_quantity = int(math.floor(
                qvalues[1] * self._agent_confidence))
        """

        # negotiate with everyone
        controller = self.add_controller(
            sell, qvalues[1], uvalues, self.agent_confidence, self.awi.current_step + 1
        )
        self.awi.loginfo_agent(
            f"Requesting {'selling' if sell else 'buying'} negotiation "
            f"on u={uvalues}, q={qvalues}, t={tvalues}"
            f" with {str(partners)} using {str(controller)}"
        )
        # partnerを信頼度順に並び替え
        ac_bind = dict()
        for pt in partners:
            if self.agent_confidence[pt] is not None:
                ac_bind[pt] = self.agent_confidence[pt]

        agent_confidence_sorted = sorted(
            ac_bind.items(), key=lambda x: x[1], reverse=True
        )
        sorted_partners = list()
        # partnerの信頼度が低いなら交渉しない．
        for acs in agent_confidence_sorted:
            if acs[1] > 0.3:
                sorted_partners.append(acs[0])

        self.awi.request_negotiations(
            is_buy=not sell,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=sorted_partners,
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
        controller_info = self.seller if is_seller else self.buyer
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        target = self.target_quantities((tmin, tmax), is_seller).sum()
        if target <= 0:
            return None
        self.awi.loginfo_agent(
            f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
            f"({Issue.num_outcomes(mechanism.issues)})"
        )
        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.add_controller(
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                self.agent_confidence,
                self.awi.current_step + 1,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator()

    def all_negotiations_concluded(
        self, controller_index: int, is_seller: bool
    ) -> None:
        """Called by the `StepController` to affirm that it is done negotiating for some time-step"""
        info = self.seller if is_seller else self.buyer
        info.done = True
        c = info.controller
        if c is None:
            return
        quantity = c.secured
        target = c.target
        time_range = info.time_range
        if is_seller:
            controllers = self.seller
        else:
            controllers = self.buyer

        self.awi.logdebug_agent(f"Killing Controller {str(controllers.controller)}")
        controllers.controller = None
        if quantity <= target:
            self._generate_negotiations(step=controller_index, sell=is_seller)
            return

    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        agent_confidence: Dict[str, int],
        next_step: int,
    ) -> MyController:
        # すでにそのステップで生成されたAgentが存在するならreturn
        if is_seller and self.seller.controller is not None:
            if next_step != self.seller.step == next_step:
                self.seller.controller.update_param(target, urange, agent_confidence)
                self.seller = ControllerInfo(
                    self.seller.controller,
                    is_seller,
                    self._trange(next_step, is_seller),
                    target,
                    agent_confidence,
                    next_step,
                    False,
                )
            return self.seller.controller
        if not is_seller and self.buyer.controller is not None:
            if next_step != self.buyer.step:
                self.buyer.controller.update_param(target, urange, agent_confidence)
                self.buyer = ControllerInfo(
                    self.buyer.controller,
                    is_seller,
                    self._trange(next_step, is_seller),
                    target,
                    agent_confidence,
                    next_step,
                    False,
                )
            return self.buyer.controller
        # controller 生成
        controller = MyController(
            is_seller=is_seller,
            agent_confidence=agent_confidence,
            step=next_step,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
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
        # set seller or buyer
        # ! delete _trange
        if is_seller:
            assert self.seller.controller is None
            self.seller = ControllerInfo(
                controller,
                is_seller,
                self._trange(next_step, is_seller),
                target,
                agent_confidence,
                next_step,
                False,
            )
        else:
            assert self.buyer.controller is None
            self.buyer = ControllerInfo(
                controller,
                is_seller,
                self._trange(next_step, is_seller),
                target,
                agent_confidence,
                next_step,
                False,
            )
        return controller

    def _get_controller(self, mechanism) -> MyController:
        neg = self._running_negotiations[mechanism.id]
        return neg.negotiator.parent


class MyTradePredictionStrategy:
    def __init__(
        self,
        *args,
        predicted_outputs: Union[int, np.ndarray] = None,
        predicted_inputs: Union[int, np.ndarray] = None,
        add_trade=True,
        agent_confidence: Dict[str, int] = dict(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.expected_outputs = predicted_outputs
        # Expected output quantity every step
        self.expected_inputs = predicted_inputs
        # Expected input quantity every step
        self.input_cost: np.ndarray = None
        # Expected unit price of the input
        self.output_price: np.ndarray = None
        # Expected unit price of the output
        self._add_trade = add_trade
        self.agent_confidence = agent_confidence

    def trade_prediction_init(self) -> None:
        """Will be called to update expected_outputs, expected_inputs,
        input_cost, output_cost during init()"""
        inp = self.awi.my_input_product

        def adjust(x, demand):
            #
            if x is None:
                # default value
                if self.awi.my_suppliers[0] == "SELLER":
                    # SELLERならライン数分購入し，販売する
                    x = max(1, self.awi.n_lines)
                else:
                    # 初期はライン数の半分購入し販売すると予測する．
                    x = max(1, self.awi.n_lines // 2)
            elif isinstance(x, Iterable):
                # filled
                return np.array(x)

            # predict(unimplement)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)

            # set 0
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

        # set input cost to catalog_price
        self.input_cost = self.awi.catalog_prices[self.awi.my_input_product] * np.ones(
            self.awi.n_steps
        )
        # set output price to catalog_price
        self.output_price = self.awi.catalog_prices[
            self.awi.my_input_product
        ] * np.ones(self.awi.n_steps)

    def trade_prediction_step(self) -> None:
        input_cor_value, output_cor_value = 0, 0  # correction value 補正値
        cur_step = self.awi.current_step

        # evaluate expected_quantity (today)
        # 今日の取引数量と，予測数量を比較し，その差によって補正値を設定
        input_diff = self.expected_inputs[cur_step] - self.inputs_secured[cur_step]
        output_diff = self.expected_outputs[cur_step] - self.outputs_secured[cur_step]
        if input_diff > 0:
            input_cor_value = 0
        elif input_diff == 0:
            input_cor_value = 2
        else:
            input_cor_value = -2

        if output_diff > 0:
            output_cor_value = 0
        elif output_diff == 0:
            output_cor_value = 2
        else:
            output_cor_value = -2

        # The secured average quantity is the expected quantity
        inp_sec_ave, oup_sec_ave = 0, 0
        inp_not_zero_num, oup_not_zero_num = 0, 0
        for inps in self.inputs_secured:
            if inps:
                inp_not_zero_num += 1
                inp_sec_ave += inps
            else:
                continue
        for oups in self.outputs_secured:
            if oups:
                oup_not_zero_num += 1
                oup_sec_ave += oups
            else:
                continue
        # calculate average
        if inp_not_zero_num == 0:
            # if no contract, set fixed quantity
            inp_sec_ave = self.awi.n_lines // 2
        else:
            inp_sec_ave = min(
                (inp_sec_ave / inp_not_zero_num) + input_cor_value, self.awi.n_lines
            )
            inp_sec_ave = max(1, inp_sec_ave)
        if oup_not_zero_num == 0:
            oup_sec_ave = self.awi.n_lines // 2
        else:
            oup_sec_ave = min(
                (oup_sec_ave / oup_not_zero_num) + output_cor_value, self.awi.n_lines
            )
            oup_sec_ave = max(1, oup_sec_ave)

        if self.awi.my_consumers[0] == "SELLER":
            inp_sec_ave = self.awi.n_lines
            oup_sec_ave = self.awi.n_lines

        # expected_inputs quantity is the average inputs_secured quantity (予想入荷量は、契約された入荷量の平均)
        self.expected_inputs[cur_step:] = inp_sec_ave
        self.expected_outputs[cur_step:] = oup_sec_ave

        self.expected_inputs[self.awi.my_input_product - self.awi.n_processes :] = 0
        self.expected_outputs[: self.awi.my_input_product + 1] = 0

    def estimate_confidence(self):
        # consumer, supplierごとに予測
        for aid in self.awi.my_consumers:
            self.agent_confidence[aid] = 1
            report = self.awi.reports_of_agent(aid)
            if report is None:
                continue
            over_n = self.awi.current_step % 5
            report_num = 0
            if over_n != 0:
                report_num = self.awi.current_step - self.awi.current_step % 5
            else:
                report_num = self.awi.current_step - 5

            self.agent_confidence[aid] = 1 - report[report_num].breach_prob

        for aid in self.awi.my_suppliers:
            self.agent_confidence[aid] = 1
            report = self.awi.reports_of_agent(aid)
            if report is None:
                continue
            over_n = self.awi.current_step % 5
            report_num = 0
            if over_n != 0:
                report_num = self.awi.current_step - self.awi.current_step % 5
            else:
                report_num = self.awi.current_step - 5

            self.agent_confidence[aid] = 1 - report[report_num].breach_prob

    @property
    def intenal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
                "agent_confidence": self.agent_confidence,
            }
        )
        return state

    def init(self):
        # initialize input_cost and output_price with catalog_price
        self.input_cost = self.awi.catalog_prices[self.awi.my_input_product] * np.ones(
            self.awi.n_steps, dtype=int
        )
        self.output_price = self.awi.catalog_prices[
            self.awi.my_output_product
        ] * np.ones(self.awi.n_steps, dtype=int)

        # initialize expected_inputs and expected_outputs
        self.trade_prediction_init()
        self.estimate_confidence()
        super().init()

    def step(self):
        self.trade_prediction_step()
        self.estimate_confidence()
        super().step()


class MyTradingStrategy(MyTradePredictionStrategy, TradingStrategy):
    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]

    def step(self):
        super().step()
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        self.outputs_needed[1:] = self.expected_inputs[:-1]
        if (
            self.awi.n_steps - (self.awi.n_processes - self.awi.my_input_product)
            >= self.awi.current_step
        ):
            # 終盤 在庫が余っているなら全て売るよう努力する
            # ? 在庫数わからない？ とりあえずライン数
            self.outputs_needed[self.awi.current_step :] = self.awi.n_lines

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        self.awi.logdebug_agent(
            f"Enter Contracts Finalized:\n"
            f"Signed {pformat([self._format(_) for _ in signed])}\n"
            f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
            f"{pformat(self.internal_state)}"
        )
        super().on_contracts_finalized(signed, cancelled, rejectors)
        total_input_price, total_output_price = 0, 0
        seller_num, buyer_num = 0, 0
        current_step = self.awi.current_step
        for contract in signed:
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            product = contract.annotation["product"]
            produce_cost = self.awi.profile.costs[0][self.awi.my_input_product]
            your_agent = contract.annotation["seller"]
            if your_agent == self.id:
                your_agent = contract.annotation["buyer"]

            my_report = self.awi.reports_of_agent(self.id)
            your_report = self.awi.reports_of_agent(your_agent)
            """
            if my_report is None or your_report is None:
                pass
            else:
                report_step = 0
                over_n = current_step % 5
                if over_n != 0:
                    report_step = current_step - current_step % 5
                else:
                    report_step = current_step - 5
                my_report_latest_step = my_report[report_step]
                your_report_latest_step = my_report[report_step]
                with open('scml2020Mercu.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([self.id, your_agent, q, u, t,
                                     self.awi.catalog_prices[product], produce_cost, your_report_latest_step.breach_prob, your_report_latest_step.breach_level, my_report_latest_step.cash, my_report_latest_step.assets])
            """
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id

            """
            your_agent_report = self.awi.reports_of_agent(
                contract.annotation["caller"])
            print(your_agent_report)
            my_agent_report = self.awi.reports_of_agent(
                self.id)[self.awi.current_step-1]
            """
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                self.outputs_secured[t] += q
                total_output_price += u
                seller_num += 1
            else:
                # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
                input_product = contract.annotation["product"]
                output_product = input_product + 1
                self.inputs_secured[t] += q
                total_input_price += u
                buyer_num += 1

        # predict output_price
        if seller_num:
            self.output_price[self.awi.current_step :] = (
                total_output_price / seller_num + 1
            )
        else:
            self.output_price = self.output_price
        # predict input_cost
        if buyer_num:
            self.input_cost[self.awi.current_step :] = total_input_price / buyer_num + 1
        else:
            self.input_cost = self.input_cost

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
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
            ),
        )
        sold, bought = 0, 0  # 今回のステップで販売(購入)した製品の数
        s = self.awi.current_step
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle
            if t < s and len(contract.issues) == 3:
                continue

            # check unit price不利益すぎる不当な契約はしない
            if is_seller and self.output_price[t] * (1 / 10) > u:
                continue
            elif self.input_cost[t] * 10 < u:
                continue

            agent_confidence = 1
            if is_seller:
                agent_confidence = self.agent_confidence[contract.annotation["buyer"]]
            else:
                agent_confidence = self.agent_confidence[contract.annotation["seller"]]
            # 信頼度が低い場合，署名しない
            if agent_confidence < 0.4:
                continue

            if is_seller:
                trange = (s, t)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            # I cant produce product, so i dont sign this contract
            if len(steps) - taken < q:
                continue
            # 今回のステップで契約した量がニーズを超えていないか
            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
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


class MyProductionStrategy(ProductionStrategy):
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
        super().on_contracts_finalized(signed, cancelled, rejectors)
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
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


class MercuAgent(
    MyNegotiationManager, MyTradingStrategy, MyProductionStrategy, SCML2020Agent
):
    def target_quantity(self, step: int, sell: bool) -> int:
        """A fixed target quantity of half my production capacity"""
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1))
        return LinearUtilityFunction((0, -0.5, -0.8))


def run(
    competition="std",
    reveal_names=True,
    n_steps=40,
    n_configs=4,
    max_n_worlds_per_config=None,
    n_runs_per_world=1,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std, 
                     collusion).        
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try. 
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value        

    """
    competitors = [MercuAgent, DecentralizingAgent, BuyCheapSellExpensiveAgent]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    run()
