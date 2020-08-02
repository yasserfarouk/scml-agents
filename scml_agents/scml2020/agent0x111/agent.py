import collections  # 7.6.20 17:00
import functools
import math
import random
import time
from abc import abstractmethod
from collections import defaultdict
from statistics import median
from typing import Tuple, Union, List, Optional, Any, Dict, Type, Callable

import numpy as np
from negmas import (
    AspirationMixin,
    LinearUtilityFunction,
    PassThroughNegotiator,
    MechanismState,
    ResponseType,
    UtilityFunction,
    AgentWorldInterface,
    Outcome,
)
from negmas import AspirationNegotiator, Issue, Negotiator, Contract
from negmas import Breach
from negmas.common import AgentMechanismInterface
from negmas.events import Notifier, Notification
from negmas.helpers import get_class
from negmas.helpers import humanize_time
from negmas.helpers import instantiate
from negmas.sao import (
    SAOController,
    SAONegotiator,
)
from scml.scml2020 import *
from scml.scml2020 import SCML2020Agent, DecentralizingAgent
from scml.scml2020.common import TIME
from scml.scml2020.components.negotiation import ControllerInfo
from scml.scml2020.utils import anac2020_std, anac2020_collusion
from tabulate import tabulate

__all__ = ["ASMASH"]


class StepController(SAOController, AspirationMixin, Notifier):
    """A controller for managing a set of negotiations about selling or buying (but not both)  starting/ending at some
    specific time-step.
    Args:
        target_quantity: The quantity to be secured
        is_seller:  Is this a seller or a buyer
        parent_name: Name of the parent
        horizon: How many steps in the future to allow negotiations for selling to go for.
        step:  The simulation step that this controller is responsible about
        urange: The range of unit prices used for negotiation
        product: The product that this controller negotiates about
        partners: A list of partners to negotiate with
        negotiator_type: The type of the single negotiator used for all negotiations.
        negotiator_params: The parameters of the negotiator used for all negotiations
        max_retries: How many times can the controller try negotiating with each partner.
        negotiations_concluded_callback: A method to be called with the step of this controller and whether is it a
                                         seller when all negotiations are concluded
        *args: Position arguments passed to the base Controller constructor
        **kwargs: Keyword arguments passed to the base Controller constructor
    Remarks:
        - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
        - Once it accumulates the required quantity, it ends all remaining negotiations
        - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
          instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.
    """

    def __init__(
        self,
        *args,
        target_quantity: int,
        is_seller: bool,
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
        # issues = [
        #   Issue(qvalues, name="quantity"),
        #   Issue(tvalues, name="time"),
        #   Issue(uvalues, name="uvalues"),
        # ]

        # ratio= self.get_ratio_of_suspects()
        # print(str("The ratio between all partners and suspects in step {} is: {}").format(step,ratio))

        if is_seller:
            self.ufun = LinearUtilityFunction((1, 1, 10))

        else:

            self.ufun = LinearUtilityFunction((1, -1, -10))

        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)
        self.completed = defaultdict(bool)
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
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.respond(offer=offer, state=state)

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
            f"secured {self.secured} of {self.target} for {self.parent_name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> PassThroughNegotiator:
        neg = super().create_negotiator(negotiator_type, name, cntxt, **kwargs)
        self.completed[neg.id] = False
        return neg

    def time_range(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self.horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

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
                self.negotiations_concluded_callback(self.step, self.is_seller)
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.time_range(self.step, self.is_seller)

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
                    extra=dict(controller_index=self.step, is_seller=self.is_seller),
                )


class ElaboratedMeanERPStrategy(ExecutionRatePredictionStrategy):
    """
    Predicts that the there is a fixed execution rate that does not change for all partners
    Args:
        execution_fraction: The expected fraction of any contract's quantity to be executed
    Provides:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract
    Hooks Into:
        - `internal_state`

        - `init`
        - `on_contract_executed`
        - `on_contract_breached`
    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    def __init__(self, *args, execution_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_breaches = []
        self.suspects_to_bankrupt = {}
        self.steps_of_suspicious = {}
        self._execution_fraction = execution_fraction
        self._total_quantity = None
        self.Contracts_situation = []
        self.executed_quantities = []

    def predict_quantity(self, contract: Contract):
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
        element = (self.awi.current_step, "Contract executed")

        self.Contracts_situation.append(element)
        if len(self.Contracts_situation) > 100:
            del self.Contracts_situation[0]
        old_total = self._total_quantity
        q = contract.agreement["quantity"]
        self.executed_quantities.append(q)
        if len(self.executed_quantities) > 100:
            del self.executed_quantities[0]
        self._total_quantity += q
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity

        # self.awi.logdebug(str("Contract executed,the execution fraction is")+str(self._execution_fraction))
        # print(str("Contract executed,the execution fraction is")+str(self._execution_fraction))
        # self.upgrade_exe_fraction(q)

    @abstractmethod
    def get_bad_situation_partners(self):
        raise ValueError("You must implement get_bad_situation_partners")

    @abstractmethod
    def upgrade_exe_fraction(self):
        raise ValueError("You must implement upgrade_exe_fraction")

    @abstractmethod
    def check_seq_of_breaches(self):
        raise ValueError("You must implement check_conseq_of_breaches")

    @abstractmethod
    def check_conseq_of_execution(self):
        raise ValueError("You must implement check_conseq_of_execution")

    @abstractmethod
    def report_money_breach(self, buyer_id):
        raise ValueError("You must implement report_money_breach")

    @abstractmethod
    def check_breaches_strategy(
        self, breach_situation: bool
    ):  # update exe fraction if the rate of the breaching is big
        raise ValueError("You must implement update_exe_fractio")

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        seller_id = contract.annotation["seller"]
        buyer_id = contract.annotation["buyer"]
        the_perpetrator = breaches[0].perpetrator
        # print("step: "+str(self.awi.current_step))
        # print("buyer is:"+buyer_id)
        # print("The perprator: "+the_perpetrator )
        type = breaches[0].type
        # print("The type: "+type)
        # if(type in "money"):
        #    print("Due to *money* breach the agent {} reported".format(buyer_id))
        #    self.report_money_breach(buyer_id)

        self.my_breaches.append((self.awi.current_step, the_perpetrator))

        element = (self.awi.current_step, "Contract breached")
        self.Contracts_situation.append(element)
        if len(self.Contracts_situation) > 100:
            del self.Contracts_situation[0]

        old_total = self._total_quantity
        q = int(contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches)))
        self._total_quantity += contract.agreement["quantity"]
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
        self._total_quantity += contract.agreement["quantity"]
        self._execution_fraction = (
            self._execution_fraction * old_total - q
        ) / self._total_quantity
        if self._execution_fraction <= 0:
            self._execution_fraction = 0
        # print(str("Contract breached,the execution fraction is")+str(self._execution_fraction))
        self.check_breaches_strategy(breach_situation=True)


class ElaboratedPredictionBasedTradingStrategy(
    FixedTradePredictionStrategy, ElaboratedMeanERPStrategy, TradingStrategy
):
    """A trading strategy that uses prediction strategies to manage inputs/outputs needed
    Hooks Into:
        - `init`
        - `on_contracts_finalized`
        - `sign_all_contracts`
        - `on_agent_bankrupt`
    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]
        self.agent_bankrupt = []

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # self.awi.logdebug_agent(
        #     f"Enter Contracts Finalized:\n"
        #     f"Signed {pformat([self._format(_) for _ in signed])}\n"
        #     f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
        #     f"{pformat(self.internal_state)}"
        # )
        super().on_contracts_finalized(signed, cancelled, rejectors)
        consumed = 0
        for contract in signed:
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                self.outputs_secured[t] += q
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, lines = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    if contract.annotation["caller"] != self.id:
                        # this is a sell contract that I did not expect yet. Update needs accordingly
                        self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.inputs_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                # this is a buy contract that I did not expect yet. Update needs accordingly
                self.outputs_needed[t + 1] += max(1, q)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
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
        sold, bought = 0, 0
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
            if is_seller:
                trange = (s, t)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
                # VER 10
                # if contract.annotation["buyer"] in self.agent_bankrupt:
                #    print("The contract with {} deleted due to money breach".format(contract.annotation["buyer"]))
                #    continue
                # VER 10
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue

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
        self.agent_bankrupt.append(agent)
        # print("The agent: {} declared bankruptcy in step: {}".format(agent,self.awi.current_step))
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


class ElaboratedStepNegotiationManager(ElaboratedMeanERPStrategy, NegotiationManager):
    """
    A negotiation manager that controls a controller and another for selling for every timestep
    Args:
        negotiator_type: The negotiator type to use to manage all negotiations
        negotiator_params: Paramters of the negotiator
    Provides:
        - `all_negotiations_concluded`
    Requires:
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`
    Hooks Into:
        - `init`
        - `respond_to_negotiation_request`
    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    # negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,

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
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
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

        self.del_no_relevant_suspects()
        partners = self.good_situation_partners(partners)
        # partners=self.get_reliable_partners(partners)
        # print("The reliable partners i want to negotiate: {}".format(partners))
        # if(len(new_partners)>=2):
        #    partners=new_partners

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

    def del_no_relevant_suspects(self):
        tmp = list()
        for (x, y) in self.steps_of_suspicious.items():
            if y - self.awi.current_step > 5:
                tmp.append(x)
        for x in tmp:
            del self.steps_of_suspicious[x]
            del self.suspects_to_bankrupt[x]
        # if(len(tmp)>0):
        # print("{} suspects returned to be normal".format(len(tmp)))

    @abstractmethod
    def get_reliable_partners(
        self,
        partners: List[str],
        threshold_of_suspects=0.8,
        ratio_between_partners_and_suspects=0.3,
    ):
        raise ValueError("You must implement get_reliable_partners")

    @abstractmethod
    def good_situation_partners(self, partners: List[str]):
        raise ValueError("You must implement good_situation_partners")

    @abstractmethod
    def get_breach_dic(self):
        raise ValueError("You must implement get_breach_dic")

    @abstractmethod
    def get_suspected_partners(self, paramater: float):
        raise ValueError("You must implement get_suspected_partners")

    @abstractmethod
    def get_bankrupt_agent(self):
        raise ValueError("You must implement get_bankrupt_agent")

    @abstractmethod
    def recognize_breaches_strategy(
        self,
    ):  ####################################################
        raise ValueError("You must implement recognize_breaches_strategy")

    def respond_to_negotiation_request(  ########################################################################################################################
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        self.del_no_relevant_suspects()

        Breaches_strategy_partners = list(self.suspects_to_bankrupt.keys())

        if Breaches_strategy_partners != None:
            if annotation["buyer"] in Breaches_strategy_partners:
                # print("Response Negative")
                return None

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


class _NegotiationCallbacks:
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class ASMASH(
    _NegotiationCallbacks,
    ElaboratedStepNegotiationManager,
    ElaboratedPredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def get_len_of_seq_of_breaches(self):
        return 4

    def check_seq_of_breaches(self):
        len_of_seq = self.get_len_of_seq_of_breaches()
        temp = False
        if len(self.Contracts_situation) >= len_of_seq:
            last_elements = self.Contracts_situation[-len_of_seq:]
            if self.awi.current_step == last_elements[-1][0]:
                temp = True
                for i in range(0, len(last_elements) - 1):
                    if (
                        ((last_elements[i + 1][0] - last_elements[i][0]) > 1)
                        or ("breach" not in last_elements[i][1])
                        or ("breach" not in last_elements[i + 1][1])
                    ):
                        temp = False
                        break

        return temp

    def get_len_of_seq_of_executions(self):
        return 12

    def check_conseq_of_execution(self):
        size_of_conseq = self.get_len_of_seq_of_executions()
        temp = False
        if len(self.Contracts_situation) >= size_of_conseq:
            last_elements = self.Contracts_situation[-size_of_conseq:]
            if self.awi.current_step == last_elements[-1][0]:
                temp = True
                for i in range(0, len(last_elements) - 1):
                    if (
                        ((last_elements[i + 1][0] - last_elements[i][0]) > 1)
                        or ("executed" not in last_elements[i][1])
                        or ("executed" not in last_elements[i + 1][1])
                    ):
                        temp = False
                        break

        return temp

    def upgrade_exe_fraction(self, quantity: float):
        threshold = 0.6
        seq_of_executions = self.check_conseq_of_execution()
        if seq_of_executions != False and self._execution_fraction < threshold:
            old_total = self._total_quantity
            # q = seq_of_executions*mean(self.executed_quantities)
            q = 8 * median(self.executed_quantities) + 2 * quantity

            temp_total = self._total_quantity + q
            self._execution_fraction = (
                self._execution_fraction * old_total + q
            ) / temp_total
            # print(str("Exe fraction upgraded: ")+str(self._execution_fraction))

    def get_attenuation_paramater(self):
        return 0.9

    def check_breaches_strategy(
        self, breach_situation: bool
    ):  # update exe fraction if the rate of the breaching is big
        # attenuation_param=self.get_attenuation_paramater()
        # if(breach_situation==True):
        # if (self._execution_fraction < 0):
        #   self._execution_fraction = 0
        #    return
        # if(self.check_seq_of_breaches()==True and self._execution_fraction>0.15):
        #    self._execution_fraction= self._execution_fraction*attenuation_param
        # self.awi.logdebug(str("Execution fraction updated!")+str( self._execution_fraction))

        # print(str("Execution fraction updated!") + str(self._execution_fraction))
        self.recognize_breaches_strategy()
        # print(str("step {}/{}. The suspects: ").format(self.awi.current_step, self.awi.n_steps) + str(self.suspects_to_bankrupt))

    def get_threshold_of_suspect(self):
        return 0.7

    def get_extended_threshold_of_suspect(self):
        return 0.9

    def good_situation_partners(
        self, partners: List[str],
    ):
        partners_after_filtering = list(set(partners))
        Breaches_strategy_partners = list(self.suspects_to_bankrupt.keys())
        # print("Bad partners: "+str(Breaches_strategy_partners))
        if Breaches_strategy_partners != None:
            # print(Breaches_strategy_partners)
            partners_after_filtering = list(
                set(partners_after_filtering) - set(Breaches_strategy_partners)
            )

        # print(str("Good partners: ")+str(partners_after_filtering))
        return partners_after_filtering

    def get_reliable_partners(
        self,
        partners: List[str],
        threshold_of_suspects=0.85,
        ratio_between_partners_and_suspects=0.3,
    ):
        # print(str("Partners before filtering: ")+str(partners))
        threshold_of_suspects = self.get_threshold_of_suspect()
        extended_threshold_of_suspect = self.get_extended_threshold_of_suspect()
        suspected_partners = self.get_suspected_partners(
            paramater=threshold_of_suspects
        )
        # print("Suspected patners: "+str(suspected_partners))
        partners_after_filtering = list(set(partners) - set(suspected_partners))
        ratio_of_partners = float(
            len(partners_after_filtering) / len(partners)
        )  # the ratio between the
        # if(ratio_of_partners<ratio_between_partners_and_suspects):
        #    suspected_partners = self.get_suspected_partners(paramater=extended_threshold_of_suspect)
        #    partners_after_filtering=list(set(partners) - set(suspected_partners))

        Breaches_strategy_partners = list(self.suspects_to_bankrupt.keys())
        if Breaches_strategy_partners != None:
            # print(Breaches_strategy_partners)
            partners_after_filtering = list(
                set(partners_after_filtering) - set(Breaches_strategy_partners)
            )
        # print(str("Partners after filtering: ")+str(partners_after_filtering))
        return partners_after_filtering

    def get_ratio_of_suspects(self, partners: List[str]):
        threshold_of_suspects = self.get_threshold_of_suspect()
        extended_threshold_of_suspect = self.get_extended_threshold_of_suspect()
        suspected_partners = self.get_suspected_partners(
            paramater=threshold_of_suspects
        )
        partners_after_filtering = list(set(partners) - set(suspected_partners))
        ratio_of_partners = float(
            len(partners_after_filtering) / len(partners)
        )  # the ratio between the suspects and all partners
        return ratio_of_partners

    def get_breach_dic(self):
        IDs = set(self.awi.my_suppliers)
        IDs = list(IDs.union(set(self.awi.my_consumers)))
        if self.awi.current_step != 0:
            reports_by_Step = dict(self.awi.reports_at_step(step=self.awi.current_step))
            new_dict = dict()
            for key, value in reports_by_Step.items():
                new_dict.update({key: value.breach_prob})
            # print(new_dict)
            return new_dict

        return -1

    def get_suspected_partners(self, paramater: float):
        breach_dic = self.get_breach_dic()
        if breach_dic != -1:
            suspected_partners = [
                id for id, val in breach_dic.items() if (val > paramater)
            ]
            # suspected_partners = [id for id, val in breach_dic.items() if (val < paramater)]

            return suspected_partners
        return []

    def recognize_breaches_strategy(self):
        ########################*******************************************************************
        ########################***********************************
        ############################## I thought about delete all suspects every 15 steps#############################
        #############################################################################################################
        # len_before_filtering =len(self.my_breaches)
        self.my_breaches = [
            (x, y) for x, y in self.my_breaches if x > self.awi.current_step - 5
        ]
        breachs = []
        # i=0
        for x, y in self.my_breaches:
            breachs.append(y)

        # for idx in indexes_to_delete:
        #    del self.my_breaches[idx]
        # if(self.awi.current_step %15 ==0):
        #    self.my_breaches.clear()

        # print("In step {}".format(self.awi.current_step))
        # print("The number of breaches before filtering was: {}".format(len_before_filtering))
        # print("The number of breaches after filtering was: {}".format(len(self.my_breaches)))
        dic_counter = collections.Counter(breachs)
        # print("The distribution of the breaches: "+str(dic_counter))

        # print(str("dic counter: ")+str(dic_counter))
        self.suspects_to_bankrupt.clear()
        self.steps_of_suspicious.clear()

        for (k, value) in dic_counter.items():
            # Check if key is even then add pair to new dictionary
            if value >= 3:
                self.suspects_to_bankrupt[k] = value
                self.steps_of_suspicious[k] = self.awi.current_step

    def get_bad_situation_partners(self):
        breach_dic = self.get_breach_dic()
        if breach_dic != -1:
            bad_situation_partners = [
                id for id, val in breach_dic.items() if (val > 0.75)
            ]

            partners = set(bad_situation_partners).union(set(self.suspects_to_bankrupt))
            partners = partners - set(
                self.agent_bankrupt
            )  # I dont want to negotiate with bankrupt agents
            return list(partners)
        return []

    def get_bankrupt_agent(self):
        return self.agent_bankrupt

    def report_money_breach(self, buyer_id):
        self.agent_bankrupt.append(buyer_id)


def run(
    competition="std",
    reveal_names=True,
    n_steps=15,
    n_configs=1,
    max_n_worlds_per_config=None,
    n_runs_per_world=2,
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
        - To speed it up, use af smaller `n_step` value

    """
    # ComponentBasedAgent
    # competitors = [MyComponentsBasedAgent, DecentralizingAgent,MovingRangeAgent, ReactiveAgent, BuyCheapSellExpensiveAgent,DoNothingAgent,IndependentNegotiationsAgent,RandomAgent]

    competitors = [
        ASMASH,
        IndDecentralizingAgent,
        DecentralizingAgent,
        MovingRangeAgent,
    ]
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
