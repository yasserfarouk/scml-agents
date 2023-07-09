import functools
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from negmas import (
    INVALID_UTILITY,
    AgentMechanismInterface,
    AgentWorldInterface,
    AspirationNegotiator,
    Breach,
    Contract,
    ControlledNegotiator,
    Issue,
    LinearUtilityFunction,
    MappingUtilityFunction,
    MechanismState,
    Negotiator,
    Outcome,
    PolyAspiration,
    ResponseType,
    SAOController,
    SAONegotiator,
    SAOResponse,
    SAOState,
    SAOSyncController,
    UtilityFunction,
    Value,
    outcome_is_valid,
)
from negmas.events import Notification, Notifier
from negmas.helpers import instantiate
from scml import SCML2020Agent
from scml.scml2020 import (
    QUANTITY,
    TIME,
    UNIT_PRICE,
    Failure,
    FinancialReport,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
)
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.components.negotiation import ControllerInfo
from scml.scml2020.services import StepController, SyncController

# improvements:
# 1. (Utility) consider agent's financial reports.
# 2. (Negotiator) param change.


class CalcTrustworthiness:
    _awi = None
    _nmi = None
    breach_level_w = 0.2
    breach_prob_w = 0.8
    last_step = True

    @property
    def awi(self):
        return self._awi

    @awi.setter
    def awi(self, value):
        self._awi = value

    @property
    def nmi(self):
        return self._nmi

    @nmi.setter
    def nmi(self, value):
        self._nmi = value

    def eval_trustworthiness(self, u, offer: Optional["Outcome"]):
        """
        u = utility for the offer
        returns new utility
        """
        # extract financial report
        rival_agent_id = (
            self.nmi.annotation["buyer"]
            if self.controller.is_seller
            else self.nmi.annotation["seller"]
        )
        financial_rep = self.awi.reports_of_agent(rival_agent_id)

        if self.last_step:
            # measure trustworthiness by the last step's financial report published
            financial_rep = [(step, fr) for step, fr in financial_rep.items()]
            financial_rep.sort(key=lambda x: x[0])
            fp: FinancialReport = financial_rep[-1][1]
            breach_level = fp.breach_level
            breach_prob = fp.breach_prob
            is_bankrupt = fp.is_bankrupt
        else:
            # measure trustworthiness by all financial report published (average)
            breach_level = sum(
                fr.breach_level for step, fr in financial_rep.items()
            ) / len(financial_rep)
            breach_prob = sum(
                fr.breach_prob for step, fr in financial_rep.items()
            ) / len(financial_rep)
            is_bankrupt = any(fr.is_bankrupt for step, fr in financial_rep.items())

        # if agent had bankrupt - don't do deals with it
        if is_bankrupt:
            return float("-inf")
        # trustworthiness = percentage of trust.
        trustworthiness = self.breach_level_w * (
            1 - breach_level
        ) + self.breach_prob_w * (1 - breach_prob)
        punishment = abs(u * (1 - trustworthiness))

        # sellers are more vulnerable to breach because they don't do the sell at all
        # (while buyers still get the product from spot market (if no bankruptcy occur))
        # if self.controller.is_seller:
        #     punishment *= 3
        u = u - punishment
        return u


class UpdateUfunc:
    def set_ufun_members(self, negotiator_id: str):
        self.ufun.nmi = self.negotiators[negotiator_id][0].nmi
        self.ufun.awi = self.awi


class DanasUtilityFunction(CalcTrustworthiness, LinearUtilityFunction):
    def __init__(self, controller, *args, **kwargs):
        self.controller = controller
        issues = kwargs.get("issues", None)
        outcomes = kwargs.get("outocmes", None)

        if self.controller.is_seller:
            super().__init__((1, 1, 10), issues=issues, outcomes=outcomes)
        else:
            super().__init__((1, -1, -10), issues=issues, outcomes=outcomes)

    def eval(self, offer: Optional["Outcome"]) -> Optional[Value]:
        u = super().eval(offer)
        return self.eval_trustworthiness(u, offer)


class DanasNegotiator(AspirationNegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_aspiration=3.5, **kwargs)


class MaintainFairPrice:
    def init_component(self, n_std, min_sample, acc_unit_prices):
        self.all_unit_prices_from_agreements = acc_unit_prices
        self.n_std = n_std
        self.min_sample = min_sample

    def achieved_agreement(self, agreement: "Outcome"):
        self.all_unit_prices_from_agreements.append(agreement[UNIT_PRICE])

    def propose_based_on_all_neg(self, outcome: "Outcome") -> Optional["Outcome"]:
        if outcome is None:
            return outcome
        if len(self.all_unit_prices_from_agreements) >= self.min_sample:
            mean = statistics.mean(self.all_unit_prices_from_agreements)
            std = statistics.stdev(self.all_unit_prices_from_agreements)
            if std == 0:
                return outcome

            if self.is_seller:
                if outcome[UNIT_PRICE] < mean - self.n_std * std:
                    # too cheap
                    return (outcome[0], outcome[1], math.ceil(mean))
            elif outcome[UNIT_PRICE] > mean + self.n_std * std:
                # too expensive
                return (outcome[0], outcome[1], int(mean))
        return outcome

    def respond_based_on_all_neg(
        self, current_resp: ResponseType, offer: "Outcome"
    ) -> ResponseType:
        if (
            current_resp != ResponseType.ACCEPT_OFFER
            or len(self.all_unit_prices_from_agreements) < self.min_sample
        ):
            return current_resp
        mean = statistics.mean(self.all_unit_prices_from_agreements)
        std = statistics.stdev(self.all_unit_prices_from_agreements)
        if std == 0:
            return ResponseType.ACCEPT_OFFER

        if self.is_seller:
            if offer[UNIT_PRICE] < mean - self.n_std * std:
                # too cheap
                return ResponseType.REJECT_OFFER
        elif offer[UNIT_PRICE] > mean + self.n_std * std:
            # too expensive
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER


class DanasController(MaintainFairPrice, UpdateUfunc, SAOController, Notifier):
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
         instantiates a single negotiator that dynnmically changes the nmi but always uses the same ufun.
    """

    def __init__(
        self,
        *args,
        acc_unit_prices: List,  # all accepted unit prices
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
        n_std: int = 2,  # standard deviation to use when deciding if unit price is acceptable
        min_sample: int = 3,  # minimum population sample to consider
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.init_component(n_std, min_sample, acc_unit_prices)
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
        self.ufun = DanasUtilityFunction(controller=self)
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)

        self.completed = defaultdict(bool)
        self.step = step
        self.retries: Dict[str, int] = defaultdict(int)
        self.max_retries = max_retries
        self.__asp = PolyAspiration(3.5, "boulware")

    def join(
        self,
        negotiator_id: str,
        nmi: AgentMechanismInterface,
        state: MechanismState,
        *,
        preferences: Optional["UtilityFunction"] = None,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        if ufun:
            preferences = ufun
        joined = super().join(
            negotiator_id, nmi, state, ufun=ufun, preferences=preferences, role=role
        )
        if joined:
            self.completed[negotiator_id] = False
        return joined

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        self.set_ufun_members(negotiator_id)
        self.__negotiator._nmi = self.negotiators[negotiator_id][0]._nmi
        outcome = self.__negotiator.propose(state)
        return self.propose_based_on_all_neg(outcome)

    def respond(self, negotiator_id: str, state: SAOState) -> ResponseType:
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.set_ufun_members(negotiator_id)
        self.__negotiator._nmi = self.negotiators[negotiator_id][0]._nmi
        resp = self.__negotiator.respond(state=state)
        return self.respond_based_on_all_neg(resp, offer)

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
            f"secured {self.secured} of {self.target} for {self.parent_name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[ControlledNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> ControlledNegotiator:
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
            self.achieved_agreement(agreement)
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

                # # todo: MADE A CHANGE
                # possible_partners = [p for p in self.partners if self.retries[p] <= self.max_retries]
                # for partner in possible_partners:
                #     self.retries[partner] += 1
                #     neg = self.create_negotiator()
                #     self.completed[neg.id] = False
                #     self.awi.loginfo(
                #         f"{str(self)} negotiating with {partner} on u={self.urange}"
                #         f", q=(1,{self.target - self.secured}), u=({tmin}, {tmax})"
                #     )
                #     self.awi.request_negotiation(
                #         not self.is_seller,
                #         product=self.product,
                #         quantity=(1, self.target - self.secured),
                #         unit_price=self.urange,
                #         time=(tmin, tmax),
                #         partner=partner,
                #         negotiator=neg,
                #         extra=dict(controller_index=self.step, is_seller=self.is_seller),
                #     )

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
