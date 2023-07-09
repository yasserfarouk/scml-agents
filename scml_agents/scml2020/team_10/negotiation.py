import os
import sys

from negmas.preferences.value_fun import asdict

sys.path.append(os.path.dirname(__file__))

from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from hyperparameters import *
from neg_model import NegModel, load_buyer_neg_model, load_seller_neg_model
from negmas import (
    AgentMechanismInterface,
    Issue,
    Negotiator,
    ResponseType,
    SAOState,
    SAOSyncController,
    outcome_is_valid,
)
from negmas.common import MechanismState
from negmas.helpers import get_class, instantiate
from negmas.outcomes import Outcome, enumerate_issues
from negmas.preferences import UtilityFunction
from negmas.sao import SAONegotiator, SAOResponse
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE
from scml.scml2020.components.negotiation import (
    AspirationNegotiator,
    NegotiationManager,
)
from utility import MyRLUtilityFunction, MyUtilityFunction


class MyFixedUtilityNegotiator(SAONegotiator):
    def __init__(
        self,
        _is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        manager=None,
        ufun: Optional[UtilityFunction] = None,
        name: Optional[str] = None,
        owner: "Agent" = None,
        horizon: int = MAX_HORIZON,
    ):
        super().__init__(
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
        )
        self.ufun = self.utility if ufun is None else ufun
        self._is_seller = _is_seller
        self.__parent = parent  # either a controller or a negotiation manager
        if manager is None:
            self.manager = parent
        else:
            self.manager = (
                manager  # the negotiation manager (parent may be a controller)
            )
        self.horizon = horizon

    def pad(self, arr, padding):
        if len(arr) == padding:
            return arr
        return np.concatenate([arr, [0] * (padding - len(arr))])

    def build_data(self, state: MechanismState, offer: "Outcome"):
        data = [None] * (RESPONSE_UTILITY + 1)

        next_step = self.manager.awi.current_step + 1
        if next_step >= self.manager.awi.n_steps - 1:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = [0] * MAX_HORIZON
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = [0] * MAX_HORIZON
        elif self._is_seller:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                self.manager.outputs_needed[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                self.manager.output_price[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
        else:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                self.manager.inputs_needed[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                self.manager.input_cost[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )

        data[OFFER_STATE_STEP] = state.step
        data[OFFER_RELATIVE_TIME] = state.relative_time
        if offer is None:
            offer = [0] * 3
        data[OFFER_QUANTITY] = offer[QUANTITY]
        data[OFFER_COST] = offer[UNIT_PRICE]
        data[OFFER_TIME] = offer[TIME]
        data[OFFER_UTILITY] = self.ufun(offer)

        if type(data[OFFER_UTILITY]) is torch.Tensor:
            data[OFFER_UTILITY] = data[OFFER_UTILITY].item()

        return data

    def fill_response_data(self, state: MechanismState, offer: "Outcome"):
        if len(self.manager.neg_history[self._is_seller][self.id]) == 0:
            return

        data = self.build_data(state, offer)
        self.manager.neg_history[self._is_seller][self.id][-1][
            RESPONSE_RELATIVE_TIME:
        ] = data[OFFER_RELATIVE_TIME : OFFER_UTILITY + 1]

    def respond(self, state: SAOState) -> "ResponseType":
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        assert len(offer) == 3

        my_offer = self.propose(state)
        data = self.build_data(state, my_offer)
        self.manager.neg_history[self._is_seller][self.id].append(data)
        self.fill_response_data(state, offer)

        return super().respond(state)

    def predict_outcome(self, history, prediction_model):
        to_features = np.array(
            [message[:RESPONSE_RELATIVE_TIME] for message in history]
        )
        to_features = torch.from_numpy(to_features).float()
        return prediction_model.predict(to_features)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Propose a set of offers

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """
        if self._is_seller:
            _needed, _secured, _price, _prediction_model = (
                self.manager.outputs_needed,
                self.manager.outputs_secured,
                self.manager.output_price,
                self.manager.seller_model,
            )
        else:
            _needed, _secured, _price, _prediction_model = (
                self.manager.inputs_needed,
                self.manager.inputs_secured,
                self.manager.input_cost,
                self.manager.buyer_model,
            )

        if self.manager.awi.current_step + 1 >= self.manager.awi.n_steps - 2:
            return None

        # TODO: check this...
        time_range = (
            self.manager.awi.current_step + 1,
            min(
                self.manager.awi.current_step + self.horizon,
                self.manager.awi.n_steps - 2,
            )
            + 1,
        )

        offer = [0] * 3

        history = self.manager.neg_history[self._is_seller][self.id]
        current_proposal = [0] * (OFFER_UTILITY + 1)
        current_proposal[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
            _needed[offer[TIME] : offer[TIME] + MAX_HORIZON], MAX_HORIZON
        )
        current_proposal[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
            _price[offer[TIME] : offer[TIME] + MAX_HORIZON], MAX_HORIZON
        )
        current_proposal[OFFER_STATE_STEP] = state.step
        current_proposal[OFFER_RELATIVE_TIME] = state.relative_time
        current_proposal[OFFER_QUANTITY] = offer[QUANTITY]
        current_proposal[OFFER_COST] = offer[UNIT_PRICE]
        current_proposal[OFFER_TIME] = offer[TIME]
        current_proposal[OFFER_UTILITY] = self.ufun(offer)

        best_proposal = current_proposal
        best_outcome = self.predict_outcome(
            history + [best_proposal], _prediction_model
        )[-1]

        offer = [0] * 3

        for time in range(*time_range):
            offer[TIME] = time
            current_proposal[OFFER_TIME] = time
            current_proposal[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                _needed[time : time + MAX_HORIZON], MAX_HORIZON
            )
            current_proposal[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                _price[time : time + MAX_HORIZON], MAX_HORIZON
            )

            quantity_range = (
                1,
                max(_needed[time : time + MAX_HORIZON]) + 1,
            )  # TODO: maybe allow even more?
            if self._is_seller:
                price_range = (ceil(_price[time]), ceil(_price[time] + 10) + 1)
            else:
                price_range = (floor(_price[time]) - 10, floor(_price[time]) + 1)
                # price_range = (floor(_price[time]), floor(_price[time]) - 10 - 1, -1)

            for quantity in range(*quantity_range):
                offer[QUANTITY] = quantity
                current_proposal[OFFER_QUANTITY] = quantity

                # if I'm a seller:
                # first, find the highest price such that the adversary agrees to buy

                # if I'm a buyer:
                # find the lowest price such that the adversary agrees to sell

                start, end = price_range

                while start < end:

                    price = (start + end) // 2
                    offer[UNIT_PRICE] = price

                    current_proposal[OFFER_COST] = price

                    current_proposal[OFFER_UTILITY] = self.ufun(offer)
                    to_predict = history + [current_proposal]
                    predicted_outcome = self.predict_outcome(
                        to_predict, _prediction_model
                    )
                    if predicted_outcome[-1] < -900:
                        # if the adversary didn't agree to buy in price p, he would not agree to buy
                        # in price greater than p
                        if self._is_seller:
                            end = price - 1
                        # if the adversary didn't agree to sell in price p, he would not agree to sell
                        # in price lower than p
                        else:
                            start = price + 1
                    else:
                        # update the best proposal
                        if predicted_outcome[-1] > best_outcome:
                            best_outcome = predicted_outcome[-1]
                            best_proposal = current_proposal
                        # if the adversary agrees to buy in price p, we'll try to sell in higher price
                        if self._is_seller:
                            start = price + 1
                        # if the adversary agrees to sell in price p, we'll try to buy in lower price
                        else:
                            end = price - 1

        offer[TIME] = best_proposal[OFFER_TIME]
        offer[UNIT_PRICE] = best_proposal[OFFER_COST]
        offer[QUANTITY] = best_proposal[OFFER_QUANTITY]

        # time = state.relative_time
        #
        # # for quantity:
        # # we want f(0) = 1, f(1) = 0 -> let's try f(x) = 1 - x**2 -> we will try to learn this function later on
        # offer = [0] * 3
        # offer[TIME] = self.manager.awi.current_step + 1
        # offer[QUANTITY] = int(round(_needed[offer[TIME]] * (1 - time**2)))
        #
        # # for buyer input cost: as time continues, we will go from input_cost * (0.5) up to input_cost
        # # for seller output price: as time continues, we will go from output_price * 2 down to output_price
        # goal_price = _price[offer[TIME]]
        # if self._is_seller:
        #     offered_price = 2*goal_price*(1-time**2) + goal_price*(time**2)
        # else:
        #     offered_price = 0.5*goal_price*(1-time**2) + goal_price*(time**2)
        # offer[UNIT_PRICE] = offered_price

        # remember the last proposal utility
        self.my_last_proposal_utility = best_proposal[OFFER_UTILITY]

        return offer

    def on_negotiation_end(self, state: MechanismState) -> None:
        self.fill_response_data(state, state.agreement)


class MyUtilityNegotiator(SAONegotiator):
    def __init__(
        self,
        _is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        manager=None,
        ufun: Optional[UtilityFunction] = None,
        name: Optional[str] = None,
        owner: "Agent" = None,
        horizon: int = MAX_HORIZON,
    ):
        super().__init__(
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
        )
        self.ufun = self.utility if ufun is None else ufun
        self._is_seller = _is_seller
        self.__parent = parent  # either a controller or a negotiation manager
        if manager is None:
            self.manager = parent
        else:
            self.manager = (
                manager  # the negotiation manager (parent may be a controller)
            )
        self.horizon = horizon

    def pad(self, arr, padding):
        if len(arr) == padding:
            return arr
        return np.concatenate([arr, [0] * (padding - len(arr))])

    def build_data(self, state: MechanismState, offer: "Outcome"):
        data = [None] * RESPONSE_RELATIVE_TIME

        next_step = self.manager.awi.current_step + 1
        if next_step >= self.manager.awi.n_steps - 1:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = [0] * MAX_HORIZON
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = [0] * MAX_HORIZON
        elif self._is_seller:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                self.manager.outputs_needed[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                self.manager.output_price[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
        else:
            data[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                self.manager.inputs_needed[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )
            data[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                self.manager.input_cost[next_step : next_step + MAX_HORIZON],
                MAX_HORIZON,
            )

        data[OFFER_STATE_STEP] = state.step
        data[OFFER_RELATIVE_TIME] = state.relative_time
        if offer is None:
            offer = [0] * 3
        data[OFFER_QUANTITY] = offer[QUANTITY]
        data[OFFER_COST] = offer[UNIT_PRICE]
        data[OFFER_TIME] = offer[TIME]
        data[OFFER_UTILITY] = self.ufun(data[:OFFER_UTILITY])

        if type(data[OFFER_UTILITY]) == torch.Tensor:
            data[OFFER_UTILITY] = data[OFFER_UTILITY].item()

        return data

    def fill_response_data(self, state: MechanismState, offer: "Outcome"):
        if len(self.manager.neg_history[self._is_seller][self.id]) == 0:
            return

        data = self.build_data(state, offer)
        self.manager.neg_history[self._is_seller][self.id][-1][
            RESPONSE_RELATIVE_TIME:
        ] = data[OFFER_RELATIVE_TIME : OFFER_UTILITY + 1]

    def respond(self, state: SAOState) -> "ResponseType":
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        my_offer = self.propose(state)
        data = self.build_data(state, my_offer)
        self.manager.neg_history[self._is_seller][self.id].append(data)
        self.fill_response_data(state, offer)

        expanded_offer = self.build_data(state, offer)[:OFFER_UTILITY]
        d = asdict(state, recurse=False)
        d["current_offer"] = expanded_offer
        state = SAOState(**d)
        return super().respond(state)

    def predict_outcome(self, history, prediction_model):
        to_features = np.array(
            [message[:RESPONSE_RELATIVE_TIME] for message in history]
        )
        to_features = torch.from_numpy(to_features).float()
        prediction = prediction_model.predict(to_features)
        return prediction

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Propose a set of offers

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """
        if self._is_seller:
            _needed, _secured, _price, _prediction_model = (
                self.manager.outputs_needed,
                self.manager.outputs_secured,
                self.manager.output_price,
                self.manager.seller_model,
            )
        else:
            _needed, _secured, _price, _prediction_model = (
                self.manager.inputs_needed,
                self.manager.inputs_secured,
                self.manager.input_cost,
                self.manager.buyer_model,
            )

        if self.manager.awi.current_step + 1 >= self.manager.awi.n_steps - 2:
            return None

        # TODO: check this...
        time_range = (
            self.manager.awi.current_step + 1,
            min(
                self.manager.awi.current_step + self.horizon,
                self.manager.awi.n_steps - 2,
            )
            + 1,
        )

        offer = [0] * 3

        history = self.manager.neg_history[self._is_seller][self.id]
        current_proposal = [0] * (OFFER_UTILITY + 1)
        current_proposal[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
            _needed[offer[TIME] : offer[TIME] + MAX_HORIZON], MAX_HORIZON
        )
        current_proposal[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
            _price[offer[TIME] : offer[TIME] + MAX_HORIZON], MAX_HORIZON
        )
        current_proposal[OFFER_STATE_STEP] = state.step
        current_proposal[OFFER_RELATIVE_TIME] = state.relative_time
        current_proposal[OFFER_QUANTITY] = offer[QUANTITY]
        current_proposal[OFFER_COST] = offer[UNIT_PRICE]
        current_proposal[OFFER_TIME] = offer[TIME]
        current_proposal[OFFER_UTILITY] = self.ufun(current_proposal[:OFFER_UTILITY])

        if type(current_proposal[OFFER_UTILITY]) is torch.Tensor:
            current_proposal[OFFER_UTILITY] = current_proposal[OFFER_UTILITY].item()

        best_proposal = current_proposal
        best_outcome = self.predict_outcome(
            history + [best_proposal], _prediction_model
        )[-1]

        for time in range(*time_range):
            current_proposal[OFFER_TIME] = time
            # TODO: actually, this should be over the time range, and the proposed time is in the interval
            current_proposal[AUX_NEEDED_START : AUX_NEEDED_END + 1] = self.pad(
                _needed[time : time + MAX_HORIZON], MAX_HORIZON
            )
            current_proposal[AUX_PRICE_START : AUX_PRICE_END + 1] = self.pad(
                _price[time : time + MAX_HORIZON], MAX_HORIZON
            )

            # TODO: actually, this should be only for _needed[time]
            quantity_range = (
                1,
                max(_needed[time : time + MAX_HORIZON]) + 1,
            )  # TODO: maybe allow even more?
            if self._is_seller:
                price_range = (ceil(_price[time]), ceil(_price[time] + 10) + 1)
            else:
                price_range = (floor(_price[time]) - 10, floor(_price[time]) + 1)
                # price_range = (floor(_price[time]), floor(_price[time]) - 10 - 1, -1)
            for quantity in range(*quantity_range):
                current_proposal[OFFER_QUANTITY] = quantity

                # if I'm a seller:
                # first, find the highest price such that the adversary agrees to buy

                # if I'm a buyer:
                # find the lowest price such that the adversary agrees to sell

                start, end = price_range
                while start < end:

                    price = (start + end) // 2

                    current_proposal[OFFER_COST] = price
                    current_proposal[OFFER_UTILITY] = self.ufun(
                        current_proposal[:OFFER_UTILITY]
                    )

                    if type(current_proposal[OFFER_UTILITY]) is torch.Tensor:
                        current_proposal[OFFER_UTILITY] = current_proposal[
                            OFFER_UTILITY
                        ].item()

                    to_predict = history + [current_proposal]
                    predicted_outcome = self.predict_outcome(
                        to_predict, _prediction_model
                    )
                    if predicted_outcome[-1] < -900:
                        # if the adversary didn't agree to buy in price p, he would not agree to buy
                        # in price greater than p
                        if self._is_seller:
                            end = price - 1
                        # if the adversary didn't agree to sell in price p, he would not agree to sell
                        # in price lower than p
                        else:
                            start = price + 1
                    else:
                        # update the best proposal
                        if predicted_outcome[-1] > best_outcome:
                            best_outcome = predicted_outcome[-1]
                            best_proposal = current_proposal
                        # if the adversary agrees to buy in price p, we'll try to sell in higher price
                        if self._is_seller:
                            start = price + 1
                        # if the adversary agrees to sell in price p, we'll try to buy in lower price
                        else:
                            end = price - 1

        offer[TIME] = best_proposal[OFFER_TIME]
        offer[UNIT_PRICE] = best_proposal[OFFER_COST]
        offer[QUANTITY] = best_proposal[OFFER_QUANTITY]

        # time = state.relative_time
        #
        # # for quantity:
        # # we want f(0) = 1, f(1) = 0 -> let's try f(x) = 1 - x**2 -> we will try to learn this function later on
        # offer = [0] * 3
        # offer[TIME] = self.manager.awi.current_step + 1
        # offer[QUANTITY] = int(round(_needed[offer[TIME]] * (1 - time**2)))
        #
        # # for buyer input cost: as time continues, we will go from input_cost * (0.5) up to input_cost
        # # for seller output price: as time continues, we will go from output_price * 2 down to output_price
        # goal_price = _price[offer[TIME]]
        # if self._is_seller:
        #     offered_price = 2*goal_price*(1-time**2) + goal_price*(time**2)
        # else:
        #     offered_price = 0.5*goal_price*(1-time**2) + goal_price*(time**2)
        # offer[UNIT_PRICE] = offered_price

        # remember the last proposal utility
        self.my_last_proposal_utility = best_proposal[OFFER_UTILITY]

        return offer

    def on_negotiation_end(self, state: MechanismState) -> None:
        self.fill_response_data(state, state.agreement)


class MyUtilityNegotiationManager(NegotiationManager):
    """
    A negotiation manager that manages independent negotiators that do not share any information once created

    Args:
        negotiator_type: The negotiator type to use to manage all negotiations
        negotiator_params: Parameters of the negotiator

    Requires:
        - `create_ufun`
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Hooks Into:
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

    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = MyUtilityNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        seller_model_path=NEG_SELL_PATH,
        buyer_model_path=NEG_BUY_PATH,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: copy into my sync controller!!!!!
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.neg_history = {True: {}, False: {}}
        self.seller_model = load_seller_neg_model(path=seller_model_path)
        self.buyer_model = load_buyer_neg_model(path=buyer_model_path)
        self.seller_model.eval()
        self.buyer_model.eval()

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

        for partner in partners:
            negotiator = self.negotiator(sell)
            self.neg_history[sell][negotiator.id] = []
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
        negotiator = self.negotiator(annotation["seller"] == self.id, issues=issues)
        self.neg_history[annotation["seller"] == self.id][negotiator.id] = []
        return negotiator

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        """Creates a utility function"""
        # return MyUtilityFunction(_is_seller=is_seller, parent=self)
        return MyRLUtilityFunction(_is_seller=is_seller, parent=self)

    def negotiator(self, is_seller: bool, issues=None, outcomes=None) -> SAONegotiator:
        """Creates a negotiator"""
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        params["_is_seller"] = is_seller
        params["parent"] = self
        return instantiate(self.negotiator_type, **params)


class MySyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent
    needs and with lowest price.

    Args:
        is_seller: Are we trying to sell (or to buy)?
        parent: The agent from which we will access `needed` and `secured` arrays
        price_weight: The importance of price in utility calculation
        utility_threshold: Accept anything with a relative utility above that
        time_threshold: Accept anything with a positive utility when we are that close
                        to the end of the negotiation
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        default_negotiator_type=MyFixedUtilityNegotiator,  # TODO: my utility negotiator
        unnecessary_percentage_threshold=0.45,
        seller_model_path=NEG_SELL_PATH,
        buyer_model_path=NEG_BUY_PATH,
        negotiation_model_class=NegModel,
        load_model=False,
        ufun=None,
        **kwargs,
    ):
        kwargs["default_negotiator_type"] = default_negotiator_type
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self._best_utils: Dict[str, float] = {}
        self.unnecessary_percentage_threshold = unnecessary_percentage_threshold
        if ufun is not None:
            self.ufun = ufun
        elif default_negotiator_type == MyFixedUtilityNegotiator:
            self.ufun = MyUtilityFunction
        else:
            self.ufun = MyRLUtilityFunction
        self.neg_history = {True: {}, False: {}}
        if load_model:
            self.seller_model = load_seller_neg_model(path=seller_model_path)
            self.buyer_model = load_buyer_neg_model(path=buyer_model_path)
        else:
            self.seller_model = negotiation_model_class(True)
            self.buyer_model = negotiation_model_class(False)
        self.seller_model.eval()
        self.buyer_model.eval()

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        """Creates a utility function"""
        return self.ufun(_is_seller=is_seller, parent=self, manager=self.__parent)

    def create_negotiator(
        self,
        negotiator_type=None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ):
        kwargs["ufun"] = self.create_ufun(is_seller=self._is_seller)
        kwargs["_is_seller"] = self._is_seller
        kwargs["manager"] = self.__parent
        negotiator = super().create_negotiator(name=name, cntxt=cntxt, **kwargs)
        negotiator.ufun.update_neg_parent(negotiator)

        self.__parent.neg_history[negotiator._is_seller][negotiator.id] = []
        return negotiator

    def utility(self, offer) -> float:
        """A simple utility function

        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract

        """

        # get my needs and secured amounts arrays
        if self._is_seller:
            _needed, _secured = (
                self.__parent.outputs_needed,
                self.__parent.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.__parent.inputs_needed,
                self.__parent.inputs_secured,
            )

        # invalide offers have no utility
        if offer is None:
            return -1000

        # offers for contracts that can never be executed have no utility
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0

        # TODO: consider the other agent's bankruptcy / finantial reports...
        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            # TODO: trying improvement
            to_produce = _needed[offer[TIME]] - _secured[t]
            unnecessary_percentage = 1 - (to_produce / offer[QUANTITY])
            if unnecessary_percentage > self.unnecessary_percentage_threshold:
                return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]

        # TODO: improve this as well
        util = self._price_weight * price + (1 - self._price_weight) * q
        return util

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        """Is this a valid offer for that negotiation"""
        issues = self.negotiators[negotiator_id][0].nmi.issues
        return outcome_is_valid(offer, issues)

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, "SAOState"]
    ) -> Dict[str, "SAOResponse"]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.

        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array([self.utility(o) for o in offers.values()])

        # TODO: maybe there is more than one best offer
        # TODO: maybe we want to continue negotiating with more than one to get better offer
        # TODO: maybe we want also to take into account the probability of breaching / bankrupting
        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.first_proposals()

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            return {
                k: SAOResponse(ResponseType.REJECT_OFFER, best_proposals[k])
                for k in offers.keys()
            }

        relative_time = min(_.relative_time for _ in states.values())

        # TODO: this is a bug!
        # assert relative_time < self._time_threshold

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility >= self._utility_threshold * self._best_utils[best_partner]
            or relative_time > self._time_threshold
        ):
            responses = {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    best_offer if self.is_valid(k, best_offer) else best_proposals[k],
                )
                for k in offers.keys()
            }
            responses[best_partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return responses

        # send the best offer to everyone else and try to improve it
        responses = {
            k: SAOResponse(
                ResponseType.REJECT_OFFER,
                best_offer if self.is_valid(k, best_offer) else best_proposals[k],
            )
            for k in offers.keys()
        }
        responses[best_partner] = SAOResponse(
            ResponseType.REJECT_OFFER, best_proposals[best_partner]
        )
        return responses

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return {nid: self.best_proposal(nid)[0] for nid in self.negotiators.keys()}

    def on_negotiation_end(self, negotiator_id: str, state: "MechanismState") -> None:
        """Update the secured quantities whenever a negotiation ends"""
        if state.agreement is None:
            return

        # TODO: is this a duplicity? I think we already update the secured in another module (PredictionBasedTradingStrategy)
        q, t = state.agreement[QUANTITY], state.agreement[TIME]
        if self._is_seller:
            self.__parent.outputs_secured[t] += q
        else:
            self.__parent.inputs_secured[t] += q

    def best_proposal(self, nid: str) -> Tuple[Optional["Outcome"], float]:
        """
        Finds the best proposal for the given negotiation

        Args:
            nid: Negotiator ID

        Returns:
            The outcome with highest utility and the corresponding utility
        """
        negotiator = self.negotiators[nid][0]
        if negotiator.nmi is None:
            return None, -1000
        utils = np.array([self.utility(_) for _ in negotiator.nmi.outcomes])
        best_indx = np.argmax(utils)
        self._best_utils[nid] = utils[best_indx]
        if utils[best_indx] < 0:
            return None, utils[best_indx]
        return negotiator.nmi.outcomes[best_indx], utils[best_indx]


class MyNegotiationManager:
    """My negotiation strategy

    Args:
        price_weight: The relative importance of price in the utility calculation.
        utility_threshold: The fraction of maximum utility above which all offers will be accepted.
        time_threshold: The fraction of the negotiation time after which any valid offers will be accepted.
        time_range: The time-range for each controller as a fraction of the number of simulation steps

    Hooks Into:
        - `init`
        - `step`

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

    def __init__(
        self,
        *args,
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        controller=MySyncController,
        seller_model_path=NEG_SELL_PATH,
        buyer_model_path=NEG_BUY_PATH,
        negotiation_model_class=NegModel,
        load_model=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.controllers: Dict[bool, controller] = {
            False: controller(
                is_seller=False,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
            True: controller(
                is_seller=True,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
        }
        self._current_end = -1
        self._current_start = -1
        self.neg_history = {True: {}, False: {}}
        if load_model:
            self.seller_model = load_seller_neg_model(path=seller_model_path)
            self.buyer_model = load_buyer_neg_model(path=buyer_model_path)
        else:
            self.seller_model = negotiation_model_class(True)
            self.buyer_model = negotiation_model_class(False)
        self.seller_model.eval()
        self.buyer_model.eval()

    def step(self):
        super().step()

        # find the range of steps about which we plan to negotiate
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
            # find the maximum amount needed at any time-step in the given range
            needs = np.max(
                needed[self._current_start : self._current_end]
                - secured[self._current_start : self._current_end]
            )
            if needs < 1:
                continue

            # set a range of prices
            if seller:
                # TODO: maybe more than that....
                # for selling set a price that is at least the catalog price
                min_price = min(
                    self.output_price[self._current_start : self._current_end]
                )
                min_price = self.awi.catalog_prices[product]
                price_range = (min_price, 2 * min_price)
            else:
                # for buying sell a price that is at most the catalog price
                price_range = (
                    0,
                    max(self.input_cost[self._current_start : self._current_end]),
                )
                price_range = (0, self.awi.catalog_prices[product])
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
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None

        # if agent is bankrupt - refuse negotiation
        caller = annotation["caller"]
        reports = self.awi.reports_of_agent(caller)
        if not reports is None:
            for step, report in reports.items():
                if report.is_bankrupt:
                    return None

        # controller = self.controllers[not annotation["is_buy"]]
        controller = self.controllers[annotation["seller"] == self.id]
        if controller is None:
            return None

        return controller.create_negotiator()
