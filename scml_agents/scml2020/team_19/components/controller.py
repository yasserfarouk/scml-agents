# required for typing
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Callable, Union, Type, Optional

from negmas import (
    AspirationMixin,
    LinearUtilityFunction,
    PassThroughNegotiator,
    MechanismState,
    ResponseType,
    UtilityFunction,
    AgentWorldInterface,
    outcome_is_valid,
    Outcome,
)
from negmas.events import Notifier, Notification
from negmas.helpers import instantiate
from negmas.common import AgentMechanismInterface
from negmas.sao import (
    SAOController,
    SAONegotiator,
    SAOSyncController,
)

import numpy as np
from typing import Dict, Tuple, List
from negmas import ResponseType, Issue
from negmas.sao import SAOState, SAOResponse
from scml.scml2020.common import TIME, QUANTITY, UNIT_PRICE


class SyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent needs and with lowest price
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self._best_utils: Dict[str, float] = {}
        # find out my needs and the amount secured lists

    def utility(self, offer: Tuple[int, int, int], max_price: int) -> float:
        """A simple utility function
        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract
        """
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
        if offer is None:
            return -1000
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0
        if self._is_seller:
            price = offer[UNIT_PRICE]
        else:
            price = max_price - offer[UNIT_PRICE]
        return self._price_weight * price + (1 - self._price_weight) * q

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        issues = self.negotiators[negotiator_id][0].ami.issues
        return outcome_is_valid(offer, issues)

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).
        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.
        Remarks:
            - The response type CANNOT be WAIT.
        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array(
            [
                self.utility(
                    o, self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value
                )
                for nid, o in offers.items()
            ]
        )

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

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        """Update the secured quantities whenever a negotiation ends"""
        if state.agreement is None:
            return

        q, t = state.agreement[QUANTITY], state.agreement[TIME]
        if self._is_seller:
            self.__parent.outputs_secured[t] += q
        else:
            self.__parent.inputs_secured[t] += q

    def best_proposal(self, nid: str) -> Tuple[Optional[Outcome], float]:
        """
        Finds the best proposal for the given negotiation
        Args:
            nid: Negotiator ID
        Returns:
            The outcome with highest utility and the corresponding utility
        """
        negotiator = self.negotiators[nid][0]
        if negotiator.ami is None:
            return None, -1000
        utils = np.array(
            [
                self.utility(_, negotiator.ami.issues[UNIT_PRICE].max_value)
                for _ in negotiator.ami.outcomes
            ]
        )
        best_indx = np.argmax(utils)
        self._best_utils[nid] = utils[best_indx]
        if utils[best_indx] < 0:
            return None, utils[best_indx]
        return negotiator.ami.outcomes[best_indx], utils[best_indx]

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return {nid: self.best_proposal(nid)[0] for nid in self.negotiators.keys()}
