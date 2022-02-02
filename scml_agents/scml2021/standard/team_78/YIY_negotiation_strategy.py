import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from negmas import (
    Issue,
    ResponseType,
    SAOSyncController,
    UtilityFunction,
    outcome_is_valid,
)
from negmas.sao import SAOResponse
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE


class ControllerUFun(UtilityFunction):
    """A utility function for the controller"""

    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller

    def eval(self, offer: "Outcome"):
        return self.controller.utility(offer)

    def xml(self, issues):
        pass


class SyncController(SAOSyncController):
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
        parent: "SklearnPredictionBasedTradingStrategy",
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
        self.utility_function = ControllerUFun(controller=self)

    def utility(self, offer: "Outcome") -> float:
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

        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]
        return self._price_weight * price + (1 - self._price_weight) * q

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
            best_utility
            >= self._utility_threshold * self.utility(best_proposals[best_partner])
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

    def on_negotiation_end(self, negotiator_id: str, state: "MechanismState") -> None:
        """Update the secured quantities whenever a negotiation ends"""
        if state.agreement is None:
            return

        if self._is_seller:
            self.__parent.sell_prices.append(state.agreement[UNIT_PRICE])
        else:
            self.__parent.buy_prices.append(state.agreement[UNIT_PRICE])


# ===================================== Negotiation Manager ============================================


class YIYNegotiationManager:
    """YIY negotiation strategy

    Args:
        price_weight: The relative importance of price in the utility calculation.
        utility_threshold: The fraction of maximum utility above which all offers will be accepted.
        time_threshold: The fraction of the negotiation time after which any valid offers will be accepted.
        time_range: The time-range for each controller as a fraction of the number of simulation steps
    """

    def __init__(
        self,
        *args,
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        breach_threshold=0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.breach_threshold = breach_threshold
        self.sell_prices = []
        self.buy_prices = []
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

    def is_breach_high(self, agent_str):
        if self.awi.current_step > 2:
            reportstep = self.awi.current_step - 1
            reports = self.awi.reports_at_step(reportstep)
            if reports:
                for agent in reports.values():
                    if (
                        agent.agent_name == agent_str
                        and agent.breach_level > self.breach_threshold
                    ):
                        return True
            return False

        return False

    def get_updated_price(self, is_sell):
        last_prices = []
        if is_sell:
            last_prices = self.sell_prices[-3:]
        else:
            last_prices = self.buy_prices[-3:]

        return sum(last_prices) / len(last_prices) if len(last_prices) > 0 else -1

    def step(self):
        super().step()

        # find the range of steps about which we plan to negotiate
        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(self.awi.n_steps - 1, self._current_start + 2)
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

            my_legal_partners = []

            # set a range of prices
            if seller:
                # request negotiations from agents with breach level less than threshold:
                partners = self.awi.my_consumers
                for partner in partners:
                    if not self.is_breach_high(partner):
                        my_legal_partners.append(partner)

                # get updated selling price
                min_price = self.awi.catalog_prices[product]
                sell_price = self.get_updated_price(seller)

                updated_price = sell_price if sell_price > 0 else min_price

                price_range = (updated_price, 2 * updated_price)
            else:
                # request negotiations from agents with breach level less than threshold:
                partners = self.awi.my_suppliers
                for partner in partners:
                    if not self.is_breach_high(partner):
                        my_legal_partners.append(partner)

                # get updated buying price
                buy_price = self.get_updated_price(seller)
                updated_price = (
                    buy_price if buy_price > 0 else self.awi.catalog_prices[product]
                )

                price_range = (0, updated_price)

            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=self.controllers[seller],
                partners=my_legal_partners,
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
        controller = self.controllers[self.id == annotation["seller"]]
        if controller is None:
            return None
        return controller.create_negotiator()
