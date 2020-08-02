import logging

from scml.scml2020 import SCML2020Agent, SCML2020World, RandomAgent, DecentralizingAgent
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
)
from scml.scml2020.components.trading import (
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
)
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.services.controllers import SyncController

from scml.scml2020 import TIME, QUANTITY, UNIT_PRICE
from negmas import ResponseType, outcome_is_valid
from negmas.sao import SAOResponse
from typing import List, Dict, Optional, Tuple, Any
from negmas import SAOSyncController
import numpy as np
import matplotlib.pyplot as plt


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
        parent: "PredictionBasedTradingStrategy",
        price_weight=0.8,
        utility_threshold=0.7,
        time_threshold=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self._max_cycles = 10
        self._best_utils: Dict[str, float] = {}
        self.seq_cntr = 0
        self.overall_best_util_sell = 0
        self.overall_best_util_buy = 0
        # find out my needs and the amount secured lists

    def clear(self):
        self.seq_cntr = 0
        self.proposals = []

    def myUtil(self, offer):
        """
        int myUtil() - returns utility of an offer

        Utility is based only on this offer being without consideration
        of other offers.
        Utility considers following parameters:
            1. Price
            2. Time - the closer the sell/buy is better.
            3. Quantity, Needed Quantity - prefer quantity that is close
            to the needed quantity

        ToDo: Currently the utility will give better score for large quantity
        without taking into consideration we can take several offers.
        """
        parent = self.__parent

        if self._is_seller:
            needed = parent.outputs_needed
            secured = parent.outputs_secured
            product_id = parent.awi.my_output_product
        else:
            needed = parent.inputs_needed
            secured = parent.inputs_secured
            product_id = parent.awi.my_input_product

        if offer is None:
            return -1000

        if (
            offer[TIME] < parent.awi.current_step
            or offer[TIME] > parent.awi.n_steps - 1
        ):
            return -1000.0

        # Since contracts are executed before production, a sell offer for today
        # for amount bigger than current output inventory is useless.
        # input_inventory = self.awi.state.inventory[self.awi.my_input_product]
        output_inventory = parent.awi.state.inventory[parent.awi.my_output_product]
        if (
            self._is_seller
            and offer[TIME] == parent.awi.current_step
            and offer[QUANTITY] > output_inventory
        ):
            return -1000.0

        price_util = offer[UNIT_PRICE] / parent.awi.catalog_prices[product_id]
        if self._is_seller == False:
            price_util = -price_util

        if offer[QUANTITY] > (needed[offer[TIME]] - secured[offer[TIME]]):
            quantity_util = (needed[offer[TIME]] - secured[offer[TIME]]) / offer[
                QUANTITY
            ]
        else:
            quantity_util = offer[QUANTITY] / (
                needed[offer[TIME]] - secured[offer[TIME]]
            )

        time_factor = (0.9) ** (offer[TIME] - parent.awi.current_step + 1)

        util = (
            price_util * self._price_weight + (1 - self._price_weight) * quantity_util
        ) * time_factor
        return util

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        """Is this a valid offer for that negotiation"""
        issues = self.negotiators[negotiator_id][0].ami.issues
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
            -counter_all intends to get the best **valid** offers. By valid it
            means that counter_all will reject offers that can't be fulfilled.
            e.g. sell tomorrow when we don't have the inputs today
            (inputs_secured[now] < output_needed[tomorrow])
            - if called more than self._max_cycles times sequentially - reject and end.
        """

        self.seq_cntr += 1

        if self._is_seller and (int(self.__parent.awi.agent.id[1]) == 5):
            a = 5

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array([self.myUtil(o) for o in offers.values()])

        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.best_offers()

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            if self.seq_cntr > self._max_cycles:
                response = ResponseType.END_NEGOTIATION
            else:
                response = ResponseType.REJECT_OFFER
            return {k: SAOResponse(response, best_proposals[k]) for k in offers.keys()}

        relative_time = min(_.relative_time for _ in states.values())

        if self._is_seller:
            if self.overall_best_util_sell < self._best_utils[best_partner]:
                self.overall_best_util_sell = self._best_utils[best_partner]
            compare_util = self.overall_best_util_sell
        else:
            if self.overall_best_util_buy < self._best_utils[best_partner]:
                self.overall_best_util_buy = self._best_utils[best_partner]
            compare_util = self.overall_best_util_buy

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility
            >= self._utility_threshold * compare_util
            # or relative_time > self._time_threshold
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
        if self.seq_cntr > self._max_cycles:
            response = ResponseType.END_NEGOTIATION
        else:
            response = ResponseType.REJECT_OFFER
        responses = {
            k: SAOResponse(
                response,
                best_offer if self.is_valid(k, best_offer) else best_proposals[k],
            )
            for k in offers.keys()
        }
        # responses[best_partner] = SAOResponse(
        #    ResponseType.REJECT_OFFER, best_proposals[best_partner]
        # )
        return responses

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return self.best_offers()

    def best_proposal_for_nid(self, nid: str) -> Tuple[Optional["Outcome"], float]:
        negotiator = self.negotiators[nid][0]
        if negotiator.ami is None:
            return None, -1000
        utils = np.array([self.myUtil(_) for _ in negotiator.ami.outcomes])
        best_indx = np.argmax(utils)
        self._best_utils[nid] = utils[best_indx]
        if utils[best_indx] < 0:
            return None, utils[best_indx]
        return negotiator.ami.outcomes[best_indx], utils[best_indx]

    def best_offers(self) -> Dict[str, "Outcome"]:
        " same as first_proposals but with myUtil "
        return {
            nid: self.best_proposal_for_nid(nid)[0] for nid in self.negotiators.keys()
        }

    def on_negotiation_end(self, negotiator_id: str, state: "MechanismState") -> None:
        """Update the secured quantities whenever a negotiation ends"""
        if state.agreement is None:
            return

        q, t = state.agreement[QUANTITY], state.agreement[TIME]
        if self._is_seller:
            self.__parent.outputs_secured[t] += q
        else:
            self.__parent.inputs_secured[t] += q


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
            min_price = self.awi.catalog_prices[product]
            if seller:
                # for selling set a price that is at least the catalog price
                price_range = (min_price, 2 * min_price)
            else:
                # for buying selling a price that is at most the catalog price
                price_range = (min_price / 3, min_price)
            # self.controllers[seller].clear()
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
        controller = self.controllers[not annotation["is_buy"]]
        if controller is None:
            return None
        return controller.create_negotiator()


class MyNewAgent(
    MyNegotiationManager,
    ReactiveTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


from collections import defaultdict


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    world = SCML2020World(
        **SCML2020World.generate(
            [MyNewAgent, DecentralizingAgent],
            n_steps=10,
            n_processes=3,
            n_agents_per_process=3,
        ),
        construct_graphs=True,
    )
    world.run_with_progress()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()
    show_agent_scores(world)
