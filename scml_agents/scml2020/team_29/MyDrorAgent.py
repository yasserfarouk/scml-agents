from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from negmas import (
    LinearUtilityFunction,
    ResponseType,
    SAOSyncController,
    outcome_is_valid,
)
from negmas.sao import SAOResponse
from playground_agent import MyAgent, MyAgentDror
from scml.scml2020 import (
    QUANTITY,
    TIME,
    UNIT_PRICE,
    DecentralizingAgent,
    IndDecentralizingAgent,
    RandomAgent,
    SCML2020Agent,
    SCML2020World,
)
from scml.scml2020.components import TradePredictionStrategy
from scml.scml2020.components.negotiation import (
    IndependentNegotiationsManager,
    MovingRangeNegotiationManager,
    StepNegotiationManager,
)
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
)
from scml.scml2020.components.trading import PredictionBasedTradingStrategy


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

        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]
        if self._is_seller:
            return (
                price
                * q
                / (
                    abs(
                        q - self.__parent.outputs_needed[self.__parent.awi.current_step]
                    )
                    + 0.1
                )
            )
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
            if seller:
                # for selling set a price that is at least the catalog price
                min_price = self.awi.catalog_prices[product]
                price_range = (min_price, 2 * min_price)
            else:
                # for buying sell a price that is at most the catalog price
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
        controller = self.controllers[not annotation["is_buy"]]
        if controller is None:
            return None
        return controller.create_negotiator()


class MyDrorAgent(
    MyNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def target_quantity(self, step: int, sell: bool) -> int:
        """A fixed target quantity of half my production capacity"""
        return self.awi.n_lines // 2

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """The catalog price seems OK"""
        return (
            1.2 * self.awi.catalog_prices[self.awi.my_output_product]
            if sell
            else 0.8 * self.awi.catalog_prices[self.awi.my_input_product]
        )

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((0, -0.5, -0.8), issues=issues, outcomes=outcomes)


class MyPredictor(TradePredictionStrategy):
    def trade_prediction_init(self):
        inp = self.awi.my_input_product
        self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
        self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)


class TheirFinalAgent(MyPredictor, DecentralizingAgent):
    pass


class DrorFinalAgent(MyPredictor, MyDrorAgent):
    pass


class DrorStepAgent(
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
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
        MovingRangeNegotiationManager.__init__(
            self,
            *args,
            price_weight=0.7,
            utility_threshold=0.9,
            time_threshold=0.9,
            time_horizon=0.1,
            **kwargs,
        )
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

    # price_weight: The relative importance of price in the utility calculation.
    # utility_threshold: The fraction of maximum utility above which all offers will be accepted.
    # time_threshold: The fraction of the negotiation time after which any valid offers will be accepted.
    # time_range: The time-range for each controller as a fraction of the number of simulation steps

    # def __init__(self, *args, negotiator_type, negotiator_params, **kwargs):
    #     super().__init__(*args, **kwargs)
    #
    #
    # def set_param(self, steps_forward):
    #         self.steps_forward = steps_forward
    #
    # def acceptable_unit_price(self, step: int, sell: bool) -> int:
    #     production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
    #     if sell:
    #         return production_cost + self.input_cost[step]
    #     return self.output_price[step] - production_cost
    #
    # def target_quantity(self, step: int, sell: bool) -> int:
    #     if sell:
    #         needed, secured = self.outputs_needed, self.outputs_secured
    #     else:
    #         needed, secured = self.inputs_needed, self.inputs_secured
    #
    #     return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    print(scores)
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


def return_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return scores


ComparisonAgent = DecentralizingAgent
comp2 = IndDecentralizingAgent

# simulation with our agent
# world = SCML2020World(
#     **SCML2020World.generate([MyDrorAgent, TheirFinalAgent, DrorFinalAgent, comp2, ComparisonAgent], n_steps=15),
#     construct_graphs=True,
# )

scores = {}
scores["DrorStepAgent"] = {}
scores["DecentralizingAgent"] = {}
for output_step in range(0, 2):
    for n_simulations in range(10):
        test_agent = DrorStepAgent
        # test_agent.set_param(test_agent, steps_forward=output_step)
        world = SCML2020World(
            **SCML2020World.generate([test_agent, DecentralizingAgent], n_steps=10),
            construct_graphs=True,
        )

        world.run()
        returned_scores = return_agent_scores(world)
        print("dror:%s" % returned_scores["DrorStepAgent"])
        print("agent:%s" % returned_scores["DecentralizingAgent"])

        if scores["DrorStepAgent"].get(output_step):
            scores["DrorStepAgent"][output_step] += returned_scores["DrorStepAgent"]
        else:
            scores["DrorStepAgent"][output_step] = returned_scores["DrorStepAgent"]

        if scores["DecentralizingAgent"].get(output_step):
            scores["DecentralizingAgent"][output_step] += returned_scores[
                "DecentralizingAgent"
            ]
        else:
            scores["DecentralizingAgent"][output_step] = returned_scores[
                "DecentralizingAgent"
            ]

print(scores)

# world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
# plt.show()


# show_agent_scores(world)
