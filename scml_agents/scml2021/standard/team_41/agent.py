import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scml.scml2020 import SCML2020Agent, SCML2021World
from scml.scml2020.agents.decentralizing import DecentralizingAgentWithLogging

__all__ = [
    "MyDoNothing",
    "MyAgent",
    "MyNewAgent",
    "AspirationAgent",
    "MyDecentralizingAgent",
    "FromScratchAgent",
    "ProactiveFromScratch",
]

# 何もしない

from scml.scml2020.agents import (
    DecentralizingAgent,
    MarketAwareDecentralizingAgent,
    RandomAgent,
)

ComparisonAgent = MarketAwareDecentralizingAgent


class MyDoNothing(SCML2020Agent):
    """My Agent that does nothing"""


world = SCML2021World(
    **SCML2021World.generate(
        agent_types=[ComparisonAgent, MyDoNothing, DecentralizingAgentWithLogging],
        n_steps=10,
    ),
    construct_graphs=True,
)

if __name__ == "__main__":
    world.run()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show() ##########1###########


# 需要優先型
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
)


class MyAgent1(DemandDrivenProductionStrategy):
    """My agent"""


class DemandDrivenProductionStrategy(ProductionStrategy):
    def on_contracts_finalized(self, signed, cancelled, rejectors):
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            # do nothing if this is not a sell contract
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # Schedule production before the delivery time
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(earliest_production, step - 1),
                line=-1,
                partial_ok=True,
            )
            # set the schedule_range which is provided for other components
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )
            # that is all folks


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([ComparisonAgent, MyDoNothing], n_steps=10),
        construct_graphs=True,
    )
    world.run()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()  ##########2##########

from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy

# PredictionBasedTradingStrategy
from scml.scml2020.components.production import DemandDrivenProductionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy


class MyAgent(
    MarketAwareTradePredictionStrategy,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    """My agent"""


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([ComparisonAgent, MyAgent], n_steps=10),
        construct_graphs=True,
    )
    world.run_with_progress()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()  ###########3############


# NegotiationControlStrategy
from scml.scml2020.components.negotiation import IndependentNegotiationsManager


class MyAgent(
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([ComparisonAgent, MyAgent], n_steps=10),
        construct_graphs=True,
    )
    try:
        world.run()
    except ValueError as e:
        print(e)

from negmas import LinearUtilityFunction


class MyAgent(
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    def target_quantity(self, step: int, sell: bool) -> int:
        """A fixed target quantity of half my production capacity"""
        return self.awi.n_lines // 2

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """The catalog price seems OK"""
        return (
            self.awi.catalog_prices[self.awi.my_output_product]
            if sell
            else self.awi.catalog_prices[self.awi.my_input_product]
        )

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((0, -0.5, -0.8), issues=issues, outcomes=outcomes)


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([ComparisonAgent, MyAgent, RandomAgent], n_steps=10),
        construct_graphs=True,
    )
    world.run_with_progress()

    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()   ##########4##########

from collections import defaultdict


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


# show_agent_scores(world)  ##########5##########


from typing import Any, Dict, List, Optional, Tuple

from negmas import ResponseType, UtilityFunction, outcome_is_valid
from negmas.sao import SAOResponse, SAOSyncController

# NegotiationControlStrategy
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

        q, t = state.agreement[QUANTITY], state.agreement[TIME]
        if self._is_seller:
            self.__parent.outputs_secured[t] += q
        else:
            self.__parent.inputs_secured[t] += q


# MyNegotiationManager
class MyNegotiationManager:
    """My negotiation strategy

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
        controller = self.controllers[self.id == annotation["seller"]]
        if controller is None:
            return None
        return controller.create_negotiator()


class MyNewAgent(
    MyNegotiationManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([ComparisonAgent, MyAgent, MyNewAgent], n_steps=10),
        construct_graphs=True,
    )
    world.run_with_progress()

    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()
    show_agent_scores(world)

from scml.scml2020 import is_system_agent


def analyze_unit_price(world, agent_type):
    """Returns the average price relative to the negotiation issues"""
    contracts = pd.DataFrame(world.saved_contracts)
    fields = ["seller_type", "buyer_type", "unit_price", "issues", "selling", "buying"]
    # Add fields indicating whether the agent_type is a seller or a buyer
    contracts["seller_type"] = contracts.seller_type.apply(lambda x: x.split(".")[-1])
    contracts["buyer_type"] = contracts.buyer_type.apply(lambda x: x.split(".")[-1])
    contracts["selling"] = contracts.seller_type == agent_type
    contracts["buying"] = contracts.buyer_type == agent_type
    # keep only contracts in which agent_type is participating
    contracts = contracts.loc[contracts.selling | contracts.buying, fields]
    # remove all exogenous contracts
    contracts = contracts.loc[contracts.issues.apply(len) > 0, fields]
    # find the minimum and maximum unit price in the negotiation issues
    min_vals = contracts.issues.apply(lambda x: x[UNIT_PRICE].min_value)
    max_vals = contracts.issues.apply(lambda x: x[UNIT_PRICE].max_value)
    # replace the unit price with its fraction of the unit-price issue range
    contracts.unit_price = (contracts.unit_price - min_vals) / (max_vals - min_vals)
    contracts = contracts.drop("issues", 1)
    contracts = contracts.rename(columns=dict(unit_price="price"))
    # group results by whether the agent is selling/buying/both
    if len(contracts) < 1:
        return ""
    print(f"{agent_type}:\n===========")
    return contracts.groupby(["selling", "buying"]).describe().round(1)


if __name__ == "__main__":
    print(analyze_unit_price(world, "MyNewAgent"))
    print(analyze_unit_price(world, "MyAgent"))
    print(analyze_unit_price(world, "DecentralizingAgent"))

    world = SCML2021World(
        **SCML2021World.generate([MyNewAgent], n_steps=10), construct_graphs=True
    )
    world.run_with_progress()
    print(analyze_unit_price(world, "MyNewAgent"))

from negmas import SAOMetaNegotiatorController

"""
controller = SAOMetaNegotiatorController(ufun=LinearUtilityFunction({
    TIME: 0.0, QUANTITY: (1-x), UNIT_PRICE: x if seller else -x
}))
"""


class YetAnotherNegotiationManager:
    """My new negotiation strategy

    Args:
        price_weight: The relative importance of price in the utility calculation.
        time_range: The time-range for each controller as a fraction of the number of simulation steps
    """

    def __init__(
        self,
        *args,
        price_weight=0.7,
        time_horizon=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._price_weight = price_weight
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
                controller = SAOMetaNegotiatorController(
                    ufun=LinearUtilityFunction(
                        {
                            TIME: 0.0,
                            QUANTITY: (1 - self._price_weight),
                            UNIT_PRICE: self._price_weight,
                        },
                    )
                )
            else:
                # for buying sell a price that is at most the catalog price
                price_range = (0, self.awi.catalog_prices[product])
                controller = SAOMetaNegotiatorController(
                    ufun=LinearUtilityFunction(
                        {
                            TIME: 0.0,
                            QUANTITY: (1 - self._price_weight),
                            UNIT_PRICE: -self._price_weight,
                        },
                    )
                )

            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=controller,
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        return None


class AspirationAgent(
    YetAnotherNegotiationManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate([MyAgent, MyNewAgent, AspirationAgent], n_steps=10),
        construct_graphs=True,
    )
    world.run_with_progress()

    show_agent_scores(world)
    for agent_type in ("MyNewAgent", "AspirationAgent"):
        print(analyze_unit_price(world, agent_type))

from scml.scml2020.components import MarketAwareTradePredictionStrategy


class MyPredictor(MarketAwareTradePredictionStrategy):
    def trade_prediction_init(self):
        inp = self.awi.my_input_product
        self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
        self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)


class MyDecentralizingAgent(MyPredictor, DecentralizingAgent):
    pass


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate(
            [AspirationAgent, ComparisonAgent, MyDecentralizingAgent], n_steps=10
        ),
        construct_graphs=True,
    )
    world.run_with_progress()

    show_agent_scores(world)

from negmas import AspirationNegotiator, MappingUtilityFunction
from scml.scml2020 import NO_COMMAND


class FromScratchAgent(SCML2020Agent):
    def init(self):
        self.prices = [
            self.awi.catalog_prices[self.awi.my_input_product],
            self.awi.catalog_prices[self.awi.my_output_product],
        ]
        self.quantities = [1, 1]

    def step(self):
        super().step()
        # update prices based on market information if available
        tp = self.awi.trading_prices
        if tp is None:
            self.prices = [
                self.awi.catalog_prices[self.awi.my_input_product],
                self.awi.catalog_prices[self.awi.my_output_product],
            ]
        else:
            self.prices = [
                self.awi.trading_prices[self.awi.my_input_product],
                self.awi.trading_prices[self.awi.my_output_product],
            ]

        # produce everything I can
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        self.awi.set_commands(commands)

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        is_seller = annotation["seller"] == self.id
        # do not engage in negotiations that obviouly have bad prices for me
        if is_seller and issues[UNIT_PRICE].max_value < self.prices[is_seller]:
            return None
        if not is_seller and issues[UNIT_PRICE].min_value > self.prices[is_seller]:
            return None
        ufun = self.create_ufun(
            is_seller,
            (issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value),
        )
        return AspirationNegotiator(ufun=ufun)

    def sign_all_contracts(self, contracts: List["Contract"]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        return [self.id] * len(contracts)

    def on_contracts_finalized(
        self,
        signed: List["Contract"],
        cancelled: List["Contract"],
        rejectors: List[List[str]],
    ) -> None:
        awi: AWI = self.awi
        for contract in signed:
            t, p, q = (
                contract.agreement["time"],
                contract.agreement["unit_price"],
                contract.agreement["quantity"],
            )
            is_seller = contract.annotation["seller"] == self.id
            oldq = self.quantities[is_seller]
            self.quantities[is_seller] += q
            self.prices[is_seller] = (
                oldq * self.prices[is_seller] + p * q
            ) / self.quantities[is_seller]

    def create_ufun(self, is_seller, prange):
        if is_seller:
            return MappingUtilityFunction(
                lambda x: -1000 if x[UNIT_PRICE] < self.prices[1] else x[UNIT_PRICE],
                reserved_value=0.0,
            )
        return MappingUtilityFunction(
            lambda x: -1000
            if x[UNIT_PRICE] > self.prices[0]
            else prange[1] - x[UNIT_PRICE],
            reserved_value=0.0,
        )


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate(
            [AspirationAgent, ComparisonAgent, FromScratchAgent], n_steps=10
        ),
        construct_graphs=True,
    )
    world.run_with_progress()
    show_agent_scores(world)


class ProactiveFromScratch(FromScratchAgent):
    def on_contracts_finalized(
        self,
        signed: List["Contract"],
        cancelled: List["Contract"],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        awi: AWI = self.awi
        for contract in signed:
            t, p, q = (
                contract.agreement["time"],
                contract.agreement["unit_price"],
                contract.agreement["quantity"],
            )
            is_seller = contract.annotation["seller"] == self.id
            if contract.annotation["caller"] == self.id:
                continue
            product = awi.my_output_product if is_seller else awi.my_input_product
            partners = awi.my_consumers if is_seller else awi.my_suppliers
            qrange = (1, q)
            prange = self.prices[not is_seller]
            trange = (awi.current_step, t) if is_seller else (t, awi.n_steps - 1)
            negotiators = [
                AspirationNegotiator(ufun=self.create_ufun(is_seller, prange))
                for _ in partners
            ]
            awi.request_negotiations(
                is_buy=is_seller,
                product=product,
                quantity=qrange,
                unit_price=prange,
                time=trange,
                controller=None,
                negotiators=negotiators,
            )


if __name__ == "__main__":
    world = SCML2021World(
        **SCML2021World.generate(
            [ProactiveFromScratch, ComparisonAgent, FromScratchAgent], n_steps=10
        ),
        construct_graphs=True,
    )
    world.run_with_progress()
    show_agent_scores(world)

    import seaborn as sns
    from scml.utils import anac2021_std

    tournament_types = [
        ProactiveFromScratch,
        FromScratchAgent,
        MyAgent,
        MyNewAgent,
        AspirationAgent,
    ]
    results = anac2021_std(
        competitors=tournament_types,
        n_configs=20,  # number of different configurations to generate
        n_runs_per_world=1,  # number of times to repeat every simulation
        n_steps=(30, 60),  # number of days (simulation steps) per simulation
    )
    print(results.total_scores)
