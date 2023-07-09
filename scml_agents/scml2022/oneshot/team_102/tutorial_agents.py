from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AspirationNegotiator,
    LinearUtilityFunction,
    MappingUtilityFunction,
    ResponseType,
    SAOMetaNegotiatorController,
    SAOResponse,
)
from scml import AWI, NO_COMMAND, SCML2020Agent
from scml.oneshot import *
from scml.scml2020.components.negotiation import IndependentNegotiationsManager
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.production import DemandDrivenProductionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from scml.scml2020.services import SyncController

__all__ = [
    "SimpleAgent",
    "BetterAgent",
    "AdaptiveAgent",
    "LearningAgent",
    "SyncAgent",
    "SampleAgent1",
    "SampleAgent2",
    "SampleAgent2a",
    "SampleAgent3",
    "AspirationAgent",
    "FromScratchAgent",
    "ProactiveFromScratch",
]


# Oneshot Agents
class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def __init__(self, *args, q_bias=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_bias = q_bias

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs * self._q_bias
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product


class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        ami = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(ami, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(ami):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, ami, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def before_step(self):
        super().before_step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state):
        """Save the best price received"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        ami = self.get_nmi(negotiator_id)
        if self._is_selling(ami):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(ami)
        if self._is_selling(ami):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


class LearningAgent(AdaptiveAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state)
        # update my current best price to use for limiting concession in other
        # negotiations
        ami = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = ami.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx


class SyncAgent(OneShotSyncAgent, BetterAgent):
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(self, *args, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    def before_step(self):
        super().before_step()
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }
        my_needs = self._needed()
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output = sorted_offers[i]
            secured += offer[QUANTITY]
            if secured >= my_needs:
                break
            chosen[k] = offer
            outputs.append(is_output)

        u = self.ufun.from_offers(tuple(chosen.values()), tuple(outputs))
        rng = self.ufun.max_utility - self.ufun.min_utility
        threshold = self._threshold * rng + self.ufun.min_utility
        if u >= threshold:
            for k, v in chosen.items():
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses


# Standard/Collusion Agents
class SampleNegotiationManager:
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
        mechanism: "NegotiatorMechanismInterface",
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


class SampleAgent1(
    MarketAwareTradePredictionStrategy,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class SampleAgent2(
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class SampleAgent2a(SampleAgent2):
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


class SampleAgent3(
    SampleNegotiationManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


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
                        (0.0, (1 - self._price_weight), 0.0, self._price_weight)
                    )
                )
            else:
                # for buying sell a price that is at most the catalog price
                price_range = (0, self.awi.catalog_prices[product])
                controller = SAOMetaNegotiatorController(
                    ufun=LinearUtilityFunction(
                        ((1 - self._price_weight), 0.0, -self._price_weight)
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
        mechanism: "NegotiatorMechanismInterface",
    ) -> Optional["Negotiator"]:
        return None


class AspirationAgent(
    YetAnotherNegotiationManager,
    PredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


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
        mechanism: "NegotiatorMechanismInterface",
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
            issues,
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

    def create_ufun(self, is_seller, prange, issues):
        if is_seller:
            return MappingUtilityFunction(
                lambda x: -1000 if x[UNIT_PRICE] < self.prices[1] else x[UNIT_PRICE],
                reserved_value=0.0,
                issues=issues,
            )
        return MappingUtilityFunction(
            lambda x: -1000
            if x[UNIT_PRICE] > self.prices[0]
            else prange[1] - x[UNIT_PRICE],
            reserved_value=0.0,
            issues=issues,
        )


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
                AspirationNegotiator(
                    ufun=self.create_ufun(is_seller, prange, issues=None)
                )
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
