from abc import ABC
from collections import defaultdict

from negmas import ResponseType, SAOResponse
from negmas.outcomes import Outcome
from scml.oneshot import *

Buy = 0
Sell = 1


class SimpleAgent(OneShotAgent, ABC):
    """A greedy agent based on OneShotAgent"""

    def __init__(self, owner=None, ufun=None, name=None):
        super().__init__(owner, ufun, name)
        self.secured = 0

    def init(self):
        pass

    def step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
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

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product


class BetterAgent(SimpleAgent, ABC):
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
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        nmi = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(nmi, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent, ABC):
    """Considers best price offers received when making its decisions"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(args, concession_exponent, kwargs)
        self._best_selling, self._best_buying = 0.0, float("inf")

    def init(self):
        super().init()

    def step(self):
        super().step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state):
        """Save the best price received"""
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


class LearningAgent(AdaptiveAgent, ABC):
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
        if offer is None:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state)
        # update my current best price to use for limiting concession in other
        # negotiations
        nmi = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
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
            partner = nmi.annotation["seller"]
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


class LearningSyncAgent(OneShotSyncAgent, LearningAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def init(self):
        super().init()

    def step(self):
        super().step()

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        SECURED_MAGNIFICATION = 1.2

        my_needs = self._needed()
        if my_needs <= 0:
            # my_needsを満たしているときは交渉終了
            responses = dict(
                zip(
                    [str(k) for k in offers.keys()],
                    [
                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for _ in offers.keys()
                    ],
                )
            )
            return responses

        counter_offers = {}
        for name, offer in offers.items():
            counter_offer = super(LearningAgent, self).propose(name, states[name])
            list(counter_offer)[QUANTITY] = min(
                self.awi.profile.n_lines, counter_offer[QUANTITY]
            )
            counter_offers[name] = tuple(counter_offer)

        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in counter_offers.items()
        }
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output = sorted_offers[i]
            secured += offer[QUANTITY]
            if secured >= my_needs * SECURED_MAGNIFICATION:
                break
            chosen[k] = offer
            outputs.append(is_output)

        for k, v in chosen.items():
            response = super(LearningAgent, self).respond(k, states[k], offers[k])
            if response == ResponseType.ACCEPT_OFFER:
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses


def print_log(names, values, on=False):
    if on:
        if type(names) == str:
            print(f"{names}:{values}")
        if type(names) == list:
            for name, value in dict(zip(names, values)).items():
                print(f"{name}:{value}", end=" ")
            print()
