from collections import defaultdict

from negmas import ResponseType
from scml.oneshot import *


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

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
            if offer[QUANTITY] <= my_needs
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
        # print(f'I am {self.awi.agent} and I am making {offer} at step {state.step}')
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
            # if state.step >= 18:
            #     print(f'I am {self.awi.agent} and would accept {th*(mx - mn) + mn} at step {state.step}')
            return (price - mn) >= th * (mx - mn)
        else:
            # if state.step >= 18:
            #     print(f'I am {self.awi.agent} and I would accept {mx - th*(mx - mn)} at step {state.step}')
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
        **kwargs
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
        # print(f'I have accepted a contract for {contract.agreement["quantity"]} at price {up} and step '
        #       f'{contract.agreement["time"]}')
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
            # print(f'I am {self.awi.agent} and trading with {partner}')
            # print([
            #         p * (1 + slack)
            #         for p, slack in (
            #             (self._best_buying, self._step_price_slack),
            #             (self._best_acc_buying, self._acc_price_slack),
            #             (self._best_opp_buying[partner], self._opp_price_slack),
            #             (
            #                 self._best_opp_acc_buying[partner],
            #                 self._opp_acc_price_slack,
            #             ),
            #         )
            #     ])
        return mn, mx


class RejectAgent(OneShotAgent):
    def before_step(self):
        if self.awi.level == 0:
            self.q = self.awi.state.exogenous_input_quantity
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price

        elif self.awi.level == 1:
            self.q = self.awi.state.exogenous_output_quantity
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price
        self.opp_q = self.q

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        self.opp_q = offer[QUANTITY]
        if self.awi.level == 0:
            if offer[UNIT_PRICE] >= self.best_price and offer[QUANTITY] <= self.q:
                return ResponseType.ACCEPT_OFFER
        elif self.awi.level == 1:
            if offer[UNIT_PRICE] <= self.best_price and offer[QUANTITY] <= self.q:
                return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, negotiator_id: str, state):
        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        offer[QUANTITY] = min(self.q, self.opp_q)
        offer[UNIT_PRICE] = self.best_price
        if state.step == 19:
            offer[UNIT_PRICE] = self.min_price
        offer = tuple(offer)
        return offer

    def on_negotiation_success(self, contract, mechanism):
        self.q -= contract.agreement["quantity"]


class PrintAgent(RejectAgent):
    def before_step(self):
        super().before_step()
        pass  # print(f'My best price is {self.best_price}')

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        response = ResponseType.ACCEPT_OFFER
        pass  # print(f'I am {self.awi.agent} and it is time step {state.step} and I am responding to {offer} with response {response}')
        return response

    def propose(self, negotiator_id: str, state):
        offer = super().propose(negotiator_id, state)
        pass  # print(f'I am {self.awi.agent} and it is time step {state.step} and I am proposing {offer}')
        return offer
