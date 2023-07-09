from collections import defaultdict

from negmas import SAONMI, MechanismState, Outcome, ResponseType, SAOResponse
from negmas.preferences import LinearAdditiveUtilityFunction, LinearUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun
from scml.oneshot import *

__all__ = [
    "MyOneShotDoNothing",
    "GreedyAgent",
    "SimpleAgent",
    "AdaptiveAgent",
    "BetterAgent",
    "SyncAgent",
    "SimpleSyncAgent",
    "MySyncAgent",
    "LearningAgent",
    "DeepSimpleAgent",
    "SimpleSingleAgreementAgent",
    "GreedyIndNeg",
]

from typing import List


class MyOneShotDoNothing(OneShotAgent):
    """My Agent that does nothing"""

    def propose(self, negotiator_id, state):
        return None

    def respond(self, negotiator_id, state, source=""):
        return ResponseType.END_NEGOTIATION


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
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


class GreedyAgent(SimpleAgent):
    def propose(self, negotiator_id: str, state: MechanismState) -> Outcome:
        nmi = self.get_nmi(negotiator_id)
        q = self._needed(negotiator_id)
        t = self.awi.current_step
        p = (
            nmi.issues[UNIT_PRICE].max_value
            if self._is_selling(nmi)
            else nmi.issues[UNIT_PRICE].min_value
        )

        return tuple([q, t, p])

    def respond(self, negotiator_id, state, source=""):
        return ResponseType.ACCEPT_OFFER


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

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state, source)
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

    def respond(self, negotiator_id, state, source=""):
        """Save the best price received"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state, source)
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

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state, source)
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


class DeepSimpleAgent(SimpleAgent):
    """A greedy agent based on OneShotSyncAgent that does something
    when in the middle of the production chain"""

    def before_step(self):
        self._sales = self._supplies = 0

    def on_negotiation_success(self, contract, mechanism):
        if contract.annotation["product"] == self.awi.my_input_product:
            self._sales += contract.agreement["quantity"]
        else:
            self._supplies += contract.agreement["quantity"]

    def _needed(self, negotiator_id):
        summary = self.awi.exogenous_contract_summary
        secured = (
            self._sales
            if self._is_selling(self.get_nmi(negotiator_id))
            else self._supplies
        )
        demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
        return demand - secured


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


class SimpleSyncAgent(SyncAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step_actions = 0

    def before_step(self):
        super().before_step()
        self.n_step_actions = 0

    def step(self):
        pass  # print(f'\rn_step_actions:{self.n_step_actions}', end='')
        super().step()

    def first_proposals(self):
        self.n_step_actions += 1
        return super().first_proposals()

    def counter_all(self, offers, states):
        self.n_step_actions += 1
        return super().counter_all(offers, states)


class MySyncAgent(SyncAgent):
    def __init__(self, *args, first_offer_q: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_offer_q = first_offer_q

    def first_proposals(self):
        proposals = super().first_proposals()
        nmi = self.get_nmi(list(proposals.keys())[0])
        price = self._best_price(nmi)

        proposals = self._distribute_quantity(list(proposals.keys()), price)

        return proposals

    def counter_all(self, offers, states):
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }
        my_needs = self._needed()
        nmi = self.get_nmi(list(offers.keys())[0])

        # sort opponent's offers by quantity
        sorted_offers = {
            k: v for k, v in sorted(offers.items(), key=lambda x: -x[1][QUANTITY])
        }

        # choose offer
        secured, chosen, unchosen = 0, {}, {}
        for k, offer in sorted_offers.items():
            if my_needs - secured - offer[QUANTITY] < 0:
                unchosen[k] = offer
            else:
                secured += offer[QUANTITY]
                chosen[k] = offer

        # make responses
        for k, v in chosen.items():
            responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        if secured <= 0:
            for k, v in unchosen.items():
                responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        else:
            # determine counter offer
            price = self._best_price(nmi)
            proposals = self._distribute_quantity(list(unchosen.keys()), price)
            for k, v in proposals.items():
                responses[k] = SAOResponse(ResponseType.REJECT_OFFER, v)

        return responses

    def _distribute_quantity(self, ids: List[str], price: int):
        n = len(ids)
        if n <= 0:
            return {}

        needs = self._needed()

        q = [0] * n
        for i in range(needs):
            q[i % n] += 1

        return {k: (v, self.awi.current_step, price) for k, v in zip(ids, q)}

    def _best_price(self, nmi: SAONMI):
        price = (
            nmi.issues[UNIT_PRICE].max_value
            if self._is_selling(nmi)
            else nmi.issues[UNIT_PRICE].min_value
        )
        return price


class SimpleSingleAgreementAgent(OneShotSingleAgreementAgent):
    """A greedy agent based on OneShotSingleAgreementAgent"""

    def before_step(self):
        self.ufun.find_limit(True)  # finds highest utility
        self.ufun.find_limit(False)  # finds lowest utility

    def is_acceptable(self, offer, source, state) -> bool:
        mx, mn = self.ufun.max_utility, self.ufun.min_utility
        u = (self.ufun(offer) - mn) / (mx - mn)
        return u >= (1 - state.relative_time)

    def best_offer(self, offers):
        ufuns = [(self.ufun(_), i) for i, _ in enumerate(offers.values())]
        keys = list(offers.keys())
        return keys[max(ufuns)[1]]

    def is_better(self, a, b, negotiator, state):
        return self.ufun(a) > self.ufun(b)


class GreedyIndNeg(OneShotIndNegotiatorsAgent):
    def generate_ufuns(self):
        d = dict()
        # generate ufuns that prefer higher prices when selling
        for partner_id in self.awi.my_consumers:
            issues = self.awi.current_output_issues
            if self.awi.is_system(partner_id):
                continue
            d[partner_id] = LinearUtilityFunction(
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                issues=issues,
                reserved_value=0.0,
            )
        # generate ufuns that prefer lower prices when selling
        for partner_id in self.awi.my_suppliers:
            issues = self.awi.current_input_issues
            if self.awi.is_system(partner_id):
                continue
            d[partner_id] = LinearAdditiveUtilityFunction(
                dict(
                    quantity=IdentityFun(),
                    time=IdentityFun(),
                    unit_price=AffineFun(slope=-1, bias=issues[UNIT_PRICE].max_value),
                ),
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                issues=issues,
                reserved_value=0.0,
            )
        return d
