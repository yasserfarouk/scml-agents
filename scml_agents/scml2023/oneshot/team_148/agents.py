from pprint import pprint

from negmas import ResponseType, SAOResponse
from scml.oneshot import *
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE

__all__ = ["AgentVSCforOneShot"]


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
        self.has_agreed = {neg_id: False for neg_id in self.negotiators.keys()}

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

        my_id = self.awi.agent.id
        negotiator_id = list(contract.partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]

        self.has_agreed[negotiator_id] = True

    def propose(self, negotiator_id: str, state) -> "Outcome":
        # (f'step{self.awi.current_step} round{state.step} {self.awi.agent.id} propose for {negotiator_id}')

        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # print(f'step{self.awi.current_step} round{state.step} {self.awi.agent.id} respond to {negotiator_id}')

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
            min(
                my_needs
                / (len(self.negotiators.keys()) - sum(list(self.has_agreed.values()))),
                quantity_issue.max_value,
            ),
            quantity_issue.min_value,
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

    def __init__(self, *args, concession_exponent=2.0, **kwargs):
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


class AgentVSCforOneShot(OneShotSyncAgent, BetterAgent):
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_step(self):
        super().before_step()
        self.ufun.find_limit(True)  # calculate self.ufun.max_utility
        self.ufun.find_limit(False)  # calculate self.ufun.min_utility

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        offers = dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )
        # print_info = {neg_id:(offer[QUANTITY] if offer else None) for neg_id,offer in offers.items()}
        # print(f'day{self.awi.current_step} {self.awi.agent.id} my_needs:{self._needed()} first_proposals:{print_info}')
        return offers

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }
        my_needs = self._needed()
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.items(), is_selling),
            key=lambda x: (-x[0][1][UNIT_PRICE]) if x[1] else x[0][1][UNIT_PRICE],
        )
        sorted_offers = sorted(sorted_offers, key=lambda x: -x[0][1][QUANTITY])
        """print('offers',offers)
        pass # print('sorted_offers',sorted_offers)"""

        secured, outputs, chosen = 0, [], dict()
        """for i, k in enumerate(offers.keys()):
            offer, is_output = sorted_offers[i]
            secured += offer[QUANTITY]
            if secured >= my_needs:
                break
            chosen[k] = offer
            outputs.append(is_output)"""

        """for (k,offer),is_output in sorted_offers:
            secured += offer[QUANTITY]
            if secured >= my_needs:
                break
            chosen[k] = offer
            outputs.append(is_output)"""
        for (k, offer), is_output in sorted_offers:
            if secured + offer[QUANTITY] <= my_needs:
                secured += offer[QUANTITY]
                chosen[k] = offer
                outputs.append(is_output)

        """print('offers',offers)
        pass # print('sorted_offers',sorted_offers)
        pass # print('chosen',chosen)
        pass # print()"""

        u = self.ufun.from_offers(tuple(chosen.values()), tuple(outputs))
        rng = self.ufun.max_utility - self.ufun.min_utility
        self._threshold = 0.1 + 0.8 * sum(
            [
                self._th(state.step, self.get_nmi(neg_id).n_steps)
                for neg_id, state in states.items()
            ]
        ) / len(states)
        threshold = self._threshold * rng + self.ufun.min_utility
        # print(f'threshold:{self._threshold:.2f}')
        if u >= threshold:
            for k, v in chosen.items():
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        # print_info = [{'neg_id':neg_id,'round':states[neg_id].step,'offers(quantity)':offers[neg_id][QUANTITY] if offers[neg_id] else None,'response':responses[neg_id].response if responses[neg_id] else None} for neg_id in states.keys()]
        # pprint(print_info,width=150)

        return responses

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
            min(
                my_needs
                / (len(self.negotiators.keys()) - sum(list(self.has_agreed.values()))),
                quantity_issue.max_value,
            ),
            quantity_issue.min_value,
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)
