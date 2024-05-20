import math
from collections import defaultdict

from negmas import ResponseType, SAOResponse
from negmas.outcomes import Outcome
from scml.oneshot import *

__all__ = ["QuantityOrientedAgent"]


class QuantityOrientedAgent(OneShotAgent):
    """Based on OneShotAgent"""

    def init(self):
        """Initializes some values needed for the negotiations"""
        if self.awi.level == 0:  # L0 agent
            self.n_partners = len(
                self.awi.all_consumers[1]
            )  # determines the number of l1 agents
        else:  # L1, Start the negotiations
            self.n_partners = len(
                self.awi.all_consumers[0]
            )  # determines the number of l0 agents

    def before_step(self):  # Resets counts
        self.secured = 0
        self.rejection = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def on_negotiation_failure(self, contract, mechanism, a, b):
        self.rejection = self.rejection + 1  # Tracks the numbers of rejections

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def propose(self, negotiator_id: str, state):
        p = 17
        step = state.step
        if (
            step <= p
        ):  # Insists on the best offer for itself for a given amount of time "p"
            return self.best_offer(negotiator_id, step)
        else:  # tries to settle using the best offer for its partners
            return self.quick_offer(negotiator_id, step)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        ami = self.get_nmi(negotiator_id)
        step = state.step
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        else:
            if offer[QUANTITY] == my_needs:  # Best possible outcome
                return ResponseType.ACCEPT_OFFER
            elif offer[QUANTITY] < my_needs:
                return (
                    ResponseType.ACCEPT_OFFER
                )  # Also accepts if the quantity is lower than needed
            elif (
                abs(offer[QUANTITY] - my_needs) < my_needs and step >= 18
            ):  # At the last rounds, might settle for a offer that minimizes the losses
                return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def best_offer(self, negotiator_id, step):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        if my_needs <= 5:
            offer[QUANTITY] = my_needs
        else:
            offer[QUANTITY] = math.ceil(my_needs / 2)  # splits demand.

        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def quick_offer(self, negotiator_id, step):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        if step <= 3:
            if my_needs <= 5:
                offer[QUANTITY] = my_needs
            else:
                offer[QUANTITY] = math.ceil(my_needs / 2)  # First offer: splits demand.
        else:
            offer[QUANTITY] = my_needs  # Offers exactly what the agent needs

        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[
                UNIT_PRICE
            ] = unit_price_issue.min_value  # Offers the best value FOR THE BUYER!!!
        else:
            offer[
                UNIT_PRICE
            ] = unit_price_issue.max_value  # Offers the best value FOR THE SELLER!!!
        return tuple(offer)

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

        secured, outputs, chosen = 0, [], dict()
        for (k, offer), is_output in sorted_offers:
            if secured + offer[QUANTITY] <= my_needs:
                secured += offer[QUANTITY]
                chosen[k] = offer
                outputs.append(is_output)

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

class KanbeAgent(AdaptiveAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._received_offers: dict[str, list[list[Outcome]]] = {}
        self._sent_offers: dict[str, list[Outcome]] = {}
        self._contracts: list[dict[str, Outcome]] = []
        self.first_target_steps = 5

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self.partners = (
            self.awi.my_consumers if self.awi.level == 0 else self.awi.my_suppliers
        )
        for partner_id in self.partners:
            self._received_offers[partner_id] = []
        self._bp_mxq_opp_acc_selling, self._bp_mnq_opp_acc_selling = defaultdict(
            int
        ), defaultdict(lambda: int(10))
        self._wp_mxq_opp_acc_selling, self._wp_mnq_opp_acc_selling = defaultdict(
            int
        ), defaultdict(lambda: int(10))
        self._bp_mxq_opp_acc_buying, self._bp_mnq_opp_acc_buying = defaultdict(
            int
        ), defaultdict(lambda: int(10))
        self._wp_mxq_opp_acc_buying, self._wp_mnq_opp_acc_buying = defaultdict(
            int
        ), defaultdict(lambda: int(10))

    def before_step(self):
        super().before_step()
        self._contracts.append({})
        for partner_id in self.partners:
            self._sent_offers[partner_id] = []
        if self.awi.level == 0:
            self.best_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.worst_price = self.awi.current_output_issues[UNIT_PRICE].min_value
        elif self.awi.level == 1:
            self.best_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.worst_price = self.awi.current_input_issues[UNIT_PRICE].max_value
        for k in self._received_offers.keys():
            self._received_offers[k].append([])
        self.contract_partners = 0
        self.remain_partners = len(self.partners)
        self.en_opponents = 0
        self.acc_worst_price = False
        self.firstproposer = False
        self.concession = False
        self.concession_step = -1
        self.second_target_steps = 15

    def propose(self, negotiator_id: str, state) -> Outcome | None:
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)

        unit_price, quantity = self._decide_price_quantity(
            self.get_nmi(negotiator_id), negotiator_id, state
        )
        new_offer = self._get_outcome(unit_price, quantity, offer[TIME])
        self._sent_offers[negotiator_id].append(new_offer)
        return new_offer

    def _decide_price_quantity(self, ami, negotiator_id, state):
        self._set_quantity_range(ami, negotiator_id, state)
        olo = self._get_olo(negotiator_id)
        if self._is_selling(ami):
            opbq = self._bp_mxq_opp_acc_selling[negotiator_id]
        else:
            opbq = self._bp_mxq_opp_acc_buying[negotiator_id]

        my_needs = self._needed(negotiator_id)
        # self.second_target_steps = min((20 - my_needs),15)
        self.second_target_steps = 20 - my_needs

        if state.step < self.first_target_steps:
            if self.concession and state.step == self.concession_step:
                price, quantity = self._mid_propose(
                    ami, negotiator_id, state, olo, opbq
                )
                return price, quantity
            price = self.best_price
            up = int(ami.issues[QUANTITY].max_value / 2)
            quantity = min(my_needs, max(up, opbq))

            if olo == None:
                self.firstproposer = True
            else:
                if olo[UNIT_PRICE] == self.best_price:
                    quantity = max(min(olo[QUANTITY], quantity), self.min_q)
        elif state.step < self.second_target_steps:
            price, quantity = self._mid_propose(ami, negotiator_id, state, olo, opbq)
        else:
            assert olo is not None
            if olo[UNIT_PRICE] == self.best_price:
                price = self.best_price
            else:
                price = self.worst_price
            if state.step >= 18:
                quantity = min(self.min_q, olo[QUANTITY])
            else:
                quantity = self.max_q
                if quantity < olo[QUANTITY]:
                    quantity = min(my_needs, olo[QUANTITY])
                else:
                    quantity = max(min(quantity, olo[QUANTITY]), self.min_q)
        return price, quantity

    def _mid_propose(self, ami, negotiator_id, state, olo, opbq):
        self._set_quantity_range(ami, negotiator_id, state)
        my_needs = self._needed(negotiator_id)
        if len(self._sent_offers[negotiator_id]) > 1:
            lp = self._sent_offers[negotiator_id][-1]
        else:
            if self.concession and state.step == self.concession_step:  # 個数はneededの半分
                if olo is not None:
                    if olo[UNIT_PRICE] == self.best_price:
                        price = self.best_price
                        quantity = max(min(olo[QUANTITY], my_needs), self.min_q)
                    else:
                        price = self.worst_price
                        up = int(self._needed(negotiator_id) / 2)
                        quantity = max(min(up, olo[QUANTITY]), self.min_q)
                else:
                    price = self.worst_price
                    quantity = int(self._needed(negotiator_id) / 2)
            else:
                price = self.best_price
                up = int(ami.issues[QUANTITY].max_value / 2)
                quantity = min(self._needed(negotiator_id), max(up, opbq))
            return price, quantity

        if self.step == self.first_target_steps:
            price = self.best_price
            quantity = self.max_q
            if olo is not None and olo[UNIT_PRICE] == self.best_price:
                quantity = max(min(olo[QUANTITY], quantity), self.min_q)
            return price, quantity
        if lp[UNIT_PRICE] == self.best_price:
            if self.concession and state.step == self.concession_step:
                if olo is not None:
                    if olo[UNIT_PRICE] == self.best_price:
                        price = self.best_price
                        quantity = max(min(olo[QUANTITY], my_needs), self.min_q)
                    else:
                        price = self.worst_price
                        quantity = max(min(self.max_q, olo[QUANTITY]), self.min_q)
                else:
                    price = self.worst_price
                    quantity = self.max_q
            else:
                price = self.best_price
                if lp[QUANTITY] <= self.min_q:
                    quantity = self.min_q
                else:
                    quantity = lp[QUANTITY] - 1
                if olo is not None:
                    if quantity < olo[QUANTITY]:
                        quantity = min(my_needs, olo[QUANTITY])
        else:
            if olo is not None:
                if olo[UNIT_PRICE] == self.best_price:
                    price = self.best_price
                    quantity = max(min(olo[QUANTITY], my_needs), self.min_q)
                    return price, quantity
            price = self.worst_price
            if lp[QUANTITY] > self.min_q:
                quantity = lp[QUANTITY] - 1
            else:
                quantity = self.min_q
            if olo is not None:
                quantity = max(self.min_q, min(quantity, olo[QUANTITY]))
        return price, quantity

    def _get_olo(self, partner_id):  # get opponent last offer
        target = self._received_offers[partner_id][self.awi.current_step]
        if len(target) == 0:
            return None
        else:
            return target[-1]

    def _get_outcome(self, unit_price: int, quantity: int, time: int) -> "Outcome":
        offer = [0, 0, 0]
        offer[UNIT_PRICE] = unit_price
        offer[QUANTITY] = quantity
        offer[TIME] = time
        return tuple(offer)

    def on_negotiation_success(self, contract, mechanism):
        # Record sales/supplies secured
        super().on_negotiation_success(contract, mechanism)
        my_id = self.awi.agent.id
        negotiator_id = (
            contract.partners[0]
            if my_id == contract.partners[1]
            else contract.partners[1]
        )
        self.contract_partners += 1
        self.remain_partners -= 1
        # update my current best price to use for limiting concession in other
        # negotiations

        up = contract.agreement["unit_price"]
        q = contract.agreement["quantity"]

        self._contracts[self.awi.current_step][negotiator_id] = self._get_outcome(
            up,
            q,
            contract.agreement["time"],
        )

        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            if up == self.best_price:
                self._bp_mxq_opp_acc_selling[partner] = max(
                    q, self._bp_mxq_opp_acc_selling[partner]
                )
                self._bp_mnq_opp_acc_selling[partner] = min(
                    q, self._bp_mnq_opp_acc_selling[partner]
                )
            else:
                self._wp_mxq_opp_acc_selling[partner] = max(
                    q, self._wp_mxq_opp_acc_selling[partner]
                )
                self._wp_mnq_opp_acc_selling[partner] = min(
                    q, self._wp_mnq_opp_acc_selling[partner]
                )
        else:
            partner = contract.annotation["seller"]
            if up == self.best_price:
                self._bp_mxq_opp_acc_buying[partner] = max(
                    q, self._bp_mxq_opp_acc_buying[partner]
                )
                self._bp_mnq_opp_acc_buying[partner] = min(
                    q, self._bp_mnq_opp_acc_buying[partner]
                )
            else:
                self._wp_mxq_opp_acc_buying[partner] = max(
                    q, self._wp_mxq_opp_acc_buying[partner]
                )
                self._wp_mnq_opp_acc_buying[partner] = min(
                    q, self._wp_mnq_opp_acc_buying[partner]
                )

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        _ = partners, annotation, mechanism, state
        self.en_opponents += 1
        self.remain_partners -= 1

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state, source)
        # update my current best price to use for limiting concession in other
        # negotiations
        self._received_offers[negotiator_id][self.awi.current_step].append(offer)
        if response == ResponseType.END_NEGOTIATION:
            return response
        my_needs = self._needed(negotiator_id)
        if my_needs > 0:
            if offer[QUANTITY] > my_needs:
                response = ResponseType.REJECT_OFFER
            else:
                self._set_quantity_range(
                    self.get_nmi(negotiator_id), negotiator_id, state
                )
                if offer[UNIT_PRICE] == self.best_price:
                    if offer[QUANTITY] >= self.min_q:
                        response = ResponseType.ACCEPT_OFFER
                    else:
                        if state.step >= 18:
                            response = ResponseType.ACCEPT_OFFER
                        else:
                            response = ResponseType.REJECT_OFFER
                else:
                    self._check_acc_worst_price(negotiator_id, state, offer)
                    if self.acc_worst_price:
                        response = ResponseType.ACCEPT_OFFER
                    else:
                        response = ResponseType.REJECT_OFFER
        """
        if response == ResponseType.ACCEPT_OFFER:
            self.contract_partners += 1
            self.remain_partners -= 1
        """
        return response

    def _check_acc_worst_price(self, negotiator_id, state, offer):
        self.acc_worst_price = False
        if state.step < self.second_target_steps:
            if len(self._sent_offers[negotiator_id]) > 0:
                lp = self._sent_offers[negotiator_id][-1]
                if lp is not None:
                    if lp[UNIT_PRICE] == self.worst_price:
                        if offer[QUANTITY] >= self.min_q:
                            self.acc_worst_price = True
                    else:
                        if self.concession and state.step == self.concession_step:
                            if offer[QUANTITY] >= self.min_q:
                                self.acc_worst_price = True
        elif state.step >= self.second_target_steps:
            if state.step >= 18:
                self.acc_worst_price = True
            else:
                if offer[QUANTITY] >= self.min_q:
                    self.acc_worst_price = True

    def _set_quantity_range(self, ami, negotiator_id, state):
        if state.step >= self.second_target_steps:
            max_opp_n = self.remain_partners
            if max_opp_n == 0:
                max_opp_n = 1
        else:
            if self.en_opponents < int(len(self.partners) / 4):
                max_opp_n = int(len(self.partners) * 3 / 4) - self.contract_partners
                if max_opp_n <= 0:
                    self.concession = True
                    max_opp_n = self.remain_partners
            else:
                self.concession = True
                max_opp_n = self.remain_partners
            if self.concession and self.concession_step < 0:
                if self.firstproposer and state.step < 19:
                    # if state.step < 19:
                    self.concession_step = state.step + 1
                else:
                    self.concession_step = state.step

        if self.en_opponents < int(len(self.partners) / 2):
            min_opp_n = int(len(self.partners) / 2) - self.contract_partners
            if min_opp_n <= 0:
                min_opp_n = max(1, int((min_opp_n + max_opp_n) / 4))
        else:
            min_opp_n = self.remain_partners
            if min_opp_n == 0:
                min_opp_n = 1

        self.min_q = max(
            ami.issues[QUANTITY].min_value, int(self._needed(negotiator_id) / max_opp_n)
        )
        self.max_q = min(
            ami.issues[QUANTITY].max_value, int(self._needed(negotiator_id) / min_opp_n)
        )

    def step(self):
        # Initialize the quantities and best prices received for next step
        super().step()