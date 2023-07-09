from collections import defaultdict

from negmas import ResponseType
from negmas.outcomes import Outcome
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE

from .base_agent import AdaptiveAgent

__all__ = ["KanbeAgent"]


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
