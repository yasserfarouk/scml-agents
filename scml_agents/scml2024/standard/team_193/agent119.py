import itertools
from math import ceil, floor

import numpy as np
from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

from .tier1_agent import AdaptiveAgent

__all__ = ["PatientAgent"]


class PatientAgent(AdaptiveAgent):
    verbose = False

    def init(self):
        # Initializes the agent.
        # Does some administrative stuff based on the level of the agent
        self.world_size = (
            f"{len(self.awi.all_consumers[0])} {len(self.awi.all_consumers[1])}"
        )
        self.max_wait = (
            len(self.awi.all_consumers[0]) + len(self.awi.all_consumers[1]) + 1
        )
        # disables the agent when it finds itself in the middle of the production chain.
        self._donothing = False
        if self.awi.level == 0:
            self.partners = self.awi.my_consumers
            self.output = [True]
            self.partner = "buyer"

        elif self.awi.level == 1:
            self.partners = self.awi.my_suppliers
            self.output = [False]
            self.partner = "seller"
        else:
            self._donothing = True

        if not self._donothing:
            self.balances = {c: [0] for c in self.partners}
            self.is_saa = {c: False for c in self.partners}
            self.n_negotiation_rounds = self.awi.settings["neg_n_steps"]

    def before_step(self):
        # Sets the agent up for the round.
        # Resets the number of proposals/offers/waits to 0
        # Finds information about the exogenous contracts for the round
        super().before_step()
        if self._donothing:
            return
        if (self.awi.current_step - 1) % 5 == 0:
            for c in self.partners:
                self.balances[c].append(
                    self.awi.reports_at_step(self.awi.current_step - 1)[c].cash
                )

        self.accepted_offers = {}
        self.received_offers = {}
        self.sent_offers = {c: "First" for c in self.partners}
        self.queue_offers = {c: "First" for c in self.partners}
        self.signed_contracts = ()

        # self.proposal_count = {c: 0 for c in self.partners}
        # self.response_count = {c: 0 for c in self.partners}
        self.wait_count = {c: 0 for c in self.partners}
        self.q_needed = {c: 10 for c in self.partners}
        self.final_offers = {}

        self.num_in = self.awi.exogenous_contract_summary[0][0]
        self.num_out = self.awi.exogenous_contract_summary[-1][0]
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

        self.patient = True

        if self.awi.level == 0:
            self.q = self.awi.current_exogenous_input_quantity
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price
            if self.num_in > self.num_out:
                self.patient = False

        elif self.awi.level == 1:
            self.q = self.awi.current_exogenous_output_quantity
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price
            if self.num_in < self.num_out:
                self.patient = False

        self.needed = self.q
        self.first = False
        self.remaining_partners = len(self.partners)
        self.calc_final = False

        if self.verbose:
            print(
                f"There are {self.num_in} produced and {self.num_out} needed. I am at {self.awi.level} and I am "
                f"{self.patient} patient"
            )

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if self._donothing:
            return None
        step = state.step
        ami = self.get_nmi(negotiator_id)
        self.cleanup(negotiator_id, offer)

        if self.needed <= 0:
            response = ResponseType.END_NEGOTIATION
        if self.first:
            response = super().respond(negotiator_id, state)
        elif self.is_saa[negotiator_id]:
            response = self.saa_response(negotiator_id, state, offer)
        elif self.patient:
            response = self.respond_patient(negotiator_id, state, offer)
        else:
            response = self.respond_impatient(negotiator_id, state, offer)

        if self.verbose:
            print(
                f"I have received {offer} from {negotiator_id} at step {state.step} "
                f"and am giving response {response}"
            )

        return response

    def saa_response(self, negotiator_id, state, offer):
        if self._donothing:
            return None
        if (self.first and state.step == 1) or (not self.first and state.step == 0):
            self.q_needed[negotiator_id] = offer[QUANTITY]
        new_offer = [-1] * 3
        q = min(self.needed, self.q_needed[negotiator_id])
        p = self.best_price
        new_offer[TIME] = self.awi.current_step
        new_offer[QUANTITY] = q
        new_offer[UNIT_PRICE] = p
        self.queue_offers[negotiator_id] = tuple(new_offer)
        if offer[UNIT_PRICE] == self.best_price:
            response = ResponseType.ACCEPT_OFFER
        else:
            response = ResponseType.REJECT_OFFER
        if self.verbose:
            pass  # print(f'I am responding to {negotiator_id} who is SAA')
        return response

    def respond_patient(self, negotiator_id, state, offer):
        if self._donothing:
            return None
        need_offer = True
        if offer[UNIT_PRICE] == self.best_price and offer[QUANTITY] <= self.needed:
            response = ResponseType.ACCEPT_OFFER
        elif state.step <= 18:
            response = ResponseType.REJECT_OFFER
        elif self.first:
            if state.step == 18:
                self.final_offers[negotiator_id] = offer
                response = self.respond_best_offers(negotiator_id, state, offer)
            else:
                self.propose_final_offer(negotiator_id, state, offer)
                need_offer = False
                if len(self.sent_offers) >= 1 and self.wait_count[negotiator_id] < (
                    self.max_wait - 1
                ):
                    response = ResponseType.WAIT
                    self.wait_count[negotiator_id] += 1
                elif self.ufun(self.queue_offers[negotiator_id]) <= self.ufun(offer):
                    response = ResponseType.ACCEPT_OFFER
                else:
                    response = ResponseType.REJECT_OFFER
        elif not self.first:
            if state.step == 18:
                response = ResponseType.REJECT_OFFER
                # self.propose_best_offer(negotiator_id, state, offer)
                # need_offer = False
                # if len(self.sent_offers) >= 1:
                #     response = ResponseType.WAIT
                # else:
                #     response = ResponseType.REJECT_OFFER
            elif state.step == 19:
                self.final_offers[negotiator_id] = offer
                response = self.respond_best_offers(negotiator_id, state, offer)
        if response == ResponseType.REJECT_OFFER and need_offer:
            self.get_offer(negotiator_id, state, offer)

        return response

    def respond_impatient(self, negotiator_id, state, offer):
        if self._donothing:
            return None
        # if offer[QUANTITY] <= self.needed:
        #     response = ResponseType.ACCEPT_OFFER
        # else:
        #     response = ResponseType.REJECT_OFFER
        response = self.respond_patient(negotiator_id, state, offer)
        return response

    def propose(self, negotiator_id: str, state):
        if self._donothing:
            return None
        if self.first:
            offer = super().propose(negotiator_id, state)
        else:
            self.wait_count[negotiator_id] = 0
            step = state.step
            ami = self.get_nmi(negotiator_id)
            offer = None
            try:
                if self.queue_offers[negotiator_id] == "First":
                    offer = self.get_first_offer(negotiator_id, state)
                    self.queue_offers[negotiator_id] = offer
                else:
                    offer = self.queue_offers[negotiator_id]
            except KeyError:
                pass
            if offer is None:
                offer = self.get_offer(negotiator_id, state, offer)
            if self.verbose:
                pass  # print(f'I am {self.awi.agent} and I am giving offer {offer} to {negotiator_id} at step {state.step}')
            self.sent_offers[negotiator_id] = offer
        return offer

    def get_first_offer(self, negotiator_id, state):
        if self._donothing:
            return None
        self.first = True
        if self.verbose:
            pass  # print('I am giving the first offer')
        offer = [-1] * 3
        offer[QUANTITY] = self.needed
        offer[TIME] = self.awi.current_step
        if self.is_saa[negotiator_id]:
            if self.awi.level == 0:
                offer[UNIT_PRICE] = (
                    self.max_price - (self.max_price - self.min_price) * 0.3
                )
            elif self.awi.level == 1:
                offer[UNIT_PRICE] = (
                    self.min_price + (self.max_price - self.min_price) * 0.3
                )
        else:
            offer[UNIT_PRICE] = self.best_price
        offer = tuple(offer)
        self.queue_offers[negotiator_id] = offer
        return offer

    def on_negotiation_success(self, contract, mechanism):
        if self._donothing:
            return None
        if self.first:
            super().on_negotiation_success(contract, mechanism)
        # When a negotiation succeeds, change what's needed
        negotiator_id = contract.annotation[self.partner]
        q = contract.agreement["quantity"]
        p = contract.agreement["unit_price"]

        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        offer[QUANTITY] = q
        offer[UNIT_PRICE] = p

        self.signed_contracts += (tuple(offer),)

        self.needed -= q
        self.accepted_offers[negotiator_id] = tuple(offer)
        self.remaining_partners -= 1

        if self.verbose:
            print(
                f"I am {self.awi.agent} have accepted a contract for {q} items at price {p} "
                f"from {negotiator_id} and I still need {self.needed}"
            )

    def on_negotiation_failure(
        self,
        partners,
        annotation,
        mechanism,
        state,
    ) -> None:
        if self._donothing:
            return None
        self.remaining_partners -= 1
        partner = [nid for nid in partners if nid != str(self.awi.agent)][0]
        if state.step <= 10:
            self.is_saa[partner] = True
        if self.verbose:
            pass  # print(f'I have failed negotiations with {partner} at {state.step} and still need {self.needed}')
        try:
            del self.sent_offers[partner]
        except KeyError:
            pass

    def cleanup(self, negotiator_id, offer):
        if self._donothing:
            return None
        try:
            del self.sent_offers[negotiator_id]
        except KeyError:
            pass
        self.received_offers[negotiator_id] = offer
        self.q_needed[negotiator_id] = offer[QUANTITY]

    def get_offer(self, negotiator_id, state, offer):
        try:
            self.sent_offers[negotiator_id]
        except KeyError:
            offer = [-1] * 3
            offer[TIME] = self.awi.current_step
            offer[QUANTITY] = min(self.needed, self.q_needed[negotiator_id])
            offer[UNIT_PRICE] = self.best_price
            offer = tuple(offer)
            self.queue_offers[negotiator_id] = offer
        return offer

    def propose_best_offer(self, negotiator_id, state, offer):
        self.get_offer(negotiator_id, state, offer)

    def respond_best_offers(self, negotiator_id, state, offer):
        if (
            self.wait_count[negotiator_id] < (self.max_wait - 1)
            and len(self.final_offers) < self.remaining_partners
            and not self.calc_final
        ):
            response = ResponseType.WAIT
            self.wait_count[negotiator_id] += 1
        elif not self.calc_final:
            max_util = self.ufun.from_offers(
                ((0, 0, 0),) + self.signed_contracts,
                outputs=tuple(self.output * (len(self.signed_contracts) + 1)),
            )
            dis_util = max_util
            self.best_nids = [None]
            for L in range(1, len(self.final_offers) + 1):
                for keys in itertools.combinations(self.final_offers, L):
                    subset = tuple([self.final_offers[nid] for nid in keys])
                    u = self.ufun.from_offers(
                        subset + self.signed_contracts,
                        outputs=tuple(self.output * (L + len(self.signed_contracts))),
                    )
                    if self.verbose:
                        pass  # print(f'{subset}: {u}')
                    if u > max_util:
                        max_util = u
                        self.best_nids = keys
                    elif u == max_util:
                        sum_balance = sum(self.balances[k][-1] for k in keys)
                        sum_prev_balance = sum(
                            self.balances[k][-1] for k in self.best_nids
                        )
                        if sum_balance < sum_prev_balance:
                            self.best_nids = keys
            self.calc_final = True
            if self.verbose:
                print(
                    f"My offers were {self.final_offers} and I chose to accept {self.best_nids} as my utility without"
                    f" any new deals is {dis_util}"
                )
        if self.calc_final and negotiator_id in self.best_nids:
            response = ResponseType.ACCEPT_OFFER
        elif self.calc_final and negotiator_id not in self.best_nids:
            response = ResponseType.END_NEGOTIATION
            # prices = [o[UNIT_PRICE] for o in list(self.final_offers.values())]
            # if offer[QUANTITY] > self.needed:
            #     response = ResponseType.REJECT_OFFER
            #     self.remaining_partners -= 1
            #     del self.final_offers[negotiator_id]
            # elif offer[UNIT_PRICE] == max(prices):
            #     response = ResponseType.ACCEPT_OFFER
            #     del self.final_offers[negotiator_id]
            # else:
            #     response = ResponseType.WAIT
        return response

    def propose_final_offer(self, negotiator_id, state, offer):
        new_offer = [-1] * 3
        if self.awi.level == 0:
            mult = 1.3
            new_offer[UNIT_PRICE] = ceil(
                max(offer[UNIT_PRICE] * mult, self.best_price / mult)
            )
        if self.awi.level == 1:
            mult = 1.3
            new_offer[UNIT_PRICE] = ceil(
                max(offer[UNIT_PRICE] / mult, self.best_price * mult)
            )
        new_offer[TIME] = self.awi.current_step
        new_offer[QUANTITY] = min(self.needed, self.q_needed[negotiator_id])
        new_offer = tuple(new_offer)
        self.queue_offers[negotiator_id] = new_offer
        return new_offer
