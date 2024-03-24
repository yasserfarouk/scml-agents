from math import floor

from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

__all__ = ["MatchingAgent"]


# noinspection PyAttributeOutsideInit
class MatchingAgent(OneShotAgent):
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

        if self.awi.level == 0:
            self.partners = self.awi.my_consumers
            self.output = [True]
            self.partner = "buyer"

        elif self.awi.level == 1:
            self.partners = self.awi.my_suppliers
            self.output = [False]
            self.partner = "seller"

        if self.verbose:
            pass  # print(f'My partners are {self.partners}')

        self.balances = {c: [0] for c in self.partners}
        self.n_negotiation_rounds = self.awi.settings["neg_n_steps"]
        self.pref_list = self.partners

    def before_step(self):
        # Sets the agent up for the round.
        # Resets the number of proposals/offers/waits to 0
        # Finds information about the exogenous contracts for the round
        if (self.awi.current_step - 1) % 5 == 0:
            for c in self.partners:
                self.balances[c].append(
                    self.awi.reports_at_step(self.awi.current_step - 1)[c].cash
                )
            sorted_balances = sorted(self.balances.items(), key=lambda x: x[1][-1])
            self.pref_list = [item[0] for item in sorted_balances]

        self.accepted_offers = {}
        self.received_offers = {}
        self.responses = {}
        self.sent_offers = {}
        self.queue_offers = {c: "First" for c in self.partners}
        self.signed_contracts = ()
        self.accepted = 0

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
        self.gap = abs(self.num_in - self.num_out)

        if self.awi.level == 0:
            self.q = self.awi.state.exogenous_input_quantity
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price
            self.worst_price = self.min_price
            self.first = False
            if self.num_in > self.num_out:
                self.patient = False
            if self.verbose:
                pass  # print(f'The viable prices are {self.awi.current_output_issues[UNIT_PRICE]}')

        elif self.awi.level == 1:
            self.q = self.awi.state.exogenous_output_quantity
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price
            self.worst_price = self.max_price
            self.first = True
            if self.num_in < self.num_out:
                self.patient = False
            if self.verbose:
                pass  # print(f'The viable prices are {self.awi.current_input_issues[UNIT_PRICE]}')

        self.needed = self.q
        self.remaining_partners = len(self.partners)
        self.calc_final = False
        if self.verbose:
            pass  # print(f'I need {self.needed} and have {self.remaining_partners} partners to trade with')

        if self.verbose:
            print(
                f"There are {self.num_in} produced and {self.num_out} needed. I am at {self.awi.level} and I am "
                f"{self.patient} patient at {self.awi.level}"
            )

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        step = state.step
        ami = self.get_nmi(negotiator_id)
        try:
            if self.verbose:
                pass  # print(self.responses)
            response = self.responses[negotiator_id]
            del self.responses[negotiator_id]
        except KeyError:
            self.cleanup(negotiator_id, offer)
            if self.needed <= 0:
                self.responses[negotiator_id] = ResponseType.END_NEGOTIATION
            elif (
                len(self.received_offers) < self.remaining_partners
                and self.wait_count.get(negotiator_id, 0) < self.max_wait
            ):
                self.responses[negotiator_id] = ResponseType.WAIT
            elif self.patient:
                self.respond_all_patient(state)
            else:
                self.respond_all_impatient()
            # elif self.patient:
            #     response = self.respond_patient(negotiator_id, state, offer)
            # else:
            #     response = self.respond_impatient(negotiator_id, state, offer)
            response = self.responses[negotiator_id]
            del self.responses[negotiator_id]
        if self.verbose:
            print(
                f"I have received {offer} from {negotiator_id} at step {state.step} "
                f"and am giving response {response}. {self.received_offers}"
            )
        return response

    def respond_all_impatient(self):
        if self.verbose:
            pass  # print(f'I am impatient and responding to offers: {self.received_offers}')
        nids = []
        qs = []
        ps = []
        self.accepted = 0
        self.n_accepted = 0
        if self.awi.level == 0:
            maximize_ps = False
        else:
            maximize_ps = True
        for nid in self.received_offers:
            offer = self.received_offers[nid]
            nids.append(nid)
            qs.append(offer[QUANTITY])
            ps.append(offer[UNIT_PRICE])
        nids_accept = best_combination(qs, ps, self.needed, maximize_ps)
        for i, nid in enumerate(nids):
            if i in nids_accept:
                self.responses[nid] = ResponseType.ACCEPT_OFFER
                self.accepted += qs[i]
                self.n_accepted += 1
            else:
                self.responses[nid] = ResponseType.REJECT_OFFER
        if self.remaining_partners - self.n_accepted != 0:
            self.propose_all()

    def respond_all_patient(self, state):
        if self.verbose:
            pass  # print(f'I am patient and responding to offers: {self.received_offers}')

        nids = []
        qs = []
        ps = []
        self.accepted = 0
        self.n_accepted = 0
        if self.awi.level == 0:
            maximize_ps = False
        else:
            maximize_ps = True

        for nid in self.received_offers:
            offer = self.received_offers[nid]
            if offer[UNIT_PRICE] != self.best_price and state.step < 18:
                self.responses[nid] = ResponseType.REJECT_OFFER
            else:
                nids.append(nid)
                qs.append(offer[QUANTITY])
                ps.append(offer[UNIT_PRICE])
        nids_accept = best_combination(
            qs, ps, self.needed, maximize_p=maximize_ps, require_target=True
        )
        for i, nid in enumerate(nids):
            if i in nids_accept:
                self.responses[nid] = ResponseType.ACCEPT_OFFER
                self.accepted += qs[i]
                self.n_accepted += 1
            else:
                self.responses[nid] = ResponseType.REJECT_OFFER
        self.propose_all()

    def propose_all(self):
        still_need = self.needed - self.accepted
        remaining_partners = self.remaining_partners - self.n_accepted
        if remaining_partners == 0:
            return None
        equal_dist = floor(still_need / remaining_partners)

        bonus = int(still_need - equal_dist * remaining_partners)
        for nid in self.responses:
            if (
                self.responses[nid] == ResponseType.ACCEPT_OFFER
                or self.responses[nid] == ResponseType.END_NEGOTIATION
            ):
                if nid in self.pref_list:
                    self.pref_list.remove(nid)

        for nid in self.responses:
            if self.responses[nid] == ResponseType.REJECT_OFFER:
                offer = [-1] * 3
                offer[UNIT_PRICE] = (
                    self.best_price if self.patient else self.worst_price
                )
                offer[QUANTITY] = equal_dist + (
                    1 if nid in self.pref_list[:bonus] else 0
                )
                offer[TIME] = self.awi.current_step
                self.sent_offers[nid] = tuple(offer)
                if offer[QUANTITY] == 0:
                    if self.wait_count.get(nid, 0) < self.max_wait:
                        self.responses[nid] = ResponseType.WAIT
                    else:
                        self.responses[nid] = ResponseType.END_NEGOTIATION
        for nid in self.partners:
            try:
                if self.responses[nid] != ResponseType.WAIT:
                    del self.received_offers[nid]
                    if self.verbose:
                        pass  # print(f'I am deleting {nid} from received offers and have {self.received_offers} left')
            except KeyError:
                pass

    def propose(self, negotiator_id: str, state):
        self.wait_count[negotiator_id] = 0
        try:
            offer = self.sent_offers[negotiator_id]
        except KeyError:
            offer = self.get_first_offer(negotiator_id, state)
        # del self.sent_offers[negotiator_id]
        if self.verbose:
            pass  # print(f'I am sending {offer} to {negotiator_id}')
        return offer

    # def respond_patient(self, negotiator_id, state, offer):
    #     if offer[UNIT_PRICE] == self.best_price and offer[QUANTITY] <= self.needed:
    #         response = ResponseType.ACCEPT_OFFER
    #     else:
    #         response = ResponseType.REJECT_OFFER
    #     return response
    #
    # def respond_impatient(self, negotiator_id, state, offer):
    #     if offer[QUANTITY] <= self.needed:
    #         response = ResponseType.ACCEPT_OFFER
    #     else:
    #         response = ResponseType.REJECT_OFFER
    #     return response

    # def propose(self, negotiator_id: str, state):
    #     self.wait_count[negotiator_id] = 0
    #     step = state.step
    #     ami = self.get_nmi(negotiator_id)
    #     offer = None
    #
    #     if self.awi.level == 1 and step == 0:
    #         offer = self.get_first_offer(negotiator_id, state)
    #     if offer is None:
    #         offer = self.get_offer(negotiator_id, state, offer)
    #     if self.verbose:
    #         print(f'I am {self.awi.agent} and I am giving offer {offer} to {negotiator_id} at step {state.step}')
    #     self.sent_offers[negotiator_id] = offer
    #     return offer

    def get_first_offer(self, negotiator_id, state):
        if self.verbose:
            pass  # print('I am giving the first offer')
        offer = [-1] * 3
        if self.remaining_partners == 0:
            return None
        offer[QUANTITY] = floor(self.needed / self.remaining_partners)
        total_q = offer[QUANTITY] * self.remaining_partners
        gap = self.needed - total_q

        if negotiator_id in self.pref_list[:gap] and total_q < self.needed:
            offer[QUANTITY] += 1
        offer[TIME] = self.awi.current_step
        if self.patient:
            offer[UNIT_PRICE] = self.best_price
        else:
            offer[UNIT_PRICE] = self.worst_price
        offer = tuple(offer)
        self.sent_offers[negotiator_id] = offer
        return offer

    def on_negotiation_success(self, contract, mechanism):
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
        self.remaining_partners -= 1
        partner = [nid for nid in partners if nid != str(self.awi.agent)][0]
        if self.verbose:
            pass  # print(f'I have failed negotiations with {partner} at {state.step} and still need {self.needed}')
        try:
            del self.sent_offers[partner]
        except KeyError:
            pass

    def cleanup(self, negotiator_id, offer):
        try:
            del self.sent_offers[negotiator_id]
        except KeyError:
            pass
        self.received_offers[negotiator_id] = offer
        self.q_needed[negotiator_id] = offer[QUANTITY]


class TestAgent(OneShotAgent):
    def propose(self, negotiator_id: str, state):
        offer = [-1] * 3
        offer[UNIT_PRICE] = self.awi.current_input_issues[UNIT_PRICE].max_value
        offer[QUANTITY] = 2
        offer[TIME] = self.awi.current_step
        offer = tuple(offer)
        return offer

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if state.step == 5:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


# Created by chatGPT
def find_combinations(qs, ps, target, current_sum=0, current_combination=None, index=0):
    if current_combination is None:
        current_combination = []

    if index == len(qs):
        if current_sum <= target:
            return [current_combination]
        else:
            return []
    else:
        without_current = find_combinations(
            qs, ps, target, current_sum, current_combination, index + 1
        )
        with_current = find_combinations(
            qs,
            ps,
            target,
            current_sum + qs[index],
            current_combination + [index],
            index + 1,
        )
        return without_current + with_current


def best_combination(qs, ps, target, maximize_p=False, require_target=False):
    combinations = find_combinations(qs, ps, target)
    valid_combinations = [
        comb for comb in combinations if sum(qs[i] for i in comb) <= target
    ]
    if not valid_combinations:
        return None
    highest_sum = max(sum(qs[i] for i in comb) for comb in valid_combinations)
    highest_sum_combinations = [
        comb for comb in valid_combinations if sum(qs[i] for i in comb) == highest_sum
    ]

    if maximize_p:
        best_comb = max(
            highest_sum_combinations, key=lambda comb: sum(ps[i] for i in comb)
        )
    else:
        best_comb = min(
            highest_sum_combinations, key=lambda comb: sum(ps[i] for i in comb)
        )
    if require_target:
        if highest_sum < target:
            return []
    return best_comb
