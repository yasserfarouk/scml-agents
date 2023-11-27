# from bettersyncagent import BetterSyncAgent
from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE


class DelayAgent(OneShotAgent):
    verbose = True

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

        self.balances = {c: [] for c in self.partners}
        self.n_negotiation_rounds = self.awi.settings["neg_n_steps"]

    def before_step(self):
        # Sets the agent up for the round.
        # Resets the number of proposals/offers/waits to 0
        # Finds information about the exogenous contracts for the round
        if (self.awi.current_step - 1) % 5 == 0:
            for c in self.partners:
                self.balances[c].append(
                    self.awi.reports_at_step(self.awi.current_step - 1)[c].cash
                )

        self.accepted_offers = {}
        self.received_offers = {}
        self.sent_offers = {c: "First" for c in self.partners}

        # self.proposal_count = {c: 0 for c in self.partners}
        # self.response_count = {c: 0 for c in self.partners}
        self.wait_count = {c: 0 for c in self.partners}

        self.num_in = self.awi.exogenous_contract_summary[0][0]
        self.num_out = self.awi.exogenous_contract_summary[-1][0]
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

        self.patient = True
        if self.verbose:
            print(
                f"There are {self.num_in} produced and {self.num_out} needed. I am at {self.awi.level} and I am "
                f"{self.patient} patient"
            )

        if self.awi.level == 0:
            self.q = self.awi.state.exogenous_input_quantity
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price
            if self.num_in > self.num_out:
                self.patient = False

        elif self.awi.level == 1:
            self.q = self.awi.state.exogenous_output_quantity
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price
            if self.num_in < self.num_out:
                self.patient = False

        self.needed = self.q
        self.first = False

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        step = state.step
        ami = self.get_nmi(negotiator_id)
        self.cleanup(negotiator_id, offer)

        if self.verbose:
            pass  # print(f'I have received {offer} from {negotiator_id} at step {state.step}')

        if self.needed <= 0:
            response = ResponseType.END_NEGOTIATION
        elif self.patient and step <= ami.n_steps * 3 / 4:
            response = ResponseType.REJECT_OFFER
        elif self.patient:
            if self.wait_count[negotiator_id] < (self.max_wait - 1) and len(
                self.received_offers
            ) < len(self.partners):
                self.wait_count[negotiator_id] += 1
                response = ResponseType.WAIT
            else:
                prices = [o[UNIT_PRICE] for o in self.received_offers.values()]
                if offer[UNIT_PRICE] == max(prices):
                    if offer[QUANTITY] > self.needed:
                        response = ResponseType.REJECT_OFFER
                        new_offer = [-1] * 3
                        new_offer[TIME] = self.awi.current_step
                        new_offer[QUANTITY] = self.needed
                        new_offer[UNIT_PRICE] = offer[UNIT_PRICE]
                        offer = tuple(offer)
                        self.sent_offers[negotiator_id] = offer
                    else:
                        if (
                            self.awi.level == 0
                            and offer[UNIT_PRICE] >= self.best_price * 0.8
                        ) or (
                            self.awi.level == 1
                            and offer[UNIT_PRICE] <= self.best_price * 1.2
                        ):
                            response = ResponseType.ACCEPT_OFFER
                        else:
                            response = ResponseType.REJECT_OFFER
                else:
                    response = ResponseType.REJECT_OFFER
        elif not self.patient:
            response = ResponseType.ACCEPT_OFFER

        if response == ResponseType.REJECT_OFFER:
            self.get_offer(negotiator_id, state, offer)

        if self.verbose:
            pass  # print(f'I am giving response {response}')
        return response

    def propose(self, negotiator_id: str, state):
        self.wait_count[negotiator_id] = 0
        step = state.step
        ami = self.get_nmi(negotiator_id)
        if self.sent_offers[negotiator_id] == "First":
            offer = self.get_first_offer(negotiator_id, state)
            self.sent_offers[negotiator_id] = offer
        else:
            offer = self.sent_offers[negotiator_id]
        return offer

    def get_first_offer(self, negotiator_id, state):
        self.first = True
        if self.verbose:
            pass  # print('I am giving the first offer')
        offer = [-1] * 3
        offer[UNIT_PRICE] = self.best_price
        offer[QUANTITY] = self.needed
        offer[TIME] = self.awi.current_step
        offer = tuple(offer)
        self.sent_offers[negotiator_id] = offer
        return offer

    def on_negotiation_success(self, contract, mechanism):
        # When a negotiation succeeds, change what's needed
        negotiator_id = contract.annotation[self.partner]
        q = contract.agreement["quantity"]
        p = contract.agreement["unit_price"]

        if self.verbose:
            pass  # print(f'I have accepted {contract} from {negotiator_id}')

        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        offer[QUANTITY] = q
        offer[UNIT_PRICE] = p

        self.needed -= q
        self.accepted_offers[negotiator_id] = offer

    def cleanup(self, negotiator_id, offer):
        try:
            del self.sent_offers[negotiator_id]
        except KeyError:
            pass
        self.received_offers[negotiator_id] = offer

    def get_offer(self, negotiator_id, state, offer):
        try:
            self.sent_offers[negotiator_id]
        except KeyError:
            offer = [-1] * 3
            offer[TIME] = self.awi.current_step
            offer[QUANTITY] = self.q
            offer[UNIT_PRICE] = self.best_price
            offer = tuple(offer)
            self.sent_offers[negotiator_id] = offer
