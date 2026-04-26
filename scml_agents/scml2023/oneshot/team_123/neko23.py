import copy
from typing import Literal

from negmas import (
    Contract,
    NegotiatorMechanismInterface,
    Outcome,
    ResponseType,
    SAOResponse,
    SAOState,
)
from scml import QUANTITY, UNIT_PRICE

from .base_oneshot import BaseAgent
from .utils.math import powerset, weighted_sample
from .utils.negutil import get_outcome

__all__ = ["AgentNeko23", "AgentNeko23Random"]


class AgentNeko23(BaseAgent):
    QUANTITY_THRESHOLD = 0.7
    CORRECTED_AGREEMENT_RATE_MIN = 0.3
    CORRECTED_AGREEMENT_RATE_FALLBACK_N_SAMPLE = 3
    CORRECTED_AGREEMENT_RATE_FALLBACK_VALUE = 0.7

    def __init__(
        self,
        quantity_algorithm: Literal[
            "random", "expected_quantity"
        ] = "expected_quantity",
    ):
        super().__init__()
        self.has_conclusion = False
        self.quantity_algorithm = quantity_algorithm

    def before_step(self):
        super().before_step()
        self.has_conclusion = False

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract, mechanism)
        self.has_conclusion = True

    def get_proposals(
        self, partners: set[str], max_quantity: int
    ) -> dict[str, Outcome]:
        self.awi.logdebug_agent(f"max_quantity: {max_quantity}")
        # negotiate with bullish stance if the world is in advantageous situation for me
        offer_price = (
            self.bullish_price() if self.is_advantageous else self.bearish_price()
        )
        time = self.awi.current_step
        if max_quantity <= 0:
            # quit negotiations
            proposal_dict = {
                partner: get_outcome(offer_price, 0, time) for partner in partners
            }
            return proposal_dict
        quantities = {partner: 1 for partner in partners}
        addible_agents: set[str] = copy.copy(partners)
        current_total = (
            self.expected_quantity_sum(quantities)
            if self.quantity_algorithm == "expected_quantity"
            else sum(quantities.values())
        )
        if current_total >= max_quantity:
            proposal_dict = {
                partner: get_outcome(offer_price, quantity, time)
                for partner, quantity in quantities.items()
            }
            self.awi.logdebug_agent(f"proposals: {str(proposal_dict)}")
            return proposal_dict
        while True:
            items = [
                (
                    partner_id,
                    self.corrected_own_opinion_rate(partner_id)
                    if self.quantity_algorithm == "expected_quantity"
                    else 1.0,
                )
                for partner_id in addible_agents
            ]
            add_to = weighted_sample(items)
            quantities[add_to] += 1
            if quantities[add_to] >= max_quantity:
                addible_agents.remove(add_to)
            if len(addible_agents) == 0:
                break
            current_total = (
                self.expected_quantity_sum(quantities)
                if self.quantity_algorithm == "expected_quantity"
                else sum(quantities.values())
            )
            if current_total >= max_quantity:
                break

        proposal_dict = {
            partner: get_outcome(offer_price, quantity, time)
            for partner, quantity in quantities.items()
        }
        self.awi.logdebug_agent(f"proposals: {str(proposal_dict)}")
        return proposal_dict

    def get_acceptable(
        self, offers: dict[str, Outcome], states: dict[str, SAOState], ultimatum: bool
    ) -> set[str]:
        if ultimatum:
            # accept offers to get as much utility as possible
            best_utility = -float("inf")
            best_set = set()
            for partner_ids in powerset(states.keys()):
                temp_offers = list(self.contracts[self.awi.current_step].values()) + [
                    offers[partner_id] for partner_id in partner_ids
                ]
                outputs = [self.is_seller] * len(temp_offers)
                utility = self.ufun.from_offers(  # noqa
                    tuple(temp_offers), tuple(outputs)
                )
                if best_utility < utility:
                    best_utility = utility
                    best_set = set(partner_ids)
                self.awi.logdebug_agent(
                    f"{best_set} {best_utility} {temp_offers} {utility}"
                )
            return best_set

        if self.is_advantageous:
            candidate_offers: list[tuple[str, Outcome]]
            mn, mx = self.get_price_range()
            if self.is_seller:
                candidate_offers = [t for t in offers.items() if t[1][UNIT_PRICE] >= mx]
            else:
                candidate_offers = [t for t in offers.items() if t[1][UNIT_PRICE] <= mn]

            for offers_set in powerset(candidate_offers):
                quantity = sum([offer[1][QUANTITY] for offer in offers_set])
                if quantity == self.needed_quantity():
                    return set([offer[0] for offer in offers_set])
            return set()  # reject all offers

        else:
            best_set: set[str] = set()
            best_quantity = -1
            for offers_set in powerset(offers.items()):
                quantity = sum([offer[1][QUANTITY] for offer in offers_set])
                if quantity > self.needed_quantity():
                    continue
                if (
                    quantity < self.needed_quantity() * self.QUANTITY_THRESHOLD
                    and not self.has_conclusion
                ):
                    continue
                if best_quantity < quantity:
                    best_quantity = quantity
                    best_set = set([offer[0] for offer in offers_set])
            return best_set

    def _counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        current_step = list(states.values())[0].step
        n_steps = self.get_nmi(list(states.keys())[0]).n_steps
        if not self.initialized_last_proposer:
            if current_step == 0:
                self.last_proposer = False
            else:
                self.last_proposer = True
        all_partners = set(offers.keys())

        # quit negotiations if already secured
        if self.needed_quantity() <= 0:
            return {
                agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for agent in all_partners
            }

        is_ultimatum = (not self.last_proposer) and current_step == n_steps - 1
        self.awi.logdebug_agent(
            f"current_step: {current_step}, n_steps: {n_steps}, "
            f"is_ultimatum: {is_ultimatum}"
        )
        acceptable_partners = self.get_acceptable(offers, states, is_ultimatum)
        acceptance_dict: dict[str, SAOResponse] = {
            agent: SAOResponse(ResponseType.ACCEPT_OFFER, None)
            for agent in acceptable_partners
        }
        accepted_quantity = 0
        for partner in acceptance_dict.keys():
            accepted_quantity += offers[partner][QUANTITY]

        needed = self.needed_quantity() - accepted_quantity
        partners_proposed_to = all_partners - acceptable_partners
        result: dict[str, SAOResponse]
        if len(partners_proposed_to) == 0:
            result = acceptance_dict
        elif needed > 0:
            proposals = self.get_proposals(partners_proposed_to, needed)
            proposal_dict: dict[str, SAOResponse] = {
                agent: SAOResponse(ResponseType.REJECT_OFFER, proposal)
                for agent, proposal in proposals.items()
            }
            result = proposal_dict | acceptance_dict
        else:
            proposal_dict: dict[str, SAOResponse] = {
                agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for agent in partners_proposed_to
            }
            result = proposal_dict | acceptance_dict

        for agent, response in result.items():
            if response.response == ResponseType.ACCEPT_OFFER:
                self.awi.logdebug_agent(f"Accept the offer of {agent}")
            elif response.response == ResponseType.REJECT_OFFER:
                self.awi.logdebug_agent(f"Reject the offer of {agent}.")
                self.awi.logdebug_agent(
                    "counter offer: (price, quantity): "
                    f"({response.outcome[UNIT_PRICE]}, {response.outcome[QUANTITY]})"
                )
            else:
                self.awi.logdebug_agent(f"End the negotiation with {agent}.")
        return result

    def _first_proposals(self) -> dict[str, Outcome]:
        return self.get_proposals(set(self.partners()), self.needed_quantity())

    def corrected_own_opinion_rate(self, partner_id: str) -> float:
        rate = self.own_opinion_rate(partner_id, situation_wise=True)
        if (
            rate is None
            or self.own_opinion_rate_denominator[partner_id]
            <= self.CORRECTED_AGREEMENT_RATE_FALLBACK_N_SAMPLE
        ):
            return self.CORRECTED_AGREEMENT_RATE_FALLBACK_VALUE
        return max(rate, self.CORRECTED_AGREEMENT_RATE_MIN)

    def expected_quantity_sum(self, quantities: dict[str, int]) -> float:
        expected_quantities = [
            quantity * self.corrected_own_opinion_rate(partner_id)
            for partner_id, quantity in quantities.items()
        ]
        return sum(expected_quantities)


class AgentNeko23Random(AgentNeko23):
    def __init__(self):
        super().__init__(quantity_algorithm="random")
