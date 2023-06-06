from abc import ABC, abstractmethod
from typing import Any, Collection, Iterable, Optional

from negmas import (
    Contract,
    Issue,
    MechanismState,
    NegotiatorMechanismInterface,
    Outcome,
    ResponseType,
    SAOResponse,
    SAOState,
)
from scml.oneshot import QUANTITY, UNIT_PRICE, OneShotSyncAgent

from .utils.enum import NegotiationStatus
from .utils.negutil import get_outcome


class BaseAgent(OneShotSyncAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secured = 0
        self.is_seller = True
        self.is_advantageous = True
        self.negotiation_status: dict[str, NegotiationStatus] = {}
        self.initialized_last_proposer = False
        self._last_proposer = True
        self.contracts: list[dict[str, Outcome]] = []

        self.own_opinion_rate_denominator: dict[str, int] = {}
        self.own_opinion_rate_numerator: dict[str, int] = {}
        self.own_opinion_rate_denominator_bullish: dict[str, int] = {}
        self.own_opinion_rate_numerator_bullish: dict[str, int] = {}
        self.own_opinion_rate_denominator_bearish: dict[str, int] = {}
        self.own_opinion_rate_numerator_bearish: dict[str, int] = {}

    def init(self):
        super().init()
        self.is_seller = True if self.awi.profile.level == 0 else False
        self.awi.logdebug_agent(f"is_seller: {self.is_seller}")

        for partner in self.partners():
            self.own_opinion_rate_denominator[partner] = 0
            self.own_opinion_rate_numerator[partner] = 0
            self.own_opinion_rate_denominator_bullish[partner] = 0
            self.own_opinion_rate_numerator_bullish[partner] = 0
            self.own_opinion_rate_denominator_bearish[partner] = 0
            self.own_opinion_rate_numerator_bearish[partner] = 0

    def before_step(self):
        self.secured = 0
        self.contracts.append({})
        self.negotiation_status = {
            agent: NegotiationStatus.CONTINUING for agent in self.partners()
        }
        exo_in = self.awi.exogenous_contract_summary[0][0]
        exo_out = self.awi.exogenous_contract_summary[2][0]
        if self.is_seller:
            self.is_advantageous = exo_in < exo_out * 1.05
        else:
            self.is_advantageous = exo_out < exo_in * 1.05
        self.awi.logdebug_agent(
            f"exo_in: {exo_in}, exo_out: {exo_out}, "
            f"advantageous: {self.is_advantageous}"
        )

        if self.awi.current_step == self.awi.n_steps - 1:
            self.awi.logdebug_agent(f"denom: {self.own_opinion_rate_denominator}")
            self.awi.logdebug_agent(f"num: {self.own_opinion_rate_numerator}")
            self.awi.logdebug_agent(
                f"denom_bullish: {self.own_opinion_rate_denominator_bullish}"
            )
            self.awi.logdebug_agent(
                f"num_bullish: {self.own_opinion_rate_numerator_bullish}"
            )
            self.awi.logdebug_agent(
                f"denom_bearish: {self.own_opinion_rate_denominator_bearish}"
            )
            self.awi.logdebug_agent(
                f"num_bearish: {self.own_opinion_rate_numerator_bearish}"
            )

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = self.get_partner_id(contract)
        if self.negotiation_status[partner_id] == NegotiationStatus.CONTINUING:
            self.negotiation_status[
                partner_id
            ] = NegotiationStatus.AGREED_IN_OWN_OPINION

        if (
            self.negotiation_status[partner_id]
            == NegotiationStatus.AGREED_IN_OWN_OPINION
        ):
            self.own_opinion_rate_denominator[partner_id] += 1
            self.own_opinion_rate_numerator[partner_id] += 1
            if self.is_advantageous:
                self.own_opinion_rate_denominator_bullish[partner_id] += 1
                self.own_opinion_rate_numerator_bullish[partner_id] += 1
            else:
                self.own_opinion_rate_denominator_bearish[partner_id] += 1
                self.own_opinion_rate_numerator_bearish[partner_id] += 1
            assert (
                self.own_opinion_rate_denominator[partner_id]
                == self.own_opinion_rate_denominator_bearish[partner_id]
                + self.own_opinion_rate_denominator_bullish[partner_id]
            )

        self.secured += contract.agreement["quantity"]
        outcome = get_outcome(
            contract.agreement["unit_price"],
            contract.agreement["quantity"],
            contract.agreement["time"],
        )
        self.contracts[self.awi.current_step][partner_id] = outcome
        self.awi.logdebug_agent(
            f"succeeded to negotiate with {partner_id}. "
            f"(price, quantity): ({outcome[UNIT_PRICE]}, {outcome[QUANTITY]})"
        )

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        partner_id = self.get_partner_id(mechanism)
        self.negotiation_status[partner_id] = NegotiationStatus.FAILED

        self.own_opinion_rate_denominator[partner_id] += 1
        if self.is_advantageous:
            self.own_opinion_rate_denominator_bullish[partner_id] += 1
        else:
            self.own_opinion_rate_denominator_bearish[partner_id] += 1

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        result = self._counter_all(offers, states)
        for partner_id, response in result.items():
            if response.response == ResponseType.ACCEPT_OFFER:
                self.negotiation_status[
                    partner_id
                ] = NegotiationStatus.AGREED_IN_PARTNER_OPINION
            elif response.response == ResponseType.END_NEGOTIATION:
                self.negotiation_status[partner_id] = NegotiationStatus.FAILED
        return result

    def first_proposals(self) -> dict[str, Outcome]:
        self.awi.logdebug_agent("first_proposals() called")
        result = self._first_proposals()
        return result

    @abstractmethod
    def _counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        raise NotImplementedError()

    @abstractmethod
    def _first_proposals(self) -> dict[str, Outcome]:
        raise NotImplementedError()

    def partners(self) -> list[str]:
        return self.awi.my_consumers if self.is_seller else self.awi.my_suppliers

    def get_partner_id(self, contract: Contract | NegotiatorMechanismInterface) -> str:
        my_id = self.awi.agent.id
        partners: list[str]
        if isinstance(contract, Contract):
            partners = list(contract.partners)
        elif isinstance(contract, NegotiatorMechanismInterface):
            partners = contract.agent_ids
        else:
            raise RuntimeError(
                "contract must be instance of Contract or NegotiatorMechanismInterface"
            )
        return partners[0] if my_id == partners[1] else partners[1]

    def get_issues(self) -> list[Issue]:
        # self.awi.logdebug_agent(f"i: {self.awi.current_input_issues}, o: {self.awi.current_output_issues}")
        if self.is_seller:
            return self.awi.current_output_issues
        else:
            return self.awi.current_input_issues

    def get_price_range(self) -> tuple[int, int]:
        price_issue = self.get_issues()[UNIT_PRICE]
        return price_issue.min_value, price_issue.max_value

    def bullish_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mx
        else:
            return mn

    def bearish_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mn
        else:
            return mx

    def needed_quantity(self) -> int:
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def own_opinion_rate(
        self, partner_id: str, situation_wise: bool = False
    ) -> Optional[float]:
        """Calculate own_opinion_rate. None if denominator == 0"""
        denominator: int
        numerator: int
        if situation_wise:
            if self.is_advantageous:
                denominator = self.own_opinion_rate_denominator_bullish[partner_id]
                numerator = self.own_opinion_rate_numerator_bullish[partner_id]
            else:
                denominator = self.own_opinion_rate_denominator_bearish[partner_id]
                numerator = self.own_opinion_rate_numerator_bearish[partner_id]
        else:
            denominator = self.own_opinion_rate_denominator[partner_id]
            numerator = self.own_opinion_rate_numerator[partner_id]
        if denominator == 0:
            return None
        return numerator / denominator

    @property
    def last_proposer(self) -> bool:
        if not self.initialized_last_proposer:
            raise RuntimeError("_last_proposer is not initialized")
        return self._last_proposer

    @last_proposer.setter
    def last_proposer(self, value):
        if self.initialized_last_proposer:
            raise RuntimeError("_last_proposer has already been initialized")
        self.initialized_last_proposer = True
        self._last_proposer = value
