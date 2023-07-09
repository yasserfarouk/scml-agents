from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from negmas import (
    Contract,
    MechanismState,
    NegotiatorMechanismInterface,
    Outcome,
    ResponseType,
    SAOState,
)
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent

from .utils import clamp, get_proposal, simple_round


class BaseAgent(OneShotAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._received_offers: Dict[str, List[List[Outcome]]] = {}
        self._contracts: List[Dict[str, Outcome]] = []
        self._is_seller = True
        self._initialized_last_proposer = False
        self._last_proposer = True
        self._secured = 0

    # -- Public Methods --

    def init(self):
        super().init()
        self._is_seller = True if self.awi.profile.level == 0 else False
        for partner_id in self._partners():
            self._received_offers[partner_id] = []

    def before_step(self):
        self._contracts.append({})
        self._initialized_last_proposer = False
        for k in self._received_offers.keys():
            self._received_offers[k].append([])
        self._secured = 0

    def propose(self, negotiator_id: str, state: MechanismState) -> Outcome:
        if not self._initialized_last_proposer:
            self._last_proposer = True
            self._after_know_offering_order()

        ami = self.get_nmi(negotiator_id)
        proposal = self._make_proposal(negotiator_id, state)

        verified_proposal = get_proposal(
            simple_round(
                clamp(
                    proposal[UNIT_PRICE],
                    (
                        ami.issues[UNIT_PRICE].min_value,
                        ami.issues[UNIT_PRICE].max_value,
                    ),
                )
            ),
            simple_round(
                clamp(
                    proposal[QUANTITY],
                    (ami.issues[QUANTITY].min_value, ami.issues[QUANTITY].max_value),
                )
            ),
            simple_round(
                clamp(
                    proposal[TIME],
                    (ami.issues[TIME].min_value, ami.issues[TIME].max_value),
                )
            ),
        )
        return verified_proposal

    def respond(self, negotiator_id: str, state: SAOState) -> ResponseType:
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if not self._initialized_last_proposer:
            self._last_proposer = False
            self._after_know_offering_order()
        self._received_offers[negotiator_id][self.awi.current_step].append(offer)
        response = self._make_response(negotiator_id, state, offer)
        return response

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract, mechanism)
        my_id = self.awi.agent.id
        negotiator_id = (
            contract.partners[0]
            if my_id == contract.partners[1]
            else contract.partners[1]
        )
        self._contracts[self.awi.current_step][negotiator_id] = get_proposal(
            contract.agreement["unit_price"],
            contract.agreement["quantity"],
            contract.agreement["time"],
        )
        self._secured = contract.agreement["quantity"]

    # -- Abstract Methods --

    @abstractmethod
    def _make_proposal(self, negotiator_id: str, state: MechanismState) -> Outcome:
        raise NotImplementedError()

    @abstractmethod
    def _make_response(
        self, negotiator_id: str, state: MechanismState, offer: Outcome
    ) -> ResponseType:
        raise NotImplementedError()

    def _after_know_offering_order(self):
        pass  # optional method

    # -- Properties --

    @property
    def _last_proposer(self) -> bool:
        if not self._initialized_last_proposer:
            raise RuntimeError("_last_proposer is not initialized")
        return self.__last_proposer

    @_last_proposer.setter
    def _last_proposer(self, value):
        if self._initialized_last_proposer:
            raise RuntimeError("_last_proposer has already been initialized")
        self._initialized_last_proposer = True
        self.__last_proposer = value

    # -- useful functions --

    def _needed_count(self) -> int:
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self._secured
        )

    def _last_received_offer(self, partner_id: str) -> Optional[Outcome]:
        target = self._received_offers[partner_id][self.awi.current_step]
        if len(target) == 0:
            return None
        else:
            return target[-1]

    def _partners(self) -> List[str]:
        return self.awi.my_consumers if self._is_seller else self.awi.my_suppliers

    def _is_better(self, a: float, b: float) -> bool:
        """Return True if argument a is "better" than argument b."""
        if self._is_seller:
            return True if a >= b else False
        else:
            return True if a <= b else False

    def _secured_count(self, step: int) -> int:
        result = 0
        for contract in self._contracts[step].values():
            result += contract[QUANTITY]
        return result
