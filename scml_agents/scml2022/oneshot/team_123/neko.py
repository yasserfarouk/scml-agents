from .base_agent import BaseAgent
from negmas import (
    MechanismState,
    Outcome,
    ResponseType,
)
from typing import Dict, List
from .utils import (
    td_concession_rate,
    get_proposal,
    get_price,
    simple_round,
)
from scml import UNIT_PRICE, QUANTITY
import random

__all__ = ["Neko"]


class Neko(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = 0.5
        self._rate_step = 0.05
        self._worst_price_rate = 0.95
        self._price_change_step = 0.05

        self._target_prices: Dict[str, float] = {}
        self._exo_quantities: List[int] = []

    def init(self):
        super().init()
        self._target_prices = {k: self._base_price() for k in self._partners()}
        if self._is_seller:
            self._worst_price_rate = 0.95 - self.awi.current_disposal_cost / 5
        else:
            self._worst_price_rate = 0.95 - self.awi.current_shortfall_penalty / 5

    def before_step(self):
        super().before_step()
        self._worst_price_rate = min(self._worst_price_rate, 1.0)
        worst_limit = self._rate_to_actual(self._base_price(), self._worst_price_rate)
        for k, v in self._target_prices.items():
            if self._is_seller:
                self._target_prices[k] = max(worst_limit, v)
            else:
                self._target_prices[k] = min(worst_limit, v)
        self._exo_quantities.append(self._needed_count())

    def _rate_to_actual(self, base_price: float, price_rate: float) -> float:
        if self._is_seller:
            return base_price * price_rate
        else:
            return base_price * (2 - price_rate)

    def _offer_price(self, negotiator_id: str, state: MechanismState) -> float:
        nmi = self.get_nmi(negotiator_id)
        target = self._target_prices[negotiator_id]
        rnd = state.step if self._last_proposer else state.step + 1
        if rnd == nmi.n_steps:
            # this is the last proposal when self._last_proposer=False
            # (and this proposal will be buried)
            rnd = nmi.n_steps - 1
        cr = td_concession_rate(rnd, nmi.n_steps, self._e)
        limit = (
            nmi.issues[UNIT_PRICE].max_value
            if self._is_seller
            else nmi.issues[UNIT_PRICE].min_value
        )
        offer_price = get_price(cr, self._is_seller, (target, limit))
        return offer_price

    def _base_price(self) -> int:
        index = (
            self.awi.my_output_product if self._is_seller else self.awi.my_input_product
        )
        if self.awi.trading_prices is None:
            return self.awi.catalog_prices[index]
        else:
            return self.awi.trading_prices[index]

    def _make_proposal(self, negotiator_id: str, state: MechanismState) -> Outcome:
        offer_price = self._offer_price(negotiator_id, state)
        time = self.awi.current_step

        quantity = self._needed_count()
        last_offer = self._last_received_offer(negotiator_id)
        if last_offer is not None:
            quantity = min(quantity, last_offer[QUANTITY])
        proposal = get_proposal(simple_round(offer_price), quantity, time)
        return proposal

    def _make_response(
        self, negotiator_id: str, state: MechanismState, offer: Outcome
    ) -> ResponseType:
        if self._needed_count() <= 0:
            return ResponseType.END_NEGOTIATION
        if offer[QUANTITY] > self._needed_count():
            return ResponseType.REJECT_OFFER
        target_price = self._offer_price(negotiator_id, state)
        if not self._is_better(offer[UNIT_PRICE], target_price):
            return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def _improve_target_price(self, negotiator_id: str):
        v = self._target_prices[negotiator_id]
        if self._is_seller:
            self._target_prices[negotiator_id] = v * (1 + self._price_change_step)
        else:
            self._target_prices[negotiator_id] = v * (1 - self._price_change_step)

    def _worsen_target_price(self, negotiator_id: str, forced: bool):
        v = self._target_prices[negotiator_id]
        next_v = (
            v * (1 - self._price_change_step)
            if self._is_seller
            else v * (1 + self._price_change_step)
        )
        if forced:
            current_limit = self._rate_to_actual(
                self._base_price(), self._worst_price_rate
            )
            if self._is_better(current_limit, next_v):
                self._worst_price_rate -= self._rate_step

        limit = self._rate_to_actual(self._base_price(), self._worst_price_rate)

        if self._is_seller:
            self._target_prices[negotiator_id] = max(next_v, limit)
        else:
            self._target_prices[negotiator_id] = min(next_v, limit)

    def step(self):
        super().step()
        last_contracts = self._contracts[self.awi.current_step]
        n_contracts = len(last_contracts)

        no_contract_sensitive = True
        if self._is_seller:
            # seller: dispose items if she makes no contract
            no_contract_sensitive = (
                True
                if self.awi.current_disposal_cost > self.awi.current_shortfall_penalty
                else False
            )
        else:
            # buyer: be penalized if she makes no contract
            no_contract_sensitive = (
                True
                if self.awi.current_disposal_cost < self.awi.current_shortfall_penalty
                else False
            )

        low_quantity = True
        low_quantity_check_window = 2 if no_contract_sensitive else 3
        no_contracts = True
        no_contracts_check_window = 1 if no_contract_sensitive else 2
        if self.awi.current_step >= low_quantity_check_window - 1:
            for i in range(low_quantity_check_window):
                tgt = self.awi.current_step - i
                if self._secured_count(tgt) >= self._exo_quantities[tgt] / 2:
                    low_quantity = False
                    break
        else:
            low_quantity = False

        if self.awi.current_step >= no_contracts_check_window - 1:
            for i in range(no_contracts_check_window):
                tgt = self.awi.current_step - i
                if len(self._contracts[tgt]) > 0:
                    no_contracts = False
                    break
        else:
            no_contracts = False

        if no_contracts or low_quantity:
            id_prices = list(self._target_prices.items())
            id_prices.sort(key=lambda x: x[1], reverse=self._is_seller)
            targets = id_prices[0 : max(1, len(id_prices)) // 2]

            for k in targets:
                self._worsen_target_price(k[0], True)

        elif n_contracts >= 2:
            targets = random.sample(
                list(self._contracts[self.awi.current_step].keys()), n_contracts - 1
            )
            for target in targets:
                self._improve_target_price(target)
