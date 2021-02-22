import sys
import numpy as np
import itertools
from negmas import (
    Issue,
    AgentMechanismInterface,
    Contract,
    Negotiator,
    MechanismState,
    Breach,
)
from scml.scml2020 import Failure
from scml.scml2020 import SCML2020Agent
from scml.scml2020 import TradingStrategy
from scml.scml2020 import SupplyDrivenProductionStrategy

from scml.scml2020 import TIME, QUANTITY, UNIT_PRICE
from negmas import ResponseType, outcome_is_valid, UtilityFunction
from negmas.sao import SAOResponse
from typing import List, Dict, Optional, Tuple, Any
from negmas import SAOSyncController
from scml.scml2020.components import SignAllPossible

__all__ = ["CrescentAgent"]


class ControllerUFun(UtilityFunction):
    """A utility function for the controller"""

    def __init__(self, controller=None):
        super().__init__(outcome_type=tuple)
        self.controller = controller

    def eval(self, offer: "Outcome"):
        return self.controller.utility(offer)

    def xml(self, issues):
        pass


class SyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent
    needs and with lowest price.

    Args:
        is_seller: Are we trying to sell (or to buy)?
        parent: The agent from which we will access `needed` and `secured` arrays
        price_weight: The importance of price in utility calculation
        utility_threshold: Accept anything with a relative utility above that
        time_threshold: Accept anything with a positive utility when we are that close
                        to the end of the negotiation
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        parent: "MyTradingStrategy",
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.utility_function = ControllerUFun(controller=self)

    def utility(self, offer: "Outcome") -> float:
        """A simple utility function

        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract

        """

        # get my needs and secured amounts arrays
        if self._is_seller:
            _needed, _secured = (
                self.__parent.outputs_needed,
                self.__parent.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.__parent.inputs_needed,
                self.__parent.inputs_secured,
            )

        # invalide offers have no utility
        if offer is None:
            return -1000

        # offers for contracts that can never be executed have no utility
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0

        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        # q=_needed[offer[TIME]]-offer[QUANTITY]
        if q < 0:
            return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]
        return self._price_weight * price + (1 - self._price_weight) * q

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        """Is this a valid offer for that negotiation"""
        issues = self.negotiators[negotiator_id][0].ami.issues
        return outcome_is_valid(offer, issues)

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, "SAOState"]
    ) -> Dict[str, "SAOResponse"]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.

        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array([self.utility(o) for o in offers.values()])

        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.first_proposals()

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            return {
                k: SAOResponse(ResponseType.REJECT_OFFER, best_proposals[k])
                for k in offers.keys()
            }

        relative_time = min(_.relative_time for _ in states.values())

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility
            >= self._utility_threshold * self.utility(best_proposals[best_partner])
            or relative_time > self._time_threshold
        ):
            responses = {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    best_offer if self.is_valid(k, best_offer) else best_proposals[k],
                )
                for k in offers.keys()
            }
            responses[best_partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return responses

        # send the best offer to everyone else and try to improve it
        responses = {
            k: SAOResponse(
                ResponseType.REJECT_OFFER,
                best_offer if self.is_valid(k, best_offer) else best_proposals[k],
            )
            for k in offers.keys()
        }
        responses[best_partner] = SAOResponse(
            ResponseType.REJECT_OFFER, best_proposals[best_partner]
        )
        return responses


class MyNegotiationManager:
    def __init__(
        self,
        *args,
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.controllers: Dict[bool, SyncController] = {
            False: SyncController(
                is_seller=False,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
            True: SyncController(
                is_seller=True,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
        }
        self._current_end = -1
        self._current_start = -1

    def step(self):
        super().step()

        # find the range of steps about which we plan to negotiate
        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return

        for seller, needed, secured, product in [
            (False, self.inputs_needed, self.inputs_secured, self.awi.my_input_product),
            (
                True,
                self.outputs_needed,
                self.outputs_secured,
                self.awi.my_output_product,
            ),
        ]:
            # find the maximum amount needed at any time-step in the given range
            needs = np.max(
                needed[self._current_start : self._current_end]
                - secured[self._current_start : self._current_end]
            )
            if needs < 1:
                continue

            # set a range of prices
            if seller:
                # for selling set a price that is at least the catalog price
                min_price = self.awi.catalog_prices[product]
                price_range = (min_price, 2 * min_price)
            else:
                # for buying sell a price that is at most the catalog price
                price_range = (0, self.awi.catalog_prices[product])
            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=self.controllers[seller],
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        controller = self.controllers[self.id == annotation["seller"]]
        if controller is None:
            return None
        return controller.create_negotiator()


class MyTradingStrategy(SignAllPossible, TradingStrategy):
    def init(self):
        super().init()
        self.inputs_needed = np.array([self.awi.n_lines] * self.awi.n_steps, dtype=int)
        self.inputs_needed[self.awi.n_steps - 1] = 0
        # self.inputs_secured=np.zeros(self.awi.n_steps,dtype=int)
        self.outputs_needed = np.array([self.awi.n_lines] * self.awi.n_steps, dtype=int)
        self.outputs_needed[0] = 0
        # self.outputs_secured=np.zeros(self.awi.n_steps,dtype=int)

    def step(self):
        super().step()

    def quantities_cost_check(self, contracts: List[Contract]) -> Tuple[int, int]:
        q_sum = 0
        u_q_sum = 0
        for contract in contracts:
            q_sum += contract.agreement["quantity"]
            u_q_sum += contract.agreement["unit_price"] * contract.agreement["quantity"]
        return q_sum, u_q_sum

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        possible_contracts = super().sign_all_contracts(contracts)
        buy_contracts = [None] * self.awi.n_steps
        sell_contracts = [None] * self.awi.n_steps
        sign_buy_contracts = []
        sign_sell_contracts = []
        # devide contracs into buy and sell
        for contract in contracts:
            if not possible_contracts:
                continue
            t = contract.agreement["time"]
            q = contract.agreement["quantity"]
            if contract.annotation["buyer"] == self.id:
                if buy_contracts[t]:
                    buy_contracts[t] += [contract]
                else:
                    buy_contracts[t] = [contract]
            else:
                if sell_contracts[t]:
                    sell_contracts[t] += [contract]
                else:
                    sell_contracts[t] = [contract]

        # conect sell_contract with buy_contracts
        for step in range(self.awi.n_steps):
            if not sell_contracts[step]:
                continue
            sell_contracts[step].sort(
                key=lambda c: c.agreement["unit_price"], reverse=True
            )
            for sell_contract in sell_contracts[step]:
                # buy contracts until step
                bc = buy_contracts[:step].copy()
                bcus = []
                for bc_bs in bc:
                    if not bc_bs:
                        continue
                    for c in bc_bs:
                        bcus += [c]
                if not bcus:
                    sell_contracts[step] = None
                    continue
                # check unit price
                bcus_over_up = [
                    c
                    for c in bcus
                    if c.agreement["unit_price"] < sell_contract.agreement["unit_price"]
                ]
                if not bcus_over_up:
                    sell_contracts[step].remove(sell_contract)
                    continue
                sign_q_sum = sys.maxsize
                sign_u_q_sum = sys.maxsize
                sign_combination = []
                for pick_times in range(1, len(bcus_over_up)):
                    contract_combinations = list(
                        itertools.combinations(bcus_over_up, pick_times)
                    )
                    # each combination check prroduction-schedule,calculate sum(q) and sum(u*q)
                    for contract_combination in contract_combinations:
                        # check q_sum u*q_sum
                        q_sum, u_q_sum = self.quantities_cost_check(
                            contract_combination
                        )
                        # if not enough quantity
                        if (
                            q_sum < sell_contract.agreement["quantity"]
                            and q_sum >= sign_q_sum
                            and u_q_sum >= sign_u_q_sum
                        ):
                            continue
                        sign_q_sum = q_sum
                        sign_u_q_sum = u_q_sum
                        sign_combination = contract_combination
                # add sign contracts list
                if sign_combination:
                    sign_sell_contracts += [sell_contract]
                    for buy_contract in sign_combination:
                        sign_buy_contracts += [buy_contract]
                        buy_contracts[buy_contract.agreement["time"]].remove(
                            buy_contract
                        )
                        if not buy_contracts[buy_contract.agreement["time"]]:
                            buy_contracts[buy_contract.agreement["time"]] = None
                sell_contracts[step].remove(sell_contract)

        # sign contracts
        results = [None] * len(contracts)
        contracts = zip(contracts, range(len(contracts)))
        for contract, i in contracts:
            if contract.annotation["buyer"] == self.id:
                for buy_contract in sign_buy_contracts:
                    if buy_contract == contract:
                        results[i] = self.id
                        break
            else:
                for sell_contract in sign_sell_contracts:
                    if sell_contract == contract:
                        results[i] = self.id
                        break
        return results


class MyProductionStrategy(SupplyDrivenProductionStrategy):
    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            input_product = contract.annotation["product"]
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(step, latest),
                line=-1,
                method="earliest",
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )


class CrescentAgent(
    MyProductionStrategy, MyNegotiationManager, MyTradingStrategy, SCML2020Agent
):
    def init(self):
        super().init()

    def step(self):
        super().step()
