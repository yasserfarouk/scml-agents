import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    ResponseType,
    SAOState,
)
from scml.scml2020 import (
    ANY_LINE,
    QUANTITY,
    TIME,
    UNIT_PRICE,
    FinancialReport,
    SCML2020Agent,
)
from scml.scml2020.services import SyncController

__all__ = ["AgentProjectGC"]


class AgentProjectGCNego(SyncController):
    def __init__(self, *args, agent_ref, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent_ref = agent_ref

    def utility(self, offer: Tuple[int, int, int], max_price: int) -> float:
        if offer is None:
            return -1000

        t = offer[TIME]
        if t < self.agent_ref.awi.current_step or t > self.agent_ref.awi.n_steps - 1:
            return -1000.0

        if self._is_seller:
            _needed, _secured = (
                self.agent_ref.outputs_needed,
                self.agent_ref.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.agent_ref.inputs_needed,
                self.agent_ref.inputs_secured,
            )

        q = offer[QUANTITY]
        # q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])

        if self.agent_ref.awi.current_step < self.agent_ref.awi.n_steps // 3:
            time_coef = 0.6
            revenue_coef = 0.4
        else:
            time_coef = 0.1
            revenue_coef = 0.9

        if self._is_seller:
            # seller
            ufunc = (offer[UNIT_PRICE] * q * revenue_coef) + (
                (1 / offer[TIME]) * time_coef
            )
        else:
            # buyer
            ufunc = ((0.5 * offer[UNIT_PRICE]) * q * revenue_coef) + (
                (1 / offer[TIME]) * time_coef
            )
            # print('>-:', time_coef, revenue_coef, self._is_seller, offer, ufunc)

        # quantity
        return ufunc


class AgentProjectGC(
    SCML2020Agent
):  # , IndependentNegotiationsManager, PredictionBasedTradingStrategy, DemandDrivenProductionStrategy):
    # name is subject to change
    schedule_range: Dict[str, Tuple[int, int, bool]] = dict()
    _current_end = -1
    _current_start = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inputs_needed: np.ndarray = None
        self.outputs_needed: np.ndarray = None
        self.inputs_secured: np.ndarray = None
        self.outputs_secured: np.ndarray = None

    def init(self):
        self.inputs_secured = np.zeros(self.awi.n_steps, dtype=int)
        self.outputs_secured = np.zeros(self.awi.n_steps, dtype=int)
        self.inputs_needed = np.zeros(self.awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)

        def adjust():
            return max(1, self.awi.n_lines // 2) * np.ones(self.awi.n_steps, dtype=int)

        expected_outputs = adjust()
        expected_inputs = adjust()

        self.inputs_needed[:-1] = expected_outputs[1:]
        self.outputs_needed[1:] = expected_inputs[:-1]

    def step(self):
        super().step()

        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(self.awi.n_steps - 1, self._current_start)
        # self._current_end = min(self.awi.n_steps - 1, self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)))

    # BEGIN TRADING STRATEGY
    """
    A trading strategy that uses market information about agents to decide on signing contracts
    """

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        super().sign_all_contracts(contracts)

        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step

        possible_contracts = []
        # decide which contracts to be signed
        sold, bought = 0, 0
        for idx, contract in enumerate(contracts):
            is_seller = contract.annotation["seller"] == self.id
            time = contract.agreement["time"]
            q = contract.agreement["quantity"]

            # skip impossible contracts
            if time > latest + 1 or time < earliest_production:
                continue

            if is_seller:
                time_range = (earliest_production, time)
                taken = sold
            else:
                time_range = (time + 1, latest)
                taken = bought

            steps, lines = self.awi.available_for_production(
                q, time_range, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue

            if is_seller:
                sold += q
            else:
                bought += q

            # push to heap queue, smallest step, will be first to pop
            heapq.heappush(possible_contracts, [int(time), idx, contract])

        return [
            heapq.heappop(possible_contracts)[1] for _ in range(len(possible_contracts))
        ]

    # END

    # BEGIN PRODUCTION STRATEGY
    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)

        latest = self.awi.n_steps - 2
        consumed = 0
        for contract in signed:
            quan = contract.agreement["quantity"]
            time = contract.agreement["time"]

            if contract.annotation["caller"] == self.id:
                continue
            if contract.annotation["seller"] == self.id:
                out_product = contract.annotation["product"]
                inp_product = out_product - 1
                self.outputs_secured[time] += quan

                if inp_product >= 0 and time > 0:
                    steps, lines = self.awi.available_for_production(
                        repeats=quan, step=(self.awi.current_step, time - 1)
                    )
                    quan = min(len(steps) - consumed, quan)
                    consumed += quan
                    if contract.annotation["caller"] != self.id:
                        self.inputs_needed[time - 1] += max(1, quan)

                steps, lines = self.awi.schedule_production(
                    process=inp_product,
                    repeats=contract.agreement["quantity"],
                    step=(time, latest),
                    line=-1,
                    partial_ok=True,
                )
                self.schedule_range[contract.id] = (
                    min(steps) if len(steps) > 0 else -1,
                    max(steps) if len(steps) > 0 else -1,
                    True,  # is seller
                )
            else:
                inp_product = contract.annotation["product"]
                output_product = inp_product + 1
                self.inputs_secured[time] += quan
                if output_product < self.awi.n_products and time < self.awi.n_steps - 1:
                    # this is a buy contract that I did not expect yet. Update needs accordingly
                    self.outputs_needed[time + 1] += max(1, quan)

    # END

    # BEGIN NEGOTIATION STRATEGY
    def utility(self, is_seller: bool, issues: List[Issue], outcomes=None):
        if self.awi.current_step < self.awi.n_steps // 3:
            time_coef = 0.6
            revenue_coef = 0.4
        else:
            time_coef = 0.1
            revenue_coef = 0.9

        if not is_seller:
            ufunc = (issues[UNIT_PRICE] * issues[QUANTITY] * -revenue_coef) + (
                (1 / issues[TIME]) * -time_coef
            )
        else:
            ufunc = (issues[UNIT_PRICE] * issues[QUANTITY] * revenue_coef) + (
                (1 / issues[TIME]) * time_coef
            )

            # quantity
        return ufunc

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        super().respond_to_negotiation_request(initiator, issues, annotation, mechanism)

        # refuse negotiation if out of time range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None

        # risk analysis
        if self.id != annotation["seller"]:  # is buyer
            # get seller
            seller_id = annotation["seller"]
            # query breach prob for seller
            seller_bb = self.awi.bb_query("reports_agent", None)
            max_breach_prob = 0.0
            if seller_id in seller_bb:
                for key in seller_bb[seller_id]:
                    report: FinancialReport = seller_bb[seller_id][key]
                    max_breach_prob = max(max_breach_prob, report.breach_prob)
            if max_breach_prob > 0.9:
                return None

            # utils = np.array([self.utility(o) for o in issues])
            # best_index = int(np.argmax(utils))
            # best_utility = utils[best_index]

            # best_partner = negotiator_ids[best_index]
            # best_offer = offers[best_partner]

            # SAOResponse(ResponseType.ACCEPT_OFFER
            # self.awi.agent.
        # find my best proposal for each negotiation
        # best_proposals = self.first_proposals()

        controller = AgentProjectGCNego(
            agent_ref=self,
            is_seller=self.id == annotation["seller"],
            parent=self,
            utility_threshold=0.1,
            time_threshold=0.5,
        )

        return controller.create_negotiator()
