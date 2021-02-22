"""
**Submitted to ANAC 2020 SCML**
*Authors* Masanori HIRANO: hirano@g.ecc.u-tokyo.ac.jp;

Your agent can learn about the state of the world and itself by accessing 
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps  
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your agent can act in the world by calling methods in the AWI it has. 
For example:

- *self.awi.request_negotiation(...)*  # requests a negotiation with one partner
- *self.awi.request_negotiations(...)* # requests a set of negotiations

 
You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list 
  [here](https://scml.readthedocs.io/en/latest/api/scml.scml2020.AWI.html)

"""
import json
import math
import argparse
import random
from scipy.stats import poisson
from tqdm import tqdm

# required for running the test tournament
import time

# required for typing
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Set

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    AgentWorldInterface,
    SAOController,
    Controller,
    SAONegotiator,
    AspirationNegotiator,
    UtilityFunction,
    LinearUtilityFunction,
    PassThroughNegotiator,
    PassThroughSAONegotiator,
    ResponseType,
)
from negmas.helpers import humanize_time, get_class
from scml.scml2020 import (
    SCML2020Agent,
    SCML2020World,
    FactoryState,
    AWI,
    FinancialReport,
    TIME,
    TradingStrategy,
    DemandDrivenProductionStrategy,
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
)
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
    IndDecentralizingAgent,
    MovingRangeAgent,
)
from scml.scml2020.utils import anac2020_collusion, anac2020_std
from scml.scml2020 import Failure
from tabulate import tabulate
from matplotlib import pylab as plt
from collections import defaultdict
import logging
from .PrintingAgent import PrintingAgent, PrintingSAOController
from .Negotiator import IntegratedNegotiationManager, SyncController, ControllerUFun

__all__ = ["MhiranoAgent"]

logger = logging.getLogger("Logging")
# sh = logging.StreamHandler()
# logger.addHandler(sh)
# logger.setLevel(20)
is_debug = False

__all__ = ["MhiranoAgent"]


class MhiranoAgent(DemandDrivenProductionStrategy, PrintingAgent):
    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, is_debug=is_debug, **kwargs)
        # 契約済(signed)の確実なinput量
        self.inputs_signed: np.ndarray = None
        # 契約済(signed)の確実なoutput量
        self.outputs_signed: np.ndarray = None
        # 交渉済(contracted)のほぼ確実なinput量
        self.inputs_contracted: np.ndarray = None
        # 交渉済(contracted)のほぼ確実なoutput量
        self.outputs_contracted: np.ndarray = None
        # 交渉完了(negotiated)のやや確実なinput量
        self.inputs_negotiated: np.ndarray = None
        # 交渉完了(negotiated)のやや確実なoutput量
        self.outputs_negotiated: np.ndaary = None
        # 交渉中(negotiating; request accepting => success/fail)のinput量
        self.inputs_negotiating: List = []
        self.outputs_negotiating: List = []
        self.controller: MhiranoController = MhiranoController(
            parent=self, default_negotiator_type=MhiranoNegotiator
        )

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        awi: AWI = self.awi
        self.inputs_signed = np.zeros(awi.n_steps, dtype=int)
        self.outputs_signed = np.zeros(awi.n_steps, dtype=int)
        self.inputs_contracted = np.zeros(awi.n_steps, dtype=int)
        self.outputs_contracted = np.zeros(awi.n_steps, dtype=int)
        self.inputs_negotiated = np.zeros(awi.n_steps, dtype=int)
        self.outputs_negotiated = np.zeros(awi.n_steps, dtype=int)
        pass

    def step(self):
        """Called at every production step by the world"""
        super().step()
        awi: AWI = self.awi
        factory_state: FactoryState = awi.state
        # contracted >= signed
        assert sum((self.inputs_contracted - self.inputs_signed) < 0) == 0
        assert sum((self.outputs_contracted - self.outputs_signed) < 0) == 0
        # assert len(self.controller._negotiating) == 0
        self.controller._agreements = {"sell": [], "buy": []}
        #####
        # ToDO 在庫の放出注文 => Done & test (価格設定の余地ある)
        #####
        more_or_less: int = sum(
            self.outputs_signed[awi.current_step + 1 : awi.current_step + 10]
        ) - sum(
            self.inputs_signed[awi.current_step + 1 : awi.current_step + 10]
        ) - factory_state.inventory[
            awi.my_input_product
        ]

        if more_or_less > 0:
            # outputの方が多い => 購入したい
            negotiator: Negotiator = self.create_negotiator(is_seller=False)
            partners: List = awi.my_suppliers
            self.request_negotiations(
                is_buy=True,
                product=awi.my_input_product,
                quantity=(1, min(more_or_less, 15)),
                unit_price=(0, awi.catalog_prices[awi.my_output_product]),
                time=(
                    awi.current_step + 1,
                    min(awi.current_step + 15, awi.n_steps - 1),
                ),
                controller=None,
                negotiators=[negotiator for _ in partners],
                partners=partners,
            )
        if more_or_less < 0:
            # inputの方が多い => 売りたい
            negotiator: Negotiator = self.create_negotiator(is_seller=True)
            partners: List = awi.my_consumers
            self.request_negotiations(
                is_buy=False,
                product=awi.my_output_product,
                quantity=(1, min(-more_or_less, 10)),
                unit_price=(
                    awi.catalog_prices[awi.my_input_product],
                    awi.catalog_prices[awi.my_output_product] * 2,
                ),
                time=(
                    awi.current_step + 1,
                    min(awi.current_step + 15, awi.n_steps - 1),
                ),
                controller=None,
                negotiators=[negotiator for _ in partners],
                partners=partners,
            )

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_accepting_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> None:
        super().on_accepting_negotiation_request(
            initiator, issues, annotation, mechanism
        )

    def create_negotiator(self, is_seller: bool) -> Negotiator:
        def create_ufun(is_seller: bool, issues=None, outcomes=None):
            if is_seller:
                return LinearUtilityFunction((1, 1, 10))
            return LinearUtilityFunction((1, -1, -10))

        ufun: UtilityFunction = create_ufun(is_seller=is_seller)
        negotiator: Negotiator = self.controller.create_negotiator(
            negotiator_type=MhiranoNegotiator, ufun=ufun
        )  # MhiranoNegotiator(parent=self, ufun=ufun)
        return negotiator

    def processing_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        awi: AWI = self.awi
        is_seller: bool = annotation["seller"] == self.id
        q_max: int = 0
        t_max: int = 0
        up_max: int = 0
        for issue in issues:
            if issue.name == "quantity":
                if isinstance(issue.values, int):
                    q_max = issue.values
                elif isinstance(issue.values, Tuple):
                    q_max = max(issue.values)
            elif issue.name == "time":
                if isinstance(issue.values, int):
                    t_max = issue.values
                elif isinstance(issue.values, Tuple):
                    t_max = max(issue.values)
            elif issue.name == "unit_price":
                if isinstance(issue.values, int):
                    up_max = issue.values
                elif isinstance(issue.values, Tuple):
                    up_max = max(issue.values)
        assert q_max != 0
        assert t_max >= self.awi.current_step - 1
        assert up_max > 0
        q: int = q_max
        t_start: int = self.awi.current_step + 1
        t_end: int = awi.n_steps - 1
        if is_seller:
            t_prod_start, _ = awi.available_for_production(
                repeats=q, step=(self.awi.current_step, t_max - 1), method="latest"
            )
            if len(t_prod_start) < 1:
                return None
            t_end = min(t_prod_start)
        else:
            t_prod_start, _ = awi.available_for_production(
                repeats=q, step=(self.awi.current_step, awi.n_steps), method="earliest"
            )
            if len(t_prod_start) < 1:
                return None
            t_start = min(t_prod_start) + 1
        production_cost: float = min(awi.profile.costs[0])
        up_max: int = up_max + production_cost * (-2 if is_seller else 10)
        up_min: int = 0 if is_seller else production_cost * 2
        is_buy: bool = is_seller
        product: int = awi.my_input_product if is_seller else awi.my_output_product
        partners: List = awi.my_suppliers if is_seller else awi.my_consumers
        negotiator: Negotiator = self.create_negotiator(is_seller=(not is_buy))
        if up_max <= 0:
            return None
        if t_start > t_end:
            return None
        if up_min > up_max:
            return None
        assert q >= 1
        assert t_end <= awi.n_steps - 1
        self.request_negotiations(
            is_buy=is_buy,
            product=product,
            quantity=(1, q),
            unit_price=(up_min, up_max),
            time=(t_start, t_end),
            controller=None,
            negotiators=[negotiator for _ in partners],
            partners=partners,
        )
        return self.create_negotiator(is_seller=(annotation["seller"] == self.id))

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        super().on_neg_request_rejected(req_id=req_id, by=by)

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """Called whenever an agent requests a negotiation with you.
        Return either a negotiator to accept or None (default) to reject it"""
        super().respond_to_negotiation_request(initiator, issues, annotation, mechanism)
        negotiator: Negotiator = self.processing_negotiation_request(
            initiator, issues, annotation, mechanism
        )

        if negotiator is None:
            self.on_rejecting_negotiation_request(
                initiator, issues, annotation, mechanism
            )
        else:
            self.on_accepting_negotiation_request(
                initiator, issues, annotation, mechanism
            )
        return negotiator

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""
        super().on_negotiation_failure(partners, annotation, mechanism, state)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""
        super().on_negotiation_success(contract, mechanism)
        is_seller: bool = contract.annotation["seller"] == self.id
        t: int = contract.agreement["time"]
        q: int = contract.agreement["quantity"]
        up: int = contract.agreement["unit_price"]
        if is_seller:
            self.outputs_contracted[t] += q
        else:
            self.inputs_contracted[t] += q

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        for contract in contracts:
            if contract.concluded_at == -1:
                is_seller: bool = contract.annotation["seller"] == self.id
                t: int = contract.agreement["time"]
                q: int = contract.agreement["quantity"]
                up: int = contract.agreement["unit_price"]
                if is_seller:
                    self.outputs_contracted[t] += q
                else:
                    self.inputs_contracted[t] += q
        return [self.id] * len(contracts)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in
        a step (day)"""
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for contract in signed:
            is_sell: bool = contract.annotation["seller"] == self.id
            t: int = contract.agreement["time"]
            q: int = contract.agreement["quantity"]
            up: int = contract.agreement["unit_price"]
            if is_sell:
                self.outputs_signed[t] += q
            else:
                self.inputs_signed[t] += q

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""
        super().on_contract_executed(contract)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution
        (i.e. resoluion is None)"""
        super().on_contract_breached(contract, breaches, resolution)

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the
        agent to change what is to be produced in its factory on that step.
        """
        return super().confirm_production(commands, balance, inventory)

    def on_failures(self, failures: List[Failure]) -> None:
        """Called when production fails. If you are careful in
        what you order in `confirm_production`, you should never see that."""
        super().on_failures(failures)

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""
        ######
        # ToDo bankruptの処理を間違えない => Done & test
        ######
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_signed[t] -= missing
                self.outputs_contracted[t] -= missing
            else:
                self.inputs_signed[t] -= missing
                self.inputs_contracted[t] -= missing


class MhiranoUFun(UtilityFunction):
    """A utility function for the controller"""

    def __init__(self, controller=None):
        super().__init__(outcome_type=tuple)
        self.controller = controller

    def eval(self, offer: "Outcome"):
        return self.controller.utility(offer)

    def xml(self, issues):
        pass


class MhiranoController(PrintingSAOController):
    def __init__(self, *args, parent: MhiranoAgent, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self._negotiating: Dict[str, Dict[str, Any]] = dict()
        self._agreements: Dict[str, List["Output"]] = {"sell": [], "buy": []}
        # self.utility_function = ControllerUFun(controller=self)
        self.ufun_max = 0.9
        self.ufun_min = 0.3
        self.power = 1.0
        self._last_evaluation = None

    def _get_best_outcome(
        self,
        is_seller: bool,
        outcomes: List["Outcome"],
        eval_outcomes: List[float] = None,
    ) -> Tuple["Outcome", Union[float, None]]:
        # 正方的なoutcomes spaceを想定する
        # (quantity, time, unit_price)のTupleのListがくる
        # 係数
        if len(outcomes) == 0:
            return None
        if eval_outcomes is None:
            coef: Tuple[int] = (0, 0, 0)
            if is_seller:
                coef = (1, 1, 10)
            else:
                coef = (1, -1, -10)
            dot = np.dot(np.array(outcomes), np.array(coef))
            return outcomes[dot.argmax()], None
        else:
            if sum(np.array(eval_outcomes) == np.array(eval_outcomes).max()) > 1:
                np_candidates: np.ndarray = np.array(outcomes)[
                    np.array(eval_outcomes) == np.array(eval_outcomes).max()
                ]
                if is_seller:
                    # 売り手の場合 納期が遅い => 量が少ない => 価格が高い
                    np_candidates = np_candidates[
                        np_candidates[:, 1] == np_candidates[:, 1].max()
                    ]
                    np_candidates = np_candidates[
                        np_candidates[:, 0] == np_candidates[:, 0].min()
                    ]
                    np_candidates = np_candidates[
                        np_candidates[:, 2] == np_candidates[:, 2].max()
                    ]
                    if len(np_candidates) == 1:
                        return tuple(np_candidates[0]), np.array(eval_outcomes).max()
                    else:
                        return (
                            tuple(np.choose(1, choices=np_candidates)),
                            np.array(eval_outcomes).max(),
                        )
                else:
                    # 買いての場合 納期が早い => 価格が安い => 量が多い
                    np_candidates = np_candidates[
                        np_candidates[:, 1] == np_candidates[:, 1].min()
                    ]
                    np_candidates = np_candidates[
                        np_candidates[:, 2] == np_candidates[:, 2].min()
                    ]
                    np_candidates = np_candidates[
                        np_candidates[:, 0] == np_candidates[:, 0].max()
                    ]
                    if len(np_candidates) == 1:
                        return tuple(np_candidates[0]), np.array(eval_outcomes).max()
                    else:
                        return (
                            tuple(np.choose(1, choices=np_candidates)),
                            np.array(eval_outcomes).max(),
                        )
            return (
                outcomes[np.array(eval_outcomes).argmax()],
                np.array(eval_outcomes).max(),
            )

    def _get_worst_outcome_over_thr(
        self,
        is_seller: bool,
        outcomes: List["Outcome"],
        eval_outcomes: List[float] = None,
        thr: float = None,
    ) -> Tuple["Outcome", Union[float, None]]:
        # 正方的なoutcomes spaceを想定する
        # (quantity, time, unit_price)のTupleのListがくる
        # 係数
        if len(outcomes) == 0:
            return None
        if thr is None:
            thr = 1.0 - 1e-9
        if eval_outcomes is None:
            coef: Tuple[int] = (0, 0, 0)
            if is_seller:
                coef = (1, 1, 10)
            else:
                coef = (1, -1, -10)
            dot = np.dot(np.array(outcomes), np.array(coef))
            new_eval = dot * (dot >= thr) + (2.0 - 1.0 * dot) * (dot < thr)
            return outcomes[new_eval.argmin()], None
        else:
            dot = np.array(eval_outcomes)
            new_eval = dot * (dot >= thr) + (2.0 - 1.0 * dot) * (dot < thr)
            return outcomes[new_eval.argmin()], dot[new_eval.argmin()]

    def _get_current_thr(self, relative_time: float) -> float:
        thr: float = (1 - (relative_time ** self.power)) * (
            self.ufun_max - self.ufun_min
        ) + self.ufun_min
        return thr

    def _cut_outcomes(
        self,
        negotiator_id: str,
        offer: "Outcome",
        is_seller: bool,
        method: str = "worse",
    ) -> None:
        assert method == "worse" or "better", "method allow only 'worse' or 'better'"

        def compare_offers(offer1, offer2) -> str:
            assert len(offer1) == len(offer2)
            judge: str = None
            for i in range(1, len(offer1)):
                if offer1[i] == offer2[i]:
                    continue
                if offer1[i] > offer2[i]:
                    if judge != None and judge != ">":
                        return "none"
                    else:
                        judge = ">"
                else:
                    if judge != None and judge != "<":
                        return "none"
                    else:
                        judge = "<"
            if judge == None:
                judge = "="
            return judge

        if is_seller:
            # 売り手の場合
            if method == "better":
                new_outcomes: List["Outcome"] = [
                    outcome
                    for outcome in self._negotiating[negotiator_id]["current_space"]
                    if compare_offers(offer, outcome) != "<"
                ]
            else:
                new_outcomes: List["Outcome"] = [
                    outcome
                    for outcome in self._negotiating[negotiator_id]["current_space"]
                    if compare_offers(offer, outcome) != ">"
                ]

        else:
            # 買い手の場合
            if method == "better":
                new_outcomes: List["Outcome"] = [
                    outcome
                    for outcome in self._negotiating[negotiator_id]["current_space"]
                    if compare_offers(offer, outcome) != ">"
                ]
            else:
                new_outcomes: List["Outcome"] = [
                    outcome
                    for outcome in self._negotiating[negotiator_id]["current_space"]
                    if compare_offers(offer, outcome) != "<"
                ]
        self._negotiating[negotiator_id]["current_space"] = new_outcomes

    def _get_current_eval(
        self, negotiator_id: str, is_seller: bool, offer: "Outcome"
    ) -> float:
        return self._get_current_eval_all(
            negotiator_id=negotiator_id, is_seller=is_seller, offers=[offer]
        )[0]

    def _time_price_probs(
        self, is_sell: bool, t_up_list: List[Tuple[int, int]]
    ) -> List[float]:
        # 時間とpriceの出現確率
        # priceに対して指数分布を仮定
        # カタログプライスはわざと逆にしている
        awi: AWI = self.parent.awi
        if is_sell:
            mean: int = int(awi.catalog_prices[awi.my_output_product])
        else:
            mean: int = int(awi.catalog_prices[awi.my_input_product])
        np_t_up: np.ndarray = np.array(t_up_list)
        origin_prob: np.ndarray = poisson.pmf(np_t_up[:, 1], mean)
        return list(origin_prob / origin_prob.sum())

    def _mk_evaluation_for_tup(
        self,
        q_max: int,
        is_seller: bool,
        t_up_list: List[Tuple[int, int]],
        input_diff_series: List[int],
        line_usage_list: List[int],
    ) -> Dict[Tuple[int, int], List[float]]:
        def _get_production_end_time_of_buy(
            _q_max: int, _start_time: int, _line_usage_list: List[int]
        ) -> np.ndarray:
            # 最も早い生産時間を考える
            production_end_time: List[Union[int, np.inf]] = []
            if _start_time < 0:
                _start_time = 0
            for i in range(_start_time, self.parent.awi.n_steps):
                if _line_usage_list[i] < self.parent.awi.n_lines:
                    can_produce: int = self.parent.awi.n_lines - _line_usage_list[i]
                    produce: int = min(can_produce, _q_max - len(production_end_time))
                    for _ in range(produce):
                        production_end_time.append(i)

                if len(production_end_time) == _q_max:
                    break
            n_left: int = _q_max - len(production_end_time)
            for _ in range(n_left):
                production_end_time.append(np.inf)
            return np.array(production_end_time) + 1

        def _get_production_start_time_of_sell(
            _q_max: int, _end_time: int, _line_usage_list: List[int]
        ) -> np.ndarray:
            # 最も遅い生産時間を考える
            production_start_time: List[int] = []
            if _end_time >= self.parent.awi.n_steps:
                _end_time = self.parent.awi.n_steps - 1
            for i in range(_end_time - 1, self.parent.awi.current_step - 1, -1):
                if _line_usage_list[i] < self.parent.awi.n_lines:
                    can_produce: int = self.parent.awi.n_lines - _line_usage_list[i]
                    produce: int = min(can_produce, _q_max - len(production_start_time))
                    for _ in range(produce):
                        production_start_time.append(i)
                if len(production_start_time) == _q_max:
                    break
            n_left: int = _q_max - len(production_start_time)
            for _ in range(n_left):
                production_start_time.append(-1)
            return production_start_time

        ####
        # 相手方のnegotiation spaceに対して，価値づけを行う関数 => (q, t, up)に対して価値をつける
        # target
        # is_sellerは評価するoffer
        t_min: int = self.parent.awi.current_step
        t_max: int = min(
            self.parent.awi.current_step + 10, self.parent.awi.n_steps
        )  # endは含まない
        #####
        # {(time, price): [[ev for agg1 for q=1, q=2, ..., q=q_max],...,[...]]}
        #
        all_q: int = 0
        agreements: List[int] = []
        negotiating_opponents: List[Tuple[int, List[Tuple[int, int]]]] = []
        if not is_seller:
            # 評価するofferが買い注文である => materialの不足を補う
            # 売り注文がopponents
            agreements = [max(0, -i) for i in input_diff_series[t_min:t_max]]
            all_q += sum(agreements)
            for data in self._negotiating.values():
                if data["is_seller"]:
                    max_q = max([q for (q, t, up) in data["current_space"]])
                    all_q += max_q
                    negotiating_opponents.append(
                        (
                            max_q,
                            list(
                                set([(t, up) for (q, t, up) in data["current_space"]])
                            ),
                        )
                    )
        else:
            # 評価するofferが売り注文である => material のあまりを消費する
            # 買い注文がopponents
            agreements = [max(0, i) for i in input_diff_series[t_min:t_max]]
            all_q += sum(agreements)
            for data in self._negotiating.values():
                if not data["is_seller"]:
                    max_q = max([q for (q, t, up) in data["current_space"]])
                    all_q += max_q
                    negotiating_opponents.append(
                        (
                            max_q,
                            list(
                                set([(t, up) for (q, t, up) in data["current_space"]])
                            ),
                        )
                    )
        #####
        # agreements: inputsベースの契約済，合意済の時系列相対契約．q at tとなる．timeのスタートは現在時刻
        # negotiating_opponents: 相対する交渉中のもの．契約の形態に合わせたinputs/outputsになっている
        #####
        # 最終的に作りたいもの
        # {(time, price): [ev for q=1, q=2, ..., q=q_max]}
        evaluation_dict: Dict[Tuple[int, int], List[float]] = {}
        #####
        # agreementsを評価 + 全てのt_upの組みを入れていく
        # inputベースの時間軸なのでそのまま比較可能
        awi: AWI = self.parent.awi
        for (t, up) in t_up_list:
            if all_q == 0:
                evaluation_dict[(t, up)] = [0.0 for i in range(1, q_max + 1)]
            else:
                if not is_seller:
                    # 評価する注文が買いの場合(agreementは売り注文)には
                    # inputのカタログプライスでないと計算しない
                    t_relative = t - t_min
                    if up <= awi.catalog_prices[awi.my_input_product]:
                        agree_tmp: List[int] = [
                            agreements[i] if t_relative <= i else 0.0
                            for i in range(t_max - t_min)
                        ]
                    else:
                        agree_tmp: List[int] = [
                            0 if t_relative <= i else 0.0 for i in range(t_max - t_min)
                        ]
                    evaluation_dict[(t, up)] = [
                        min(sum(agree_tmp), i) / all_q for i in range(1, q_max + 1)
                    ]
                else:
                    # 評価する注文が売りの場合(agreementは買い注文)にはには
                    # outputのカタログプライス以上でないと計算しない
                    start_time: List[int] = _get_production_start_time_of_sell(
                        _q_max=q_max, _end_time=t, _line_usage_list=line_usage_list
                    )
                    relative_start_time: List[int] = [x - t_min for x in start_time]
                    evaluation_dict[(t, up)] = [0.0 for i in range(1, q_max + 1)]
                    if up >= awi.catalog_prices[awi.my_output_product]:
                        # priceはOK
                        for j in range(q_max):
                            t_relative = relative_start_time[j]
                            agree_tmp: List[int] = [
                                agreements[i] if t_relative >= i else 0.0
                                for i in range(t_max - t_min)
                            ]
                            if sum(agree_tmp) > j + 1:
                                evaluation_dict[(t, up)][j] += j / all_q
            #####
            # ToDo agreementの価格評価基準の追加 => done & test
            #####
        #####
        # negotiatingの状態を反映していく
        # 買いの場合は，売りを評価すればいいので，買いofferを最短の生産を考える
        # 売りの場合は，買いを評価すればいいので，売りofferを最遅の生産を考える
        if all_q == 0:
            return evaluation_dict

        def _get_profit_prob(
            _t: int, _up: int, _is_sell: bool, _outcome_space: List[Tuple[int, int]]
        ) -> float:
            # t, upを与えたら，ターゲットとなる注文1つのoutcome spaceの何割をカバーできるかを計算
            # is_sell　は (_t, _up)が売り注文かどうか
            _prob_list: List[float] = self._time_price_probs(
                is_sell=_is_sell, t_up_list=_outcome_space
            )
            _results: float = 0.0
            production_cost: float = min(self.parent.awi.profile.costs[0]) * 1.5
            if not is_seller:
                _up += production_cost
            else:
                _up -= production_cost

            np_outcome_space: np.ndarray = np.array(_outcome_space)
            np_prob: np.ndarray = np.array(_prob_list)
            if not is_seller:
                # ターゲットは売り注文
                _results: float = sum(
                    (np_outcome_space[:, 0] >= _t)
                    * (np_outcome_space[:, 1] > _up)
                    * np_prob
                )
            else:
                # ターゲットは買い注文
                _results: float = sum(
                    (np_outcome_space[:, 0] <= _t)
                    * (np_outcome_space[:, 1] < _up)
                    * np_prob
                )
            assert _results >= 0.0
            assert _results <= 1.0 + 1e-9
            return _results

        # negotiating_opponents: 相対する交渉中のもの．契約の形態に合わせたinputs/outputsになっている
        # [(個数, [current outcome space])] となっている
        ######
        # ToDo tqdm消す
        ######
        if not is_seller:
            # 買いの場合は，売りを評価すればいいので，買いofferを最短の生産を考える
            for (t, up), ev_list in evaluation_dict.items():
                products_available_time_at_q: List[
                    int
                ] = _get_production_end_time_of_buy(
                    _q_max=q_max, _start_time=t, _line_usage_list=line_usage_list
                )
                possible_time: Set[int] = set(products_available_time_at_q)
                for (q_target, outcome_space) in negotiating_opponents:
                    probs_by_time: Dict[int, float] = {}
                    for _t in possible_time:
                        prob = _get_profit_prob(
                            _t=t,
                            _up=up,
                            _is_sell=is_seller,
                            _outcome_space=outcome_space,
                        )
                        probs_by_time[_t] = prob
                    for i in range(q_max):
                        q = i + 1
                        _t = products_available_time_at_q[i]
                        if q <= q_target:
                            ev_list[i] += probs_by_time[_t] / all_q
        else:
            # 売りの場合は，買いを評価すればいいので，売りofferを最遅の生産を考える
            for (t, up), ev_list in evaluation_dict.items():
                products_need_time_at_q: List[int] = _get_production_start_time_of_sell(
                    _q_max=q_max, _end_time=t, _line_usage_list=line_usage_list
                )
                possible_time: Set[int] = set(products_need_time_at_q)
                for (q_target, outcome_space) in negotiating_opponents:
                    probs_by_time: Dict[int, float] = {}
                    for _t in possible_time:
                        prob = _get_profit_prob(
                            _t=t,
                            _up=up,
                            _is_sell=is_seller,
                            _outcome_space=outcome_space,
                        )
                        probs_by_time[_t] = prob
                    for i in range(q_max):
                        q = i + 1
                        _t = products_need_time_at_q[i]
                        if q <= q_target:
                            ev_list[i] += probs_by_time[_t] / all_q
        return evaluation_dict

    def _get_current_eval_all(
        self, negotiator_id: str, is_seller: bool, offers: List["Outcome"]
    ) -> List[float]:
        # offerは(q, t, uo)
        # 在庫確認
        awi: AWI = self.parent.awi
        parent: MhiranoAgent = self.parent
        factory_state: FactoryState = awi.state
        material_inventory: int = awi.state.inventory[awi.my_input_product]
        # productはsing済なので breachしても出ていくので在庫は発生し得ない．現在持ってる在庫は今回のstepで出すためのもの
        # product_inventory: int = awi.state.inventory[awi.my_output_product]
        # # もし全てのmaterialを生産するならいつ使えるようになるか
        # production_step, _ = awi.available_for_production(repeats=material_inventory,
        #                                                   step=(awi.current_step, awi.n_steps - 1), method="earliest")
        # products_available_step: List[int] = production_step + 1
        # 余剰を確認(時系列のinputの余剰(マイナスは不足を確認))
        # outputの契約は契約時に生産のscheduleしているので，生産計画を確認すれば良い
        line_usage_list: List[int] = []
        for i in range(awi.n_steps):
            i_line_working = factory_state.commands[i]
            line_usage_list.append(-sum(i_line_working == -1) + factory_state.n_lines)
        input_diff_series: List[int] = parent.inputs_signed - np.array(line_usage_list)
        # 1step前までのdiffは過去のもの
        input_diff_series[: awi.current_step - 1] = 0
        # 現在のstepにmaterial inventoryを追加
        input_diff_series[awi.current_step] += material_inventory
        # agreementの処理
        for sell_agreement in self._agreements["sell"]:
            (q, t, up) = sell_agreement
            assert t >= awi.current_step
            line_usage_list[t - 1] += q
            input_diff_series[t - 1] -= q
        # とりあえず追加して，ラインが足りないところは前倒す
        for i in range(awi.n_steps - 1, awi.current_step - 1, -1):
            if line_usage_list[i] > awi.n_lines:
                move_num: int = line_usage_list[i] - awi.n_lines
                line_usage_list[i] -= move_num
                line_usage_list[i - 1] += move_num
                input_diff_series[i] += move_num
                input_diff_series[i] -= move_num
        for buy_agreement in self._agreements["buy"]:
            (q, t, up) = buy_agreement
            assert t >= awi.current_step
            input_diff_series[t] += q
        #####################
        # ここまででできたもの agreementまでを考慮に入れた
        # input_diff_series: materialの過不足
        # line_usage_list: ラインの使用状況
        #####################
        q_max: int = 0
        t_up_list: List[Tuple[int, int]] = []
        for offer in offers:
            (q, t, up) = offer
            if q > q_max:
                q_max = q
            t_up_list.append((t, up))
        # qが1-q_maxまでの評価値(t,up)の評価値を作成
        # {(time, price): [ev for q=1, q=2, ..., q=q_max]}
        evaluation_dict: Dict[
            Tuple[int, int], List[float]
        ] = self._mk_evaluation_for_tup(
            q_max=q_max,
            is_seller=is_seller,
            t_up_list=t_up_list,
            input_diff_series=input_diff_series,
            line_usage_list=line_usage_list,
        )
        results_list: List[float] = []
        for offer in offers:
            (q, t, up) = offer
            evl: float = sum(evaluation_dict[(t, up)][: q - 1])
            results_list.append(evl)
        return results_list

    def propose_(
        self, negotiator_id: str, state: MechanismState
    ) -> Optional["Outcome"]:
        outcomes: List["Outcome"] = self._negotiating[negotiator_id]["current_space"]
        if len(outcomes) == 0:
            # もし，space cutでスペースがなくなってしまったら
            return None
        is_seller: bool = self._negotiating[negotiator_id]["is_seller"]
        eval_outcomes: List[float] = self._get_current_eval_all(
            negotiator_id=negotiator_id, is_seller=is_seller, offers=outcomes
        )
        best_outcome, utility = self._get_best_outcome(
            is_seller=is_seller, outcomes=outcomes, eval_outcomes=eval_outcomes
        )
        if self._negotiating[negotiator_id]["max_utility"] < utility:
            self._negotiating[negotiator_id]["max_utility"] = utility
        if utility <= 1e-9:
            return None
        thr_p: float = self._get_current_thr(state.relative_time)
        thr_eval: float = thr_p * self._negotiating[negotiator_id]["max_utility"]
        propose_outcome, propose_outcom_eval = self._get_worst_outcome_over_thr(
            is_seller=is_seller,
            outcomes=outcomes,
            eval_outcomes=eval_outcomes,
            thr=thr_eval,
        )
        return propose_outcome

    def record_offer_(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> None:
        self._negotiating[negotiator_id]["last_offer"] = offer

    def respond_(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        self._negotiating[negotiator_id]["last_opponent_offer"] = offer
        # ここにきている時点で前回のproposeがrejectされているか，新しいpropose
        is_seller: bool = self._negotiating[negotiator_id]["is_seller"]
        if self._negotiating[negotiator_id]["last_offer"] is not None:
            # 前回のofferがrejectされた場合には
            self._cut_outcomes(
                negotiator_id=negotiator_id,
                offer=self._negotiating[negotiator_id]["last_offer"],
                is_seller=is_seller,
                method="better",
            )
        else:
            # 前回のofferがない場合はmax_utilityを計算していないので
            outcomes: List["Outcome"] = self._negotiating[negotiator_id][
                "current_space"
            ]
            # slime outcomesで0にはならないようにしているので
            assert len(outcomes) != 0
            eval_outcomes: List[float] = self._get_current_eval_all(
                negotiator_id=negotiator_id, is_seller=is_seller, offers=outcomes
            )
            best_outcome, utility = self._get_best_outcome(
                is_seller=is_seller, outcomes=outcomes, eval_outcomes=eval_outcomes
            )
            if self._negotiating[negotiator_id]["max_utility"] < utility:
                self._negotiating[negotiator_id]["max_utility"] = utility
        self._cut_outcomes(
            negotiator_id=negotiator_id,
            offer=offer,
            is_seller=self._negotiating[negotiator_id]["is_seller"],
            method="worse",
        )
        # spaceのカット終了
        offer_eval: float = self._get_current_eval(
            negotiator_id=negotiator_id, is_seller=is_seller, offer=offer
        )
        if offer_eval <= 1e-9:
            return ResponseType.REJECT_OFFER
        thr_p: float = self._get_current_thr(state.relative_time)
        thr_eval: float = thr_p * self._negotiating[negotiator_id]["max_utility"]
        if offer_eval >= thr_eval:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def on_negotiation_end_(self, negotiator_id: str, state: MechanismState) -> None:
        if negotiator_id not in self._negotiating:
            return None
        if state.agreement is not None:
            (q, t, up) = state.agreement
            is_sell: bool = self._negotiating[negotiator_id]["is_seller"]
            if is_sell:
                self.parent.outputs_negotiated[t] += q
                self._agreements["sell"].append(state.agreement)
            else:
                self.parent.inputs_negotiated[t] += q
                self._agreements["buy"].append(state.agreement)
        self._negotiating.pop(negotiator_id)

    def join_(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        issues: List[Issue],
        outcomes: List["Outcome"],
    ) -> None:
        # スタート時点の issueとoutcomes, amiを保存
        is_seller = ami.annotation["seller"] == self.parent.id
        outcomes = self._slim_outcomes(outcomes=outcomes)
        self._negotiating[negotiator_id] = {
            "state": state,
            "ami": ami,
            "is_seller": is_seller,
            "issues": issues,
            "initial_space": outcomes,
            "current_space": outcomes,
            "last_offer": None,
            "last_opponent_offer": None,
            "max_utility": 0.0,
        }

    def on_negotiation_start_(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        issues: List[Issue],
        outcomes: List["Outcome"],
    ) -> None:
        if negotiator_id not in self._negotiating:
            is_seller = ami.annotation["seller"] == self.parent.id
            ######
            # outcomesのスリム化
            ######
            outcomes = self._slim_outcomes(outcomes=outcomes)
            self._negotiating[negotiator_id] = {
                "state": state,
                "ami": ami,
                "is_seller": is_seller,
                "issues": issues,
                "initial_space": outcomes,
                "current_space": outcomes,
                "last_offer": None,
                "last_opponent_offer": None,
                "max_utility": 0.0,
            }
            return None
        pass

    def _slim_outcomes(self, outcomes: List["Outcome"]):
        q_max: int = 0
        up_max: int = 0
        up_min: int = np.inf
        t_max: int = 0
        t_min: int = np.inf
        _outcomes = outcomes
        for (q, t, up) in _outcomes:
            if q > q_max:
                q_max = q
            if up > up_max:
                up_max = up
            if up < up_min:
                up_min = up
            if t > t_max:
                t_max = t
            if t_min > t:
                t_min = t
        if (t_max - t_min) >= 10:
            if t_min < self.parent.awi.current_step + 10:
                _outcomes = [
                    (q, t, up)
                    for (q, t, up) in _outcomes
                    if t < self.parent.awi.current_step + 10
                ]
            else:
                interval: int = int(math.sqrt(t_max - t_min)) + 1
                _outcomes = [
                    (q, t, up)
                    for (q, t, up) in _outcomes
                    if (t - t_min) % interval == 0
                ]
        if q_max >= 10:
            interval: int = int(math.sqrt(q_max)) + 1
            _outcomes = [(q, t, up) for (q, t, up) in _outcomes if q % interval == 0]
        if (up_max - up_min) >= 10:
            interval: int = int(math.sqrt(up_max - up_min)) + 1
            _outcomes = [
                (q, t, up) for (q, t, up) in _outcomes if (up - up_min) % interval == 0
            ]
        assert len(_outcomes) != 0
        return _outcomes


class MhiranoNegotiator(AspirationNegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__parent = self.parent

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        result: bool = super().join(ami=ami, state=state, ufun=ufun, role=role)
        ami: AgentMechanismInterface = self.ami
        issues = ami.issues
        outcomes = ami.outcomes
        if outcomes is None:
            return None
        self.__parent.join_(
            negotiator_id=self.name,
            ami=ami,
            state=state,
            issues=issues,
            outcomes=outcomes,
        )
        return result

    def on_ufun_changed(self):
        super().on_ufun_changed()

    def on_negotiation_start(self, state: MechanismState) -> None:
        ami: AgentMechanismInterface = self.ami
        issues = ami.issues
        outcomes = ami.outcomes
        self.__parent.on_negotiation_start_(
            negotiator_id=self.name,
            ami=ami,
            state=state,
            issues=issues,
            outcomes=outcomes,
        )
        super().on_negotiation_start(state=state)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return self.__parent.respond_(negotiator_id=self.name, state=state, offer=offer)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        offer: "Outcome" = self.__parent.propose_(negotiator_id=self.name, state=state)
        if offer is not None:
            self.__parent.record_offer_(
                negotiator_id=self.name, state=state, offer=offer
            )
        return offer

    def on_negotiation_end(self, state: MechanismState) -> None:
        self.__parent.on_negotiation_end_(negotiator_id=self.name, state=state)
        super().on_negotiation_end(state=state)


def run(
    competition="std",
    reveal_names=True,
    n_steps=100,
    n_configs=1,
    n_runs_per_world=1,
    is_many: bool = False,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std, 
                     collusion).        
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try. 
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value        

    """
    competitors = [MhiranoAgent, DecentralizingAgent, BuyCheapSellExpensiveAgent]
    if is_many:
        competitors += [IndDecentralizingAgent, MovingRangeAgent]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
            parallelism="serial",
            agent_names_reveal_type=reveal_names,
            compact=True,
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
            parallelism="serial",
            agent_names_reveal_type=reveal_names,
            compact=True,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    result_dict = {}
    for (name, score) in zip(
        results.total_scores["agent_type"].apply(lambda x: x.split(".")[-1]),
        results.total_scores["score"],
    ):
        result_dict[name] = score
    json.dump(result_dict, open("_output.json", "w"))


def debug():
    agent_types = [
        MhiranoAgent,
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
    ]  # + [IndDecentralizingAgent, MovingRangeAgent]
    world = SCML2020World(**SCML2020World.generate(agent_types=agent_types, n_steps=50))
    _, _ = world.draw()
    plt.show()
    if is_debug:
        world.run()
    else:
        world.run_with_progress()  # may take few minutes
    print(world.winners[0])
    fig, (score, profit) = plt.subplots(1, 2)
    final_scores = [
        100 * world.stats[f"score_{_}"][-1] for _ in world.non_system_agent_names
    ]
    final_profits = [
        100 * world.stats[f"balance_{_}"][-1] / world.stats[f"balance_{_}"][0] - 100
        for _ in world.non_system_agent_names
    ]
    plt.setp(score.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(profit.xaxis.get_majorticklabels(), rotation=45)
    score.bar(world.non_system_agent_names, final_scores)
    profit.bar(world.non_system_agent_names, final_profits)
    score.set(ylabel="Final Score (%)")
    profit.set(ylabel="Final Profit (%)")
    fig.show()

    def show_agent_scores(world):
        scores = defaultdict(list)
        for aid, score in world.scores().items():
            scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
        scores = {k: sum(v) / len(v) for k, v in scores.items()}
        plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
        plt.show()

    show_agent_scores(world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("arg1", help="is_std")
    parser.add_argument("arg2", help="is_many")
    parser.add_argument("arg3", help="dummy")
    parser.add_argument("arg4", help="seeds")
    args = parser.parse_args()
    seed: int = int(args.arg4)
    is_std: bool = (args.arg1 == "1")
    is_many: bool = (args.arg2 == "1")
    print("seed: {}, is_std:{}, is_many:{}".format(seed, is_std, is_many))
    random.seed(seed)
    np.random.seed(seed)
    run(competition=("std" if is_std else "collusion"), is_many=is_many)
    # debug()
