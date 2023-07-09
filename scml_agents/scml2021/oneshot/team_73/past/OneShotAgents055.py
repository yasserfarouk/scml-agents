import itertools
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import Iterable, Tuple, Union

import matplotlib.colors
from matplotlib import pyplot as plt
from negmas import MechanismState, ResponseType, SAOResponse
from negmas.outcomes import Outcome
from OneShotAgents.nego_utils import *
from scml.oneshot import *

Buy = 0
Sell = 1
Offer = 0
Accept = 1
INF = 1000


class SimpleAgent(OneShotAgent, ABC):
    """A greedy agent based on OneShotAgent"""

    def __init__(self, owner=None, ufun=None, name=None):
        super().__init__(owner, ufun, name)
        self.secured = 0

    def init(self):
        pass

    def step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product


class BetterAgent(SimpleAgent, ABC):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        nmi = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(nmi, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent, ABC):
    """Considers best price offers received when making its decisions"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(args, concession_exponent, kwargs)
        self._best_selling, self._best_buying = 0.0, float("inf")

    def init(self):
        super().init()

    def step(self):
        super().step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state):
        """Save the best price received"""
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


class AdaptiveSyncAgent(OneShotSyncAgent, AdaptiveAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def init(self):
        AdaptiveAgent.init(self)

    def step(self):
        AdaptiveAgent.step(self)

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        SECURED_MAGNIFICATION = 1.2

        my_needs = self._needed()
        if my_needs <= 0:
            # my_needsを満たしているときは交渉終了
            responses = dict(
                zip(
                    [str(k) for k in offers.keys()],
                    [
                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for _ in offers.keys()
                    ],
                )
            )
            return responses

        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v) for k, v in offers.items()
        }
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling, offers.keys()),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output, name = sorted_offers[i]
            response = AdaptiveAgent.respond(self, name, states[name], offer)

            if response == ResponseType.ACCEPT_OFFER:
                if secured < my_needs:
                    secured += offer[QUANTITY]
                    responses[name] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        for name, offer in offers.items():
            counter_offer = AdaptiveAgent.propose(self, name, states[name])
            list(counter_offer)[QUANTITY] = min(
                self.awi.profile.n_lines, counter_offer[QUANTITY]
            )
            if responses[name] != SAOResponse(ResponseType.ACCEPT_OFFER, None):
                responses[name] = SAOResponse(
                    ResponseType.REJECT_OFFER, tuple(counter_offer)
                )

        return responses


class LearningAgent(AdaptiveAgent, ABC):
    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state)
        # update my current best price to use for limiting concession in other
        # negotiations
        nmi = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = nmi.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx


class AgentT055(AdaptiveAgent, ABC):
    better_agent = BetterAgent(concession_exponent=0.2)

    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=float("inf"),
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        concession_exponent=0.1,
        worst_opp_acc_price_slack=0.0,
        first_offer_price_slack=INF,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack
        self.new_price_selling, self.new_price_buying = float("inf"), 0.0  # 価格変化後の交渉価格
        self.new_price_slack = 0.05
        self.concession_threshold = 3  # 譲歩の変化率の閾値
        self.worst_opp_acc_price_slack = (
            worst_opp_acc_price_slack  # 相手にとって最も良い合意価格に関するSlack変数
        )
        self.first_offer_price_slack = (
            first_offer_price_slack  # 合意のない相手に対するoffer価格に関するslack変数
        )

        # 取引情報
        self.success_list = defaultdict(lambda: list())  # 交渉成功した際の取引データ
        self.success_contracts = []  # 交渉成功した契約のリスト
        self.failure_opp_list = []  # 各日の交渉失敗した相手のリスト
        self.opp_offer_list = defaultdict(lambda: list())  # 相手のOfferのリスト
        self.my_offer_list = defaultdict(lambda: list())  # 自分のOfferのリスト
        self.nego_info = {}  # 交渉情報

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self.failure_opp_list = []
        self._opp_price_slack = 0.0
        self._opp_acc_price_slack = 0.2
        self.worst_opp_acc_price_slack = 0.0
        self.first_offer_price_slack = INF

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        is_selling = self._is_selling(self.get_nmi(negotiator_id))
        nmi = self.get_nmi(negotiator_id)
        if is_selling:
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["seller"]
            )
        else:
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["buyer"]
            )

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

        # 取引データを記録
        if self._is_selling(mechanism):
            self.success_list[shorten_name(contract.partners[0])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
        else:
            self.success_list[shorten_name(contract.partners[1])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
        self.success_contracts.append(contract)

    def propose(self, negotiator_id: str, state) -> "Outcome":
        self.nego_info["negotiation_step"] = state.step  # 交渉ステップを記録

        offer = super().propose(negotiator_id, state)

        if offer is None:
            return None

        offer = list(offer)
        offer[QUANTITY] = min(self.awi.profile.n_lines, offer[QUANTITY])

        self._record_information({shorten_name(negotiator_id): offer}, True)  # offerの保存

        # デバッグ用
        # if self.nego_info["negotiation_step"] == 19:
        #     print_log(["step", "proposer name"], [self.awi.current_step, self.nego_info["my_name"]])
        #     print_log("offer", offer)

        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        self.nego_info["negotiation_step"] = state.step  # 交渉ステップを記録
        self._record_information(
            {shorten_name(negotiator_id): offer}, False
        )  # offerの保存

        # update my current best price to use for limiting concession in other
        # negotiations
        nmi = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)

        response = super().respond(negotiator_id, state)

        # # opp_best_offerのリセット
        # if self.nego_info["negotiation_step"] <= nmi.n_steps / 2:
        #     self._best_selling, self._best_buying = 0.0, float("inf")

        # # 最終ステップかつこれ以上相手のOfferがない場合は受け入れ
        # if response != ResponseType.END_NEGOTIATION:
        #     response = self.final_answer(negotiator_id, response)

        # デバッグ用
        # if self.nego_info["negotiation_step"] == 19:
        #     print_log(["step", "responder name", "to"], [self.awi.current_step, self.nego_info["my_name"], negotiator_id])

        return response

    def _find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        is_selling = self._is_selling(nmi)
        # offer a price that is around th of your best possible price

        # パラメタの設定
        name = nmi.annotation["buyer"] if is_selling else nmi.annotation["seller"]
        success_agreements = opponent_agreements(
            nmi, is_selling, self.success_contracts
        )
        accept_agreements = [
            _
            for _ in success_agreements
            if _.mechanism_state["current_proposer"] == self.nego_info["my_name"]
        ]
        offer_agreements = [
            _
            for _ in success_agreements
            if _.mechanism_state["current_proposer"] != self.nego_info["my_name"]
        ]
        step = state.step
        rank = opponent_rank(
            list(self.active_negotiators.keys()), is_selling, self.success_contracts
        )
        good_price_range = self._good_price_range(nmi)
        std = good_price_range["max"] if is_selling else good_price_range["min"]
        pattern = ["offer"]
        if self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
            if self._environment_factor(nmi) >= 0.5:
                pattern.append("good_env")
                if success_agreements:
                    if offer_agreements:
                        pattern.append("offer_agreements")
                    else:
                        pattern.append("accept_agreements")
                        if price_comparison(
                            is_selling,
                            worst_opp_acc_price(
                                nmi, is_selling, self.success_contracts
                            ),
                            std,
                        ):
                            pattern.append("good")
                        else:
                            pattern.append("bad")

                    if list(rank.keys())[0] == name:
                        step = min(state.step + 1, nmi.n_steps - 1)
                elif self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
                    pattern.append("first_offer")
                else:
                    pattern.append("")  # パラメタを変更しない
            else:
                pattern.append("bad_env")
                if success_agreements:
                    if offer_agreements:
                        pattern.append("offer_agreements")
                    else:
                        pattern.append("accept_agreements")
                        if price_comparison(
                            is_selling,
                            worst_opp_acc_price(
                                nmi, is_selling, self.success_contracts
                            ),
                            std,
                        ):
                            pattern.append("good")
                        else:
                            pattern.append("bad")
                    if list(rank.keys())[0] == name:
                        step = min(state.step + 1, nmi.n_steps - 1)
                elif self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
                    pattern.append("first_offer")
                else:
                    pattern.append("")  # パラメタを変更しない
            self._set_param(pattern)

        mn, mx = self._price_range(nmi)
        th = self._th(step, nmi.n_steps)

        if is_selling:
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        is_selling = self._is_selling(nmi)

        # 相手の譲歩率に応じて判断
        name = nmi.annotation["buyer"] if is_selling else nmi.annotation["seller"]
        success_agreements = opponent_agreements(
            nmi, is_selling, self.success_contracts
        )
        pattern = ["accept"]
        if self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
            if self._opp_concession_rate_change(name):
                pattern.append("concession")
            else:
                if success_agreements:
                    pattern.append("persist")
                else:
                    up = nmi.issues[UNIT_PRICE]
                    if is_selling:
                        return price >= (up.min_value + up.max_value) / 2
                    else:
                        return price <= (up.min_value + up.max_value) / 2
            self._set_param(pattern)

        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _good_price_range(self, nmi: SAONMI):
        """エージェントにとって良い価格帯を見つける"""
        is_selling = self._is_selling(nmi)
        mx = nmi.issues[UNIT_PRICE].max_value
        mn = nmi.issues[UNIT_PRICE].min_value

        price_range = {
            "min": mn
            + AgentT055.better_agent._th(nmi.n_steps - 2, nmi.n_steps) * (mx - mn),
            "max": mx
            - AgentT055.better_agent._th(nmi.n_steps - 2, nmi.n_steps) * (mx - mn),
        }

        return price_range

    def _first_offer_price(self, name: str):
        """合意のない相手に対するofferの価格を決定"""
        nmi = self.get_nmi(name)
        good_price_range = self._good_price_range(nmi)
        is_selling = self._is_selling(nmi)
        time = t(self.awi.current_step, self.awi.n_steps)

        # 基準値の決定
        std = (good_price_range["min"] + good_price_range["max"]) / 2
        # std = good_price_range["max"] if is_selling else good_price_range["min"]  # 正しい方
        # std = good_price_range["min"] if is_selling else good_price_range["max"]

        # # 価格を決定（段階的に譲歩）
        # th = [0.0, 0.2]
        # if time < th[0]:
        #     price = std * (1 + TF_sign(is_selling) * 0.1)
        # elif th[0] <= time <= th[1] or 0.5 < self._self_factor(nmi):
        #     price = std * (1 + TF_sign(is_selling) * 0.0)
        # elif 0.3 <= self._self_factor(nmi) <= 0.5:
        #     price = std * (1 + TF_sign(is_selling) * (-0.1))
        # else:
        #     price = std * (1 + TF_sign(is_selling) * (-0.2))

        # 価格を決定（滑らかに譲歩）
        strong_range = {"max": 0.3, "min": -0.3}
        rng = strong_range["max"] - strong_range["min"]
        th = [0.0, 0.3]
        if time < th[0]:
            strong_degree = strong_range["max"]
        elif th[0] <= time <= th[1] or 0.5 < self._self_factor(nmi):
            strong_degree = strong_range["max"] - rng * min(
                time - th[0] / th[1] - th[0], 1
            )
        else:
            strong_degree = strong_range["min"] - 0.1

        price = std * (1 + TF_sign(is_selling) * strong_degree)

        return price

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        concession_degree = 1 - self._strong_degree(nmi)
        is_selling = self._is_selling(nmi)
        name = nmi.annotation["buyer"] if is_selling else nmi.annotation["seller"]

        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                # self.new_price_selling * (1 - self.new_price_slack),
                # min([_[UNIT_PRICE] for _ in self.my_offer_list[partner]] + [float("inf")]),
                # min([_.agreement["unit_price"] for _ in success_agreements] + [float("inf")]),
                worst_opp_acc_price(nmi, is_selling, self.success_contracts)
                * (1 + self.worst_opp_acc_price_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            # (self._best_selling, self._step_price_slack),
                            # (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                            (
                                self._first_offer_price(name),
                                self.first_offer_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = nmi.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                # self.new_price_buying * (1 + self.new_price_slack),
                # max([_[UNIT_PRICE] for _ in self.my_offer_list[partner]] + [0]),
                # max([_.agreement["unit_price"] for _ in success_agreements] + [0]),
                worst_opp_acc_price(nmi, is_selling, self.success_contracts)
                * (1 - self.worst_opp_acc_price_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            # (self._best_buying, self._step_price_slack),
                            # (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                            # (self.opp_next_price(partner), self._opp_price_slack),
                            (
                                self._first_offer_price(name),
                                self.first_offer_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx

    def _self_factor(self, nmi):
        """自身の交渉の進捗を評価"""
        prev_agreement = 0  # 前日合意できたか
        agreement_ratio = 0  # 相手との交渉成功割合
        good_agreement = 0  # 良い値段で合意できたか
        w_prev = 4
        w_good = 2
        w_prev, w_good = param_normalization([w_prev, w_good])

        # これまでの交渉成功割合
        success_agreements = [
            [_.agreement, _.mechanism_state["current_proposer"]]
            for _ in self.success_contracts
        ]
        if success_agreements:
            simulation_steps = {lis[0]["time"] for lis in success_agreements}
            prev_agreement = len(simulation_steps) / (self.awi.current_step + 1)
        else:
            prev_agreement = 1

        # 良い値段で合意できているか
        success_agreements = opponent_agreements(
            nmi, self._is_selling(nmi), self.success_contracts
        )
        if success_agreements:
            tp = self.awi.trading_prices[1]
            prev_up = success_agreements[-1].agreement["unit_price"]
            if self._is_selling(nmi):
                good_agreement = 0.5 - (prev_up - tp) / tp
            else:
                good_agreement = 0.5 + (prev_up - tp) / tp
        else:
            good_agreement = 0.5

        # デバッグ用
        # print_log("params", param_normalization([w_prev, w_ratio, w_good]))
        # print_log("params", [prev_agreement, agreement_ratio, good_agreement])
        # print_log("self_factor", w_prev * prev_agreement + w_good * good_agreement)

        # 重み付けして足す
        return w_prev * prev_agreement + w_good * good_agreement

    def _environment_factor(self, nmi):
        """マーケットの状況を評価"""
        if self._is_selling(nmi):
            n_sellers = len(self.awi.all_suppliers[1])
            n_buyers = len(self.awi.my_consumers)
            return min(n_buyers / n_sellers / 2, 1)
        else:
            n_sellers = len(self.awi.my_suppliers)
            n_buyers = len(self.awi.all_consumers[1])
            return min(n_sellers / n_buyers / 2, 1)

    def _opp_concession_rate_change(self, name: str):
        """相手の譲歩の変化率を計算"""
        nmi = self.get_nmi(name)
        offers = [
            _
            for _ in self.opp_offer_list[shorten_name(name)]
            if _[TIME] == self.awi.current_step
        ]
        if len(offers) >= 3:
            prev = offers[-2][UNIT_PRICE] - offers[-3][UNIT_PRICE]
            now = offers[-1][UNIT_PRICE] - offers[-2][UNIT_PRICE]
            if prev == 0:
                return 0
            rng = nmi.issues[UNIT_PRICE].max_value - nmi.issues[UNIT_PRICE].min_value

            # return now / prev > self.concession_threshold and abs(now) >= rng / 8
            return now / prev > self.concession_threshold
        else:
            return False

    def _record_information(self, offers: dict, mine: bool):
        """offer や utilを保存"""
        # offerを保存
        if mine:
            for k, v in offers.items():
                o = list(v)
                o.append(self.nego_info["negotiation_step"])
                self.my_offer_list[shorten_name(k)].append(o)
        else:
            for k, v in offers.items():
                o = list(v)
                o.append(self.nego_info["negotiation_step"])
                self.opp_offer_list[shorten_name(k)].append(o)

    def _set_param(self, pattern: List[str]) -> None:
        """
        各種パラメタを引数によって設定
        :param pattern:
        :return: None
        """

        r = 0.1
        if pattern[0] == "accept":
            self.first_offer_price_slack = INF
            if pattern[1] == "concession":
                # self._step_price_slack = 0.1
                self._opp_price_slack = 0.0
                self._opp_acc_price_slack = 0.2
                self.worst_opp_acc_price_slack = INF

            elif pattern[1] == "persist":
                # self._step_price_slack = 0.0
                self._opp_price_slack = 0.0
                self._opp_acc_price_slack = 0.0
                self.worst_opp_acc_price_slack = INF

        elif pattern[0] == "offer":
            if pattern[1] == "good_env":
                if pattern[2] == "offer_agreements":
                    # self._step_price_slack = 0.0
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = -INF
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = INF

                elif pattern[2] == "accept_agreements":
                    self.first_offer_price_slack = INF
                    if pattern[3] == "good":
                        # self._step_price_slack = 0.0
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                    elif pattern[3] == "bad":
                        # self._step_price_slack = 0.0
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                elif pattern[2] == "first_offer":
                    # self._step_price_slack = 0.0
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = 0.0
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = 0.0

            elif pattern[1] == "bad_env":
                if pattern[2] == "offer_agreements":
                    # self._step_price_slack = 0.0
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = -INF
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = INF

                elif pattern[2] == "accept_agreements":
                    self.first_offer_price_slack = INF
                    if pattern[3] == "good":
                        # self._step_price_slack = 0.0
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                    elif pattern[3] == "bad":
                        # self._step_price_slack = 0.0
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                elif pattern[2] == "first_offer":
                    # self._step_price_slack = 0.0
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = 0.0
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = 0.0

    def _strong_degree(self, nmi):
        # w_self = 0.2
        # w_env = 1 - w_self
        # return w_self * self.self_factor(nmi) + w_env * self.environment_factor(nmi)
        environment_factor = self._environment_factor(nmi)
        self_factor = self._self_factor(nmi)
        strong_degree = environment_factor + (self_factor - 0.5)
        # print_log("strong_degree", strong_degree)
        return min(1.0, max(0.0, strong_degree))

    def _opp_next_price(self, name):
        delta = 1
        nmi = self.get_nmi(name)
        offers = [
            _
            for _ in self.opp_offer_list[shorten_name(name)]
            if _[TIME] == self.awi.current_step
        ]

        if len(offers) == 0 or self.nego_info["negotiation_step"] == nmi.n_steps:
            if self._is_selling(nmi):
                return nmi.issues[UNIT_PRICE].min_value
            else:
                return nmi.issues[UNIT_PRICE].max_value

        if len(offers) > delta:
            prev = offers[-delta - 1][UNIT_PRICE]
            now = offers[-1][UNIT_PRICE]
            next_price = now + (now - prev) / delta
        else:
            if self._is_selling(nmi):
                prev = nmi.issues[UNIT_PRICE].min_value
            else:
                prev = nmi.issues[UNIT_PRICE].max_value
            now = offers[-1][UNIT_PRICE]
            next_price = now + (now - prev) / delta

        return next_price

    def _final_answer(self, name: str, response: ResponseType):
        # 最終ステップかつこれ以上相手のOfferがない場合は受け入れ
        nmi = self.get_nmi(name)
        is_selling = self._is_selling(nmi)
        if (
            self.nego_info["negotiation_step"] >= nmi.n_steps - 1
            and len(self.failure_opp_list) == len(self.active_negotiators.keys()) - 1
        ):
            response = ResponseType.ACCEPT_OFFER

        # 受け入れない場合はリストに追加
        if response != ResponseType.ACCEPT_OFFER:
            self.failure_opp_list.append(name)

        return response

    def _utility_check(self, name: str, offer: tuple):
        """提案or受諾するOfferの効用値を何もしない時と比べる"""
        nmi = self.get_nmi(name)

        util = self.ufun.from_offers((offer,), (self._is_selling(nmi),))
        do_nothing_util = self.ufun.from_offers(tuple(), (self._is_selling(nmi),))
        # print_log(["util", "do_nothing_util"], [util, do_nothing_util])

        if util <= do_nothing_util:
            return False
        else:
            return True

    def _change_trading_price(self, name: str, offer: list):
        """交渉価格を大幅に変化させる"""
        nmi = self.get_nmi(name)
        is_selling = self._is_selling(nmi)
        slack = 0.2

        # stepに応じたofferを生成
        timing = int(self.awi.n_steps * 0.7)
        # if self.awi.current_step < timing:
        #     if self.nego_info["negotiation_step"] < 2:
        #         offer[UNIT_PRICE] = self.awi.trading_prices[1]
        #         offer[QUANTITY] = 1
        if self.awi.current_step == timing:
            if self.nego_info["negotiation_step"] < 2:
                if is_selling:
                    self.new_price_selling = min(
                        nmi.issues[UNIT_PRICE].max_value * 0.9,
                        min(_.agreement["unit_price"] for _ in self.success_contracts)
                        * (1 - slack),
                    )
                    offer[UNIT_PRICE] = self.new_price_selling
                # else:
                #     self.new_price_buying = \
                #         max(nmi.issues[UNIT_PRICE].min_value * 1.1,
                #             max([_.agreement["unit_price"] for _ in self.success_contracts]) * (1 + slack))
                #     offer[UNIT_PRICE] = self.new_price_buying
                offer[QUANTITY] = 1
        elif self.awi.current_step > timing:
            if self.nego_info["negotiation_step"] < 2:
                if is_selling:
                    offer[UNIT_PRICE] = self.new_price_selling
                # else:
                #     offer[UNIT_PRICE] = self.new_price_buying
                offer[QUANTITY] = 1

        return offer


class LearningSyncAgent_(OneShotSyncAgent, ABC):
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(
        self,
        *args,
        delta=0.02,
        concession_exponent=0.2,
        util_exponent=1,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.03,
        opp_util_slack=0.01,
        agr_util_slack=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # ハイパーパラメタ
        self._threshold = 0.7
        self._delta = delta
        self._e = concession_exponent
        self.e = util_exponent  # 閾値の決定に用いる
        self.strong_degree = 1.0  # offerのprice rangeを決定する際に用いる（どれだけ強気かどうか）
        self.parameter_min_max = defaultdict(lambda: list())  # デバッグ用
        self.min_max_delta = 0.1  # strong_degreeの変化の度合い
        self.successful_times_thr = 3  # 連続交渉成功回数の閾値

        # Slack変数，Delta
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack
        self.opp_util_slack = opp_util_slack  # opp_utilのSlack変数
        self.agr_util_slack = agr_util_slack  # agr_utilのSlack変数

        # 取引データの管理
        self.secured = 0
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self.success_list = defaultdict(lambda: list())  # 交渉成功した際の取引データ
        self.success_contracts = []  # 交渉成功した契約のリスト
        self.total_trade_quantity = [0, 0]  # 総取引量（外的契約を含む）
        self.my_offer_list = defaultdict(lambda: list())  # 相手ごとの自分のOfferのリスト
        self.opp_offer_list = defaultdict(lambda: list())  # 相手のOfferのリスト
        self.best_opp_util = -float("inf")  # その日の相手のOfferの効用値の最大値
        self.best_agr_util = -float("inf")  # 合意に達した契約の効用値の最大値
        self.successful_times = [0, 0]  # 合意に達した回数

        # その他
        self.my_name = None  # 自分の名前（短縮版）

    def init(self):
        # デバッグ用
        self.parameter_min_max["threshold"] = [0, float("inf")]
        self.parameter_min_max["strong degree"] = [0, float("inf")]

    def step(self):
        # 外的契約の総取引量を保存
        self.total_trade_quantity[Buy] += self.awi.current_exogenous_input_quantity
        self.total_trade_quantity[Sell] += self.awi.current_exogenous_output_quantity

        # その日の取引の分析結果を表示
        # self.print_analyze()
        if self.awi.current_step == self.awi.n_steps - 1:
            self.display_offers_scatter()
            self.display_contracts()

        # パラメタのリセット
        self.secured = 0
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self.best_opp_util = -float("inf")

        ## デバッグ用
        self.parameter_min_max["threshold"][0] = max(
            self.parameter_min_max["threshold"][0], self._threshold
        )
        self.parameter_min_max["strong degree"][0] = max(
            self.parameter_min_max["strong degree"][0], self.strong_degree
        )
        self.parameter_min_max["threshold"][1] = min(
            self.parameter_min_max["threshold"][1], self._threshold
        )
        self.parameter_min_max["strong degree"][1] = min(
            self.parameter_min_max["strong degree"][1], self.strong_degree
        )

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

        # 合意を表示
        # print()
        # print(contract.partners, f"mechanism.state.step:{mechanism.state.step}", contract.agreement)

        # 取引データを記録
        if self._is_selling(mechanism):
            self.success_list[self.shorten_name(contract.partners[0])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
            self.total_trade_quantity[Sell] += contract.agreement["quantity"]
        else:
            self.success_list[self.shorten_name(contract.partners[1])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
            self.total_trade_quantity[Buy] += contract.agreement["quantity"]

        self.success_contracts.append(contract)
        print_log("contract.annotation", contract.annotation)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        if self.my_name is None:
            if self._is_selling(self.get_nmi(negotiator_id)):
                self.my_name = self.shorten_name(
                    self.get_nmi(negotiator_id).annotation["seller"]
                )
            else:
                self.my_name = self.shorten_name(
                    self.get_nmi(negotiator_id).annotation["buyer"]
                )

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_, float("inf")) for _ in self.negotiators.keys()),
            )
        )

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        # 変数
        SECURED_MAGNIFICATION = 1.2

        n = [_ for _ in states.keys()][0]
        state = [_ for _ in states.values()][0]
        opponent_names = self.opponent_names(self.get_nmi(n))

        # print("\n")
        # print(
        #     f"my name:{self.get_nmi(n).negotiator_names[0] if self._is_selling(self.get_nmi(n)) else self.get_nmi(n).negotiator_names[1]}")
        # print(f"step:{state.step}, n_steps:{self.get_nmi(n).n_steps},")
        # f" t:{self.t(state.step, self.get_nmi(n).n_steps)}")
        # print(f"offers:{offers}")
        # print(f"_threshold:{self._threshold}, delta:{self._delta}")
        # print(f"strong degree:{self.strong_degree}, delta:{self.min_max_delta}")
        # print(f"parameter [max, min]:{self.parameter_min_max.items()}")
        # print(f"my_opponents:{self.awi.my_consumers, self.awi.my_suppliers}")
        # print(f"my_opponents full:{self.opponent_names(self.get_nmi(n))}")

        # 良いUnit Priceを記録

        print_log(
            "my name",
            self.get_nmi(n).negotiator_names[0]
            if self._is_selling(self.get_nmi(n))
            else self.get_nmi(n).negotiator_names[1],
        )
        print_log(["step", "n_steps"], [state.step, self.get_nmi(n).n_steps])
        print_log("offers", offers)

        for n in offers.keys():
            self.record_best_price(n, offers[n])

        my_needs = self._needed()
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output = sorted_offers[i]
            secured += offer[QUANTITY]
            if (
                secured
                >= max(my_needs, self.awi.profile.n_lines) * SECURED_MAGNIFICATION
            ):
                break
            chosen[k] = offer
            outputs.append(is_output)

        if my_needs > 0:
            u, producible = self.from_offers(
                tuple(chosen.values()), tuple(outputs), True
            )
        else:
            # my_needsを満たしているときは交渉終了
            responses = dict(
                zip(
                    [str(k) for k in offers.keys()],
                    [
                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for _ in offers.keys()
                    ],
                )
            )
            return responses

        # LearningAgentのproposeでOfferを取得
        keys = opponent_names
        counter_offers = []
        for n in keys:
            counter_offers.append(self.counter_propose(n, state.step, producible))

        # レスポンスを決定する
        responses = dict(
            zip(
                [str(k) for k in keys],
                [SAOResponse(ResponseType.REJECT_OFFER, v) for v in counter_offers],
            )
        )

        # print(f"counter_offers:{counter_offers}")
        # print(f"responses:{responses.items()}")
        # print(f"my_need:{my_needs}, secured:{self.secured}")
        print_log("counter_offers", counter_offers)
        print_log("responses", responses)
        print_log(["my_need", "secured"], [my_needs, self.secured])

        # offerを保存
        for n in keys:
            o = list(responses[n].outcome)
            o.append(state.step)
            if (
                self.my_offer_list[self.shorten_name(n)] == []
                or o[3] != self.my_offer_list[self.shorten_name(n)][-1][3]
            ):
                self.my_offer_list[self.shorten_name(n)].append(o)
            # print(f"my offer list:{self.my_offer_list[self.shorten_name(n)]}")
        for k, v in offers.items():
            o = list(v)
            o.append(state.step)
            # o[TIME] = state.step
            self.opp_offer_list[self.shorten_name(k)].append(o)

        # その日の良い効用値を保存
        self.best_opp_util = max(u, self.best_opp_util)

        for k, v in chosen.items():
            up = v[UNIT_PRICE]
            if self._is_good_price(self.get_nmi(k), state.step, up):
                responses[str(k)] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        # print("\n", chosen.values(), sorted_offers, offers)
        # print(f"\nchosen.values():{chosen.values()}")
        # print(
        #     f"u:{u}, threshold:{threshold}, u >= threshold:{u >= threshold}, "
        #     f"my_need:{my_needs}, producible:{producible}, secured {secured}, "
        #     f"\ndetect_min_utility:{self.detect_min_utility(my_needs, self.get_nmi(n))}, "
        #     f"default_min_utility: {self.ufun.min_utility}")
        # print(f"u:{u}, threshold:{threshold}, u >= threshold:{u >= threshold}, "
        #       f"\nmy_need:{my_needs}, producible:{producible}, secured {secured}, "
        #       f"\nbest_opp_util:{self.best_opp_util}, best_agr_util:{self.best_agr_util}, "
        #       f"\ndetect_max_min_utility:{mx, mn}, std:{std} "
        #       f"detect_std_utility:{self.detect_std_utility(my_needs, self.get_nmi(n))})")
        # print(f"total trade quantity:{self.total_trade_quantity}")
        print_log("total trade quantity", self.total_trade_quantity)
        print_log("responses", responses)

        if self.success_list.keys():
            success_simulation_steps = set(
                itertools.chain.from_iterable(
                    [[_[TIME] for _ in lis] for lis in self.success_list.values()]
                )
            )
            # print(f"agreement　rate:{len(success_simulation_steps) / self.awi.n_steps}")
            print_log(
                "agreement rate", len(success_simulation_steps) / self.awi.n_steps
            )
            success_unit_prices = itertools.chain.from_iterable(
                [[_[UNIT_PRICE] for _ in lis] for lis in self.success_list.values()]
            )
            # print(f"average unit price:{mean(success_unit_prices)}")
            print_log("average unit price", mean(success_unit_prices))

        return responses

    def counter_propose(self, negotiator_id: str, step, producible: int) -> "Outcome":
        offer = self.best_offer(negotiator_id, producible)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(
            self.get_nmi(negotiator_id), max(step, 0)
        )
        return tuple(offer)

    def record_best_price(self, negotiator_id: str, offer: "Outcome") -> "None":
        # 取引におけるBestな値を記録
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)

        # update my current best price to use for limiting concession in other
        # negotiations
        nmi = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)

    def best_offer(self, negotiator_id, producible):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value, producible),
            quantity_issue.min_value,
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product

    def _is_good_price(self, nmi, step, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, step):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

    @staticmethod
    def t(step, n_steps):
        return step / n_steps

    def util_th(self, step, n_steps):
        return ((n_steps - step - 1) / (n_steps - 1)) ** self.e

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = nmi.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx

    def adjust_offer_price(self, counter_offers, nmi):
        # offerする価格を調整する
        adjust_offers = []
        counter_offers = [list(_) for _ in counter_offers]

        if self._is_selling(nmi):
            mn = min(_[UNIT_PRICE] for _ in counter_offers) + self.strong_degree
            mn = min(mn, nmi.issues[UNIT_PRICE].max_value)
            for offer in counter_offers:
                if offer[UNIT_PRICE] < mn:
                    offer[UNIT_PRICE] = mn
                adjust_offers.append(tuple(offer))
        else:
            mx = max(_[UNIT_PRICE] for _ in counter_offers) - self.strong_degree
            mx = max(mx, nmi.issues[UNIT_PRICE].min_value)
            for offer in counter_offers:
                if offer[UNIT_PRICE] > mx:
                    offer[UNIT_PRICE] = mx
                adjust_offers.append(tuple(offer))

        # print(f"counter_offers:{counter_offers}")

        return adjust_offers

    def detect_max_min_utility(self, need, nmi):
        max_offer = (
            min(need, self.awi.profile.n_lines),
            self.awi.current_step,
            nmi.issues[UNIT_PRICE].max_value
            if self._is_selling(nmi)
            else nmi.issues[UNIT_PRICE].min_value,
        )
        min_offer = (
            0,
            self.awi.current_step,
            nmi.issues[UNIT_PRICE].min_value
            if self._is_selling(nmi)
            else nmi.issues[UNIT_PRICE].max_value,
        )

        return (
            self.from_offers(
                (max_offer,), (True,) if self._is_selling(nmi) else (False,)
            ),
            self.from_offers(
                (min_offer,), (True,) if self._is_selling(nmi) else (False,)
            ),
        )

    def detect_std_utility(self, need, nmi):
        # my_needまたはn_linesの量の製品が相手にとってもっともよい値段で売れた時の効用値を下限として設定
        # offer = (max(need, self.awi.profile.n_lines),
        #          self.awi.current_step,
        #          nmi.issues[UNIT_PRICE].min_value)
        offer = [
            min(need, self.awi.profile.n_lines),
            self.awi.current_step,
            self.awi.trading_prices[1],
        ]

        slack = 0.3
        if self._is_selling(nmi):
            offer[UNIT_PRICE] *= 1 - slack
        else:
            offer[UNIT_PRICE] *= 1 + slack

        return self.from_offers(
            (tuple(offer),), (True,) if self._is_selling(nmi) else (False,)
        )

    # その日の交渉の結果を表示
    def print_analyze(self):
        print("\n ~~~~  analyze  ~~~~")
        print("day", self.awi.current_step, " agent name:", self.short_type_name)
        # print("exogenous contract summary", self.awi.exogenous_contract_summary)
        print(
            "exogenous input",
            self.awi.current_exogenous_input_quantity,
            self.awi.current_exogenous_input_price,
        )
        print(
            "exogenous output",
            self.awi.current_exogenous_output_quantity,
            self.awi.current_exogenous_output_price,
        )
        # print("current issues", self.awi.current_input_issues, self.awi.current_output_issues)
        print(
            "current agreement",
            [
                [k, v[-1]]
                for k, v in self.success_list.items()
                if v[-1][TIME] == self.awi.current_step
            ],
        )
        print("current balance", self.awi.current_balance)
        # print("cost of production", self.awi.profile.cost)
        # print("current disposal cost", self.awi.current_disposal_cost)
        # print("current shortfall penalty", self.awi.current_shortfall_penalty)
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)
        print("ufun", self.ufun.max_utility, self.ufun.min_utility)
        # offers = [v[-1] for k, v in self.success_list.items() if v[-1][TIME] == self.awi.current_step]
        # print("current profit", self.ufun.from_offers(tuplw(offers), tuple([True] * len(offers))))
        # print("success contracts", self.success_contracts)
        print(
            "current profit",
            self.ufun.from_contracts(
                [
                    _
                    for _ in self.success_contracts
                    if _.agreement["time"] == self.awi.current_step
                ],
                False,
            ),
        )
        if self.awi.is_first_level:
            q_in = self.awi.current_exogenous_input_quantity
            q_out = sum(
                v[-1][QUANTITY]
                for k, v in self.success_list.items()
                if v[-1][TIME] == self.awi.current_step
            )
            print("breach_level", self.ufun.breach_level(q_in, q_out))
        else:
            q_in = sum(
                v[-1][QUANTITY]
                for k, v in self.success_list.items()
                if v[-1][TIME] == self.awi.current_step
            )
            q_out = self.awi.current_exogenous_output_quantity
            print("breach_level", self.ufun.breach_level(q_in, q_out))
        print("total trade quantity", self.total_trade_quantity)
        print(f"trading prices:{self.awi.trading_prices}")
        # print("my offers", self.my_offer_list)
        if self.success_list.keys():
            success_simulation_steps = set(
                itertools.chain.from_iterable(
                    [[_[TIME] for _ in lis] for lis in self.success_list.values()]
                )
            )
            print(f"agreement　rate:{len(success_simulation_steps) / self.awi.n_steps}")
            success_unit_prices = itertools.chain.from_iterable(
                [[_[UNIT_PRICE] for _ in lis] for lis in self.success_list.values()]
            )
            print(f"average unit price:{mean(success_unit_prices)}")
        # print(f"_best_selling:{self._best_selling}, _best_buying:{self._best_buying}, \n"
        #       f"_best_acc_selling:{self._best_acc_selling}, _best_acc_buying:{self._best_acc_buying}, \n"
        #       f"_best_opp_selling:{self._best_opp_selling.items()}, _best_opp_buying:{self._best_opp_buying.items()}, \n"
        #       f"_best_opp_acc_selling:{self._best_opp_acc_selling.items()}, _best_opp_acc_buying:{self._best_opp_acc_buying.items()}")

    def display_offers_scatter(self):
        STEP = 3
        # colors = ["blue", "orange", "pink", "brown", "red", "grey", "yellow", "green", "black"]
        colors = list(matplotlib.colors.CSS4_COLORS.keys())
        markers = [
            ".",
            "o",
            "^",
            "s",
            "*",
            "'",
            "v",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "s",
        ]
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        for k, v in self.my_offer_list.items():
            # print(self.my_offer_list.keys())
            x = [_[STEP] for _ in v]
            y = [_[UNIT_PRICE] for _ in v]
            ax.scatter(
                x,
                y,
                c="blue",
                marker=markers.pop(),
                alpha=0.5,
                label="my offer to " + k,
            )

        for k, v in self.opp_offer_list.items():
            # print(self.opp_offer_list.keys())
            x = [_[STEP] for _ in v]
            y = [_[UNIT_PRICE] for _ in v]
            ax.scatter(x, y, c=colors.pop(), alpha=0.3, label=k)

        ax.set_title(self.shorten_name(self.short_type_name))
        ax.set_xlabel("negotiation step")
        ax.set_ylabel("unit price")

        ax.set_xlim(0, 20)
        plt.legend(loc="center left", fontsize=10)

        fig.show()

    def display_contracts(self):
        colors = list(matplotlib.colors.CSS4_COLORS.keys())
        markers = [
            ".",
            "o",
            "^",
            "s",
            "*",
            "'",
            "v",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "s",
        ]
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        for k, v in self.success_list.items():
            # print(self.my_offer_list.keys())
            x = [_[TIME] for _ in v]
            y = [_[UNIT_PRICE] for _ in v]
            ax.scatter(
                x,
                y,
                c=colors.pop(),
                marker=markers.pop(),
                label="contract with " + k,
                edgecolors="black",
            )

        ax.set_title("agent" "s offers")
        ax.set_xlabel("simulation step")
        ax.set_ylabel("unit price")
        ax.set_ylim(0, 40)
        plt.legend(loc="lower left", fontsize=14)

        fig.show()

    @staticmethod
    def shorten_name(name: str):
        return name.split("-")[0]

    def opponent_names(self, nmi):
        if self._is_selling(nmi):
            consumers = self.awi.my_consumers
            return list(self.negotiators.keys())[-len(consumers) :]
        else:
            suppliers = self.awi.my_suppliers
            return list(self.negotiators.keys())[-len(suppliers) :]

    # Offerから効用値を計算
    def from_offers(
        self, offers: Iterable[Tuple], outputs: Iterable[bool], return_producible=False
    ) -> Union[float, Tuple[float, int]]:
        """
        Calculates the utility value given a list of offers and whether each
        offer is for output or not (= input).

        Args:
            offers: An iterable (e.g. list) of tuples each with three values:
                    (quantity, time, unit price) IN THAT ORDER. Time is ignored
                    and can be set to any value.
            outputs: An iterable of the same length as offers of booleans
                     specifying for each offer whether it is an offer for buying
                     the agent's output product.
            return_producible: If true, the producible quantity will be returned
        Remarks:
            - This method takes into account the exogenous contract information
              passed when constructing the ufun.
        """

        def order(x):
            """A helper function to order contracts in the following fashion:
            1. input contracts are ordered from cheapest to most expensive.
            2. output contracts are ordered from highest price to cheapest.
            3. The relative order of input and output contracts is indeterminate.
            """
            offer, is_output, is_exogenous = x
            # if is_exogenous and self.force_exogenous:
            #     return float("-inf")
            return -offer[UNIT_PRICE] if is_output else offer[UNIT_PRICE]

        # 変数の設定
        ex_qin = self.awi.current_exogenous_input_quantity
        ex_qout = self.awi.current_exogenous_output_quantity
        ex_pin = self.awi.current_exogenous_input_price
        ex_pout = self.awi.current_exogenous_output_price
        current_balance = self.awi.current_balance
        production_cost = self.awi.profile.cost
        n_lines = self.awi.profile.n_lines
        output_penalty_scale = self.ufun.output_penalty_scale
        input_penalty_scale = self.ufun.input_penalty_scale
        shortfall_penalty = self.ufun.shortfall_penalty
        disposal_cost = self.ufun.disposal_cost

        # copy inputs because we are going to modify them.
        offers, outputs = deepcopy(list(offers)), deepcopy(list(outputs))
        # print(f"offers {offers}, outputs {outputs}")

        # indicate that all inputs are not exogenous and that we are adding two
        # exogenous contracts after them.
        exogenous = [False] * len(offers) + [True, True]
        # add exogenous contracts as offers one for input and another for output
        offers += [
            (ex_qin, 0, ex_pin / ex_qin if ex_qin else 0),
            (ex_qout, 0, ex_pout / ex_qout if ex_qout else 0),
        ]
        outputs += [False, True]
        # initialize some variables
        qin, qout, pin, pout = 0, 0, 0, 0
        qin_bar, going_bankrupt = 0, current_balance < 0
        pout_bar = 0
        # we are going to collect output contracts in output_offers
        output_offers = []
        # sort contracts in the optimal order of execution: from cheapest when
        # buying and from the most expensive when selling. See `order` above.
        sorted_offers = list(sorted(zip(offers, outputs, exogenous), key=order))
        # print(f"sorted_offers {sorted_offers}")

        # we calculate the total quantity we are are required to pay for `qin` and
        # the associated amount of money we are going to pay `pin`. Moreover,
        # we calculate the total quantity we can actually buy given our limited
        # money balance (`qin_bar`).
        for offer, is_output, is_exogenous in sorted_offers:
            offer = self.ufun.outcome_as_tuple(offer)
            if is_output:
                output_offers.append((offer, is_exogenous))
                continue
            topay_this_time = offer[UNIT_PRICE] * offer[QUANTITY]
            if not going_bankrupt and (
                pin + topay_this_time + offer[QUANTITY] * production_cost
                > current_balance
            ):
                unit_total_cost = offer[UNIT_PRICE] + production_cost
                can_buy = int((current_balance - pin) // unit_total_cost)
                qin_bar = qin + can_buy
                going_bankrupt = True
            pin += topay_this_time
            qin += offer[QUANTITY]

        if not going_bankrupt:
            qin_bar = qin

        # print(f"output_offers {output_offers}\nqin {qin}, qin_ber {qin_bar}, pin {pin}, going_bankrupt {going_bankrupt}")

        # calculate the maximum amount we can produce given our limited production
        # capacity and the input we CAN BUY
        n_lines = n_lines
        producible = min(qin_bar, n_lines)
        # print(f"n_lines {n_lines}, producible(min(qin_bar, n_lines)) {producible}")

        # No need to this test now because we test for the ability to produce with
        # the ability to buy items. The factory buys cheaper items and produces them
        # before attempting more expensive ones. This may or may not be optimal but
        # who cars. It is consistent that it is all that matters.
        # # if we do not have enough money to pay for production in full, we limit
        # # the producible quantity to what we can actually produce
        # if (
        #     self.production_cost
        #     and producible * self.production_cost > current_balance
        # ):
        #     producible = int(current_balance // self.production_cost)

        # find the total sale quantity (qout) and money (pout). Moreover find
        # the actual amount of money we will receive
        done_selling = False
        for offer, is_exogenous in output_offers:
            if not done_selling:
                if qout + offer[QUANTITY] >= producible:
                    assert producible >= qout, f"producible {producible}, qout {qout}"
                    can_sell = producible - qout
                    done_selling = True
                else:
                    can_sell = offer[QUANTITY]
                pout_bar += can_sell * offer[UNIT_PRICE]
            pout += offer[UNIT_PRICE] * offer[QUANTITY]
            qout += offer[QUANTITY]

        # print(f"qout {qout}, pout {pout}")

        # should never produce more than we signed to sell
        producible = min(producible, qout)
        # print(f"producible(min(producible, qout)) {producible}")

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        producible = min(qin, n_lines, producible)
        # print(f"producible(min(qin, n_lines, producible)) {producible}")

        # the scale with which to multiply disposal_cost and shortfall_penalty
        # if no scale is given then the unit price will be used.
        output_penalty = output_penalty_scale
        if output_penalty is None:
            output_penalty = pout / qout if qout else 0
        output_penalty *= shortfall_penalty * max(0, qout - producible)
        input_penalty = input_penalty_scale
        if input_penalty is None:
            input_penalty = pin / qin if qin else 0
        input_penalty *= disposal_cost * max(0, qin - producible)

        # print(f"production_cost {production_cost}")
        # print(f"output_penalty {output_penalty}, output_penalty_scale {output_penalty_scale}, shortfall_penalty {shortfall_penalty}")
        # print(f"input_penalty {input_penalty}, input_penalty_scale {input_penalty_scale}, disposal_cost {disposal_cost}")

        # call a helper method giving it the total quantity and money in and out.
        u = self.ufun.from_aggregates(
            qin, qout, producible, pin, pout_bar, input_penalty, output_penalty
        )
        # print(f"u {u}")

        if return_producible:
            # the real producible quantity is the minimum of what we can produce
            # given supplies and production capacity and what we can sell.
            return u, producible
        return u


class LearningSyncAgent(OneShotSyncAgent, LearningAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def init(self):
        LearningAgent.init(self)

    def step(self):
        LearningAgent.step(self)

    def on_negotiation_success(self, contract, mechanism):
        LearningAgent.on_negotiation_success(self, contract, mechanism)

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        SECURED_MAGNIFICATION = 1.2

        my_needs = self._needed()
        if my_needs <= 0:
            # my_needsを満たしているときは交渉終了
            responses = dict(
                zip(
                    [str(k) for k in offers.keys()],
                    [
                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for _ in offers.keys()
                    ],
                )
            )
            return responses

        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v) for k, v in offers.items()
        }
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling, offers.keys()),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output, name = sorted_offers[i]
            response = LearningAgent.respond(self, name, states[name], offer)

            if response == ResponseType.ACCEPT_OFFER:
                if secured < my_needs:
                    secured += offer[QUANTITY]
                    responses[name] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        for name, offer in offers.items():
            counter_offer = LearningAgent.propose(self, name, states[name])
            list(counter_offer)[QUANTITY] = min(
                self.awi.profile.n_lines, counter_offer[QUANTITY]
            )
            if responses[name] != SAOResponse(ResponseType.ACCEPT_OFFER, None):
                responses[name] = SAOResponse(
                    ResponseType.REJECT_OFFER, tuple(counter_offer)
                )

        return responses


class LearningSyncAgentT(LearningSyncAgent, ABC):
    def __init__(
        self, acc_price_slack=float("inf"), opp_acc_price_slack=0.2, *args, **kwargs
    ):
        super().__init__(
            *args,
            acc_price_slack=acc_price_slack,
            opp_acc_price_slack=opp_acc_price_slack,
            **kwargs,
        )

        self.success_list = defaultdict(lambda: list())  # 交渉成功した際の取引データ
        self.success_contracts = []  # 交渉成功した契約のリスト
        self.opp_offer_list = defaultdict(lambda: list())  # 相手のOfferのリスト
        self.best_agr_util = -float("inf")  # 合意に達した契約の効用値の最大値
        self.best_agr_util_slack = 0.15
        self.trading_price_util_slack = 0.2

        # その他
        self.nego_info = {}

    def init(self):
        super().init()
        # 効用値の最大値と最小値を計算
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

        self.nego_info["first"] = False

    def first_proposals(self):
        self.nego_info["first"] = True
        return super().first_proposals()

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        if self._is_selling(self.get_nmi(negotiator_id)):
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["seller"]
            )
        else:
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["buyer"]
            )

        if self._is_selling(self.get_nmi(negotiator_id)):
            self._opp_price_slack = 0.1
            self._step_price_slack = 0.1
        else:
            self._opp_price_slack = 0.1
            self._step_price_slack = 0.1

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        # 合意案のutilを計算し，best_utilを更新
        util = -float("inf")
        if self.success_list:
            agreements = [
                _ for _ in self.success_list if _[TIME] == self.awi.current_step
            ]
            is_selling = [
                self._is_selling(self.get_nmi(negotiator_id)) for _ in agreements
            ]
            if agreements:
                util = self.ufun.from_offers(tuple(agreements), tuple(is_selling))

        self.best_agr_util = max(self.best_agr_util, util)

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)

        # 取引データを記録
        if self._is_selling(mechanism):
            self.success_list[shorten_name(contract.partners[0])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
        else:
            self.success_list[shorten_name(contract.partners[1])].append(
                [
                    contract.agreement["quantity"],
                    self.awi.current_step,
                    contract.agreement["unit_price"],
                ]
            )
        self.success_contracts.append(contract)

    def counter_all(self, offers, states):
        print_log("", "")
        # 交渉ステップを記録
        self.nego_info["negotiation_step"] = list(states.values())[0].step

        self.record_information(offers)

        # responsesを生成
        responses = super().counter_all(offers, states)

        # my_needを満たしていたら交渉終了
        if responses[list(offers.keys())[0]] == SAOResponse(
            ResponseType.END_NEGOTIATION, None
        ):
            return responses

        # 良いOfferの組み合わせを選ぶ
        best_util, good_offers = self.select_good_offers(offers)

        # 受け入れるかどうかの判定
        threshold = self.util_th(states)
        if best_util > threshold:
            for k, v in offers.items():
                if k in good_offers.keys():
                    # 受け入れる相手にAccept
                    responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                else:
                    # 受け入れない相手は交渉終了
                    responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            self.best_agr_util = max(self.best_agr_util, best_util)
        else:
            # 受け入れない相手にcounter_offerを設定
            for name in offers.keys():
                if responses[name] == SAOResponse(ResponseType.ACCEPT_OFFER, None):
                    counter_offer = LearningAgent.propose(self, name, states[name])
                    list(counter_offer)[QUANTITY] = min(
                        self.awi.profile.n_lines, counter_offer[QUANTITY]
                    )
                    responses[name] = SAOResponse(
                        ResponseType.REJECT_OFFER, tuple(counter_offer)
                    )

        # デバッグ用
        print_log("step", self.nego_info["negotiation_step"])
        print_log("name", self.nego_info["my_name"])
        print_log("first proposal", self.nego_info["first"])
        print_log(["best_util", "threshold"], [best_util, threshold])
        print_log("responses", responses)
        # if self.awi.current_step > 0:
        #     print_log("financial report", self.awi.reports_at_step(self.awi.current_step))

        return responses

    def select_good_offers(self, offers: dict):
        """
        効用値の良いOfferの組み合わせを探す
        :param offers:
        :return: best util and good offers
        """
        names = offers.keys()
        best_util = -float("inf")
        best_opponents = []
        for n in range(1, max(len(names) + 1, 3 + 1)):
            for comb in itertools.combinations(names, n):
                is_selling = (self._is_selling(self.get_nmi(_)) for _ in comb)
                select_offers = [offers[_] for _ in comb]
                util = self.ufun.from_offers(tuple(select_offers), tuple(is_selling))

                if util > best_util:
                    best_util = util
                    best_opponents = comb

        return best_util, dict(zip(best_opponents, [offers[_] for _ in best_opponents]))

    def util_th(self, states):
        """効用値の閾値を決定"""
        nmi = self.get_nmi(list(states.keys())[0])
        is_selling = self._is_selling(nmi)
        my_need = max(min(self._needed(), self.awi.profile.n_lines), 0)
        edge_offers = [
            (my_need, 0, nmi.issues[UNIT_PRICE].max_value),
            (my_need, 0, nmi.issues[UNIT_PRICE].min_value),
        ]

        # max_utilityを決定
        if self._is_selling(nmi):
            max_utility = self.ufun.from_offers((edge_offers[0],), (is_selling,))
        else:
            max_utility = self.ufun.from_offers((edge_offers[1],), (is_selling,))

        # min_utilityを決定
        do_nothing_util = self.ufun.from_offers(tuple(), (is_selling,))
        tp_util = self.ufun.from_offers(
            ((my_need, 0, self.awi.trading_prices[1]),), (is_selling,)
        )
        mn = do_nothing_util
        min_utility = min(
            max_utility * (1 - self._range_slack),
            max(
                (u - mn) * (1 - slack) + mn
                for u, slack in (
                    # (self.best_agr_util, self.best_agr_util_slack),
                    (do_nothing_util, 0),
                    # (tp_util, self.trading_price_util_slack)
                )
            ),
        )

        # デバッグ用
        print_log(
            ["before : best_agr_util", "do_nothing_util", "tp_util"],
            [self.best_agr_util, do_nothing_util, tp_util],
        )
        print_log(
            ["after : best_agr_util", "do_nothing_util", "tp_util"],
            [
                (u - mn) * (1 - slack) + mn
                for u, slack in (
                    (self.best_agr_util, self.best_agr_util_slack),
                    (do_nothing_util, 0),
                    (tp_util, self.trading_price_util_slack),
                )
            ],
        )

        # 譲歩関数により閾値を決定
        step = self.nego_info["negotiation_step"]
        n_steps = nmi.n_steps

        # 売り手と買い手で閾値を変更
        step = min(0, step - 1) if not self._is_selling(nmi) else step

        return min_utility + (max_utility - min_utility) * self._th(step, n_steps)

    def _price_range(self, nmi):
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        concession_degree = 1 - self.strong_degree(nmi)

        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack * concession_degree)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            # (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            # (self.opp_next_price(partner), self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = nmi.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack * concession_degree)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            # (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (self.opp_next_price(partner), self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )

        # デバッグ用
        # print_log(["self factor", "env factor"], [self.self_factor(nmi), self.environment_factor(nmi)])
        # print_log("concession_degree", concession_degree)
        return mn, mx

    # def _find_good_price(self, nmi, state):
    #     mn, mx = self._price_range(nmi)
    #     # step = min(state.step + 1, nmi.n_steps - 1) if self._is_selling(nmi) else state.step
    #     step = state.step
    #     th = self._th(step, nmi.n_steps)
    #     # offer a price that is around th of your best possible price
    #     if self._is_selling(nmi):
    #         return mn + th * (mx - mn)
    #     else:
    #         return mx - th * (mx - mn)

    def strong_degree(self, nmi):
        w_self = 0.5
        w_env = 1 - w_self
        return w_self * self.self_factor(nmi) + w_env * self.environment_factor(nmi)

    def self_factor(self, nmi):
        """自身の交渉の進捗を評価"""
        prev_agreement = 0  # 前日合意できたか
        agreement_ratio = 0  # 相手との交渉成功割合
        good_agreement = 0  # 良い値段で合意できたか
        w_prev = 3
        w_ratio = 4 * t(self.nego_info["negotiation_step"], nmi.n_steps)
        # w_ratio = 1
        w_good = 3
        w_prev, w_ratio, w_good = param_normalization([w_prev, w_ratio, w_good])

        # 前日合意できたか
        success_agreements = [
            [_.agreement, _.mechanism_state["current_proposer"]]
            for _ in self.success_contracts
        ]
        if success_agreements:
            accept_simulation_steps = {
                lis[0]["time"]
                for lis in success_agreements
                if lis[1] == self.nego_info["my_name"]
            }
            offer_simulation_steps = {
                lis[0]["time"]
                for lis in success_agreements
                if lis[1] != self.nego_info["my_name"]
            }
            if self.awi.current_step - 1 in accept_simulation_steps:
                # 自分の合意による交渉成功時
                prev_agreement = 1
            elif self.awi.current_step - 1 in offer_simulation_steps:
                # 自分のOfferによる交渉成功時
                prev_agreement = 1
            else:
                # 交渉失敗時
                prev_agreement = 0
        else:
            prev_agreement = 0.5

        # 相手との交渉成功割合
        if self._is_selling(nmi):
            opponent_name = nmi.annotation["buyer"]
            success_agreements = [
                _ for _ in self.success_contracts if _.partners[0] == opponent_name
            ]
        else:
            opponent_name = nmi.annotation["seller"]
            success_agreements = [
                _ for _ in self.success_contracts if _.partners[1] == opponent_name
            ]
        agreement_ratio = len(success_agreements) / (self.awi.current_step + 1)

        # 良い値段で合意できているか
        if self.success_contracts:
            tp = self.awi.trading_prices[1]
            prev_up = self.success_contracts[-1].agreement["unit_price"]
            if self._is_selling(nmi):
                good_agreement = 0.5 - (prev_up - tp) / tp
            else:
                good_agreement = 0.5 + (prev_up - tp) / tp
        else:
            good_agreement = 0.5

        # デバッグ用
        # print_log("params", param_normalization([w_prev, w_ratio, w_good]))

        # 重み付けして足す
        return (
            w_prev * prev_agreement
            + w_ratio * agreement_ratio
            + w_good * good_agreement
        )

    def environment_factor(self, nmi):
        """マーケットの状況を評価"""
        if self._is_selling(nmi):
            n_sellers = len(self.awi.all_suppliers[1])
            n_buyers = len(self.awi.my_consumers)
            return min(n_buyers / n_sellers / 2, 1)
        else:
            n_sellers = len(self.awi.my_suppliers)
            n_buyers = len(self.awi.all_consumers[1])
            return min(n_sellers / n_buyers / 2, 1)

    def opp_next_price(self, name):
        delta = 1
        offers = [
            _
            for _ in self.opp_offer_list[shorten_name(name)]
            if _[TIME] == self.awi.current_step
        ]
        nmi = self.get_nmi(name)
        if len(offers) == 0:
            if self._is_selling(nmi):
                return nmi.issues[UNIT_PRICE].min_value
            else:
                return nmi.issues[UNIT_PRICE].max_value
        elif len(offers) > delta:
            prev = offers[-delta - 1][UNIT_PRICE]
        else:
            if self._is_selling(nmi):
                prev = nmi.issues[UNIT_PRICE].min_value
            else:
                prev = nmi.issues[UNIT_PRICE].max_value

        now = offers[-1][UNIT_PRICE]
        return now + (now - prev) / delta

    def record_information(self, offers):
        """offer や utilを保存"""
        # offerを保存
        for k, v in offers.items():
            o = list(v)
            o.append(self.nego_info["negotiation_step"])
            self.opp_offer_list[shorten_name(k)].append(o)


def print_log(names, values, on=False):
    if on:
        if type(names) == str:
            print(f"{names}:{values}")
        if type(names) == list:
            for name, value in dict(zip(names, values)).items():
                print(f"{name}:{value}", end=" ")
            print()
