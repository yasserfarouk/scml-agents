import itertools
from abc import ABC
from collections import defaultdict
from typing import Any, Optional, Union

from negmas import (
    Breach,
    Contract,
    LinearUtilityFunction,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
    RenegotiationRequest,
    ResponseType,
    SAONegotiator,
    SAOResponse,
    SAOState,
    UtilityFunction,
)
from negmas.outcomes import Outcome
from scml import (
    AWI,
    ExecutionRatePredictionStrategy,
    FixedERPStrategy,
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    ProductionStrategy,
    SCML2020Agent,
    SupplyDrivenProductionStrategy,
)
from scml.oneshot import *

from .nego_utils import *
from .tutorial_agents import *

Buy = 0
Sell = 1
Offer = 0
Accept = 1
INF = 1000

__all__ = [
    "GentleS",
    "LearningSyncAgent",
    # "Daruma",
]


class GentleS(LearningAgent, ABC):
    better_agent = BetterAgent(concession_exponent=0.2)

    def __init__(self, *args, round_th=19, q_bias=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._concession_threshold = 3  # 譲歩の変化率の閾値
        self._round_th = round_th
        self._q_bias = q_bias
        self._trading_price_slack = 0.0

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
        self.failure_opp_list = []

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        is_selling = self._is_selling(self.get_nmi(negotiator_id))
        ami = self.get_nmi(negotiator_id)
        # my_nameを記録
        if is_selling:
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["seller"]
            )
        else:
            self.nego_info["my_name"] = shorten_name(
                self.get_nmi(negotiator_id).annotation["buyer"]
            )

        # 初期のprice rangeを記録
        if self.awi.current_step == 0:
            self.nego_info["init_up_issue"] = ami.issues[UNIT_PRICE]

        # デバッグ用
        # print_log(["max price", "min price"], [ami.issues[UNIT_PRICE].max_value, ami.issues[UNIT_PRICE].min_value])

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

        offer = super().propose(negotiator_id, state)  # 情報の記録

        if offer is None:
            return None

        offer = list(offer)
        offer[QUANTITY] = self._find_good_quantity(
            self.get_nmi(negotiator_id), state, offer[QUANTITY]
        )

        self._record_information({shorten_name(negotiator_id): offer}, True)  # offerの保存

        # デバッグ用
        # if self.nego_info["negotiation_step"] == 19:
        #     print_log(["step", "proposer name"], [self.awi.current_step, self.nego_info["my_name"]])
        #     print_log("offer", offer)

        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        self.nego_info["negotiation_step"] = state.step  # 交渉ステップを記録
        self._record_information(
            {shorten_name(negotiator_id): offer}, False
        )  # offerの保存

        ami = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)

        response = super().respond(negotiator_id, state)

        # デバッグ用
        # if self.nego_info["negotiation_step"] == 19:
        #     print_log(["step", "responder name", "to"], [self.awi.current_step, self.nego_info["my_name"], negotiator_id])

        return response

    def _find_good_price(self, ami, state):
        """Finds a good-enough price conceding linearly over time"""
        is_selling = self._is_selling(ami)
        # offer a price that is around th of your best possible price

        # パラメタの設定
        name = ami.annotation["buyer"] if is_selling else ami.annotation["seller"]
        success_agreements = opponent_agreements(
            ami, is_selling, self.success_contracts
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
        time = t(self.awi.current_step, self.awi.n_steps)
        up = ami.issues[UNIT_PRICE]
        worst_contract_price = worst_opp_acc_price(ami, is_selling, success_agreements)

        if self.nego_info["negotiation_step"] < self._round_th:
            # 前半
            if is_selling:
                return up.max_value
            else:
                return up.min_value
        else:
            # 後半
            if offer_agreements:
                # 自分のオファーによる合意がある時
                return worst_contract_price
            elif accept_agreements:
                # 相手のオファーによる合意がある時
                if price_comparison(
                    is_selling, worst_contract_price, self.awi.trading_prices[1]
                ):
                    # 価格が良い時
                    return worst_contract_price
                else:
                    # 価格が悪い時
                    return worst_contract_price * (1.0 + 0.1 * TF_sign(is_selling))
            else:
                # 合意がない時
                # 基準値の決定
                std = self.awi.trading_prices[1]

                # 価格を決定（滑らかに譲歩）
                strong_range = {"max": 0.1, "min": -0.3}
                rng = strong_range["max"] - strong_range["min"]
                th = [0.0, 0.3]
                if time < th[0]:
                    strong_degree = strong_range["max"]
                elif th[0] <= time <= th[1]:
                    strong_degree = strong_range["max"] - rng * min(
                        time - th[0] / th[1] - th[0], 1
                    )
                else:
                    strong_degree = strong_range["max"] - rng

                price = std * (1 + TF_sign(is_selling) * strong_degree)

                return price

    def _find_good_quantity(self, ami, state, quantity: int):
        """これまでの相手のオファーを確認し，相手のオファー量の方が小さければそちらに合わせる"""
        # 規定量（10個）を超えていないか確認
        result = min(self.awi.profile.n_lines, quantity)

        # 直近の相手のオファー量を確認する
        is_selling = self._is_selling(ami)
        opp_name = opp_name_from_ami(is_selling, ami)

        if self.opp_offer_list[opp_name]:
            opp_offer_quantity = self.opp_offer_list[opp_name][-1][QUANTITY]
            result = min(opp_offer_quantity, result)

        return result

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        is_selling = self._is_selling(ami)

        # 誰とも合意しない場合と比べて良いかどうか判定
        if is_selling and not self.ufun.ok_to_sell_at(price):
            return False
        elif not is_selling and not self.ufun.ok_to_buy_at(price):
            return False

        # 相手の譲歩率に応じて判断
        name = ami.annotation["buyer"] if is_selling else ami.annotation["seller"]
        success_agreements = opponent_agreements(
            ami, is_selling, self.success_contracts
        )
        if self.nego_info["negotiation_step"] < self._round_th:
            # 前半
            # 譲歩率の変化
            self._opp_price_slack = 0.0
            self._opp_acc_price_slack = 0.0
            self._trading_price_slack = 0.0
        else:
            # 後半
            if self._opp_concession_rate_change(name):
                if success_agreements:
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = 0.2
                    self._trading_price_slack = INF
                else:
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = INF
                    self._trading_price_slack = 0.0

        # a good price is one better than the threshold
        mn, mx = self._price_range(ami)
        if self._is_selling(ami):
            return price >= mn
        else:
            return price <= mx

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        is_selling = self._is_selling(ami)
        name = ami.annotation["buyer"] if is_selling else ami.annotation["seller"]

        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
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
                            (self.awi.catalog_prices[1], self._trading_price_slack),
                        )
                    ]
                ),
            )
        else:
            partner = ami.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
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
                            (self.awi.catalog_prices[1], self._trading_price_slack),
                        )
                    ]
                ),
            )
        return mn, mx

    def _opp_concession_rate_change(self, name: str):
        """相手の譲歩の変化率を計算"""
        ami = self.get_nmi(name)
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
            return now / prev > self._concession_threshold
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


class LearningSyncAgent(OneShotSyncAgent, GentleS, ABC):
    def __init__(
        self, best_agr_util_slack=0.15, trading_price_util_slack=0.2, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.best_agr_util = -float("inf")  # 合意に達した契約の効用値の最大値
        self.best_agr_util_slack = best_agr_util_slack
        self.trading_price_util_slack = trading_price_util_slack

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
        GentleS.init(self)
        # 効用値の最大値と最小値を計算
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

    def step(self):
        GentleS.step(self)

    def on_negotiation_success(self, contract, mechanism):
        GentleS.on_negotiation_success(self, contract, mechanism)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        GentleS.on_negotiation_start(self, negotiator_id, state)

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
                util = self.ufun.from_offers(agreements, is_selling)

        self.best_agr_util = max(self.best_agr_util, util)

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        SECURED_MAGNIFICATION = 1.2

        # print_log("step", [_.step for _ in states.values()])
        # print_log("offers", offers.keys())
        # print_log("active_negotiators", self.active_negotiators.keys())

        # counter_all用にnegotiation_stepを設定
        self.nego_info["counter_all_step"] = min([_.step for _ in states.values()])

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
            response = GentleS.respond(self, name, states[name])

            if response == ResponseType.ACCEPT_OFFER:
                if secured < my_needs:
                    secured += offer[QUANTITY]
                    responses[name] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        for name, offer in offers.items():
            counter_offer = GentleS.propose(self, name, states[name])
            list(counter_offer)[QUANTITY] = min(
                self.awi.profile.n_lines, counter_offer[QUANTITY]
            )
            if responses[name] != SAOResponse(ResponseType.ACCEPT_OFFER, None):
                responses[name] = SAOResponse(
                    ResponseType.REJECT_OFFER, tuple(counter_offer)
                )

        # # 良いOfferの組み合わせを選ぶ
        # best_util, good_offers = self.select_good_offers(offers)
        #
        # # 受け入れるかどうかの判定
        # threshold = self.util_th(states)
        # if best_util > threshold:
        #     for k, v in offers.items():
        #         if k in good_offers.keys():
        #             # 受け入れる相手にAccept
        #             responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        #         else:
        #             # 受け入れない相手は交渉終了
        #             responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        #     self.best_agr_util = max(self.best_agr_util, best_util)
        #     # print_log("response", responses)
        # else:
        #     # 受け入れない相手にcounter_offerを設定
        #     for name in offers.keys():
        #         if responses[name] == SAOResponse(ResponseType.ACCEPT_OFFER, None):
        #             counter_offer = Welf.propose(self, name, states[name])
        #             responses[name] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)

        return responses

    def select_good_offers(self, offers: dict):
        """
        効用値の良いOfferの組み合わせを探す
        :param offers:
        :return: best util and good offers
        """
        # すでにある合意を取得
        contracts = todays_contract(self.awi.current_step, self.success_contracts)
        is_selling = self._is_selling(self.get_nmi(list(offers.keys())[0]))
        agreement_offers = dict(
            zip(
                [opp_name_from_contract(is_selling, _) for _ in contracts],
                [tuple(list(_.agreement.values())) for _ in contracts],
            )
        )

        # 効用値の良い組み合わせを探す
        names = offers.keys()
        best_opponents = []
        is_selling = [
            self._is_selling(self.get_nmi(_)) for _ in agreement_offers.keys()
        ]
        if agreement_offers:
            best_util = self.ufun.from_offers(agreement_offers.values(), is_selling)
        else:
            best_util = -float("inf")
        # if agreement_offers:
        #     print_log("agreement_offers", agreement_offers)
        for n in range(1, max(len(names) + 1, 3 + 1)):
            for comb in itertools.combinations(names, n):
                is_sellings = [self._is_selling(self.get_nmi(_)) for _ in comb]
                select_offers = [offers[_] for _ in comb]
                util = self.ufun.from_offers(
                    select_offers + list(agreement_offers.values()),
                    is_sellings + is_selling,
                )

                if util > best_util:
                    best_util = util
                    best_opponents = comb

        return best_util, dict(zip(best_opponents, [offers[_] for _ in best_opponents]))

    def util_th(self, states):
        """効用値の閾値を決定"""
        ami = self.get_nmi(list(states.keys())[0])
        is_selling = self._is_selling(ami)
        my_need = max(min(self._needed(), self.awi.profile.n_lines), 0)
        edge_offers = [
            (my_need, 0, ami.issues[UNIT_PRICE].max_value),
            (my_need, 0, ami.issues[UNIT_PRICE].min_value),
        ]

        # max_utilityを決定
        if self._is_selling(ami):
            max_utility = self.ufun.from_offers([edge_offers[0]], [is_selling])
        else:
            max_utility = self.ufun.from_offers([edge_offers[1]], [is_selling])

        # min_utilityを決定
        do_nothing_util = self.ufun.from_offers([], [])
        tp_util = self.ufun.from_offers(
            [(my_need, 0, self.awi.trading_prices[1])], [is_selling]
        )
        mn = do_nothing_util
        min_utility = min(
            max_utility * (1 - self._range_slack),
            max(
                [
                    (u - mn) * (1 - slack) + mn
                    for u, slack in (
                        # (self.best_agr_util, self.best_agr_util_slack),
                        (do_nothing_util, 0),
                        # (tp_util, self.trading_price_util_slack)
                    )
                ]
            ),
        )

        # 譲歩関数により閾値を決定
        step = self.nego_info["counter_all_step"]
        n_steps = ami.n_steps

        # デバッグ用
        # print_log(["before : best_agr_util", "do_nothing_util", "tp_util"],
        #           [self.best_agr_util, do_nothing_util, tp_util])
        # print_log(["after : best_agr_util", "do_nothing_util", "tp_util"],
        #           [(u - mn) * (1 - slack) + mn
        #            for u, slack in (
        #                (self.best_agr_util, self.best_agr_util_slack),
        #                (do_nothing_util, 0),
        #                (tp_util, self.trading_price_util_slack)
        #            )])
        # print_log("util", min_utility + (max_utility - min_utility) * self._th(step, n_steps))

        return min_utility + (max_utility - min_utility) * self._th(step, n_steps)


class MyNegotiator(SAONegotiator):
    def __init__(
        self,
        awi: AWI,
        best_opp_acc_selling: dict,
        best_opp_acc_buying: dict,
        acceptable_unit_price: int,
        target_quantity: int,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        round_th=19,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Negotiator
        self.awi = awi

        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self.round_th = round_th

        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = best_opp_acc_selling
        self._best_opp_acc_buying = best_opp_acc_buying

        self.opp_offer_list = defaultdict(lambda: list())  # 相手のOfferのリスト
        self.my_offer_list = defaultdict(lambda: list())  # 自分のOfferのリスト
        self._is_selling = True

        self.acceptable_unit_price = acceptable_unit_price
        self.target_quantity = target_quantity

    def on_negotiation_start(self, state: MechanismState) -> None:
        self.opp_name = opp_name_from_ami(self._is_selling, self.ami)
        self._is_selling = self.ami.annotation["seller"] == self.id

    def propose(self, state: SAOState) -> Optional[Outcome]:
        offer = super(MyNegotiator, self).propose(state)

        self._record_information({self.opp_name: offer}, True)  # offerの保存

        offer = list(offer)
        offer[QUANTITY] = self._find_good_quantity()
        offer[TIME] = min(self.awi.current_step + 3, self.awi.n_steps)
        offer[UNIT_PRICE] = self._find_good_price()

        return tuple(offer)

    def respond(self, state: SAOState) -> ResponseType:
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        self._record_information({self.opp_name: offer}, False)  # offerの保存

        # unit price の保存
        up = offer[UNIT_PRICE]
        if self._is_selling:
            self._best_opp_selling[self.opp_name] = max(
                up, self._best_opp_selling[self.opp_name]
            )
        else:
            self._best_opp_buying[self.opp_name] = min(
                up, self._best_opp_buying[self.opp_name]
            )

        respond = super(MyNegotiator, self).respond(state)

        if respond == ResponseType.REJECT_OFFER:
            return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def _is_good_price(self, price: int):
        """良い値段かどうか判定"""
        acceptable_up = self.acceptable_unit_price
        return price_comparison(self._is_selling, price, acceptable_up)

    def _find_good_price(self):
        """オファーする値段を決定"""
        step = self.ami.state.step
        up_range = self.ami.issues[UNIT_PRICE]
        price = up_range.max_value if self._is_selling else up_range.min_value
        std = (up_range.max_value + up_range.min_value) / 2.0

        if step >= self.round_th:
            if self._is_selling:
                price = std
            else:
                price = std

        return price

    def _is_good_quantity(self, quantity: int):
        """良い製品量かどうか判定"""
        return True

    def _find_good_quantity(self):
        """オファーする量を決定"""
        # 直近の相手のオファー量を確認する
        quantity = self.target_quantity

        if self.opp_offer_list[self.opp_name]:
            opp_offer_quantity = self.opp_offer_list[self.opp_name][-1][QUANTITY]
            quantity = min(opp_offer_quantity, quantity)

        return quantity

    def _record_information(self, offers: dict, mine: bool):
        """offer や utilを保存"""
        # offerを保存
        if mine:
            for k, v in offers.items():
                o = list(v)
                o.append(self.ami.state.step)
                self.my_offer_list[shorten_name(k)].append(o)
        else:
            for k, v in offers.items():
                o = list(v)
                o.append(self.ami.state.step)
                self.opp_offer_list[shorten_name(k)].append(o)


class MyNegotiationManager(IndependentNegotiationsManager):
    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = MyNegotiator,
        agreeable_price_slack=0.2,
        time_th=0.8,
        **kwargs,
    ):
        super(MyNegotiationManager, self).__init__(
            *args, negotiator_type=negotiator_type, **kwargs
        )
        # 取引情報
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self.success_list = defaultdict(lambda: list())  # 交渉成功した際の取引データ
        self.success_contracts = []  # 交渉成功した契約のリスト

        # Negotiation Manager
        self.agreeable_price_slack = agreeable_price_slack
        self.time_th = time_th

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """交渉における合意可能価格"""
        current_time = t(step, self.awi.n_steps)
        if current_time < self.time_th:
            if sell:
                return self.awi.trading_prices[self.awi.my_output_product]
            else:
                return self.awi.trading_prices[self.awi.my_input_product]
        else:
            if sell:
                return (
                    self.awi.trading_prices[self.awi.my_input_product]
                    + self.awi.profile.cost[self.awi.process]
                )
            else:
                return (
                    self.awi.trading_prices[self.awi.my_output_product]
                    - self.awi.profile.cost[self.awi.level]
                )

    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        """交渉で用いる効用関数"""
        if is_seller:
            return LinearUtilityFunction((1, 1, 3), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((1, -1, -3), issues=issues, outcomes=outcomes)

    def _urange(self, step, is_seller, time_range):
        """交渉の価格帯を設定"""
        prices = (
            self.awi.catalog_prices
            if not self._use_trading
            or not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        aprice = self.acceptable_unit_price(step, is_seller)
        if is_seller:
            cprice = prices[self.awi.my_output_product]
            return int(cprice * (1.0 - self.agreeable_price_slack)), int(
                self._max_margin * cprice + 0.5
            )

        cprice = prices[self.awi.my_input_product]
        return int(cprice * self._min_margin), int(
            aprice * (1.0 + self.agreeable_price_slack)
        )

    def negotiator(
        self, is_seller: bool, issues=None, outcomes=None, partner=None
    ) -> SAONegotiator:
        add_param = {
            "awi": self.awi,
            "best_opp_acc_selling": self._best_opp_acc_selling,
            "best_opp_acc_buying": self._best_opp_acc_buying,
            "acceptable_unit_price": self.acceptable_unit_price(
                self.awi.current_step, is_seller
            ),
            "target_quantity": self.target_quantity(self.awi.current_step, is_seller),
        }
        self.negotiator_params.update(add_param)
        return super(MyNegotiationManager, self).negotiator(
            is_seller, issues, outcomes, partner
        )


class MyProductionStrategy(SupplyDrivenProductionStrategy):
    pass


class MyTradingStrategy(PredictionBasedTradingStrategy):
    pass


class MyPredictionStrategy(FixedERPStrategy):
    pass


class Daruma(
    MyNegotiationManager,
    MyTradingStrategy,
    MyProductionStrategy,
    SCML2020Agent,
):
    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super(Daruma, self).on_negotiation_success(contract, mechanism)
        up = contract.agreement["unit_price"]
        _is_selling = contract.annotation["seller"] == self.id
        if _is_selling:
            partner = contract.annotation["buyer"]
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

        # 取引データを記録
        if _is_selling:
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

    def target_quantity(self, step: int, sell: bool) -> int:
        """交渉における取引量の最大値を設定"""
        need = self.outputs_needed[step] if sell else self.inputs_needed[step]
        return need
