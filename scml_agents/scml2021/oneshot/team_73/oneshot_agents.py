from abc import ABC
from collections import defaultdict

from negmas import MechanismState, ResponseType
from negmas.outcomes import Outcome
from scml.oneshot import *

from .nego_utils import *

Buy = 0
Sell = 1
Offer = 0
Accept = 1
INF = 1000

__all__ = [
    "SimpleAgent",
    "BetterAgent",
    "AdaptiveAgent",
    "Gentle",
]


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
        super().__init__(*args, concession_exponent=concession_exponent, **kwargs)
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


class Gentle(AdaptiveAgent, ABC):
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
        self._opp_acc_price_slack = 0.0
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
        # std = (good_price_range["min"] + good_price_range["max"]) / 2
        # std = self.awi.trading_prices[1]
        pattern = ["offer"]
        if self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
            if self._environment_factor(nmi) >= 0.5:
                pattern.append("good_env")
                if success_agreements:
                    if accept_agreements:
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
                    else:
                        pattern.append("offer_agreements")

                    if list(rank.keys())[0] == name:
                        step = min(state.step + 1, nmi.n_steps - 1)
                elif self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
                    pattern.append("first_offer")
                else:
                    pattern.append("")  # パラメタを変更しない
            else:
                pattern.append("bad_env")
                if success_agreements:
                    if accept_agreements:
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
                    else:
                        pattern.append("offer_agreements")
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
        is_selling = self._is_selling(nmi)

        # 相手の譲歩率に応じて判断
        name = nmi.annotation["buyer"] if is_selling else nmi.annotation["seller"]
        success_agreements = opponent_agreements(
            nmi, is_selling, self.success_contracts
        )
        pattern = ["accept"]
        if self.nego_info["negotiation_step"] >= nmi.n_steps - 1:
            # 譲歩率の変化
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
            # env-factor
            if self._environment_factor(nmi) > 0.5:
                pattern.append("good_env")
            else:
                pattern.append("bad_env")

            self._set_param(pattern)

        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)

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
            + Gentle.better_agent._th(nmi.n_steps - 2, nmi.n_steps) * (mx - mn),
            "max": mx
            - Gentle.better_agent._th(nmi.n_steps - 2, nmi.n_steps) * (mx - mn),
        }

        return price_range

    def _first_offer_price(self, name: str):
        """合意のない相手に対するofferの価格を決定"""
        nmi = self.get_nmi(name)
        is_selling = self._is_selling(nmi)
        time = t(self.awi.current_step, self.awi.n_steps)

        # 基準値の決定
        up = nmi.issues[UNIT_PRICE]
        std = (up.max_value + up.min_value) / 2

        # 価格を決定（滑らかに譲歩）
        strong_range = {"max": 0.2, "min": -0.3}
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
                            (self._best_selling, self._step_price_slack),
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
                            (self._best_buying, self._step_price_slack),
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
                good_agreement = min(max(0.5 - (prev_up - tp) / tp, 0), 1)
            else:
                good_agreement = max(min(0.5 + (prev_up - tp) / tp, 1), 0)
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
                self._opp_price_slack = 0.0
                self._opp_acc_price_slack = 0.2
                self.worst_opp_acc_price_slack = INF

            elif pattern[1] == "persist":
                # self._step_price_slack = 0.0
                self._opp_price_slack = 0.0
                self._opp_acc_price_slack = 0.0
                self.worst_opp_acc_price_slack = INF

        elif pattern[0] == "offer":
            self._step_price_slack = INF
            if pattern[1] == "good_env":
                if pattern[2] == "accept_agreements":
                    self.first_offer_price_slack = INF
                    if pattern[3] == "good":
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                    elif pattern[3] == "bad":
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = r

                elif pattern[2] == "offer_agreements":
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = -INF
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = INF

                elif pattern[2] == "first_offer":
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = 0.0
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = 0.0

            elif pattern[1] == "bad_env":
                if pattern[2] == "accept_agreements":
                    self.first_offer_price_slack = INF
                    if pattern[3] == "good":
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = 0.0

                    elif pattern[3] == "bad":
                        self._opp_price_slack = 0.0
                        self._opp_acc_price_slack = -INF
                        self.worst_opp_acc_price_slack = r

                elif pattern[2] == "offer_agreements":
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = -INF
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = INF

                elif pattern[2] == "first_offer":
                    self._opp_price_slack = 0.0
                    self._opp_acc_price_slack = 0.0
                    self.worst_opp_acc_price_slack = 0.0
                    self.first_offer_price_slack = 0.0


def print_log(names, values, on=False):
    if on:
        if type(names) == str:
            print(f"{names}:{values}")
        if type(names) == list:
            for name, value in dict(zip(names, values)).items():
                print(f"{name}:{value}", end=" ")
            print()
