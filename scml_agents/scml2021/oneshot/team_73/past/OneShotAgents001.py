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
from scml.oneshot import *

Buy = 0
Sell = 1


class LearningSyncAgent(OneShotSyncAgent, ABC):
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(
        self,
        *args,
        threshold=0.5,
        delta=0.01,
        concession_exponent=0.1,
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
        self._threshold = threshold
        self._delta = delta
        self.secured = 0
        self._e = concession_exponent
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

        self.success_list = defaultdict(lambda: list())  # 交渉成功した際の取引データ
        # self.failure_list = defaultdict(lambda: list())  # 交渉失敗した際の取引データ
        self.success_contracts = []  # 交渉成功した契約のリスト
        self.total_trade_quantity = [0, 0]  # 総取引量（外的契約を含む）
        self.my_offer_list = defaultdict(lambda: list())  # 相手ごとの自分のOfferのリスト
        self.opp_offer_list = defaultdict(lambda: list())  # 相手のOfferのリスト

        self.best_opp_util = -float("inf")  # その日の相手のOfferの効用値の最大値
        self.best_agr_util = -float("inf")  # 合意に達した契約の効用値の最大値
        self.opp_util_slack = opp_util_slack  # opp_utilのSlack変数
        self.agr_util_slack = agr_util_slack  # agr_utilのSlack変数
        self.e = util_exponent  # 閾値の決定に用いる

        self.my_name = None  # 自分の名前（短縮版）

    def init(self):
        # 効用値の最大値と最小値を計算
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

        # _thresholdをレベル別に設定
        if self.awi.is_first_level:
            self._threshold = 0.6
        else:
            self._threshold = 0.4

    def step(self):
        # 外的契約の総取引量を保存
        self.total_trade_quantity[Buy] += self.awi.current_exogenous_input_quantity
        self.total_trade_quantity[Sell] += self.awi.current_exogenous_output_quantity

        # その日の取引の分析結果を表示
        # self.print_analyze()
        # if self.awi.current_step == self.awi.n_steps - 1:
        #     self.display_offers_scatter()
        #     self.display_contracts()

        # パラメタのリセット
        self.secured = 0
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self.best_opp_util = -float("inf")

        # 合意に達しなかった場合，thresholdを下げる
        success_agreements = [
            [_.agreement, _.mechanism_state["current_proposer"]]
            for _ in self.success_contracts
        ]
        accept_simulation_steps = {
            lis[0]["time"] for lis in success_agreements if lis[1] == self.my_name
        }
        offer_simulation_steps = {
            lis[0]["time"] for lis in success_agreements if lis[1] != self.my_name
        }
        # print(f"success_simulation_steps:{success_simulation_steps}")
        if self.awi.current_step in accept_simulation_steps:
            self._threshold += self._delta
        elif self.awi.current_step not in offer_simulation_steps:
            self._threshold -= self._delta

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

        # # 合意を表示
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
        # print(f"contract annotation:{contract.annotation}")

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
        CHANGE_TIME = 0.8

        n = [_ for _ in states.keys()][0]
        state = [_ for _ in states.values()][0]
        opponent_names = self.opponent_names(self.get_nmi(n))

        # print("\n")
        # print(
        #     f"my name:{self.get_nmi(n).negotiator_names[0] if self._is_selling(self.get_nmi(n)) else self.get_nmi(n).negotiator_names[1]}")
        # print(f"step:{state.step}, n_steps:{self.get_nmi(n).n_steps},"
        #       f" t:{self.t(state.step, self.get_nmi(n).n_steps)}")
        # print(f"offers:{offers}")
        # print(f"_threshold:{self._threshold}")
        # print(f"my_opponents:{self.awi.my_consumers, self.awi.my_suppliers}")
        # print(f"my_opponents full:{self.opponent_names(self.get_nmi(n))}")

        # 良いUnit Priceを記録
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
            u, producible = self.from_offers(list(chosen.values()), outputs, True)
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

        responses = dict(
            zip(
                [str(k) for k in keys],
                [SAOResponse(ResponseType.REJECT_OFFER, v) for v in counter_offers],
            )
        )
        # print(f"counter_offers:{counter_offers}")
        # print(f"responses:{responses.items()}")
        # print(f"my_need:{my_needs}, secured:{self.secured}")

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

        # 相手のOfferの効用値で受け入れるかどうか決定
        mx, mn = self.detect_max_min_utility(my_needs, self.get_nmi(n))
        std = max(
            (self.best_opp_util - mn) * (1 - self.opp_util_slack),
            self._threshold * (mx - mn),
            min(
                (self.best_agr_util - mn) * (1 - self.agr_util_slack),
                (self.detect_std_utility(my_needs, self.get_nmi(n)) - mn),
            ),
        )
        # std = mn + self._threshold * (mx - mn)
        std += mn
        # mn = self.detect_min_utility(my_needs, self.get_nmi(n))
        rng = mx - std

        threshold = std + rng * self.util_th(state.step, self.get_nmi(n).n_steps)
        # print(f"threshold:{threshold}")

        if u >= threshold:
            for k, v in chosen.items():
                responses[str(k)] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            self.best_agr_util = max(u, self.best_agr_util)

        # for k, v in chosen.items():
        #     up = v[UNIT_PRICE]
        #     if self._is_good_price(self.get_nmi(n), state.step, up):
        #         responses[str(k)] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

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
            self.from_offers([max_offer], [True] if self._is_selling(nmi) else [False]),
            self.from_offers([min_offer], [True] if self._is_selling(nmi) else [False]),
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
            [tuple(offer)], [True] if self._is_selling(nmi) else [False]
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
        print("cost of production", self.awi.profile.cost)
        # print("current disposal cost", self.awi.current_disposal_cost)
        # print("current shortfall penalty", self.awi.current_shortfall_penalty)
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)
        print("ufun", self.ufun.max_utility, self.ufun.min_utility)
        # offers = [v[-1] for k, v in self.success_list.items() if v[-1][TIME] == self.awi.current_step]
        # print("current profit", self.ufun.from_offers(offers, [True] * len(offers)))
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
                alpha=0.3,
                label="contract with " + k,
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
