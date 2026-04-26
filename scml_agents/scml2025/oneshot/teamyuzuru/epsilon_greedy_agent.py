#!/usr/bin/env python
"""

**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* Yuzuru Kitamura (kitamura@katfuji.lab.tuat.ac.jp)

"""

from __future__ import annotations

import math
import random

# required for typing
from negmas import Contract, ResponseType, SAOResponse
from scml.oneshot import *

# required for typing
# required for development
from scml.oneshot import OneShotSyncAgent


def log_message(*args, **kwargs):
    " ".join(str(arg) for arg in args)
    # print(message, **kwargs)


def distribute_uneven(q: int, n: int) -> list[int]:
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    elif q == n:
        return [1] * n

    else:
        lst = [1] * n
        lst[random.randint(0, n - 1)] += q - n
        return lst


def distribute_evenly(total: int, n: int) -> list[int]:
    base_value = total // n
    # 余りを計算
    remainder = total % n

    # 基本値をリストに追加し、余りがある限り1を加える
    distribution = [base_value + (1 if i < remainder else 0) for i in range(n)]
    # リストをシャッフル
    random.shuffle(distribution)

    return distribution


def distribute(q: int, n: int) -> list[int]:
    from collections import Counter

    from numpy.random import choice

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


from itertools import chain, combinations


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    入力されたiterableから冪集合を求める関数

    :param iterable: 冪集合を求めたいiterable
    :return: 冪集合
    :rtype: iterable
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class EpsilonGreedyAgent(OneShotSyncAgent):
    # 総初期オファー数
    _total_first_propose = 0
    # 総初期オファー合意数(0除算を防ぐために1で初期化)
    _total_agreements = 1

    # 現在がfirst_proposalの直後かどうかを判定するフラグ
    _is_after_first_proposal = True

    # 自身がsecond levelの時にfirst proposalを受理した回数をカウントする辞書(key:相手のID, value:回数)
    _first_proposal_accepted = dict()
    # これまでのstepにおいて受理された割合を記録する辞書(key:相手のID, value:割合)
    _first_proposal_accepted_ratio = dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_proposal_accepted.clear()
        self._first_proposal_accepted_ratio.clear()

    def before_step(self):
        if len(self._first_proposal_accepted_ratio) == 0:
            if self.awi.is_first_level:
                self._first_proposal_accepted = {k: 0 for k in self.awi.my_consumers}
                self._first_proposal_accepted_ratio = {
                    k: 0 for k in self.awi.my_consumers
                }

            if self.awi.is_last_level:
                self._first_proposal_accepted = {k: 0 for k in self.awi.my_suppliers}
                self._first_proposal_accepted_ratio = {
                    k: 0 for k in self.awi.my_suppliers
                }

        return super().before_step()

    def distribute_needs_evenly(self) -> dict[str, int]:
        """Distributes my needs evenly over all my partners
        (日本語) 自分のニーズを(おおよそ)均等にすべてのパートナーに分配する
        """

        dist = dict()
        for i, (needs, all_partners) in enumerate(
            [
                (self.awi.needed_supplies, self.awi.my_suppliers),
                (self.awi.needed_sales, self.awi.my_consumers),
            ]
        ):
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]

            partners = len(partner_ids)
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))

                # log_message(i,"|",self.awi.is_first_level)
                continue

            # needsの倍率
            needs_multiplier = 1.3
            if self.awi.is_last_level:  # Second levelの時
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute_evenly(int(needs * needs_multiplier), partners),
                        )
                    )
                )
            else:  # First levelの時
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute_evenly(int(needs / needs_multiplier), partners),
                        )
                    )
                )

        return dist

    def distribute_needs_smartly(self) -> dict[str, int]:
        dist = dict()
        for i, (needs, all_partners) in enumerate(
            [
                (
                    self.awi.needed_supplies,
                    self.awi.my_suppliers,
                ),  # 自分がsecond levelの時にはこっち
                (
                    self.awi.needed_sales,
                    self.awi.my_consumers,
                ),  # 自分がfirst levelの時にはこっちだけが実行される
            ]
        ):
            # (日本語) 自分と交渉中のサプライヤーと消費者を見つける(self.negotiators が自分の取引相手)
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]

            # patrners : 交渉中のパートナーの数
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            # (日本語) needsがない場合、すべての交渉を終了する
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))

                # log_message(i,"|",self.awi.is_first_level)
                continue

            accepted_ratio_list = [
                self._first_proposal_accepted_ratio[k] for k in partner_ids
            ]
            accepted_ratio_sum = sum(accepted_ratio_list)
            # ゼロ除算回避
            if accepted_ratio_sum == 0:
                accepted_ratio_sum = 1
            # 最低1は入れる
            accepted_ratio_list = [
                max(1, math.ceil(needs * ratio / accepted_ratio_sum))
                for ratio in accepted_ratio_list
            ]

            dist.update(dict(zip(partner_ids, accepted_ratio_list)))

        return dist

    def distribute_needs_to_mynegotiator(
        self, partner_ids: list[str]
    ) -> dict[str, int]:
        dist = dict()
        for i, (needs, all_partners) in enumerate(
            [
                (self.awi.needed_supplies, self.awi.my_suppliers),
                (self.awi.needed_sales, self.awi.my_consumers),
            ]
        ):
            # num_patrners : 交渉中のパートナーの数
            num_partners = len(partner_ids)
            all_partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            num_all_partners = len(all_partner_ids)

            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * num_partners)))
                continue

            needs_multiplier = 1.0
            # 最初に全員に1を分配しておく
            dist.update(
                dict(
                    zip(all_partner_ids, distribute(num_all_partners, num_all_partners))
                )
            )
            log_message("initial dist:", dist)
            # 自分にオファーを出した相手にのみニーズを分配
            if self.awi.is_last_level:  # Second levelの時
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute(int(needs * needs_multiplier), num_partners),
                        )
                    )
                )
                return dist
            else:  # First levelの時
                dist.update(
                    dict(
                        zip(
                            partner_ids,
                            distribute(int(needs / needs_multiplier), num_partners),
                        )
                    )
                )
                # dist.update(dict(zip(partner_ids, distribute(0, num_partners))))
                return dist

        # なぜかこれがないとエラーが起こる
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        # (日本語) 私のニーズを私にとって最高の価格でランダムにパートナー?に分配する。
        current_step, price = self._step_and_price(best_price=False)
        n_steps = self.awi.n_steps
        # current_step/n_step の確率でTrueになる乱数を生成
        if random.random() < current_step / n_steps:
            distribution = self.distribute_needs_smartly()
            log_message("smart proposal")
        else:
            distribution = self.distribute_needs_evenly()
            log_message("even proposal")
        # k: パートナーID
        d = {
            k: (quantity, current_step, price) if quantity > 0 else None
            for k, quantity in distribution.items()
        }

        # first proposalの個数をカウント
        self._total_first_propose += len(d)
        # 最初の提案をしたのでフラグをTrueにする
        self._is_after_first_proposal = True
        # log_message("total_first_propose:",self._total_first_propose)

        return d

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        buyer = contract.annotation["buyer"]
        seller = contract.annotation["seller"]
        contract.annotation["caller"]
        # callerと一致する方のみを出力
        if self.awi.is_first_level:
            log_message("agreement with", buyer)

        else:
            log_message("agreement with", seller)
            if self._is_after_first_proposal:
                self._first_proposal_accepted[seller] = 1
                log_message("first counted!")

        if self._is_after_first_proposal:
            self._total_agreements += 1

        return super().on_negotiation_success(contract, mechanism)

    def counter_all(self, offers, states):
        # counter の場合、responseはfirst_proposalでないためフラグをFalseにする
        self._is_after_first_proposal = False

        response = dict()
        # process for sales and supplies independently
        # このfor文は、販売と供給を独立して処理する.実質for文を2回回しているだけ
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            price = issues[UNIT_PRICE].rand()

            partners = {_ for _ in all_partners if _ in offers.keys()}
            # log_message("partners",partners)

            # 自分にとって最適なオファーの組み合わせを探す
            plist = list(powerset(partners))

            best_diff, best_indx = float("inf"), -1
            # sf -> shortfall
            min_shortfall, min_sf_index = 10, -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                shortfall = needs - offered
                estimated_cost = self._calc_estimated_cost(offered, needs)
                utility_value = self.ufun.from_offers(
                    tuple(offers[k] for k in partner_ids),
                    tuple([self.awi.is_first_level] * len(partner_ids)),
                    False,
                    False,
                )
                # utilityinfo = self.ufun.from_offers(tuple(offers[k] for k in partner_ids), tuple([self.awi.is_first_level]*len(partner_ids)),True,False)
                normalized_utility = 1 - utility_value / 100  # 勝手に規格化

                diff = normalized_utility
                # ec -> estimated_cost
                log_message(
                    f"estimated_cost: {estimated_cost:>5.3f} utility: {utility_value:>8.3f} shortfall: {shortfall} offers",
                    str([offers[_] for _ in partner_ids]),
                )
                # log_message("utility_info",str(self.ufun.from_offers(tuple(offers[k] for k in partner_ids), tuple([self.awi.is_first_level]*len(partner_ids)),True,False)))

                if shortfall < min_shortfall and shortfall >= 0:
                    min_shortfall, min_sf_index = shortfall, i
                if diff < best_diff:
                    best_diff, best_indx = diff, i

            # 現在時刻
            current_time = min([_.relative_time for _ in states.values()])
            log_message("current_time:", current_time)
            self._current_threshold(current_time)

            time_threshold = 0.2

            # log_message("th:",th)
            # first levelの時は、min_shortfallを使う が、min_shortfallが0の時は0決着のため、best_diffを使い最良価格を取る
            if min_shortfall != 0 and current_time <= time_threshold:
                partner_ids = plist[min_sf_index]
                others = list(partners.difference(partner_ids))
            else:  # min_shortfall == 0 or current_time > time_threshold
                if min_shortfall == 0 and needs > 0:
                    log_message("expected 0 diff agreement")
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))

            if len(others) == 0 or (
                current_time > time_threshold and len(partner_ids) > 0
            ):
                if len(others) == 0:
                    log_message("agreement with all partners")
                elif current_time > time_threshold:
                    log_message(
                        "time is over. So accept or end negotiation with all offers."
                    )
                else:
                    log_message("unexpected case")

                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            elif shortfall <= 3:
                log_message("num_others:", len(others))
                distribution = dict(zip(others, distribute(int(needs), len(others))))
                log_message("distribution:", distribution)

                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                excess_needs = needs - offered

                if excess_needs < 0:
                    response |= {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    } | {
                        k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for k, q in distribution.items()
                    }
                    log_message("!!End negotiation with", others)
                else:
                    # ここで自分のニーズを満たすために必要な量を計算
                    distribution = dict(
                        zip(others, distribute(int(excess_needs), len(others)))
                    )
                    # log_message("distribution!!!!!!!!!!!!!!!!:",distribution)

                    response |= {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    } | {
                        k: SAOResponse(ResponseType.REJECT_OFFER, None)
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                        for k, q in distribution.items()
                    }
                continue

            else:
                distribution = self.distribute_needs_to_mynegotiator(partners)
                log_message("distribution:", distribution)

                response.update(
                    {
                        k: SAOResponse(ResponseType.REJECT_OFFER, None)
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                        for k, q in distribution.items()
                    }
                )
                # print("response:",response)
        return response

    def _calc_estimated_cost(self, offerd: int, needs: int):
        shortage = 0
        # 自分の工場レベルを確認
        is_first_level = self.awi.is_first_level
        # 自分の工場レベルがfirst levelの時
        if is_first_level:
            # 売る量-買う量
            shortage = offerd - needs
        elif self.awi.is_last_level:
            # 売る量-買う量
            shortage = needs - offerd
        else:
            # log_message("Error: 工場レベルが不明です")
            pass

        # 余った時
        if shortage < 0:
            # 一旦平均のdisposal_costを使う
            disposal_cost = self.awi.current_disposal_cost

            return -disposal_cost * shortage
        # 足りない時
        elif shortage > 0:
            short_fall_penalty = self.awi.current_shortfall_penalty

            return short_fall_penalty * shortage
        # ピッタリの時
        else:
            return 0

    def _current_threshold(self, r: float):
        # mn, mx = 0, self.awi.n_lines // 2
        mn, mx = 2.5, self.awi.n_lines // 2
        # (0, 0.5),(1,3)を通る線(r=1の時直線)
        return mn + (mx - mn) * (r**1)

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

    def step(self):
        log_message(
            "last needed supplies:",
            self.awi.needed_supplies,
            "last needed sales:",
            self.awi.needed_sales,
        )
        # first offerの受け入れ比率を計算
        for partner_id, one_if_accepted in self._first_proposal_accepted.items():
            self._first_proposal_accepted_ratio[partner_id] += (
                one_if_accepted - self._first_proposal_accepted_ratio[partner_id]
            ) / (self.awi.current_step + 1)

        sumsfp = (
            self.awi.current_shortfall_penalty * self.awi.needed_supplies
            if self.awi.needed_supplies > 0
            else 0 - self.awi.current_shortfall_penalty * self.awi.needed_sales
            if self.awi.needed_sales < 0
            else 0
        )
        sumdc = (
            -self.awi.current_disposal_cost * self.awi.needed_supplies
            if self.awi.needed_supplies < 0
            else 0 + self.awi.current_disposal_cost * self.awi.needed_sales
            if self.awi.needed_sales > 0
            else 0
        )

        # sumsfpとsumdcをそれぞれ3桁にパディングして表示
        log_message(f"🔳shortfall penalty: {sumsfp:3.3f}, disposal cost: {sumdc:3.3f}")
        # log_message("agreements ratio:",self._total_agreements/self._total_first_propose)
        log_message("first_proposal_accepted:", self._first_proposal_accepted)
        log_message(
            "first_proposal_accepted_ratio:", self._first_proposal_accepted_ratio
        )
        # 辞書のvalueをゼロにする
        self._first_proposal_accepted = {
            k: 0 for k, v in self._first_proposal_accepted.items()
        }


if __name__ == "__main__":
    import sys

    # run をインポート
    from helpers.runner import run

    for i in range(1):
        run([EpsilonGreedyAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
