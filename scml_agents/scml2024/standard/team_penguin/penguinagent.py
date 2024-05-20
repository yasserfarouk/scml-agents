#!/usr/bin/env python
# type: ignore
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing
import random

# required for development
from scml.std import *

# required for typing
from negmas import *


from itertools import chain, combinations, repeat

from numpy.random import choice
from collections import Counter

__all__ = ["PenguinAgent"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
    # q:needs,n:n_partners
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


# 部分集合を返すメソッド
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class PenguinAgent(StdSyncAgent):
    """An agent that distributes today's needs randomly over 75% of its partners and
    samples future offers randomly."""

    # 初期値分散70%,生産性70%
    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1  # ここではawiが使えないから暫定、stepで決める
        self._threshold = threshold
        self._ptoday = ptoday
        self._productivity = productivity

    def step(self):
        super().step()
        self._threshold = self.awi.n_lines * 0.1  # 閾値更新

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        partners = self.negotiators.keys()
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

        # 今日のオファーと将来のオファーをするパートナーごとに分ける
        first_dict = dict()
        future_supplie_partner = []
        future_consume_partner = []

        for k, q in distribution.items():
            if q > 0:
                first_dict[k] = (q, s, self.best_price(k))
            elif self.is_supplier(k):
                future_supplie_partner.append(k)
            elif self.is_consumer(k):
                future_consume_partner.append(k)

        response = dict()
        future_supplie_dict = dict()
        future_consume_dict = dict()
        response |= first_dict
        future_supplie_dict |= self.future_supplie_offer(future_supplie_partner)
        response |= future_supplie_dict
        future_consume_dict |= self.future_consume_offer(future_consume_partner)
        response |= future_consume_dict

        return response

    def counter_all(self, offers, states):
        response = dict()
        awi = self.awi
        # process for sales and supplies independently
        for edge_needs, all_partners, issues in [
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
            # 今日必要な量を計算
            day_production = self.awi.n_lines * self._productivity
            needs = 0
            if self.is_supplier(all_partners[0]):
                needs = int(
                    day_production
                    - awi.current_inventory_input
                    - awi.total_supplies_at(awi.current_step)
                )
            elif self.is_consumer(all_partners[0]):
                if awi.total_sales_at(
                    awi.total_sales_at(awi.current_step) <= awi.n_lines
                ):  # すでに契約済みの量が一日に可能な生産量を下回る場合
                    needs = int(
                        max(
                            0,
                            min(
                                self.awi.n_lines,
                                day_production + awi.current_inventory_input,
                            )
                            - awi.total_sales_at(awi.current_step),
                        )
                    )  # 今日売る量(上限値は生産ラインの数)

            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}

            # 現在のステップのオファーだけ抽出(将来のオファーに対する返答は後から考える)
            current_step_offers = dict()  # 現在のステップのオファー
            future_step_offers = dict()  # 将来のステップのオファー

            for p in partners:
                if offers[p] is None:
                    continue
                if offers[p][TIME] == self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    current_step_offers[p] = offers[p]
                elif offers[p][TIME] != self.awi.current_step and self.is_valid_price(
                    offers[p][UNIT_PRICE], p
                ):
                    future_step_offers[p] = offers[p]

            # 現在のステップのオファーのエージェントのリスト
            current_step_partners = {
                _ for _ in partners if _ in current_step_offers.keys()
            }
            future_step_partners = {
                _ for _ in partners if _ in future_step_offers.keys()
            }

            # 将来の契約返す(ここで返したものを後から再度オファーするエージェントの候補から外す)
            duplicate_list = [0 for _ in range(awi.n_steps)]  # 重複分
            for p, x in future_step_offers.items():
                step = future_step_offers[p][TIME]
                if step <= awi.n_steps:  # 必要
                    if (
                        future_step_offers[p][QUANTITY] + duplicate_list[step - 1]
                        <= self.needs_at(step, p)
                    ):  # オファーの商品量+すでに受け入れたオファーの商品量が必要量を下回ってた場合
                        response[p] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, future_step_offers[p]
                        )
                        duplicate_list[step - 1] += future_step_offers[p][
                            QUANTITY
                        ]  # 重複しないように保存

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)

            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = current_step_partners.difference(partner_ids)
                offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)

                diff = abs(offered - needs)
                if offered - needs >= 0:
                    if diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = diff, i
                else:
                    if diff < best_minus_diff and offered > 0:
                        best_minus_diff, best_index_minus = diff, i

            has_accept_offer = True
            # ここら辺でオファーの数がニードを超えるばあい、超えないリストとかで一番いいやつを選ぶ
            if best_plus_diff <= self._threshold and len(plist[best_index_plus]) > 0:
                best_indx = best_index_plus
            elif len(plist[best_index_minus]) > 0:
                best_indx = best_index_minus
            else:
                has_accept_offer = False

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = plist[best_indx]
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))
                others_s = []
                others_c = []
                response.update(
                    {
                        k: SAOResponse(
                            ResponseType.ACCEPT_OFFER, current_step_offers[k]
                        )
                        for k in partner_ids
                    }
                )
                # othersに将来の契約をオファーする
                for x in others:
                    if self.is_supplier(x):
                        others_s.append(x)
                    if self.is_consumer(x):
                        others_c.append(x)

                others_s_dict = dict()
                others_s_dict |= self.future_supplie_offer(others_s)
                others_c_dict = dict()
                others_c_dict |= self.future_consume_offer(others_c)
                for k, x in others_s_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in others_c_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                flag = 1  # 無視したい処理のためのフラグ

            if flag != 1:
                # If I still do not have a good enough offer, distribute my current needs
                # randomly over my partners.
                # たりてないならまだオファーを返していないORこの関数で受け取っていない相手にオファーを出す(再度firstproposal)
                # Responseのキー値を一意にするためにまだの相手に出す
                other_partners = {
                    _
                    for _ in all_partners
                    if _ not in response.keys() and _ in self.negotiators.keys()
                }  # 後ろの条件式がないとエラーが出る:(
                distribution = self.distribute_todays_needs(other_partners)
                future_supplie_partner = []
                future_consume_partner = []

                for k, q in distribution.items():
                    if q > 0:
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, self.price(k)),
                        )
                    elif self.is_supplier(k):
                        future_supplie_partner.append(k)
                    elif self.is_consumer(k):
                        future_consume_partner.append(k)

                future_supplie_offer_dict = dict()
                future_supplie_offer_dict |= self.future_supplie_offer(
                    future_supplie_partner
                )
                future_consume_offer_dict = dict()
                future_consume_offer_dict |= self.future_consume_offer(
                    future_consume_partner
                )
                for k, x in future_supplie_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

                for k, x in future_consume_offer_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    # 法外な値段を排除するためのプログラム
    def is_valid_price(self, price, partner):
        nmi = self.get_nmi(partner)
        issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return price >= minp
        elif self.is_supplier(partner):
            return price <= maxp
        else:
            False

    def needs_at(self, step, partner):  # stepでの必要量を返すメソッド
        need = 0
        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        if self.is_supplier(partner):
            need = int(
                day_production
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
            )
        elif self.is_consumer(partner):
            need = int(
                max(
                    0,
                    min(self.awi.n_lines, day_production + awi.current_inventory_input)
                    - awi.total_sales_at(step),
                )
            )

        return need

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """Distributes my urgent (today's) needs randomly over some my partners"""
        if partners is None:
            partners = self.negotiators.keys()

        # initialize all quantities to zero
        response = dict(zip(partners, repeat(0)))

        # partnersをconsumeとsupplierで分ける
        supplie_partners = []
        consume_partners = []
        for x in partners:
            if self.is_supplier(x):
                supplie_partners.append(x)
            elif self.is_consumer(x):
                consume_partners.append(x)

        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        supplie_needs = int(
            day_production
            - awi.current_inventory_input
            - awi.total_supplies_at(awi.current_step)
        )
        consume_needs = int(
            max(
                0,
                min(self.awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step),
            )
        )
        n_supplier = len(supplie_partners)
        n_consumer = len(consume_partners)

        if n_supplier > 0 and supplie_needs > 0:
            response |= self.distribute_todays_supplie_consume_needs(
                supplie_partners, supplie_needs
            )

        if (
            n_consumer > 0
            and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= self.awi.n_lines
        ):  # 供給者が１人以上かつ需要が１以上かつすでに契約済みの量が一日に可能な生産量を下回るとき
            response |= self.distribute_todays_supplie_consume_needs(
                consume_partners, consume_needs
            )

        return response

    def distribute_todays_supplie_consume_needs(
        self, partners, needs
    ) -> dict[str, int]:
        response = dict(zip(partners, repeat(0)))
        random.shuffle(partners)
        partners = partners[: max(1, int(self._ptoday * len(partners)))]
        n_partners = len(partners)

        # if my needs are small, use a subset of negotiators
        if needs < n_partners <= 0:
            partners = random.sample(partners, random.randint(1, needs))
            n_partners = len(partners)

        response |= dict(zip(partners, distribute(needs, n_partners)))

        return response

    def future_supplie_offer(self, list):
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        d1 = dict()
        d2 = dict()
        d3 = dict()
        response = dict()
        # リストを50%,30%,20%に分ける
        step1_list = list[: int(len(list) * 0.5)]
        n_step1 = len(step1_list)
        step2_list = list[int(len(list) * 0.5) : int(len(list) * 0.8)]
        n_step2 = len(step2_list)
        step3_list = list[int(len(list) * 0.8) :]
        n_step3 = len(step3_list)
        # 一日の生産量
        p = awi.n_lines * self._productivity
        # 1step先で必要な量
        if s + 1 < n:
            step1_needs = int(
                (p - awi.current_inventory_input - awi.total_supplies_at(s + 1)) / 3
            )
            if step1_needs > 0 and n_step1 > 0:  # 必要
                d1 |= dict(zip(step1_list, distribute(step1_needs, n_step1)))
                for k, q in d1.items():
                    if q > 0:
                        response[k] = (q, s + 1, self.best_price(k))

        # 2step先で必要な量
        if s + 2 < n:
            step2_needs = int(
                (p - awi.current_inventory_input - awi.total_supplies_at(s + 2)) / 3
            )
            if step2_needs > 0 and n_step2 > 0:
                d2 |= dict(zip(step2_list, distribute(step2_needs, n_step2)))
                for k, q in d2.items():
                    if q > 0:
                        response[k] = (q, s + 2, self.best_price(k))

        # 3step先で必要な量
        if s + 3 < n:
            step3_needs = int(
                (p - awi.current_inventory_input - awi.total_supplies_at(s + 3)) / 3
            )
            if step3_needs > 0 and n_step3 > 0:
                d3 |= dict(zip(step3_list, distribute(step3_needs, n_step3)))
                for k, q in d3.items():
                    if q > 0:
                        response[k] = (q, s + 3, self.best_price(k))

        return response

    def future_consume_offer(self, list):
        response = dict()
        awi = self.awi
        s = awi.current_step  # 現在のステップ
        n = awi.n_steps  # 総ステップ数
        d1 = dict()
        d2 = dict()
        d3 = dict()
        # リストを50%,30%,20%に分ける
        step1_list = list[: int(len(list) * 0.5)]
        n_step1 = len(step1_list)
        step2_list = list[int(len(list) * 0.5) : int(len(list) * 0.8)]
        n_step2 = len(step2_list)
        step3_list = list[int(len(list) * 0.8) :]
        n_step3 = len(step3_list)
        # 一日の生産量
        p = awi.n_lines * self._productivity
        # 1step先で売る量
        if (
            s + 1 < n and awi.total_sales_at(s + 1) <= self.awi.n_lines
        ):  # 総ステップ数を超えないときとすでに契約済みの量が生産量を超えない場合
            step1_needs = int(
                max(
                    0,
                    min(self.awi.n_lines, p + awi.current_inventory_input)
                    - awi.total_sales_at(s + 1),
                )
                / 3
            )  # ステップで売る量(上限値は生産ラインの数)契約が重複する?3分の1
            if step1_needs > 0 and n_step1 > 0:
                d1 |= dict(zip(step1_list, distribute(step1_needs, n_step1)))
                for k, q in d1.items():
                    if q > 0:
                        response[k] = (q, s + 1, self.best_price(k))

        # 2step先で売る量
        if s + 2 < n and awi.total_sales_at(s + 2) <= self.awi.n_lines:
            step2_needs = int(
                max(
                    0,
                    min(self.awi.n_lines, p + awi.current_inventory_input)
                    - awi.total_sales_at(s + 2),
                )
                / 3
            )
            if step2_needs > 0 and n_step2 > 0:
                d2 |= dict(zip(step2_list, distribute(step2_needs, n_step2)))
                for k, q in d2.items():
                    if q > 0:
                        response[k] = (q, s + 2, self.best_price(k))

        # 3step先で売る量
        if s + 3 < n and awi.total_sales_at(s + 2) <= self.awi.n_lines:
            step3_needs = int(
                max(
                    0,
                    min(self.awi.n_lines, p + awi.current_inventory_input)
                    - awi.total_sales_at(s + 3),
                )
                / 3
            )
            if step3_needs > 0 and n_step3 > 0:
                d3 |= dict(zip(step3_list, distribute(step3_needs, n_step3)))
                for k, q in d3.items():
                    if q > 0:
                        response[k] = (q, s + 3, self.best_price(k))

        return response

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def best_price(self, partner):
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner):
        # 反対提案で用いる
        # 今日足りていない分のためのオファーのため少し譲歩する

        nmi = self.get_nmi(partner)
        issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value

        if self.is_consumer(partner):
            return max(maxp * 0.7, minp)
        else:
            return min(minp * 1.2, maxp)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([PenguinAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
