#!/usr/bin/env python
"""
last update:2023/4/1/15:02
- 閾値を0.1から2.5に変更
- estimated_costの計算式を修正

last update:2023/4/1/15:45
- distribute_needsの値に1.5をかける/割るように変更
- first_proposalにおけるbest_price=True をFalseに変更

- couter_allにおいてReject時の分配の方法を、全員に分配するのではなく、自分にオファーを出した相手にのみ分配するように変更
- current_thresholdの計算式を変更

- thの関数の値を低めに変更

**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations
from negmas import SAOResponse, ResponseType, Outcome, SAOState

from scml.oneshot.world import SCML2024OneShotWorld as W
from scml.oneshot import *
from scml.runner import WorldRunner
import pandas as pd
from rich.jupyter import print
import time


# required for typing
from typing import Any

# required for development
from scml.oneshot import OneShotAWI, OneShotSyncAgent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState

import random


def message(tmp: Any):
    # 提出前にここをコメントアウトすること!!
    pass # print(tmp)
    """
    
    詳細説明
    
    :param int 引数(arg1)の名前: 引数(arg1)の説明
    :param 引数(arg2)の名前: 引数(arg2)の説明
    :type 引数(arg2)の名前: 引数(arg2)の型
    :return: 戻り値の説明
    :rtype: 戻り値の型
    :raises 例外の名前: 例外の定義
    """

def distribute(q: int, n: int) -> list[int]:
    """何かを分配する関数

    Distributes n values over m bins with at least one item per bin assuming q > n
    (日本語) n個の値をm個のビンに少なくとも1つのアイテムを含むように分配する(q > nを仮定)

    :param int q: 分配するものの総数?
    :param int n: 分配する場所(or 相手)の数?
    :return: なんじゃこりゃ
    :rtype: list[int]
    """
    from numpy.random import choice
    from collections import Counter

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

class YuzuAgent5(OneShotSyncAgent):
    static_stepcounter=0

    def before_step(self):
        pass # print("=======current_step:",self.awi.current_step,end = "")
        if self.awi.is_first_level:
            pass # print(" This is first level=======")
        else:
            pass # print(" This is second level=======")
        pass # print("first needed supplies:",self.awi.needed_supplies,"first needed sales:",self.awi.needed_sales)
        
        return super().before_step()

    def distribute_needs(self) -> dict[str, int]:
        """Distributes my needs randomly over all my partners
        (日本語) 自分のニーズをランダムにすべてのパートナーに分配する
        """

        dist = dict()
        for i,(needs, all_partners) in enumerate([
            (self.awi.needed_supplies, self.awi.my_suppliers), # 自分がsecond levelの時にはこっち
            (self.awi.needed_sales, self.awi.my_consumers),# 自分がfirst levelの時にはこっちだけが実行される
        ]):
            #4つをそれぞれ出力
            #print("needs",needs,"partners:", all_partners)
            # find suppliers and consumers still negotiating with me
            # (日本語) 自分と交渉中のサプライヤーと消費者を見つける(self.negotiators が自分の取引相手)
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            
            # patrners : 交渉中のパートナーの数
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            # (日本語) needsがない場合、すべての交渉を終了する
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                
                #print(i,"|",self.awi.is_first_level)
                continue

            # distribute my needs over my (remaining) partners.
            # (日本語) 自分のニーズを(残っている)パートナーに分配する。
            # Second levelの時は、ニーズを多く受注し、First levelの時は、ニーズを少なく発注する
            # needsの倍率
            needs_multiplier = 1.5
            if self.awi.is_last_level:# Second levelの時
                dist.update(dict(zip(partner_ids, distribute(int(needs*needs_multiplier), partners))))
            else:# First levelの時
                dist.update(dict(zip(partner_ids, distribute(int(needs/needs_multiplier), partners))))
            
            #print(dist)
        return dist
    
    def distribute_needs_to_mynegotiator(self,partner_ids:list[str]) -> dict[str, int]:
        """
        自分にオファーを出した相手にのみニーズを分配する関数

        :param list[str] partner_ids: 自分にオファーを出した相手のIDのリスト
        :return: ニーズの分配
        :rtype: dict[str, int]
        """

        dist = dict()
        for i,(needs, all_partners) in enumerate([
            (self.awi.needed_supplies, self.awi.my_suppliers), # 自分がsecond levelの時にはこっち
            (self.awi.needed_sales, self.awi.my_consumers),# 自分がfirst levelの時にはこっちだけが実行される
        ]):
            #4つをそれぞれ出力
            
            # num_patrners : 交渉中のパートナーの数
            num_partners = len(partner_ids)
            all_partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            num_all_partners = len(all_partner_ids)

            # if I need nothing, end all negotiations
            # (日本語) needsがない場合、すべての交渉を終了する
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * num_partners)))
                continue

            # (日本語) 自分のニーズを(残っている)パートナーに分配する。
            # Second levelの時は、ニーズを多く受注し、First levelの時は、ニーズを少なく発注する
            # needsの倍率
            needs_multiplier = 1.0
            # 最初に全員に1を分配しておく
            dist.update(dict(zip(all_partner_ids, distribute(num_all_partners, num_all_partners))))
            pass # print("initial dist:",dist)
            # 自分にオファーを出した相手にのみニーズを分配
            if self.awi.is_last_level:# Second levelの時
                dist.update(dict(zip(partner_ids, distribute(int(needs*needs_multiplier), num_partners))))
                return dist
            else:# First levelの時
                dist.update(dict(zip(partner_ids, distribute(int(needs/needs_multiplier), num_partners))))
                return dist
            
        # なぜかこれがないとエラーが起こる
        return dist
            

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        # (日本語) 私のニーズを私にとって最高の価格でランダムにパートナー?に分配する。
        current_step, price = self._step_and_price(best_price=False)
        distribution = self.distribute_needs()
        # k: パートナーID
        d = {k: (quantity, current_step, price) if quantity > 0 else None for k, quantity in distribution.items()}
        pass # print("my first proposal:",d)

        return d

    def counter_all(self, offers, states):
        # offers の中身を全て出力
        pass # print("offers:",offers)
        pass # print("needed_supplies:",self.awi.needed_supplies,"needed_sales:",self.awi.needed_sales)
        
        
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
            # get a random price
            price = issues[UNIT_PRICE].rand()
            #print("price:",price)
            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}
            #print("partners",partners)

            # 自分にとって最適なオファーの組み合わせを探す
            plist = list(powerset(partners))
            
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                estimated_cost = self._calc_estimated_cost(offered, needs)

                # ここを変更すれば元に戻る
                diff = estimated_cost

                pass # print(f"estimated_cost:{estimated_cost:.5f} offers",str([offers[_] for _ in partner_ids]))
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            # If the best combination of offers is good enough, accept them and end all other negotiations
            # (日本語) 最適なオファーの組み合わせが十分に良い場合、それらを受け入れて他のすべての交渉を終了します
            #print([_.relative_time for _ in states.values()])
            #print("min:",min([_.relative_time for _ in states.values()]))
            th = self._current_threshold(
                min([_.relative_time for _ in states.values()])
            )
            pass # print("th:",th)
            if best_diff <= th:
                #print("disporsal cost:",self.awi.current_disposal_cost,"shortfall penalty:",self.awi.current_shortfall_penalty)
                #print("the cost({}) is under the threshold({})".format(best_diff,th))
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs randomly over my partners.
            # (日本語) まだ十分なオファーがない場合、現在のニーズをランダムにパートナーに分配します。
            distribution = self.distribute_needs_to_mynegotiator(partners)
            pass # print("distribution:",distribution)
            
            response.update(
                {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                    )
                    for k, q in distribution.items()
                }
            )
            pass # print("response:",response)
        return response
    
    def _calc_estimated_cost(self, offerd:int, needs:int):
        """
        disposal costとimprovement costを考慮してdiffを計算する関数
        （単にabsを計算するだけでなく商品の過剰ペナルティと不足ペナルティを加味する）
        
        :param int offerd: 提案された数量
        :param int needs: 外生契約で必要とされる数量
        :return: diff (本当はestimated_cost_indexとかにした方がいいかも)
        :rtype: int
        """
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
            #print("Error: 工場レベルが不明です")
            pass
        
        #余った時
        if shortage < 0:
            # 一旦平均のdisposal_costを使う
            disposal_cost = self.awi.current_disposal_cost

            return -disposal_cost*shortage
        #足りない時
        elif shortage > 0:
            short_fall_penalty = self.awi.current_shortfall_penalty

            return short_fall_penalty*shortage
        #ピッタリの時
        else:
            return 0


    def _current_threshold(self, r: float):
        #mn, mx = 0, self.awi.n_lines // 2
        mn, mx = 0.5, 2.5
        #(0, 0.5),(1,3)を通る線(r=1の時直線)
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
        pass # print("last needed supplies:",self.awi.needed_supplies,"last needed sales:",self.awi.needed_sales)
        # 実際に支払ったshort fall penalty と disposal costを計算
        # sumsp = sum of shortfall penalty
        # sumdc = sum of disposal cost
        sumsfp = ( self.awi.current_shortfall_penalty*self.awi.needed_supplies if self.awi.needed_supplies > 0 else 0
                    - self.awi.current_shortfall_penalty*self.awi.needed_sales if self.awi.needed_sales < 0 else 0)
        sumdc  = (-self.awi.current_disposal_cost*self.awi.needed_supplies if self.awi.needed_supplies < 0 else 0
                    + self.awi.current_disposal_cost*self.awi.needed_sales if self.awi.needed_sales > 0 else 0)

        # sumsfpとsumdcをそれぞれ3桁にパディングして表示
        pass # print(f"🔳shortfall penalty: {sumsfp:3.3f}, disposal cost: {sumdc:3.3f}")
        

if __name__ == "__main__":
    import sys

    # run をインポート
    from helpers.runner import run
    for i in range(30):
        run([YuzuAgent5], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
    