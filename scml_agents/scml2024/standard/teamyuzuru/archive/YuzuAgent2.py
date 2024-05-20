#!/usr/bin/env python
"""
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

class YuzuAgent2(OneShotSyncAgent):

    

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
            dist.update(dict(zip(partner_ids, distribute(needs, partners))))
            #print(dist)
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        # (日本語) 私のニーズを私にとって最高の価格でランダムにパートナー?に分配する。
        current_step, price = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        # k: パートナーID
        d = {k: (quantity, current_step, price) if quantity > 0 else None for k, quantity in distribution.items()}
        
        # 静的な情報を出力する
        # 出力する情報の辞書を作成
        outputdict_Static_A = {
            "n_products":self.awi.n_products,
            "n_processes":self.awi.n_processes,
            "n_competitors":self.awi.n_competitors,
            "all_suppliers":self.awi.all_suppliers,
            "all_consumers":self.awi.all_consumers,
            "Production_Capacities":self.awi.production_capacities,
            "is_system":self.awi.is_system,
            "is_bankrupt":self.awi.is_bankrupt(None),# Agent_id を入れるとそれが破産しているかどうかがわかる。Noneを入れると自分が破産しているかどうかがわかる
            "catalog_prices":self.awi.catalog_prices,
            "price_multiplier":self.awi.price_multiplier,
            "is_exogenous_forced":self.awi.is_exogenous_forced,
            "current_step":self.awi.current_step,
            "n_steps":self.awi.n_steps,
            "relative_time":self.awi.relative_time,
            #"state":self.awi.state,
            #"settings":self.awi.settings,
            "price_range":self.awi.price_range,
        }

        dict_dynamic_C = {
            "current_input_outcome_space":self.awi.current_input_outcome_space,
            "current_output_outcome_space":self.awi.current_output_outcome_space,
            #"current_negotiation_details":self.awi.current_negotiation_details,
            "current_input_issues":self.awi.current_input_issues,
            "current_output_issues":self.awi.current_output_issues,
            #"current_buy_nmis":self.awi.current_buy_nmis,
            #"current_sell_nmis":self.awi.current_sell_nmis,
            #"current_nmis":self.awi.current_nmis,
            #"current_buy_states":self.awi.current_buy_states,
            #"current_sell_states":self.awi.current_sell_states,
            #"current_states":self.awi.current_states,
            "current_buy_offers":self.awi.current_buy_offers,
            "current_sell_offers":self.awi.current_sell_offers,
            "current_offers":self.awi.current_offers,
            #"running_buy_nmis":self.awi.running_buy_nmis,
            #"running_sell_nmis":self.awi.running_sell_nmis,
            #"running_nmis":self.awi.running_nmis,
            #"running_buy_states":self.awi.running_buy_states,
            #"running_sell_states":self.awi.running_sell_states,
            #"running_states":self.awi.running_states,

        }

        dict_dynamic_D = {
            "current_exogenous_input_quantity":self.awi.current_exogenous_input_quantity,
            "current_exogenous_input_price":self.awi.current_exogenous_input_price,
            "current_exogenous_output_quantity":self.awi.current_exogenous_output_quantity,
            "current_exogenous_output_price":self.awi.current_exogenous_output_price,
            "current_disposal_cost":self.awi.current_disposal_cost,
            "current_shortfall_penalty":self.awi.current_shortfall_penalty,
            "current_balance":self.awi.current_balance,
            "current_score":self.awi.current_score,
            "current_inventory_input":self.awi.current_inventory_input,
            "current_inventory_output":self.awi.current_inventory_output,
            "current_inventory":self.awi.current_inventory,
        }

        dict_dynamic_E = {
            "sales":self.awi.sales,
            "supplies":self.awi.supplies,
            "total_sales":self.awi.total_sales,
            "total_supplies":self.awi.total_supplies,
            "needed_sales":self.awi.needed_sales,
            "needed_supplies":self.awi.needed_supplies,

        }

        #print("========")
        """
        for string,method in dict_dynamic_C.items():
            pass # print("{}:{}".format(string.ljust(20),method))
            #print("🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳🔳")
        """


        return d

    def counter_all(self, offers, states):
        # offers の中身を全て出力
        #print("current_step:",self.awi.current_step)
        
        
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
            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}
            #print("partners",partners)

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            # 私に最適なオファーセットを提供したパートナーのセットを見つけることです（つまり、私のニーズに最も近い合計数量）
            plist = list(powerset(partners))

            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                estimated_cost = self._calc_estimated_cost(offered, needs)

                # ここを変更すれば元に戻る
                diff = estimated_cost

                #print("abs:",diff,"estimated_cost:",estimated_cost)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            # If the best combination of offers is good enough, accept them and end all other negotiations
            # (日本語) 最適なオファーの組み合わせが十分に良い場合、それらを受け入れて他のすべての交渉を終了します
            
            th = self._current_threshold(
                min([_.relative_time for _ in states.values()])
            )
            # 一旦th = 1に固定
            th = 0.1
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
            distribution = self.distribute_needs()
            #print("the cost is over the threshold!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
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
        return response
    
    def _calc_estimated_cost(self, offerd:int, needs:int):
        """
        disposal costとimprovement costを考慮してdiffを計算する関数
        （単にabsを計算するだけだと商品の過剰ペナルティと不足ペナルティを加味する）
        
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
            diff = offerd - needs
        elif self.awi.is_last_level:
            # 売る量-買う量
            diff = needs - offerd
        else:
            #print("Error: 工場レベルが不明です")
            pass
        
        #余った時
        if diff > 0:
            # 一旦平均のdisposal_costを使う
            disposal_cost = self.awi.current_disposal_cost
            #disposal_cost = self.awi.profile.disposal_cost_mean
            return disposal_cost*diff
        #足りない時
        elif diff < 0:
            short_fall_penalty = self.awi.current_shortfall_penalty
            #short_fall_penalty = self.awi.profile.shortfall_penalty_mean
            return -short_fall_penalty*diff
        #ピッタリの時
        else:
            return 0
            
        
        diff = abs(offerd - needs)
        return diff

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return mn + (mx - mn) * (r**4.0)

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

if __name__ == "__main__":
    import sys

    # run をインポート
    from helpers.runner import run

    run([YuzuAgent2], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
    