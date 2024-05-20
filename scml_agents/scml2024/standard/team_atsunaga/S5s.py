from __future__ import annotations
from negmas.sao import SAONMI

from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent
# required for typing
from typing import Any
import random
import numpy as np
# from scipy.stats import linregress
# required for development
from scml.std import StdSyncAgent, StdAgent
from scml.scml2020.components import *
# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState
from negmas import LinearUtilityFunction

from typing import List, Dict, Any
from negmas import Contract, ResponseType
class Neg_list:
    # 
    def __init__(self,neg_id):
        self.neg_id = neg_id
        self.q_dif = [] # 量の差分
        self.p_dif = [] # 価格の差分
        self.t_dif = [] # 納期の差分
#他のエージェントに関する情報を管理するクラス
class AgentInfo:
    def __init__(self,n_steps) -> None:
        self.finances = [None for _ in range(n_steps)] # 日にちごとの財務状況
        self.inventory = [0 for _ in range(n_steps)] # 日にちごとの製品の在庫
class Neg_info:
    def __init__(self, negotiator_id) -> None:
        best_offer = None # 最良の提案（自分の提案）
        input = False # 買い契約
        output = False # 売り契約
        negotiator = negotiator_id # 交渉相手
        send_list = [] # 送信した提案のリスト
        get_list = []   # 受信した提案のリスト
        pass
class cont:
    def __init__(self,quantity,time,price):
        self.price = price
        self.quantity = quantity
        self.time = time

class loss_prd:
    def __init__(self,sell_time,quantity):
        self.sell_time = sell_time
        self.quantity = quantity
class prd:
    def __init__(self,ID,buy_time,sell_time,buy_price,sell_price):
        self.ID = ID
        self.buy_time = buy_time
        self.sell_time = sell_time
        self.buy_price = buy_price
        self.sell_price = sell_price

class S5s(StdAgent):
    def init(self):
        super().init()
        # print(self.awi.level)
        #最大日数分のリストを作成
        self.neg_list = [] # 交渉のリスト-> 交渉相手ごとにNeg_listを作成
        self.first_proposal = None
        self.neg_list = [] # 交渉のリスト-> 交渉相手ごとにNeg_listを作成
        self.fac_lines_cap =[self.awi.n_lines for _ in range(self.awi.n_steps)] # 日にちごとの工場の生産ラインのキャパシティ
        self.inventory =[[] for _ in range(self.awi.n_steps)] # 日にちごとの製品の未契約在庫
        self.all_inventory = [0 for _ in range(self.awi.n_steps)] # 日にちごとの在庫（売約済みも含む）
        self.raw_inventory = [0 for _ in range(self.awi.n_steps)]# 日にちごとの原料の在庫
        # print(self.awi.n_steps)
        self.buy_inventory= [0 for _ in range(self.awi.n_steps)] # 日にちごとの購入在庫
        self.sold_contract = [0 for _ in range(self.awi.n_steps)] # 日にちごとの販売契約
        self.buy_contract = [0 for _ in range(self.awi.n_steps)] # 日にちごとの購入契約
        self.neg_place = [[] for _ in range(self.awi.n_steps)] # 交渉を行う日
        # print(self.awi.my_suppliers)
        # print(self.awi.my_consumers)
        self.buy_day = self.awi.catalog_prices[self.awi.my_input_product] # その日の販売額
        self.sell_day = self.awi.catalog_prices[self.awi.my_input_product] #その日の
        self.buy_num_day = 0 # その日の購入数
        self.sell_num_day = 0   # その日の販売数
        self.buy_cont_day = 0 # その日の購入額
        self.sell_cont_day = 0  # その日の販売額
        self.supplier_neginf = [i for i in self.awi.my_suppliers] # 供給先の交渉情報
        self.consumer_neginf = [i for i in self.awi.my_consumers]
        self.pro_ID = 111 # プロダクトID
        self.ave_buy_price = self.awi.catalog_prices[self.awi.my_input_product]# 平均購入価格
        self.ave_sell_price = self.awi.catalog_prices[self.awi.my_output_product] # 平均販売価格
        # self.ave_buy_quantity = 0 # 平均購入量
        self.buy_num = 0 # 購入総数
        self.sell_num = 0 # 販売総数
        self.sell_quantity = 0 # 平均販売量
        self.inve_cost = 0 # 在庫コスト
        self.inve_cost_caluculate = 0 # 未来の在庫コスト
        self.current_finances = 0 # 現在の財務状況 -> 在庫コストを算出するために使用
        self.ave_storage_cost = 0 # 平均在庫コスト
        self.storage_num = 0 #在庫コストを払った在庫の総数
        self.process_cost = self.awi.profile.cost # 1日の生産コスト
        self.target_quantity = self.awi.n_lines/3 # 目標在庫量
        self.sold_prd = [] # 売却済み製品
        self.need_quantity = 0 # 必要な製品の量
        self.need_price = 0 # 必要な製品の価格(最下層用)
        self.less_inve = [] # 在庫不足量
        self.buy_cot_num = 0 # 購入契約数
        self.sell_cot_num = 0 # 販売契約数
        self.buy_cont_num_day = 0 # 今日の販売契約数
        self.sell_cont_num_day = 0  # 今日の購入契約数
        self.ip = self.awi.catalog_prices[self.awi.my_input_product] # 原料の名前
        self.op = self.awi.catalog_prices[self.awi.my_output_product] # 製品の名前
        self.offer_counter = 0 # 提案数
        

    def before_step(self):
        # 1日の最初に呼び出される
        # 在庫コストの計算
        # print(len(self.inventory[self.awi.current_step])-self.awi.current_inventory_output)
        # print(self.awi.trading_prices[self.awi.my_input_product]-self.awi.catalog_prices[self.awi.my_input_product])
        self.offer_counter = 0
        self.inve_cost = self.awi.current_balance - self.current_finances
        # self.inve_cost = self.current_finances
        # 在庫1つあたりのコストを計算
        if self.awi.current_inventory_input + self.awi.current_inventory_output > 0:
            self.ave_storage_cost = (self.ave_storage_cost * self.storage_num + self.inve_cost) / (self.storage_num + self.awi.current_inventory_input + self.awi.current_inventory_output)
        self.storage_num += self.awi.current_inventory_input + self.awi.current_inventory_output
        # print(self.ave_storage_cost)
        # print(self.inventory)
        # print(self.awi.current_step)
        # print(self.raw_inventory)
        # self.awi.current_input_issues
        # print(self.awi.current_input_issues[0].values[0])
        if self.awi.is_first_level:
            self.raw_inventory[self.awi.current_step] += self.awi.current_exogenous_input_quantity
            Q = self.awi.current_exogenous_input_quantity
            T = self.awi.current_step
            P = self.awi.current_exogenous_input_price
            for time in range(T, self.awi.n_steps):
                if Q > self.fac_lines_cap[T]:
                    Q -= self.fac_lines_cap[T]
                    for i in range(self.fac_lines_cap[T]):
                        op_pd =prd(self.pro_ID, T, None, P, None)
                        for j in range(T, self.awi.n_steps):
                            self.inventory[j].append(op_pd)
                        self.pro_ID += 1 
                    self.fac_lines_cap[T] = 0
                else:
                    self.fac_lines_cap[T] -= Q
                    for i in range(Q):
                        op_pd =prd(self.pro_ID, T, None, P, None)
                        for j in range(T, self.awi.n_steps):
                            self.inventory[j].append(op_pd)
                        self.pro_ID += 1
                    break
                self.buy_contract[T] += Q
                
        elif self.awi.is_last_level:
            if self.awi.current_inventory_output < self.awi.current_exogenous_output_quantity:
                self._update_sell_contract(self.awi.current_inventory_output, self.awi.current_step)
                self.need_quantity = self.awi.current_exogenous_output_quantity - self.awi.current_inventory_output
            else:
                self._update_sell_contract(self.awi.current_exogenous_output_quantity, self.awi.current_step)
            self.less_inve = sorted(self.less_inve, key=lambda x: x.sell_time)    
        self.buy_num_day = 0
        self.sell_num_day = 0
        self.buy_cont_day = 0
        self.sell_cont_day = 0

        return super().before_step()

    def step(self):
        pass # print(self.awi.current_inventory_input)
        self.current_finances = self.awi.current_balance
        if self.buy_num_day > 0:
            self.buy_day = self.buy_cont_day /self.buy_num_day
        if self.sell_num_day > 0:
            self.sell_day = self.sell_cont_day /self.sell_num_day
        # self.update_price()
        self.less_inve = sorted(self.less_inve, key=lambda x: x.sell_time)
        for ll in self.less_inve:
            if ll.sell_time == self.awi.current_step:
                self.less_inve.remove(ll)
            elif ll.sell_time > self.awi.current_step:
                break
        # print(self.inventory)
        return super().step()
    def respond(self, negotiator_id: str, state: SAOState,  source="") -> ResponseType:
        neg_info = None
        # print(self.awi.current_offers)
        #self.neg_placeのリスト内にあるclassのnegotiator_idがnegotiator_idと一致するものがあるかどうかを調べる
        for place in self.neg_place[self.awi.current_step]:
            if place.neg_id == negotiator_id:
                neg_info = place
                break
        if neg_info is None:
            neg_info = Neg_list(negotiator_id)
            self.neg_place[self.awi.current_step].append(neg_info)
        neg_info.q_dif.append(state.current_offer[QUANTITY])
        neg_info.p_dif.append(state.current_offer[UNIT_PRICE])
        neg_info.t_dif.append(state.current_offer[TIME])

        Q = state.current_offer[0]
        T = state.current_offer[1]
        P = state.current_offer[2]
        if T > self.awi.n_steps-1:
            return ResponseType.REJECT_OFFER
        if negotiator_id in self.awi.my_consumers:

        # print("完成品の販売")
            if T == 0:
                return ResponseType.REJECT_OFFER
            elif len(self.inventory[T]) >= Q:
                if self.ave_buy_price + self.awi.profile.cost + self.ave_storage_cost < 0.9 * P:
                    return ResponseType.ACCEPT_OFFER
                elif self.awi.current_step/self.awi.n_steps > 0.8:
                    return ResponseType.ACCEPT_OFFER
                else:    
                    return ResponseType.REJECT_OFFER
            else:
                if P > self.awi.trading_prices[self.awi.my_output_product]*1.2 and T - self.awi.current_step > self.awi.n_steps/5:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER
    
        else:
            # print("原料の購入")
            # return ResponseType.ACCEPT_OFFER
            if len(self.inventory[self.awi.n_steps-1]) >= self.awi.n_lines*1.5:
                return ResponseType.REJECT_OFFER
            if len(self.less_inve) > 0:
                if self.less_inve[0].sell_time == self.awi.current_step:
                    if P < self.awi.trading_prices[self.awi.my_input_product]*1.2:
                        return ResponseType.ACCEPT_OFFER
            
            if T > self.awi.n_steps/3*2 :
                if state.current_offer[UNIT_PRICE] > self.awi.trading_prices[self.awi.my_input_product]-5:
                    return ResponseType.REJECT_OFFER
                else:
                    return ResponseType.ACCEPT_OFFER
            else:
                count_positive = len([x for x in self.buy_inventory if x > 0])
                if count_positive == 0:
                    if P < self.awi.trading_prices[self.awi.my_input_product]*1.2:
                        return ResponseType.ACCEPT_OFFER
                elif len(self.inventory[T]) >= self.sell_num/count_positive:
                    return ResponseType.REJECT_OFFER
                else:
                    if P < self.awi.trading_prices[self.awi.my_input_product]*1.1:
                        return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER
        #     print("原料の購入")
        # else:
        #     print("その他")

    def propose(self, negotiator_id: str, state) :
        # print(self.awi.current_nmis)
        nmi = self.get_nmi(negotiator_id)
        t = -1
        # print(nmi.issues[UNIT_PRICE])
        offer_list =[]
        t = 0
        q = 0
        p = 0
        if negotiator_id not in self.neg_place[self.awi.current_step]:
            neg_info = Neg_list(negotiator_id)
            self.neg_place[self.awi.current_step].append(neg_info)

        if negotiator_id in self.awi.my_suppliers:
            highest_price = int(self.ave_sell_price - self.awi.profile.cost)
            lowest_price = int(min(self.awi.current_input_issues[0].values[0], self.awi.trading_prices[self.awi.my_input_product]*0.7))
            n = self.offer_counter // 5 
            if len(self.less_inve) > 0:
                if state.step < 5:
                    t = random.randint(self.awi.current_step, self.less_inve[0].sell_time)
                    q = self.less_inve[0].quantity
                    p = random.randint(lowest_price,highest_price)
                    offer = self.check_proposal([q, t, p],negotiator_id)
                    return tuple(offer)
                else:
                    p = random.randint(lowest_price,highest_price)
                    for i in self.neg_place[self.awi.current_step][negotiator_id]:
                        for time in i.t_dif:
                            if time < self.less_inve[n].sell_time:
                                for quantity in i.q_dif:
                                    if quantity <= self.less_inve[n].quantity + 2:
                                        offer = self.check_proposal([quantity, time, p],negotiator_id)
                                        offer_list.append(tuple(offer))
                                        break
                offer_list = sorted(offer_list, key=lambda x: x[1])
                self.offer_counter += 1
                return offer_list[n]
            else:
                need_buy = 0
                if len(self.inventory[-1]) > self.awi.n_lines * 2:
                    return None
                if len(self.buy_inventory[self.awi.current_step:]) > 0:
                    need_buy = sum(self.buy_inventory[self.awi.current_step:])/len(self.buy_inventory[self.awi.current_step:])
                    if need_buy > self.awi.n_lines:
                        need_buy = self.awi.n_lines
                else:
                    need_buy = self.awi.n_lines
                for i in range(self.awi.current_step, self.awi.n_steps):
                    if len(self.inventory[i]) < need_buy:
                        t = i
                        q = need_buy - len(self.inventory[i])
                        p = random.randint(lowest_price,highest_price)
                        break
                offer = self.check_proposal([q, t, p],negotiator_id)
                return tuple(offer)
        else:
            #販売時
            much_inve = sorted(self.inventory[self.awi.current_step:], key=len ,reverse=True)
            indexed_lists = [(index, sublist) for index, sublist in enumerate(self.inventory[self.awi.current_step:])]
            sorted_indexed_lists = sorted(indexed_lists, key=lambda x: len(x[1]), reverse=True)

            # ソートされたリストからインデックスだけを抽出
            much_inve = [index for index, sublist in sorted_indexed_lists]


            t = much_inve[state.step//5]
            q = min(int(len(self.inventory[t])/3),self.awi.n_lines)
            p = random.randint(int(self.awi.trading_prices[self.awi.my_output_product]*0.8),int(self.awi.trading_prices[self.awi.my_output_product]*1.5))
            return tuple([q, t, p])


    def check_proposal(self, offer, nego_id):
        
        if nego_id in self.awi.my_suppliers:
            current_issue = self.awi.current_input_issues
            if offer[0] < current_issue[0].values[0]:
                offer[0] = current_issue[0].values[0]
            elif offer[0] > current_issue[0].values[1]:
                offer[0] = current_issue[0].values[1]
            
            if offer[1] < current_issue[1].values[0]:
                offer[1] = current_issue[1].values[0]
            elif offer[1] > current_issue[1].values[1]:
                offer[1] = current_issue[1].values[1]
            
            if offer[2] < current_issue[2].values[0]:
                offer[2] = current_issue[2].values[0]
            elif offer[2] > current_issue[2].values[1]:
                offer[2] = current_issue[2].values[1]
        else:
            current_issue = self.awi.current_output_issues
            if offer[0] < current_issue[0].values[0]:
                offer[0] = current_issue[0].values[0]
            elif offer[0] > current_issue[0].values[1]:
                offer[0] = current_issue[0].values[1]
            
            if offer[1] < current_issue[1].values[0]:
                offer[1] = current_issue[1].values[0]
            elif offer[1] > current_issue[1].values[1]:
                offer[1] = current_issue[1].values[1]
            
            if offer[2] < current_issue[2].values[0]:
                offer[2] = current_issue[2].values[0]
            elif offer[2] > current_issue[2].values[1]:
                offer[2] = current_issue[2].values[1]
        return offer
    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        # 交渉が成功したときに呼び出される
        #合意に至った契約に基づいて、自身の内部状態を更新する
        #購入契約の場合は，入荷日の原料の在庫を更新する
        quantity = contract.agreement["quantity"]
        price = contract.agreement["unit_price"]
        time = contract.agreement["time"]
        if contract.annotation["buyer"] == self.id:
            self._update_buy_contract(quantity, time,price)
            self.buy_num += quantity
            self.ave_buy_price = (self.ave_buy_price * (self.buy_num - quantity) + price * quantity) / self.buy_num
            self.buy_contract[time] += quantity
            self.buy_cont_day += price * quantity
            self.buy_num_day += quantity
            self.buy_cot_num += 1
        #販売契約の場合は，出荷日の製品の在庫を更新する
        else:
            self._update_sell_contract(quantity, time)
            self.sell_num += quantity
            self.ave_sell_price = (self.ave_sell_price * (self.sell_num - quantity) + price * quantity) / self.sell_num
            self.sell_cont_day += price * quantity
            self.sell_num_day += quantity
            self.sell_cot_num += 1
        return super().on_negotiation_success(contract, mechanism)

          
    def _update_buy_contract(self, quantity: int, time: int, price) -> None:
        for i in range(time, self.awi.n_steps):
            self.raw_inventory[i] += quantity
        
        for i in range(time, self.awi.n_steps-1):
            if quantity <= 0:
                break
            if quantity > self.fac_lines_cap[i]:
                quantity -= self.fac_lines_cap[i]
                # self.inventory[i+1] += self.fac_lines_cap[i]
                for j in range(self.fac_lines_cap[i]):
                    op_pd =prd(self.pro_ID, time, None, self.ave_buy_price, None)
                    for k in range(i, self.awi.n_steps):
                        self.inventory[k].append(op_pd)
                    # self.inventory[i+1].append(op_pd)
                    self.pro_ID += 1
                self.raw_inventory[i] -= self.fac_lines_cap[i]
                self.fac_lines_cap[i] = 0
            else:
                self.fac_lines_cap[i] -= quantity
                # self.inventory[i+1] += quantity
                for j in range(quantity):
                    op_pd =prd(self.pro_ID, time, None, self.ave_buy_price, None)
                    for k in range(i, self.awi.n_steps):
                        self.inventory[k].append(op_pd)
                    self.pro_ID += 1
                self.raw_inventory[i] -= quantity
                break
        pass

    def _update_sell_contract(self, quantity: int, time: int) -> None:
        # 出荷日の製品の在庫を更新するロジックをここに実装する
        # ｎ日目の在庫は，self.inventory[n]に格納されている
        # 出荷日〜最終日までの在庫を更新する
        if len(self.inventory[time]) < quantity:
            self.less_inve.append(loss_prd(time, quantity - len(self.inventory[time])))
            quantity = len(self.inventory[time])
        self.inventory[time] = sorted(self.inventory[time], key=lambda x: x.buy_time)
        for element in self.inventory[time][-quantity:]:
            element.sell_time = time
            # self.delete_prd(time, element.ID)
            self.delete_prd(element)
            self.sold_prd.append(element)
        self.buy_inventory[time] += quantity

    def delete_prd(self, del_prd):
        # del_prd = None
        # for i in range(len(self.inventory[sell_time])):
        #     if self.inventory[sell_time][i].ID == ID:
        #         del_prd = self.inventory[sell_time][i]
        #         break
        for i in range(del_prd.buy_time, self.awi.n_steps):
            if del_prd in self.inventory[i]:
                self.inventory[i].remove(del_prd)
        return del_prd


