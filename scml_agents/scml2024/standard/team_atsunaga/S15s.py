#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations



from negmas.sao import SAONMI
import threading
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent
# required for typing
from typing import Any
import random
import numpy as np
# from scipy.stats import linregress
# required for development
from scml.std import StdSyncAgent, StdAWI
from scml.scml2020.components import *
# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState
from negmas import LinearUtilityFunction
import uuid
from typing import List, Dict, Any
from negmas import Contract, ResponseType
import copy
class offer_neg():
    def __init__(self,negotiator_id,offer:tuple):
        self.negotiator_id = negotiator_id
        self.offer = offer
        self.price = offer[UNIT_PRICE]
        self.quantity = offer[QUANTITY]
        self.time = offer[TIME]
class Neg_list:
    # 
    def __init__(self,neg_id):
        self.neg_id = neg_id
        self.q_dif = [] # 量の差分
        self.p_dif = [] # 価格の差分
        self.t_dif = [] # 納期の差分

class S7s(StdSyncAgent):

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        self.in_schedule = copy.deepcopy(self.buy_num)
        self.out_schedule = copy.deepcopy(self.sell_num)
        offers = {}
        for neg_id in self.awi.my_suppliers:
            new_offer = self.make_first(neg_id,self.raw_inventory,self.fac_lines_cap)
            if new_offer is not None:
                new_offer = self.check_proposal(new_offer,neg_id)
            offers[neg_id] = new_offer
        for neg_id in self.awi.my_consumers:
            new_offer = self.make_first(neg_id,self.raw_inventory,self.fac_lines_cap)
            if new_offer is not None:
                new_offer = self.check_proposal(new_offer,neg_id)
            offers[neg_id] = new_offer


        return offers
    
    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        # print(offers)
        self.in_schedule = copy.deepcopy(self.buy_num)
        self.out_schedule = copy.deepcopy(self.sell_num)
        respond_offer = {}
        inventory = copy.deepcopy(self.raw_inventory)
        fac_cap = copy.deepcopy(self.fac_lines_cap)
        buy_neg = []
        sell_neg = []
        for negotiator_id,values in offers.items():
            if negotiator_id in self.awi.my_suppliers:
                buy_neg.append(offer_neg(negotiator_id,values)) 
            else:
                sell_neg.append(offer_neg(negotiator_id,values))
        buy_neg = sorted(buy_neg,key=lambda x:x.price ,reverse=True)
        buy_date = {}
        sell_date ={}
        # max_ip = 0
        # max_ip_id =
        if len(buy_neg) > 0:
            buy_neg = sorted(buy_neg,key=lambda x:x.price)
            type = self.responsses(buy_neg[0].negotiator_id,states[buy_neg[0].negotiator_id],inventory,fac_cap)
            if type == ResponseType.ACCEPT_OFFER:
                respond_offer[buy_neg[0].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                inventory,fac_cap = self.Update_inventory(buy_neg[0].offer,inventory,fac_cap,buy_neg[0].negotiator_id)
            else:
                new_offer = self.make_offer(buy_neg[0].negotiator_id,states[buy_neg[0].negotiator_id],inventory,fac_cap)
                if new_offer is not None:
                #     inventory,fac_cap = selfventory(new_o.Update_inffer,inventory,fac_cap,sell_neg[0].negotiator_id)
                    respond_offer[buy_neg[0].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)

        if len(sell_neg) > 0:
            sell_neg = sorted(sell_neg,key=lambda x:x.price,reverse=True)
            type = self.responsses(sell_neg[0].negotiator_id,states[sell_neg[0].negotiator_id],inventory,fac_cap)
            if type == ResponseType.ACCEPT_OFFER:
                respond_offer[sell_neg[0].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                inventory,fac_cap = self.Update_inventory(sell_neg[0].offer,inventory,fac_cap,sell_neg[0].negotiator_id)
            else:
                new_offer = self.make_offer(sell_neg[0].negotiator_id,states[sell_neg[0].negotiator_id],inventory,fac_cap)
                if new_offer is not None:
                #     inventory,fac_cap = selfventory(new_o.Update_inffer,inventory,fac_cap,sell_neg[0].negotiator_id)
                    respond_offer[sell_neg[0].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)



        return respond_offer
    def Update_inventory(self, contract,inventory,fac_cap,neg_id) -> None:
        # 交渉が成功したときに呼び出される
        #合意に至った契約に基づいて、自身の内部状態を更新する
        #購入契約の場合は，入荷日の原料の在庫を更新する
        quantity = contract[QUANTITY]
        price = contract[UNIT_PRICE]
        time = int(contract[TIME])
        time = min(time,self.awi.n_steps-1)
        # print(contract)
        if neg_id in self.awi.my_suppliers:
            self.in_schedule[time] += quantity

            
        #販売契約の場合は，出荷日の製品の在庫を更新する
        else:
            fac_cap[time] -= quantity
            self.out_schedule[time] += quantity

        for i in range(time,self.awi.n_steps):
            if i == self.awi.n_steps-1:
                inventory[i] = sum(self.in_schedule)-sum(self.out_schedule)
            else:
                inventory[i] = sum(self.in_schedule[:i+1]) - sum(self.out_schedule[:i+1])
        return inventory,fac_cap


    def make_offer(self, negotiator_id: str, state,inventory,fac_cap) -> Outcome | None:
        neg_info = None
        for place in self.neg_place[self.awi.current_step]:
            if place.neg_id == negotiator_id:
                neg_info = place
                break
        if neg_info is None:
            neg_info = Neg_list(negotiator_id)
            self.neg_place[self.awi.current_step].append(neg_info)
        if negotiator_id in self.awi.my_suppliers:
            # 材料購入戦略
            need_num = 0
            if self.awi.is_last_level:
                if inventory[self.awi.current_step] < 0:
                    need_num = -inventory[self.awi.current_step]/len(self.awi.my_suppliers)
                    date_time = self.awi.current_step
                    p = self.ip
                    offer = self.check_proposal([need_num,p,date_time],negotiator_id)
                    return tuple(offer)
            if sum(self.buy_num) - sum(self.sell_num )> 2*self.awi.n_lines:
                return None
            
            for t in range(self.awi.current_step, self.awi.n_steps):
                if sum(self.buy_num[:t+1]) - sum(self.sell_num[:t]) < need_num:
                    need_num = need_num - sum(self.buy_num[:t+1]) + sum(self.sell_num[:t])
                    if need_num > self.awi.n_steps:
                        need_num = self.awi.n_lines
                    date_time = t
                    p = int(state.step/20 * (((self.awi.catalog_prices[self.awi.my_output_product] + self.awi.trading_prices[self.awi.my_output_product])/2) - self.op)+self.op)
                    offer = self.check_proposal([need_num,p,date_time],negotiator_id)
                    return tuple(offer)
                
        else:
            # 製品販売戦略
            if self.awi.current_step < self.awi.n_steps:
                emp_lines = [0 for _ in range(self.awi.n_steps)]
                t =0
                q =0
                for time in range(self.awi.current_step, self.awi.n_steps):
                    if inventory[time] > 0 and fac_cap[time] > 0:
                        q = min(fac_cap[time], inventory[time])
                        t = time
                        break
                if q == 0:
                    return None
                p = int(self.op*1.1)
                offer = self.check_proposal([q, p, t], negotiator_id)
            else:
                q = 0
                time = 0
                for t in range(self.awi.current_step, self.awi.n_steps):
                    if inventory[t] > 0:
                        time = t
                        q = min(fac_cap[t], inventory[t])
                        break
                p = int(self.op*1.1)
                offer = self.check_proposal([q, p, time], negotiator_id)
            return tuple(offer)
        
    def make_first(self, negotiator_id: str,inventory,fac_cap) -> Outcome | None:
        neg_info = None
        for place in self.neg_place[self.awi.current_step]:
            if place.neg_id == negotiator_id:
                neg_info = place
                break
        if neg_info is None:
            neg_info = Neg_list(negotiator_id)
            self.neg_place[self.awi.current_step].append(neg_info)
        if negotiator_id in self.awi.my_suppliers:
            # 材料購入戦略
            need_num = 0
            if self.awi.is_last_level:
                if inventory[self.awi.current_step] < 0:
                    need_num = -inventory[self.awi.current_step]/len(self.awi.my_suppliers)
                    date_time = self.awi.current_step
                    p = self.ip
                    offer = self.check_proposal([need_num,p,date_time],negotiator_id)
                    return tuple(offer)
            if sum(self.buy_num) - sum(self.sell_num )> 2*self.awi.n_lines:
                return None
            
            for t in range(self.awi.current_step, self.awi.n_steps):
                if sum(self.buy_num[:t+1]) - sum(self.sell_num[:t]) < need_num:
                    need_num = need_num - sum(self.buy_num[:t+1]) + sum(self.sell_num[:t])
                    if need_num > self.awi.n_steps:
                        need_num = self.awi.n_lines
                    date_time = t
                    p = int(min(self.ip * 0.8, self.awi.trading_prices[self.awi.my_input_product]*0.9)) * 1.2
                    offer = self.check_proposal([need_num,p,date_time],negotiator_id)
                    return tuple(offer)

        else:
            # 製品販売戦略
            if self.awi.current_step < self.awi.n_steps:
                # emp_lines = [0 for _ in range(self.awi.n_steps)]
                t =0
                q=0
                for time in range(self.awi.current_step, self.awi.n_steps):
                    if inventory[time] > 0 and fac_cap[time] > 0:
                        q = min(fac_cap[time], inventory[time])
                        t = time
                        break
                    if q == 0:
                        return None
                p = int(self.op*1.2)
                offer = self.check_proposal([q, p, t], negotiator_id)
            return tuple(offer)
           
    def responsses(self, negotiator_id: str, state: SAOState, inventory,fac_cap) -> ResponseType:
        
        if state.current_offer is None:
            return ResponseType.REJECT_OFFER
        neg_info = None
        # if self.neg_place[self.awi.current_step] is None:
        #     neg_info = Neg_list(negotiator_id)
        #     self.neg_place[self.awi.current_step].append(neg_info)
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

        quantity = state.current_offer[QUANTITY]
        time = state.current_offer[TIME]
        price = state.current_offer[UNIT_PRICE]
        if time > self.awi.n_steps-1:
            return ResponseType.REJECT_OFFER

        if negotiator_id in self.awi.my_consumers:
        # print("完成品の販売")
            p = int(state.step/20 * (((self.awi.catalog_prices[self.awi.my_output_product] + self.awi.trading_prices[self.awi.my_output_product])/2) - self.op)+self.op)
            if fac_cap[time] < quantity:
                return ResponseType.REJECT_OFFER
            if self.awi.current_step < self.awi.n_steps*0.7:
                
                if quantity < inventory[time]:
                        if price > p:
                            return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER
            else:
                if quantity < inventory[time]:
                    return ResponseType.ACCEPT_OFFER

        else:
            #材料購入
            if self.awi.is_last_level:
                if inventory[self.awi.current_step] < 0:
                    if time == self.awi.current_step:
                        # if price < self.ip*1.1:
                        return ResponseType.ACCEPT_OFFER
            min_ip = min(self.ip * 0.8, self.awi.trading_prices[self.awi.my_input_product]*0.9)
            p = int(state.step/20 * (self.ip - min_ip) + min_ip)
            if time > self.awi.n_steps*0.7:
                return ResponseType.REJECT_OFFER
            if self.awi.current_step < self.awi.n_steps*0.3:
                need_buy = self.awi.n_lines/3
            elif self.awi.current_step != 0:
                need_buy = sum(self.sell_num[:self.awi.current_step])/self.awi.current_step
                if need_buy == 0:
                    need_buy = self.awi.n_lines/3
            else:
                need_buy = self.awi.n_lines
            if sum(self.buy_num[:time+1]) - sum(self.sell_num[:time]) < need_buy :
                if price < p:
                    return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER
# =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        # print(self.awi.level)
        #最大日数分のリストを作成
        self.raw_inventory = [0 for _ in range(self.awi.n_steps)] # 原料の在庫
        self.buy_cont = [0 for _ in range(self.awi.n_steps)] # 購入数
        self.sell_cont = [0 for _ in range(self.awi.n_steps)] # 販売数
        self.fac_lines_cap = [self.awi.n_lines for _ in range(self.awi.n_steps)] # 工場の生産能力
        self.loss_prd = [] # 不足量のリスト
        self.buy_p_sum = 0 # 購入価格の合計
        self.buy_q_num = 0 # 購入数
        self.sell_p_sum = 0 # 販売価格の合計
        self.sell_q_num = 0 # 販売数
        self.buy_day = [0 for _ in range(self.awi.n_steps+5)] #その日ごとの購入価格の合計
        self.buy_num = [0 for _ in range(self.awi.n_steps+5)] #その日ごとの購入数
        self.sell_day = [0 for _ in range(self.awi.n_steps+5)]   #その日ごとの販売価格の合計
        self.sell_num = [0 for _ in range(self.awi.n_steps+5)]    #その日ごとの販売数
        self.neg_place = [[] for _ in range(self.awi.n_steps+5)] # 交渉相手のリスト
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product]
        self.storage_num = 0
        self.ave_storage_cost = 0
        self.buy_num_day = 0    # 当日の購入数
        self.sell_num_day = 0   # 当日の販売数
        self.buy_cont_day = 0   # 当日の購入契約数
        self.sell_cont_day = 0  # 当日の販売契約数
        self.buy_p_day = 0    # その日の購入額
        self.sell_p_day = 0 # その日の販売額
        self.buy_ave_day = 0 # その日の購入平均価格
        self.sell_ave_day = 0 # その日の販売平均価格
        self.buy_ave_cont = 0 # その日の購入契約平均個数
        self.sell_ave_cont = 0 # その日の販売契約平均個数
        self.current_finances = self.awi.current_balance # 現在の資金
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product]
        self.in_schedule = []
        self.out_schedule = []

    def before_step(self):
        self.offer_counter = 0
        self.inve_cost = self.awi.current_balance - self.current_finances
        # 在庫1つあたりのコストを計算
        if self.awi.current_inventory_input  > 0:
            self.ave_storage_cost = (self.ave_storage_cost * self.storage_num + self.inve_cost) / (self.storage_num + self.awi.current_inventory_input)
        self.storage_num += self.awi.current_inventory_input
        self.buy_num_day = 0    # 当日の購入数
        self.sell_num_day = 0   # 当日の販売数
        self.buy_cont_day = 0   # 当日の購入契約数
        self.sell_cont_day = 0  # 当日の販売契約数
        self.buy_p_day = 0    # その日の購入額
        self.sell_p_day = 0 # その日の販売額

        if self.awi.is_first_level:
            quantity = self.awi.current_exogenous_input_quantity
            time = self.awi.current_step
            price = self.awi.current_exogenous_input_price
    
            self.buy_num[time] += quantity
            self.buy_day[time] += price * quantity
            self.buy_q_num += quantity
            self.buy_p_sum += quantity * price
            self.buy_cont_day += 1
            self.buy_num_day += quantity
            self.buy_p_day += price * quantity

        elif self.awi.is_last_level:
            quantity = self.awi.current_exogenous_output_quantity
            time = self.awi.current_step
            price = self.awi.current_exogenous_output_price

            self.sell_num[time] += quantity
            self.sell_day[time] += price * quantity
            self.sell_q_num += quantity
            self.sell_p_sum += quantity * price
            self.sell_cont_day += 1
            self.sell_num_day += quantity
            self.sell_p_day += price * quantity

        for i in range(self.awi.current_step,self.awi.n_steps):

            self.raw_inventory[i] = sum(self.buy_num[:i+1]) - sum(self.sell_num[:i+1])

        return super().before_step()
    def step(self):
        # if self.awi.current_step == self.awi.n_steps - 1:
        #     # print(self.buy_num)
        #     # print(self.sell_num)
        #     print(sum(self.buy_num)-sum(self.sell_num))
        self.current_finances = self.awi.current_balance
        self._update_iop()

        self.buy_ave_day = 0 # その日の購入平均価格
        self.sell_ave_day = 0   # その日の販売平均価格
        self.buy_ave_cont = 0   # その日の購入契約平均個数
        self.sell_ave_cont = 0  # その日の販売契約平均個数
        if self.buy_num_day > 0:
            #1日あたりの購入契約合意数
            # self.buy_day = self.buy_cont_day /self.buy_num_day
            self.buy_ave_day = self.buy_p_day / self.buy_num_day # その日の購入平均価格
            self.buy_ave_cont = self.buy_num_day / self.buy_cont_day # その日の購入契約平均個数
        else:
            self.buy_ave_day = 0
            self.buy_ave_cont = 0
        if self.sell_num_day > 0:
            #1日あたりの販売契約合意数
            self.sell_ave_day = self.sell_p_day / self.sell_num_day
            self.sell_ave_cont = self.sell_num_day / self.sell_cont_day
        else:
            self.sell_ave_day = 0
            self.sell_ave_cont = 0
        self._update_iop()
    
            # if i == 0:
            #     self.raw_inventory[i] = sum(self.buy_num[:i]) 
            # else:
            #     self.raw_inventory[i] = sum(self.buy_num[:i]) - sum(self.sell_num[:i-1])
        return super().step()
    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        # 交渉が成功したときに呼び出される
        #合意に至った契約に基づいて、自身の内部状態を更新する
        #購入契約の場合は，入荷日の原料の在庫を更新する

        quantity = contract.agreement["quantity"]
        price = contract.agreement["unit_price"]
        time = contract.agreement["time"]

        if contract.annotation['buyer'] == self.id:
            length = len(self.buy_num)
            if time >= length:
                for i in range(length, time+1):
                    self.buy_num.append(0)
                    self.buy_day.append(0)
            self.buy_num[time] += quantity
            self.buy_day[time] += price * quantity
            self.buy_p_sum += price * quantity
            self.buy_q_num += quantity
            self.buy_cont_day += 1
            self.buy_num_day += quantity
            self.buy_p_day += price * quantity
            
        #販売契約の場合は，出荷日の製品の在庫を更新する
        else:
            self.fac_lines_cap[time] -= quantity

            self.sell_num [time] += quantity
            self.sell_day[time] += price * quantity
            self.sell_p_sum += price * quantity
            self.sell_q_num += quantity
            self.sell_cont_day += 1
            self.sell_num_day += quantity
            self.sell_p_day += price * quantity

        return super().on_negotiation_success(contract, mechanism)

    def _update_iop(self):
        if self.buy_cont_day > 0:
            if self.buy_ave_day/self.ip > 0.9:
                self.ip *= 1.05
            else:
                self.ip *= 0.95
        else:
            self.ip *= 1.1
        
        if self.sell_cont_day > 0:
            if self.sell_ave_day/self.op > 1.1:
                self.op *= 1.1
            else:
                self.op *= 0.9
        else:
            self.op *= 0.9
        if self.sell_q_num != 0:
            if self.ip < self.sell_p_sum/self.sell_q_num - self.awi.profile.cost:
                self.ip = self.sell_p_sum/self.sell_q_num - self.awi.profile.cost
        if self.buy_q_num != 0:
            if self.op > self.buy_p_sum/self.buy_q_num + self.awi.profile.cost:
                self.op = self.buy_p_sum/self.buy_q_num + self.awi.profile.cost
    
    def check_proposal(self, offer, nego_id):
        if isinstance(offer, tuple):
            offer = list(offer)
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
                if offer[2] > self.awi.n_steps-1:
                    offer[2] = self.awi.n_steps-1
        return tuple(offer)

