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
        for neg in buy_neg:
            if neg.time not in buy_date:
                buy_date[neg.time] = []
            buy_date[neg.time].append(neg)
        for neg in sell_neg:
            if neg.time not in sell_date:
                sell_date[neg.time] = []
            sell_date[neg.time].append(neg)
        
        #購入の交渉
        for date,o_list in buy_date.items():
            if len(o_list) == 1:
                type = self.responsses(o_list[0].negotiator_id,states[o_list[0].negotiator_id],inventory,fac_cap)
                if type == ResponseType.ACCEPT_OFFER:
                    respond_offer[o_list[0].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                    inventory,fac_cap = self.Update_inventory(o_list[0].offer,inventory,fac_cap,o_list[0].negotiator_id)
                else:
                    new_offer = self.make_offer(o_list[0].negotiator_id,states[o_list[0].negotiator_id],inventory,fac_cap)
                    if new_offer is not None:
                        inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[0].negotiator_id)
                    respond_offer[o_list[0].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
            else:
                o_list = sorted(o_list,key=lambda x:x.price)
                for i in range(len(o_list)):
                    if self.responsses(o_list[i].negotiator_id,states[o_list[i].negotiator_id],inventory,fac_cap) == ResponseType.ACCEPT_OFFER:
                        respond_offer[o_list[i].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                        inventory,fac_cap = self.Update_inventory(o_list[0].offer,inventory,fac_cap,o_list[0].negotiator_id)
                        for j in range(i+1,len(o_list)):
                            new_offer = self.make_offer(o_list[j].negotiator_id,states[o_list[j].negotiator_id],inventory,fac_cap)
                            if new_offer is not None:
                                inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[j].negotiator_id)
                            respond_offer[o_list[j].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
                        break
                    else:
                        new_offer = self.make_offer(o_list[i].negotiator_id,states[o_list[i].negotiator_id],inventory,fac_cap)
                        if new_offer is not None:
                            inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[i].negotiator_id)
                        respond_offer[o_list[i].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
        
        for date,o_list in sell_date.items():
            if len(o_list)==1:
                type = self.responsses(o_list[0].negotiator_id,states[o_list[0].negotiator_id],inventory,fac_cap)
                if type == ResponseType.ACCEPT_OFFER:
                    respond_offer[o_list[0].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                else:
                    new_offer = self.make_offer(o_list[0].negotiator_id,states[o_list[0].negotiator_id],inventory,fac_cap)
                    if new_offer is not None:
                        inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[0].negotiator_id)
                    respond_offer[o_list[0].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
            else:
                o_list = sorted(o_list,key=lambda x:x.price)
                for i in range(len(o_list)):
                    if self.responsses(o_list[i].negotiator_id,states[o_list[i].negotiator_id],inventory,fac_cap) == ResponseType.ACCEPT_OFFER:
                        respond_offer[o_list[i].negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER,None)
                        for j in range(i+1,len(o_list)):
                            new_offer = self.make_offer(o_list[j].negotiator_id,states[o_list[j].negotiator_id],inventory,fac_cap)
                            if new_offer is not None:
                                inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[j].negotiator_id)
                            respond_offer[o_list[j].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
                        break
                    else:
                        new_offer = self.make_offer(o_list[i].negotiator_id,states[o_list[i].negotiator_id],inventory,fac_cap)
                        if new_offer is not None:
                            inventory,fac_cap = self.Update_inventory(new_offer,inventory,fac_cap,o_list[i].negotiator_id)
                        respond_offer[o_list[i].negotiator_id] = SAOResponse(ResponseType.REJECT_OFFER,new_offer)
            # for neg in buy_neg:
                # for neg_id,values in neg.items():
                #     if values.outcomes[0][0] < self.ip:
                #         return {neg_id:SAOResponse(False,None)}
                #     if values.outcomes[0][0] > self.ip:
                #         return {neg_id:SAOResponse(True,values)}
        
        return respond_offer
    def Update_inventory(self, contract,inventory,fac_cap,neg_id) -> None:
        # 交渉が成功したときに呼び出される
        #合意に至った契約に基づいて、自身の内部状態を更新する
        #購入契約の場合は，入荷日の原料の在庫を更新する
        quantity = contract[QUANTITY]
        price = contract[UNIT_PRICE]
        time = contract[TIME]
        
        # print(contract)
        if neg_id in self.awi.my_suppliers:
            for t in range(time, self.awi.n_steps):
                inventory[t] += quantity

            
        #販売契約の場合は，出荷日の製品の在庫を更新する
        else:
            fac_cap[time] -= quantity
            for t in range(time, self.awi.n_steps):
                inventory[t] -= quantity

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
            if inventory[-1] > 2*self.awi.n_lines:
                return None
            if state.step > 10:
                inde = 0
                for t_offer in neg_info.t_dif:
                    for time in range(t_offer,min(int(t_offer + self.awi.n_steps/10),self.awi.n_steps)):
                        if inventory[time] < 0:
                            need_num = neg_info.q_dif[inde]
                            first_loss = time
                            break
                    if need_num > 0:
                        break
                    inde += 1
            # current_step + (n_steps/10)以内で不足があれば率先して提案
            if self.awi.current_step + self.awi.n_steps/10 > self.awi.n_steps*0.8:
                return None
            if need_num == 0:
                first_loss = 0
                min_ip = min(self.ip * 0.8, self.awi.trading_prices[self.awi.my_input_product]*0.9)
                for t in range(self.awi.current_step, int(self.awi.current_step + self.awi.n_steps/10)):
                    if inventory[t] < 0:
                        if first_loss == 0:
                            first_loss = t
                        if need_num < -inventory[t]:
                            need_num = -inventory[t]
                if need_num > 0:
                    need_num = min(need_num, self.awi.n_lines/3)
                else:
                    need_num = self.awi.n_lines/3
                if first_loss == 0:
                    first_loss = random.randint(self.awi.current_step, int(self.awi.current_step + self.awi.n_steps/10))
                
                    need_num = min(need_num, self.awi.n_lines/3)
            min_ip = min(self.ip * 0.8, self.awi.trading_prices[self.awi.my_input_product]*0.9)
            p = int(state.step/20 * (self.ip - min_ip) + min_ip)
            offer = self.check_proposal([need_num, p, first_loss], negotiator_id)
            return tuple(offer)
        else:
            # 製品販売戦略
            if self.awi.current_step < self.awi.n_steps*2/3:
                emp_lines = [0 for _ in range(self.awi.n_steps)]
                for time in range(self.awi.current_step, int(self.awi.current_step + self.awi.n_steps/5)):
                    if inventory[time] == 0:
                        emp_lines[time] = 0
                    emp_lines[time] = min (fac_cap[time], inventory[time])
                emp_lines_index = sorted(range(len(emp_lines)), key=lambda k: emp_lines[k],reverse=True)
                time = emp_lines_index[state.step//2]
                q = min(emp_lines[time],self.awi.n_lines/3)
                if q == 0:
                    return None
                p = int(state.step/20 * (((self.awi.catalog_prices[self.awi.my_output_product] + self.awi.trading_prices[self.awi.my_output_product])/2) - self.op)+self.op)
                offer = self.check_proposal([q, p, time], negotiator_id)
            else:
                q = 0
                time = 0
                for t in range(self.awi.current_step, self.awi.n_steps):
                    if inventory[t] > 0:
                        time = t
                        q = min(fac_cap[t], inventory[t])
                        break
                p = int(state.step/20 * (((self.awi.catalog_prices[self.awi.my_output_product] + self.awi.trading_prices[self.awi.my_output_product])/2) - self.op)+self.op)
                offer = self.check_proposal([q, p, time], negotiator_id)
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
        with self.used:
            if negotiator_id in self.awi.my_consumers:

            # print("完成品の販売")
                if fac_cap[time] < quantity:
                    return ResponseType.REJECT_OFFER
                if self.awi.current_step < self.awi.n_steps*2/3:
                    if time == 0:
                        if inventory[time] > quantity:
                            if price > self.op:
                                return ResponseType.ACCEPT_OFFER
                        return ResponseType.REJECT_OFFER
                    if time < self.awi.current_step + self.awi.n_steps/10:
                        if quantity < inventory[time]:
                            if price > self.op:
                                return ResponseType.ACCEPT_OFFER
                    elif time < self.awi.current_step + self.awi.n_steps/5:
                        if quantity < inventory[time]:
                            if price > self.op:
                                return ResponseType.ACCEPT_OFFER
                        else:
                            if price > self.op*1.2:
                                return ResponseType.ACCEPT_OFFER
                    else:
                        if price > self.op:
                            return ResponseType.ACCEPT_OFFER

                    return ResponseType.REJECT_OFFER
                else:
                    if inventory[time] > quantity:
                        return ResponseType.ACCEPT_OFFER
                    
            else:
                #材料購入
                for t in range(time, min(int(time + self.awi.n_steps/10),self.awi.n_steps)):
                    if inventory[t] < 0:
                        if inventory[t] + quantity < self.awi.n_lines/2:
                            if price < self.ip:
                                return ResponseType.ACCEPT_OFFER
                        else:
                            if price < self.ip*0.9:
                                return ResponseType.ACCEPT_OFFER
                #全日程を通した，生産ライン数の60％を超える購入は拒否
                # if sum(self.buy_num[self.awi.current_step:]) + Q  > self.awi.n_lines * (self.awi.n_steps - self.awi.current_step) * 0.4:
                # if inventory[-1] + quantity  > sum(fac_cap[self.awi.current_step:]) * 0.6:
                if self.buy_q_num + quantity > self.awi.n_lines * (self.awi.n_steps) * 0.6:
                    # print("too much")
                    return ResponseType.REJECT_OFFER
                if time < self.awi.n_steps*2/3:
                    max_q = max(inventory[time:int(self.awi.n_steps*2/3)+1]) 
                    if max_q + quantity < 2 * self.awi.n_lines:
                        if price < self.ip:
                            return ResponseType.ACCEPT_OFFER
                else:
                    max_q = max(inventory[time:self.awi.n_steps])
                    if max_q + quantity < self.awi.n_lines:
                        if price < self.ip:
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
        self.buy_day = [0 for _ in range(self.awi.n_steps)] #その日ごとの購入価格の合計
        self.buy_num = [0 for _ in range(self.awi.n_steps)] #その日ごとの購入数
        self.sell_day = [0 for _ in range(self.awi.n_steps)]   #その日ごとの販売価格の合計
        self.sell_num = [0 for _ in range(self.awi.n_steps)]    #その日ごとの販売数
        self.neg_place = [[] for _ in range(self.awi.n_steps)] # 交渉相手のリスト
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
        self.used = threading.Lock()

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
            Q = self.awi.current_exogenous_input_quantity
            T = self.awi.current_step
            P = self.awi.current_exogenous_input_price
            for t in range(T, self.awi.n_steps):
                self.raw_inventory[t] += Q
            self.buy_num[T] += Q
            self.buy_day[T] += P * Q
            self.buy_q_num += Q
            self.buy_p_sum += Q * P
            self.buy_cont_day += 1
            self.buy_num_day += Q
            self.buy_p_day += P * Q

        elif self.awi.is_last_level:
            Q = self.awi.current_exogenous_output_quantity
            T = self.awi.current_step
            P = self.awi.current_exogenous_output_price
            for t in range(T, self.awi.n_steps):
                self.raw_inventory[t] -= Q
            self.sell_num[T] += Q
            self.sell_day[T] += P * Q
            self.sell_q_num += Q
            self.sell_p_sum += Q * P
            self.sell_cont_day += 1
            self.sell_num_day += Q
            self.sell_p_day += P * Q


        return super().before_step()
    def step(self):
        # if self.awi.current_step == self.awi.n_steps - 1:
        #     print(self.fac_lines_cap)
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
        if self.raw_inventory[self.awi.current_step] < 0:
            for t in range(self.awi.current_step, self.awi.n_steps):
                self.raw_inventory[t] += -1*self.raw_inventory[self.awi.current_step]
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
        with self.used:

            quantity = contract.agreement["quantity"]
            price = contract.agreement["unit_price"]
            time = contract.agreement["time"]
            if contract.partners[0] == self.id:
                neg_id = contract.partners[1]
            else:
                neg_id = contract.partners[0]
            # print(contract)
            if contract.annotation['buyer'] == self.id:
                for t in range(time, self.awi.n_steps):
                    self.raw_inventory[t] += quantity
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
                for t in range(time, self.awi.n_steps):
                    self.raw_inventory[t] -= quantity

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
                self.ip = self.sell_p_sum/self.sell_q_num
        if self.buy_q_num != 0:
            if self.op > self.buy_p_sum/self.buy_q_num + self.awi.profile.cost:
                self.op = self.buy_p_sum/self.buy_q_num + self.awi.profile.cost
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

