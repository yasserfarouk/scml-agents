# -*- coding: utf-8 -*-
import math
import statistics
from typing import Dict
import negmas
import scml

from collections import defaultdict
import random
from negmas import ResponseType
from scml.oneshot import *
from scml.utils import anac2024_oneshot

__all__ = ["RTAgent"]
#BaseAgent
class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
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
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]

        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = self._find_good_price(ami)
        return tuple(offer)

    def _find_good_price(self, ami):
        """Finds a good-enough price."""
        unit_price_issue = ami.issues[UNIT_PRICE]
        if self._is_selling(ami):
            return unit_price_issue.max_value
        return unit_price_issue.min_value

    def is_seller(self, negotiator_id):
        return negotiator_id in self.awi.current_negotiation_details["sell"].keys()

    def _needed(self, negotiator_id=None):
        return (
            self.awi.needed_sales
            if self.is_seller(negotiator_id)
            else self.awi.needed_supplies
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product


class RTAgent(SimpleAgent):
    DIST = 0.5
    #MAX_LEN = 5
    class Set_List:
        def __init__(self):
            self.evals = []

        def add_evalation(self, evalation):
            self.evals.append(evalation)

        def get_average(self):
            return statistics.mean(self.evals)
        
        def len(self):
            return len(self.evals)
        

    def __init__(self):
        super().__init__()
        self.seller_list: Dict[str, RTAgent.Set_List] = {} # buy from supplier 購入元
        self.buyer_list: Dict[str, RTAgent.Set_List] = {} # sell to consumer 販売先

    def before_step(self):
        super().before_step()
        #print('ステップ '+ str(self.awi.current_step))
        self.seller_list.clear()
        self.buyer_list.clear()
        if self.awi.level == 0:
            self.q_need = self.awi.current_exogenous_input_quantity
        elif self.awi.level == 1:
            self.q_need = self.awi.current_exogenous_output_quantity
    
    def propose(self, negotiator_id: str, state) -> "Outcome": #提案戦略 相手の提案履歴をもとに作成
        #print('提案')
        my_needs = self.q_need
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        offer = list(super().propose(negotiator_id,state))
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            if partner not in self.buyer_list:
                return tuple(offer)
            else:
                average = self.buyer_list[partner].get_average()
                if type(average) != float:
                    return tuple(offer)
                if average == 0: # 相手の要求数が多すぎる 売りすぎない
                    offer[QUANTITY] = my_needs + 1
                    offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value if offer[TIME] / nmi.n_steps < 0.7 else nmi.issues[UNIT_PRICE].min_value 
                else:
                    for i in range(-2, 0):
                        new_offer_q = my_needs + i
                        if new_offer_q >= nmi.issues[UNIT_PRICE].min_value:
                            q_evaluation = math.exp(-(new_offer_q - my_needs)**2/2*RTAgent.DIST)/math.sqrt(2*math.pi*(RTAgent.DIST**2))
                            if q_evaluation*nmi.issues[UNIT_PRICE].max_value > average: #少なくてもいいから少しでも良くなるように売る
                                offer[QUANTITY] = new_offer_q
                                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value if offer[TIME] / nmi.n_steps < 0.7 else nmi.issues[UNIT_PRICE].min_value 
                                break
        else:
            partner = nmi.annotation["seller"]
            if partner not in self.seller_list:
                return tuple(offer)
            else:
                average = self.seller_list[partner]
                if type(average) != float:
                    return tuple(offer)
                if average == 0: #大量に売ってくる 買いすぎない
                    offer[QUANTITY] = my_needs + 1
                    offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value if offer[TIME] / nmi.n_steps < 0.7 else nmi.issues[UNIT_PRICE].max_value
                else:
                    for i in range(-2, 0):
                        new_offer_q = my_needs + i 
                        if new_offer_q >= nmi.issues[UNIT_PRICE].min_value:
                            q_evaluation = math.exp(-(new_offer_q - my_needs)**2/2*RTAgent.DIST)/math.sqrt(2*math.pi*(RTAgent.DIST**2))
                            pass # print(average)
                            if q_evaluation*nmi.issues[UNIT_PRICE].max_value > average: #少なくてもいいから少しでも良くなるように買う
                                offer[QUANTITY] = new_offer_q
                                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value if offer[TIME] / nmi.n_steps < 0.7 else nmi.issues[UNIT_PRICE].max_value 
                                break                                 
        return tuple(offer)
    
    def respond(self, negotiator_id, state): #受け入れ戦略
        """取引情報の価値を保存する"""
        #print('受け入れ')
        q_evaluation = 0
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if self.q_need <= 0:
            return ResponseType.END_NEGOTIATION
        response = super().respond(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if response == ResponseType.REJECT_OFFER:
            q_evaluation = math.exp(-(offer[QUANTITY]-self.q_need)**2/2*RTAgent.DIST)/math.sqrt(2*math.pi*(RTAgent.DIST**2)) if offer[QUANTITY] - self.q_need <= 1 else 0 #多すぎるオファーは拒否
            if self._is_selling(nmi): #売るときは高いほうがいい
                partner = nmi.annotation["buyer"]
                #print('buyer '+ str(partner) +' 数 '+str(offer[QUANTITY])+ ' 値段 '+ str(offer[UNIT_PRICE]))
                if partner not in self.buyer_list:
                    self.buyer_list[partner] = RTAgent.Set_List()
                    self.buyer_list[partner].add_evalation(q_evaluation * offer[UNIT_PRICE])
                    return response
                else:
                    #if self.buyer_list[partner].len() == RTAgent.MAX_LEN: #5回分の取引価値を保存
                    last_ave = self.buyer_list[partner].get_average()
                    self.buyer_list[partner].add_evalation(q_evaluation * offer[UNIT_PRICE])
                    if self.buyer_list[partner].get_average() > last_ave: #平均がよくなる→相手が譲歩した
                        return ResponseType.ACCEPT_OFFER
                    else:
                        return response 

            else: #買うときは安いほうがいい
                partner = nmi.annotation["seller"]
                #print('seller '+ str(partner) +' 数 '+str(offer[QUANTITY])+ ' 値段 '+ str(offer[UNIT_PRICE]))
                if partner not in self.seller_list:
                    self.seller_list[partner] = RTAgent.Set_List()
                    self.seller_list[partner].add_evalation(-1 * (1 - q_evaluation) * offer[UNIT_PRICE])
                    return response
                else:
                    #if self.seller_list[partner].len() == RTAgent.MAX_LEN: #5回分の取引価値を保存
                    last_ave = self.seller_list[partner].get_average()
                    self.seller_list[partner].add_evalation(-1 * (1 - q_evaluation) * offer[UNIT_PRICE])
                    if self.seller_list[partner].get_average() > last_ave:
                        return ResponseType.ACCEPT_OFFER  
                    else:
                        return response                  
        else:
            good_price = super()._find_good_price(nmi)
            if self._is_selling(nmi):
                partner = nmi.annotation["buyer"]
                #print('goodbuyer '+ str(partner) +' 数 '+str(offer[QUANTITY])+ ' 値段 '+ str(offer[UNIT_PRICE]))
                if partner not in self.buyer_list:
                    self.buyer_list[partner] = RTAgent.Set_List()
                    self.buyer_list[partner].add_evalation(q_evaluation*good_price)
                    #return ResponseType.REJECT_OFFER
                else:
                    self.buyer_list[partner].add_evalation(q_evaluation*good_price)
            else:
                partner = nmi.annotation["seller"]
                #print('goodseller '+ str(partner) +' 数 '+str(offer[QUANTITY])+ ' 値段 '+ str(offer[UNIT_PRICE]))
                if partner not in self.seller_list:
                    self.seller_list[partner] = RTAgent.Set_List()
                    self.seller_list[partner].add_evalation(-1 * (1 - q_evaluation) * good_price)
                    #return ResponseType.REJECT_OFFER
                else:
                    self.seller_list[partner].add_evalation(-1 * (1 - q_evaluation) * good_price)
            return response


    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)    
        #update q_need
        self.q_need -= contract.agreement["quantity"] #必要量の更新