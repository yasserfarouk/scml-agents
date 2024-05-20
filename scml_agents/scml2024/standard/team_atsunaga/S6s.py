from __future__ import annotations
from negmas.sao import SAONMI
import uuid


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
class InventoryManager:
    def __init__(self, agent):
        self.agent = agent
        self.products = [[] for _ in range(agent.awi.n_steps)]
        self.raw_materials = [0 for _ in range(agent.awi.n_steps)]

    def add_product(self, product):
        for i in range(product.production_time, self.agent.awi.n_steps):
            self.products[i].append(product)

    def remove_product(self, product):
        for i in range(product.sell_time, self.agent.awi.n_steps):
            if product in self.products[i]:
                self.products[i].remove(product)

    def add_raw_material(self, quantity, time):
        for i in range(time, self.agent.awi.n_steps):
            self.raw_materials[i] += quantity

    def remove_raw_material(self, quantity, time):
        self.raw_materials[time] -= quantity

class Product:
    def __init__(self, product_id, production_time, quantity, buy_price, sell_price=None, sell_time=None):
        self.id = product_id
        self.production_time = production_time
        self.quantity = quantity
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.sell_time = sell_time

class NegotiationManager:
    def __init__(self, agent):
        self.agent = agent
        self.buy_negotiations = {}
        self.sell_negotiations = {}
        self.negotiations = {}  # 追加

    def add_negotiation(self, negotiation, is_buy):
        if is_buy:
            self.buy_negotiations[negotiation.id] = negotiation
        else:
            self.sell_negotiations[negotiation.id] = negotiation
        self.negotiations[negotiation.id] = negotiation  # 追加

    def remove_negotiation(self, negotiation, is_buy):
        if is_buy:
            del self.buy_negotiations[negotiation.id]
        else:
            del self.sell_negotiations[negotiation.id]
        del self.negotiations[negotiation.id]  # 追加

class Negotiation:
    def __init__(self, negotiation_id, is_buy, quantity=0, price=0, time=0):
        self.id = negotiation_id
        self.is_buy = is_buy
        self.quantity = quantity
        self.price = price
        self.time = time
        self.counter = 0  # 追加

class S6s(StdAgent):

    def init(self):
        super().init()
        self.inventory_manager = InventoryManager(self)
        self.negotiation_manager = NegotiationManager(self)
        self.process_cost = self.awi.profile.cost
        self.production_capacity = [self.awi.n_lines for _ in range(self.awi.n_steps)]
        self.neg_place = [[] for _ in range(self.awi.n_steps)]
        self.less_inve = []
        self.buy_inventory = [0 for _ in range(self.awi.n_steps)]
        self.buy_num_day = 0
        self.buy_cont_day = 0
        self.sell_num_day = 0
        self.sell_cont_day = 0

    def propose(self, negotiator_id: str, state):
        nmi = self.get_nmi(negotiator_id)
        offer_list = []
        q = 0
        t = 0
        p = 0
        negotiation = None
        for nego in self.negotiation_manager.buy_negotiations.values():
            if nego.id == negotiator_id:
                negotiation = nego
                break
        if negotiation is None:
            for nego in self.negotiation_manager.sell_negotiations.values():
                if nego.id == negotiator_id:
                    negotiation = nego
                    break
        if negotiation is None:
            negotiation = Negotiation(negotiator_id, negotiator_id in self.awi.my_suppliers)
            self.negotiation_manager.add_negotiation(negotiation, negotiator_id in self.awi.my_suppliers)

        if negotiator_id in self.awi.my_suppliers:
            highest_price = int(self.awi.trading_prices[self.awi.my_output_product] - self.process_cost)
            lowest_price = int(min(self.awi.current_input_issues[0].values[0], self.awi.trading_prices[self.awi.my_input_product] * 0.7))
            n = self.negotiation_manager.negotiations[negotiator_id].counter // 5

            if len(self.less_inve) > 0:
                if state.step < 5:
                    t = random.randint(self.awi.current_step, self.less_inve[0].sell_time)
                    q = self.less_inve[0].quantity
                    p = random.randint(lowest_price, highest_price)
                    offer = self.check_proposal([q, t, p], negotiator_id)
                    return tuple(offer)
                else:
                    p = random.randint(lowest_price, highest_price)
                    for time in negotiation.time:
                        if time < self.less_inve[n].sell_time:
                            for quantity in negotiation.quantity:
                                if quantity <= self.less_inve[n].quantity + 2:
                                    offer = self.check_proposal([quantity, time, p], negotiator_id)
                                    offer_list.append(tuple(offer))
                                    break
                    offer_list = sorted(offer_list, key=lambda x: x[1])
                    self.negotiation_manager.negotiations[negotiator_id].counter += 1
                    return offer_list[n]
            else:
                need_buy = 0
                if len(self.inventory_manager.products[-1]) > self.awi.n_lines * 2:
                    return None
                if len(self.buy_inventory[self.awi.current_step:]) > 0:
                    need_buy = sum(self.buy_inventory[self.awi.current_step:]) / len(self.buy_inventory[self.awi.current_step:])
                    if need_buy > self.awi.n_lines:
                        need_buy = self.awi.n_lines
                else:
                    need_buy = self.awi.n_lines
                for i in range(self.awi.current_step, self.awi.n_steps):
                    if len(self.inventory_manager.products[i]) < need_buy:
                        t = i
                        q = need_buy - len(self.inventory_manager.products[i])
                        p = random.randint(lowest_price, highest_price)
                        break
                offer = self.check_proposal([q, t, p], negotiator_id)
                return tuple(offer)
        else:
            much_inve = sorted(self.inventory_manager.products[self.awi.current_step:], key=len, reverse=True)
            indexed_lists = [(index, sublist) for index, sublist in enumerate(self.inventory_manager.products[self.awi.current_step:])]
            sorted_indexed_lists = sorted(indexed_lists, key=lambda x: len(x[1]), reverse=True)
            much_inve = [index for index, sublist in sorted_indexed_lists]

            t = much_inve[state.step // 5]
            q = min(int(len(self.inventory_manager.products[t]) / 3), self.awi.n_lines)
            p = random.randint(int(self.awi.trading_prices[self.awi.my_output_product] * 0.8), int(self.awi.trading_prices[self.awi.my_output_product] * 1.5))
            return tuple([q, t, p])

    def respond(self, negotiator_id: str, state: SAOState, source="") -> ResponseType:
        negotiation = None
        for nego in self.negotiation_manager.buy_negotiations.values():
            if nego.id == negotiator_id:
                negotiation = nego
                break
        if negotiation is None:
            for nego in self.negotiation_manager.sell_negotiations.values():
                if nego.id == negotiator_id:
                    negotiation = nego
                    break
        if negotiation is None:
            negotiation = Negotiation(negotiator_id, negotiator_id in self.awi.my_suppliers)
            self.negotiation_manager.add_negotiation(negotiation, negotiator_id in self.awi.my_suppliers)

        negotiation.quantity = state.current_offer[QUANTITY]
        negotiation.price = state.current_offer[UNIT_PRICE]
        negotiation.time = state.current_offer[TIME]

        Q = state.current_offer[0]
        T = state.current_offer[1]
        P = state.current_offer[2]
        if T > self.awi.n_steps - 1:
            return ResponseType.REJECT_OFFER
        if negotiator_id in self.awi.my_consumers:
            if T == 0:
                return ResponseType.REJECT_OFFER
            elif len(self.inventory_manager.products[T]) >= Q:
                if self.inventory_manager.products[T][0].buy_price + self.process_cost + self.inventory_manager.products[T][0].buy_price * 0.01 < 0.9 * P:
                    return ResponseType.ACCEPT_OFFER
                elif self.awi.current_step / self.awi.n_steps > 0.8:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            else:
                if P > self.awi.trading_prices[self.awi.my_output_product] * 1.2 and T - self.awi.current_step > self.awi.n_steps / 5:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER
        else:
            if len(self.inventory_manager.products[self.awi.n_steps - 1]) >= self.awi.n_lines * 1.5:
                return ResponseType.REJECT_OFFER
            if len(self.less_inve) > 0:
                if self.less_inve[0].sell_time == self.awi.current_step:
                    if P < self.awi.trading_prices[self.awi.my_input_product] * 1.2:
                        return ResponseType.ACCEPT_OFFER
            if T > self.awi.n_steps / 3 * 2:
                if P > self.awi.trading_prices[self.awi.my_input_product] - 5:
                    return ResponseType.REJECT_OFFER
                else:
                    return ResponseType.ACCEPT_OFFER
            else:
                count_positive = len([x for x in self.buy_inventory if x > 0])
                if count_positive == 0:
                    if P < self.awi.trading_prices[self.awi.my_input_product] * 1.2:
                        return ResponseType.ACCEPT_OFFER
                elif len(self.inventory_manager.products[T]) >= self.sell_num_day / count_positive:
                    return ResponseType.REJECT_OFFER
                else:
                    if P < self.awi.trading_prices[self.awi.my_input_product] * 1.1:
                        return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

    def on_negotiation_success(self, contract, mechanism):
        is_buy = contract.annotation["buyer"] == self.id
        quantity = contract.agreement["quantity"]
        price = contract.agreement["unit_price"]
        time = contract.agreement["time"]

        if is_buy:
            self.inventory_manager.add_raw_material(quantity, time)
            for i in range(time, self.awi.n_steps - 1):
                if quantity <= 0:
                    break
                if quantity > self.production_capacity[i]:
                    produced_quantity = self.production_capacity[i]
                    quantity -= produced_quantity
                    self.production_capacity[i] = 0
                else:
                    produced_quantity = quantity
                    self.production_capacity[i] -= quantity
                    quantity = 0
                for _ in range(produced_quantity):
                    product = Product(self.generate_product_id(), time, 1, price)
                    self.inventory_manager.add_product(product)
        else:
            products_to_sell = self.inventory_manager.products[time][:quantity]
            for product in products_to_sell:
                if product.sell_time is not None:  # 追加
                    self.inventory_manager.remove_product(product)
                    product.sell_price = price
                    product.sell_time = time



    def step(self):
        self.current_finances = self.awi.current_balance
        if self.buy_num_day > 0:
            self.buy_day = self.buy_cont_day / self.buy_num_day
        if self.sell_num_day > 0:
            self.sell_day = self.sell_cont_day / self.sell_num_day
        self.buy_num_day = 0
        self.buy_cont_day = 0
        self.sell_num_day = 0
        self.sell_cont_day = 0
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


    def generate_product_id(self):
        return str(uuid.uuid4())

