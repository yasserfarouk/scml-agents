from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from negmas import LinearUtilityFunction
from scml import SupplyDrivenProductionStrategy
from scml.scml2020 import SCML2020Agent, SCML2023World
from scml.scml2020.components import IndependentNegotiationsManager
from scml.scml2020.components.trading import (
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
)

__all__ = ["AgentSDH"]


class AgentSDHStd(
    IndependentNegotiationsManager,
    SupplyDrivenProductionStrategy,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
):
    def init(self):
        self.buy_costs = []
        self.buy_num = 0
        self.sell_costs = []
        self.sell_num = 0
        self.ave_buy = 0
        self.ave_sell = 0
        self.buy_flag = 0
        self.sell_flag = 0
        self.cost = self.awi.profile.costs[0][self.awi.my_input_product]
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product] - 1
        self.in_schedule = [0 for _ in range(self.awi.n_steps)]
        self.out_schedule = [0 for _ in range(self.awi.n_steps)]
        self.inve_schedule = [0 for _ in range(self.awi.n_steps)]
        self.state_lines = [0 for _ in range(self.awi.n_steps)]
        self.inputs = [0 for _ in range(self.awi.n_steps)]
        self.outputs = [0 for _ in range(self.awi.n_steps)]
        self.estimated_outputs = [0 for _ in range(self.awi.n_steps)]
        self.cost = self.awi.profile.costs[0][self.awi.my_input_product]
        self.highest_price = self.awi.catalog_prices[self.awi.my_input_product]
        self.lowest_price = (
            self.awi.catalog_prices[self.awi.my_input_product] + self.cost
        ) * 1.05
        self.balance = 0
        self.n_steps = 0
        super().init()

    def on_contracts_finalized(self, signed, cancelled, rejectors) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        for c in signed:
            q = c.agreement["quantity"]
            t = c.agreement["time"]
            if c.annotation["buyer"] == self.id:
                self.in_schedule[t] += q
                self.inputs[t] += q
                self.balance -= c.agreement["unit_price"] * q
                self.buy_costs.append(c.agreement["unit_price"] * q)
                self.buy_num += q
                if self.in_schedule[t] <= self.awi.n_lines:
                    self.inve_schedule[t + 1 :] = [
                        x + q for x in self.inve_schedule[t + 1 :]
                    ]
                else:
                    if t + 1 < self.awi.n_steps:
                        self.in_schedule[t + 1] += (
                            self.in_schedule[t] - self.awi.n_lines
                        )
                        self.inputs[t + 1] += self.in_schedule[t] - self.awi.n_lines
                        self.in_schedule[t] = self.awi.n_lines
                        self.inputs[t] = self.awi.n_lines
                        self.inve_schedule[t + 1 :] = [
                            x + self.awi.n_lines for x in self.inve_schedule[t + 1 :]
                        ]
                        self.inve_schedule[t + 2 :] = [
                            x + (q - self.awi.n_lines)
                            for x in self.inve_schedule[t + 2 :]
                        ]
            else:
                self.out_schedule[t] += q
                self.balance += c.agreement["unit_price"] * q
                self.sell_costs.append(c.agreement["unit_price"] * q)
                self.sell_num += q

                self.inve_schedule[t:] = [x - q for x in self.inve_schedule[t:]]

                sell_q = q
                for j in range(t - 1, self.awi.current_step, -1):
                    if self.inputs[j] != 0:
                        if self.inputs[j] < sell_q:
                            self.inputs[j] = 0
                            self.inve_schedule[j + 1 : t] = [
                                x - self.in_schedule[j]
                                for x in self.inve_schedule[j + 1 : t]
                            ]
                            sell_q -= self.in_schedule[j]
                        elif j != t - 1:
                            self.inputs[j] -= sell_q
                            self.inve_schedule[j + 1 : t] = [
                                x - sell_q for x in self.inve_schedule[j + 1 : t]
                            ]
                            break
                    if sell_q <= 0:
                        break

    def sign_all_contracts(self, contracts):
        inventory = self.inve_schedule.copy()
        input = self.in_schedule.copy()
        ins = self.inputs.copy()
        buy = []
        sell = []
        signed = [None] * len(contracts)
        for c in contracts:
            if self.awi.current_step <= c.agreement["time"] < self.awi.n_steps:
                if c.annotation["buyer"] == self.id:
                    buy.append(c)

                else:
                    sell.append(c)
        buy = sorted(buy, key=lambda x: x.agreement["unit_price"])
        sell = sorted(sell, key=lambda x: x.agreement["unit_price"], reverse=True)
        ave_buy = 0

        if self.buy_num != 0:
            # ave_buy = sum((lambda x:(x.agreement["unit_price"] * x.agreement["quantity"]) for x in self.buy_costs)) / (len(buy)*sum((lambda x: x.agreement["quantity"] for x in self.buy_costs)))
            ave_buy = sum(self.buy_costs) / self.buy_num
        sell_q = 0
        for i in sell:
            if self.awi.current_step <= c.agreement["time"] < self.awi.n_steps:
                q = i.agreement["quantity"]
                t = i.agreement["time"]
                if q <= inventory[t]:
                    if i.agreement["unit_price"] - self.cost > ave_buy:
                        signed[contracts.index(i)] = self.id
                        self.sell_flag = 1
                        inventory[t:] = [x - q for x in inventory[t:]]
                        sell_q = q
                        for j in range(t - 1, self.awi.current_step, -1):
                            if ins[j] != 0:
                                if ins[j] < sell_q:
                                    ins[j] = 0
                                    inventory[j + 1 : t] = [
                                        x - self.in_schedule[j]
                                        for x in inventory[j + 1 : t]
                                    ]
                                    sell_q -= self.in_schedule[j]
                                elif j != t - 1:
                                    ins[j] -= sell_q
                                    inventory[j + 1 : t] = [
                                        x - sell_q for x in inventory[j + 1 : t]
                                    ]
                                    break
                                if sell_q <= 0:
                                    break
                    else:
                        break

        if self.sell_num != 0:
            ave_sell = sum(self.sell_costs) / self.sell_num
            need_buy = sum(self.out_schedule[: self.awi.current_step]) / len(
                self.out_schedule[: self.awi.current_step]
            )
            if need_buy > self.awi.n_lines:
                need_buy = self.awi.n_lines
        else:
            ave_sell = self.awi.catalog_prices[self.awi.my_output_product] + 3
            need_buy = self.awi.n_lines

        for i in buy:
            q = i.agreement["quantity"]
            t = i.agreement["time"]
            if inventory[-1] > self.awi.n_lines * 4:
                break
            if t < self.awi.n_steps * 0.9:
                if need_buy >= i.agreement["quantity"] + inventory[t]:
                    if i.agreement["unit_price"] + self.cost < ave_sell:
                        self.buy_flag = 1
                        signed[contracts.index(i)] = self.id
                        input[t] += q
                        inventory[t + 1 :] = [x + q for x in inventory[t + 1 :]]
                elif t + 2 < self.awi.n_steps - 1:
                    if (
                        i.agreement["unit_price"] <= (ave_sell - self.cost) * 0.7
                        or i.agreement["unit_price"] <= ave_buy * 0.7
                    ):
                        self.buy_flag = 1
                        signed[contracts.index(i)] = self.id
                        input[t] += q
                        if input[t] <= self.awi.n_lines:
                            inventory[t + 1 :] = [x + q for x in inventory[t + 1 :]]
                        else:
                            input[t + 1] += input[t] - self.awi.n_lines
                            input[t] = self.awi.n_lines
                            inventory[t + 1 :] = [
                                x + self.awi.n_lines for x in inventory[t + 1 :]
                            ]
                            inventory[t + 2 :] = [
                                x + q - self.awi.n_lines for x in inventory[t + 2 :]
                            ]
        self.ave_buy = ave_buy
        self.ave_sell = ave_sell
        return signed

    def step(self):
        self.update_price()
        super().step()

    def update_price(self):
        # if self.ip ==0 :
        #    self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        if self.buy_flag > 0:
            if self.ave_buy / self.ip > 0.9:
                self.ip *= 1.05
            else:
                self.ip *= 0.95
        else:
            self.ip *= 1.1
        # if self.op ==0:
        #    self.op = self.awi.catalog_prices[self.awi.my_output_product]-1
        if self.sell_flag > 0:
            if self.ave_sell / self.op > 1.1:
                # self.op = max(self.op*1.05, self.lowest_price)
                self.op = self.op * 1.1
            else:
                # self.op = max(self.ave_sell * 0.9,self.lowest_price)
                self.op = self.ave_sell * 0.9
        else:
            self.op *= 0.9
        # print(self.ip)
        # print(self.op)

    def acceptable_unit_price(self, step, sell):
        if sell:
            return int(self.op)
        return int(self.ip)

    def target_quantity(self, step, sell):
        if sell:
            return self.inve_schedule[step] / 3
        return self.awi.n_lines / 3

    def create_ufun(self, is_seller, issues, outcomes):
        if is_seller:
            # (q,t,p)?
            return LinearUtilityFunction((3, 0, 2), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((1, 0, -2), issues=issues, outcomes=outcomes)


class AgentSDH(AgentSDHStd):
    our_id = set()

    def init(self):
        super().init()
        self.our_id.add(self.id)

    def target_quantity(self, step, sell):
        if sell:
            if len(self.our_id) > 1:
                self.n_steps += 1
                return self.awi.n_lines / 3
            return self.inve_schedule[step] / 3
        return self.awi.n_lines / 3

    def before_step(self):
        if self.awi.current_step == self.awi.n_steps - 1:
            self.our_id.discard(self.id)
        return super().before_step()