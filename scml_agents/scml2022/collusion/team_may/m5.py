import numpy as np
from negmas import LinearUtilityFunction
from scml.scml2020 import SCML2020Agent
from scml import SupplyDrivenProductionStrategy
from scml.scml2020.components import IndependentNegotiationsManager

__all__ = ["M5Collusion"]

# Standard Strategy
class M5Std(
    IndependentNegotiationsManager, SupplyDrivenProductionStrategy, SCML2020Agent
):
    def init(self):
        self.n_shipments = 0
        self.n_arrivals = 0
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product] - 1
        self.buyschedule = [0 for _ in range(self.awi.n_steps)]
        self.sellschedule = [0 for _ in range(self.awi.n_steps)]
        self.inputs = [0 for _ in range(self.awi.n_steps)]
        self.outputs = [0 for _ in range(self.awi.n_steps)]
        self.estimated_outputs = [0 for _ in range(self.awi.n_steps)]
        self.initbal = self.awi.state.balance
        self.balance = self.awi.state.balance
        self.cost = self.awi.profile.costs[0][self.awi.my_input_product]
        self.breachflg = [0, 0]
        self.min_score = -0.25 * self.awi.n_steps / 200
        self.slope = 0.55 * self.awi.n_steps / 200
        self.purchase_period = 0.6
        self.agreement_rate = 0.5
        self.highest_price = self.awi.catalog_prices[self.awi.my_input_product]
        self.lowest_price = (
            self.awi.catalog_prices[self.awi.my_input_product] + self.cost
        ) * 1.05
        super().init()

    def step(self):
        self.update_price()
        super().step()

    def sign_all_contracts(self, contracts):
        # future inventory
        self.inputs[self.awi.current_step] = self.awi.current_inventory[
            self.awi.my_input_product
        ]
        self.estimated_outputs[self.awi.current_step] = self.awi.current_inventory[
            self.awi.my_output_product
        ]
        self.breachflg = [0, 0]
        for i in range(self.awi.current_step + 1, self.awi.n_steps):
            self.inputs[i] = max(
                self.inputs[i - 1] + self.buyschedule[i - 1] - self.awi.n_lines, 0
            )
            tmp = (
                self.estimated_outputs[i - 1]
                + min(self.inputs[i - 1] + self.buyschedule[i - 1], self.awi.n_lines)
                - self.sellschedule[i - 1]
            )
            if tmp < 0:
                self.breachflg = [i, -tmp]
            self.estimated_outputs[i] = max(tmp, 0)

        self.outputs[self.awi.current_step] = self.awi.current_inventory[
            self.awi.my_input_product
        ]
        inputs = self.awi.current_inventory[self.awi.my_input_product]
        for i in range(self.awi.current_step + 1, self.awi.n_steps):
            self.outputs[i] = (
                self.estimated_outputs[i - 1]
                + min(inputs, self.awi.n_lines)
                - self.sellschedule[i - 1]
            )
            inputs = max(inputs - self.awi.n_lines, 0)

        # signing strategy
        signed = [None] * len(contracts)
        buy, sell = [], []
        qb = [0 for _ in range(self.awi.n_steps)]
        qs = [0 for _ in range(self.awi.n_steps)]
        for c in contracts:
            if self.awi.current_step <= c.agreement["time"] < self.awi.n_steps:
                if c.annotation["buyer"] == self.id:
                    buy.append(c)
                else:
                    sell.append(c)
        # buy  = sorted(buy,  key=lambda x: x.agreement["unit_price"])
        # sell = sorted(sell, key=lambda x: -x.agreement["unit_price"])
        dbal = 0
        for b in buy:
            [q, t] = [b.agreement["quantity"], b.agreement["time"]]
            arrivals = sum(self.buyschedule[t + 1 :])
            productivity = self.awi.n_lines * (self.awi.n_steps - t - 1)
            if (
                self.inputs[t] + arrivals + sum(qb) + q <= productivity
                and (self.balance - dbal - self.initbal) / self.initbal
                > self.min_score + self.slope * self.awi.current_step / self.awi.n_steps
                and t < self.awi.n_steps * self.purchase_period
                and self.op > self.lowest_price
            ) or (
                t <= self.breachflg[0]
                and sum(qb[:t]) * self.agreement_rate < self.breachflg[1]
            ):
                dbal += (b.agreement["unit_price"] + self.cost) * q
                signed[contracts.index(b)] = self.id
                qb[t] += q
        for s in sell:
            [q, t] = [s.agreement["quantity"], s.agreement["time"]]
            if self.outputs[t] >= sum(self.sellschedule[t:]) + sum(qs) + q or (
                self.awi.my_input_product == 0
                and self.estimated_outputs[t]
                >= sum(self.sellschedule[t:]) + sum(qs) + q
            ):
                signed[contracts.index(s)] = self.id
                qs[t] += q
        return signed

    def update_price(self):
        # acceptable input price
        if self.n_arrivals > 0:
            self.ip *= 0.95
        else:
            self.ip = min(self.ip * 1.05, self.highest_price)
        # acceptable output price
        if self.awi.current_step / self.awi.n_steps < 0.2 or self.n_shipments > 0:
            self.op = min(
                self.op * 1.05, self.awi.catalog_prices[self.awi.my_output_product] - 1
            )
        elif self.awi.current_step / self.awi.n_steps < 0.7:
            self.op = max(self.op * 0.95, self.lowest_price)
        else:
            self.op = max(
                self.op * 0.95,
                self.awi.catalog_prices[self.awi.my_output_product] * 0.55,
            )
        self.n_arrivals, self.n_shipments = 0, 0

    def acceptable_unit_price(self, step, sell):
        if sell:
            return int(self.op)
        return int(self.ip)

    def target_quantity(self, step, sell):
        return 2 * self.awi.n_lines

    def create_ufun(self, is_seller, issues, outcomes):
        if is_seller:
            return LinearUtilityFunction(
                (10, 2, 1), issues=issues, outcomes=outcomes
            )  # (q,t,p)?
        return LinearUtilityFunction((1, -2, -4), issues=issues, outcomes=outcomes)

    def on_contracts_finalized(self, signed, cancelled, rejectors):
        for c in signed:
            if self.awi.current_step <= c.agreement["time"] < self.awi.n_steps:
                if c.annotation["seller"] == self.id:
                    self.sellschedule[c.agreement["time"]] += c.agreement["quantity"]
                    self.balance += c.agreement["unit_price"] * c.agreement["quantity"]
                    self.n_shipments += c.agreement["quantity"]
                elif c.annotation["buyer"] == self.id:
                    self.buyschedule[c.agreement["time"]] += c.agreement["quantity"]
                    self.balance -= (
                        c.agreement["unit_price"] + self.cost
                    ) * c.agreement["quantity"]
                    self.n_arrivals += c.agreement["quantity"]

    def on_agent_bankrupt(self, agent, contracts, quantities, compensation_money):
        for c in contracts:
            if c.agreement["time"] >= self.awi.current_step:
                if c.annotation["seller"] == self.id:
                    self.sellschedule[c.agreement["time"]] -= c.agreement["quantity"]
                    self.balance -= c.agreement["unit_price"] * c.agreement["quantity"]
                elif c.annotation["buyer"] == self.id:
                    self.buyschedule[c.agreement["time"]] -= c.agreement["quantity"]
                    self.balance += (
                        c.agreement["unit_price"] + self.cost
                    ) * c.agreement["quantity"]


# Collusion Strategy
class M5Collusion(M5Std):
    # Shared Information
    my_friends = dict()
    total_initbal = 0

    def init(self):
        super().init()
        self.my_friends[self.id] = dict(
            level=self.awi.my_input_product,
            cost=self.cost,
            current_balance=self.initbal,
        )
        self.total_initbal += self.initbal
        self.dcost = 0

    def total_balance(self):
        return sum([d["current_balance"] for d in self.my_friends.values()])

    def is_collusion(self):
        return len(self.my_friends) > 1

    def step(self):
        if self.awi.current_step == 0:
            min_cost = self.cost
            for d in self.my_friends.values():
                if d["level"] == self.awi.my_input_product and min_cost > d["cost"]:
                    min_cost = d["cost"]
            self.dcost = self.cost - min_cost

            if self.is_collusion():
                self.min_score = -0.3 * self.awi.n_steps / 200
                self.slope = 0.7 * self.awi.n_steps / 200
                self.highest_price = max(
                    self.awi.catalog_prices[self.awi.my_input_product] - self.dcost, 1
                )

        super().step()
        self.my_friends[self.id].update(dict(current_balance=self.balance))
