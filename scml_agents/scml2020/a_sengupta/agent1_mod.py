"""
**Submitted to ANAC 2020 SCML**
*Authors* Ayan Sengupta (a-sengupta), email: a-sengupta@nec.com

Agent Information
-----------------

  - Agent Name: Merchant
  - Team Name: a-sengupta
  - Affiliation: NEC Corporation
  - Country: Japan
  - Team Members:
    1. Ayan Sengupta <a-sengupta@nec.com>

"""
import copy
import itertools
from abc import ABC
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Contract,
    Issue,
    LinearUtilityFunction,
    Negotiator,
    ResponseType,
    SAONegotiator,
    ToughNegotiator,
)
from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.helpers import get_class, instantiate
from negmas.outcomes.base_issue import make_issue
from scml.scml2020 import (
    DecentralizingAgent,
    DoNothingAgent,
    NegotiationManager,
    RandomAgent,
    SCML2020Agent,
    SCML2020World,
    Simulation,
)

# required for development
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)
from scml.scml2020.components import (
    IndependentNegotiationsManager,
    NegotiationManager,
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
    SignAll,
    SupplyDrivenProductionStrategy,
    TradingStrategy,
)

__all__ = ["Merchant"]


QUANTITY = 0
TIME = 1
PRICE = 2
row_list = []


class ToughAspirationNegotiator(AspirationNegotiator):
    def __init__(
        self,
        brother=False,
        threshold=0.8,
        behave="natural",
        all_money=10000,
        exec_time=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.brother = brother
        self.threshold = threshold
        self.behave = behave
        self.all_money = all_money
        self.exec_time = exec_time

    def respond(self, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.brother == True and offer[0] == 1 and offer[2] > 1000:
            return ResponseType.ACCEPT_OFFER
        if self.brother == True and offer[2] == 1 and offer[0] == 2:
            return ResponseType.ACCEPT_OFFER

        if self.ufun_max is None or self.ufun_min is None:
            self.on_preferences_changed(
                [PreferencesChange(PreferencesChangeType.General)]
            )
        if self.ufun is None:
            return ResponseType.REJECT_OFFER
        u = self.ufun(offer)
        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER
        asp = (
            self.utility_at(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state):
        if self.behave == "suicidal":
            return (1, self.exec_time, self.all_money)
        if self.behave == "help_me":
            return (2, self.exec_time, 1)

        asp = (
            self.utility_at(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )

        if asp < self.threshold and self.brother == False:
            asp = self.threshold
        if self.presorted:
            if len(self.ordered_outcomes) < 1:
                return None
            for i, (u, o) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.ordered_outcomes[i][1]
                    return self.ordered_outcomes[i - 1][1]
            return self.ordered_outcomes[-1][1]


class GryffindorIndependentNegotiationsManager(IndependentNegotiationsManager):
    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = ToughAspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

    def step(self):
        my_brother_consumers = []
        my_brother_suppliers = []
        price_max = []

        def exploit_condition():
            for agent in self.awi.my_consumers:
                if agent in self.brothers and agent not in my_brother_consumers:
                    for item in self.brothers_nt:
                        if item.id == agent:
                            price_max.append(item.balance)
                    my_brother_consumers.append(agent)

            for agent in self.awi.my_suppliers:
                if agent in self.brothers and agent not in my_brother_suppliers:
                    my_brother_suppliers.append(agent)

            if len(my_brother_consumers) > 0:
                return 1
            else:
                return 2

        price_range = (
            int(self.awi.catalog_prices[self.awi.my_output_product] * 0),
            int(self.awi.catalog_prices[self.awi.my_output_product] * 1.5),
        )
        if exploit_condition() == 1:
            price_range = (1, int(min(price_max)))

        elif exploit_condition() == 2:
            price_range = (1, int(self.awi.state.balance))

        def test_proximity():
            _ = []
            for item in self.brothers_nt:
                _.append(item.in_product)
            if self.awi.my_input_product + 1 in _:
                return False
            if self.awi.my_input_product - 1 in _:
                return False
            return True

        def same_level_brother():
            for item in self.brothers_nt:
                if item.in_product == self.awi.my_input_product and item.id != self.id:
                    return True

            return False

        def check_profit():
            if self.awi.current_step > 1:
                if self.awi.state.balance == self.init_balance:
                    return False
                if self.awi.state.balance > self.init_balance:
                    return True

        # (my_brother_suppliers)
        if (
            len(my_brother_suppliers) > 0
            and same_level_brother()
            and not check_profit()
            and self.awi.current_step > 3
        ):
            if (
                self.awi.state.inventory[self.awi.my_input_product]
                + self.awi.state.inventory[self.awi.my_output_product]
                < 10
            ):
                self.neg_extras = {
                    "threshold": 0.8,
                    "behave": "help_me",
                    "exec_time": self.awi.current_step + 1,
                    "brother": True,
                    "all_money": int(self.awi.state.balance * 0.8)
                    // (len(my_brother_suppliers)),
                }
                self._start_negotiations(
                    self.awi.my_input_product,
                    False,
                    self.awi.current_step + 1,
                    (2, 2),
                    (1, 1),
                    (self.awi.current_step + 1, self.awi.current_step + 2),
                    my_brother_suppliers,
                )
            else:
                self._start_negotiations(
                    self.awi.my_output_product,
                    True,
                    self.awi.current_step + 1,
                    (1, 2),
                    (
                        int(self.awi.catalog_prices[self.awi.my_output_product] * 0.5),
                        self.awi.catalog_prices[self.awi.my_output_product] * 2,
                    ),
                    (self.awi.current_step + 1, self.awi.current_step + 5),
                    self.awi.my_consumers,
                )

        # standard
        if len(self.brothers) < 2 or test_proximity():
            self._start_negotiations(
                self.awi.my_input_product,
                False,
                self.awi.current_step + 1,
                (1, 100),
                (1, 2),
                (self.awi.current_step + 1, self.awi.n_steps),
                self.awi.my_suppliers,
            )

            self._start_negotiations(
                self.awi.my_output_product,
                True,
                self.awi.current_step + 1,
                (1, 2),
                (500, 1000),
                (self.awi.current_step, self.awi.n_steps),
                self.awi.my_consumers,
            )

            for _ in range(4):
                if self.did_buy == False:
                    self._start_negotiations(
                        self.awi.my_input_product,
                        False,
                        self.awi.current_step + 1,
                        (1, 10),
                        (
                            int(
                                self.awi.catalog_prices[self.awi.my_output_product] * 0
                            ),
                            int(
                                self.awi.catalog_prices[self.awi.my_output_product] * 2
                            ),
                        ),
                        (self.awi.current_step + 1, self.awi.n_steps),
                        self.awi.my_suppliers,
                    )
                if self.did_buy == False:
                    if self.awi.current_step + 2 < self.awi.n_steps - 2:
                        self._start_negotiations(
                            self.awi.my_output_product,
                            True,
                            self.awi.current_step + 1,
                            (1, 10),
                            (
                                int(
                                    self.awi.catalog_prices[self.awi.my_output_product]
                                    * 0.5
                                ),
                                int(
                                    self.awi.catalog_prices[self.awi.my_output_product]
                                    * 2.5
                                ),
                            ),
                            (self.awi.current_step + 2, self.awi.n_steps - 2),
                            self.awi.my_consumers,
                        )

            for _ in range(7):
                self._start_negotiations(
                    self.awi.my_input_product,
                    False,
                    self.awi.current_step + 1,
                    (1, 5),
                    (
                        int(self.awi.catalog_prices[self.awi.my_output_product] * 0),
                        int(self.awi.catalog_prices[self.awi.my_output_product] * 1.4),
                    ),
                    (
                        self.awi.current_step + 1,
                        min(self.awi.current_step + 3 + _, self.awi.n_steps),
                    ),
                    self.awi.my_suppliers,
                )

                if (
                    min(self.awi.current_step + 3 + _, self.awi.n_steps - 2)
                    > self.awi.current_step + 1
                ):
                    self._start_negotiations(
                        self.awi.my_output_product,
                        True,
                        self.awi.current_step + 1,
                        (1, 5),
                        (
                            int(
                                self.awi.catalog_prices[self.awi.my_output_product]
                                * 0.8
                            ),
                            int(
                                self.awi.catalog_prices[self.awi.my_output_product]
                                * 2.5
                            ),
                        ),
                        (
                            self.awi.current_step + 1,
                            min(self.awi.current_step + 3 + _, self.awi.n_steps - 2),
                        ),
                        self.awi.my_consumers,
                    )
        # collusion
        else:
            if self.awi.state.balance > 100 and len(my_brother_suppliers) > 0:
                self.neg_extras = {
                    "threshold": 0.8,
                    "behave": "suicidal",
                    "exec_time": 1,
                    "brother": True,
                    "all_money": int(self.awi.state.balance * 0.8)
                    // (len(my_brother_suppliers)),
                }
                self._start_negotiations(
                    self.awi.my_input_product,
                    False,
                    self.awi.current_step,
                    (1, 1),
                    (
                        int(self.awi.state.balance * 0.8) // (len(my_brother_suppliers))
                        - 1,
                        int(self.awi.state.balance * 0.8)
                        // (len(my_brother_suppliers)),
                    ),
                    (1, 1),
                    my_brother_suppliers,
                )

        commands = -1 * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = -1
        self.awi.set_commands(
            self.awi.my_input_product * np.ones(self.awi.n_lines, dtype=int)
        )
        if self.awi.current_step == self.awi.n_steps - 1:
            Merchant.brothers = []

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        for contract in signed:
            if contract.agreement["time"] >= self.awi.current_step:
                if contract.annotation["buyer"] == self.id:
                    self.step_wise_inventory[
                        contract.agreement["time"]
                    ] += contract.agreement["quantity"]
                    self.step_wise_buy_inventory[
                        contract.agreement["time"]
                    ] += contract.agreement["quantity"]
                    self.step_wise_balance[contract.agreement["time"]] -= (
                        contract.agreement["quantity"]
                        * contract.agreement["unit_price"]
                    )
                    self.product_bought += contract.agreement["quantity"]
                    self.did_buy = True
                else:
                    self.did_buy = False
                if contract.annotation["seller"] == self.id:
                    self.step_wise_sell_inventory[
                        contract.agreement["time"]
                    ] += contract.agreement["quantity"]
                    self.step_wise_inventory[
                        contract.agreement["time"]
                    ] -= contract.agreement["quantity"]
                    self.step_wise_balance[contract.agreement["time"]] += (
                        contract.agreement["quantity"]
                        * contract.agreement["unit_price"]
                    )
                    self.did_sell = True
                else:
                    self.did_sell = False

    def _start_negotiations(
        self, product, sell, step, qvalues, uvalues, tvalues, partners
    ):
        issues = [
            make_issue((int(qvalues[0]), int(max(qvalues))), name="quantity"),
            make_issue((int(tvalues[0]), int(max(tvalues))), name="time"),
            make_issue((int(uvalues[0]), int(max(uvalues))), name="unit_price"),
        ]

        for partner in partners:
            if partner in self.brothers:
                self.neg_extras["brother"] = True
            else:
                self.neg_extras["brother"] = False

            if sell:
                if self.did_sell:
                    self.neg_extras["threshold"] = min(
                        0.9, self.neg_extras["threshold"] + 0.05 * 2
                    )
                else:
                    self.neg_extras["threshold"] = max(
                        0.4, self.neg_extras["threshold"] - 0.05 * 4
                    )
            else:
                if self.did_buy:
                    self.neg_extras["threshold"] = min(
                        0.9, self.neg_extras["threshold"] + 0.05 * 2
                    )
                else:
                    self.neg_extras["threshold"] = max(
                        0.4, self.neg_extras["threshold"] - 0.05 * 4
                    )

            neg = self.negotiator(sell, issues=issues)

            self.awi.request_negotiation(
                is_buy=not sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                extra=self.neg_extras,
                negotiator=neg,
            )

    def respond_to_negotiation_request(self, initiator, issues, annotation, mechanism):
        if initiator not in self.brothers:
            self.negotiator_params["brother"] = False
            if len(self.brothers) < 2:
                pass
            else:
                return self.negotiator(annotation["seller"] == self.id, issues=issues)

        else:
            self.negotiator_params["brother"] = True
            return self.negotiator(annotation["seller"] == self.id, issues=issues)


class GryffindorTradingStrategy(TradingStrategy, SignAll):
    def init(self):
        super().init()
        self.step_wise_balance = [0] * self.awi.n_steps
        self.step_wise_inventory = [0] * self.awi.n_steps
        self.step_wise_buy_inventory = [0] * self.awi.n_steps
        self.step_wise_sell_inventory = [0] * self.awi.n_steps
        self.step_wise_available_inventory = [0] * self.awi.n_steps
        self.step_wise_remaining_inventory = [0] * self.awi.n_steps
        self.target = [10] * self.awi.n_steps
        self.product_bought = 0
        self.did_buy = False
        self.did_sell = False
        self.brothers += [self.id]
        self.init_balance = self.awi.state.balance
        Brother = namedtuple("Brother", ["id", "in_product", "balance"])
        self.brothers_nt += [
            Brother(
                id=self.id,
                in_product=self.awi.my_input_product,
                balance=self.awi.state.balance,
            )
        ]
        self.neg_extras = {"threshold": 0.8}

    def _greedy(self, contract_buy, contract_sell, production_cost, target):
        if (
            self.awi.current_step < contract_buy["time"] < self.awi.n_steps
            and self.awi.current_step < contract_sell["time"] < self.awi.n_steps
        ):
            if (
                contract_sell["time"] > contract_buy["time"]
                and contract_sell["unit_price"]
                > contract_buy["unit_price"] + production_cost
                and target[contract_buy["time"]] >= contract_buy["quantity"]
                and contract_buy["quantity"] > contract_sell["quantity"]
            ):
                return True

    def _over_sold_late_buy(self, buy_contract, spec):
        if buy_contract["time"] < spec[0]:
            return True

    def sign_all_contracts(self, contracts):
        # super().sign_all_contracts(contracts)

        sign = [None] * len(contracts)
        production_cost = self.awi.profile.costs[0][self.awi.profile.input_products][0]
        buy = [_ for _ in contracts if _.annotation["buyer"] == self.id]
        buy_from_brother = [
            _
            for _ in contracts
            if _.annotation["buyer"] == self.id
            and _.annotation["seller"] in self.brothers
        ]
        sell = [_ for _ in contracts if _.annotation["seller"] == self.id]
        sell_to_brother = [
            _
            for _ in contracts
            if _.annotation["seller"] == self.id
            and _.annotation["buyer"] in self.brothers
        ]
        temp_b, temp_s = [], []
        buy.sort(key=lambda x: x.agreement["unit_price"])
        sell.sort(key=lambda x: x.agreement["unit_price"], reverse=True)
        max_sell_possible = (
            (self.awi.n_steps - self.awi.current_step - 2) * 10
            + self.awi.state.inventory[self.awi.my_output_product]
            - sum(self.step_wise_sell_inventory[self.awi.current_step :])
        )

        def profit_estimate():
            basic_future_balance_change_estimate = sum(
                self.step_wise_balance[self.awi.current_step :]
            )
            total_production_cost = (
                sum(self.step_wise_buy_inventory[self.awi.current_step :])
                * production_cost
            )
            cum_buy = np.cumsum([0] + self.step_wise_buy_inventory[:-1])
            cum_sell = np.cumsum(self.step_wise_sell_inventory)
            over_sold = [i - j for i, j in zip(cum_sell, cum_buy)]
            if over_sold[-1] > 0:
                spot_market_loss = (
                    over_sold[-1]
                    * self.awi.catalog_prices[self.awi.my_input_product]
                    * 1.2
                )
            else:
                spot_market_loss = 0

            estimated_profit = (
                self.awi.state.balance
                + basic_future_balance_change_estimate
                - self.init_balance
                - total_production_cost
                - spot_market_loss
            ) / self.init_balance
            return estimated_profit

        def test_proximity():
            _ = []
            for item in self.brothers_nt:
                _.append(item.in_product)
            if self.awi.my_input_product + 1 in _:
                return False
            if self.awi.my_input_product - 1 in _:
                return False
            return True

        if len(self.brothers) < 2 or test_proximity() and profit_estimate() < 1:
            # greedy signing policy
            for b in buy:
                for s in sell:
                    if (
                        self._greedy(
                            b.agreement, s.agreement, production_cost, self.target
                        )
                        is True
                    ):
                        sign[contracts.index(b)] = self.id
                        sign[contracts.index(s)] = self.id
                        temp_b.append(b.id)
                        temp_s.append(s.id)
                        self.target[b.agreement["time"]] -= b.agreement["quantity"]
                        sell.remove(s)
                        break

            # inventory profit component
            for b in buy:
                if (
                    b.agreement["unit_price"] + production_cost
                    < self.awi.catalog_prices[self.awi.my_output_product] // 2
                ):
                    sign[contracts.index(b)] = self.id

            _ = (
                self.awi.state.inventory[self.awi.my_output_product]
                + self.awi.state.inventory[self.awi.my_input_product]
            )

            # sell extra bought products
            if (
                _ > sum(self.step_wise_sell_inventory[self.awi.current_step :])
                and sum(self.step_wise_sell_inventory[self.awi.current_step :])
                < max_sell_possible
            ):
                selling_values = [s.agreement["quantity"] for s in sell]
                cum_selling_price = np.cumsum(selling_values)
                should_sign = np.sum(cum_selling_price < _)
                for i in range(should_sign):
                    sign[contracts.index(sell[i])] = self.id
                sell = sell[should_sign:]

            # start signing something
            if (
                sum(self.step_wise_sell_inventory[self.awi.current_step :]) == 0
                and sum(self.step_wise_buy_inventory[self.awi.current_step :]) == 0
            ):
                for i, b in enumerate(buy):
                    sign[contracts.index(b)] = self.id
                    if i == 2:
                        break
                for i, b in enumerate(sell):
                    sign[contracts.index(b)] = self.id
                    if i == 2:
                        break

            # buy for over selling contracts signed
            if sum(self.step_wise_sell_inventory[self.awi.current_step + 1 :]) > sum(
                self.step_wise_buy_inventory[self.awi.current_step :]
                + self.awi.state.inventory[self.awi.my_output_product]
            ):
                cum_buy = np.cumsum([0] + self.step_wise_buy_inventory[:-1])
                cum_sell = np.cumsum(self.step_wise_sell_inventory)
                over_sold = [i - j for i, j in zip(cum_sell, cum_buy)]
                buy_specs = []
                for i, v in enumerate(over_sold):
                    if v > 0:
                        buy_specs.append([i, v])

                for specs in buy_specs:
                    for b in buy:
                        if (
                            self._over_sold_late_buy(b.agreement, specs) is True
                            and specs[1] > 0
                        ):
                            specs[1] -= b.agreement["quantity"]
                            sign[contracts.index(b)] = self.id
                            self.target[b.agreement["time"]] -= b.agreement["quantity"]
                            buy.remove(b)

            assert len(temp_b) == len(set(temp_b))
            assert len(temp_s) == len(set(temp_s))

        # for collusion
        for agent in self.awi.my_consumers:
            if agent in self.brothers:
                for i, b in enumerate(sell_to_brother):
                    sign[contracts.index(b)] = self.id
                    break
                break

        # for collusion
        for agent in self.awi.my_suppliers:
            if agent in self.brothers:
                for i, b in enumerate(buy_from_brother):
                    sign[contracts.index(b)] = self.id

        return sign


class Merchant(
    GryffindorIndependentNegotiationsManager,  # TODO: reject all negotiation requests
    GryffindorTradingStrategy,  # TODO: implement price distribution, quantity distribution
    SupplyDrivenProductionStrategy,  # Produce all the time
    SCML2020Agent,
):
    brothers = []
    brothers_nt = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_wise_balance = None
        self.step_wise_inventory = None
        self.step_wise_buy_inventory = None
        self.step_wise_sell_inventory = None
        self.step_wise_available_inventory = None
        self.step_wise_remaining_inventory = None

    """def target_quantity(self, step: int, sell: bool) -> int:

        return self.awi.n_lines"""

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """The catalog price seems OK"""
        return (
            self.awi.catalog_prices[self.awi.my_output_product]
            if sell
            else self.awi.catalog_prices[self.awi.my_input_product]
        )

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
        if is_seller:
            return LinearUtilityFunction((0, 0.1, 1), issues=issues, outcomes=outcomes)
        return LinearUtilityFunction((0, -0.1, -1), issues=issues, outcomes=outcomes)
