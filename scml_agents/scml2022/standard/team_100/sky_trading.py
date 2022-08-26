from typing import List
from typing import Optional

import numpy as np
from negmas import Contract

from scml.scml2020.common import ANY_LINE
from scml.scml2020.common import is_system_agent

# from sky_prediction import SkyPredictionStrategy
from .sky_prediction import SkyPredictionStrategy

INF = 256


class TradingStrategy:
    """Base class for all trading strategies.

    Provides:
        - `inputs_needed` (np.ndarray):  How many items of the input product do
          I need to buy at every time step (n_steps vector).
          This should be read **but not updated** by the `NegotiationManager`.
        - `outputs_needed` (np.ndarray):  How many items of the output product
          do I need to sell at every time step (n_steps vector).
          This should be read **but not updated** by the `NegotiationManager`.
        - `inputs_secured` (np.ndarray):  How many items of the input product I
          already contracted to buy (n_steps vector) [out of `input_needed`].
          This can be read **but not updated** by the `NegotiationManager`.
        - `outputs_secured` (np.ndarray):  How many units of the output product
          I already contracted to sell (n_steps vector) [out of `outputs_secured`]
          This can be read **but not updated** by the `NegotiationManager`.

    Hooks Into:
        - `init`
        - `internal_state`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs_needed: np.ndarray = None
        """How many items of the input product do I need at every time step"""
        self.outputs_needed: np.ndarray = None
        """How many items of the output product do I need at every time step"""
        self.inputs_secured: np.ndarray = None
        """How many units of the input product I have already secured per step"""
        self.outputs_secured: np.ndarray = None
        """How many units of the output product I have already secured per step"""

    def init(self):
        super().init()
        awi = self.awi
        # initialize needed/secured for inputs and outputs to all zeros
        self.inputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.outputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.inputs_needed = np.zeros(awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(awi.n_steps, dtype=int)

    def step(self):
        super().step()
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        if s > n - 2:
            return
        inventory = awi.current_inventory
        n_in, n_out = inventory[awi.my_input_product], inventory[awi.my_output_product]

        for t in range(s + 1, n - 1):
            if self.inputs_needed[t] >= n_out:
                n_out = 0
                self.inputs_needed[t] -= n_out
                break
            n_out -= self.inputs_needed[t]
            self.inputs_needed[t] = 0

        need_to_sell = n_in + n_out
        if need_to_sell < 1:
            return

        self.inputs_secured[s + 1] += n_in

        total_to_sell = self.outputs_secured[s + 1 :].sum()
        need_to_sell -= total_to_sell
        if need_to_sell <= 0:
            return

        self.outputs_needed[s + 1] += need_to_sell

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "inputs_secured": self.inputs_secured
                if self.inputs_secured is not None
                else None,
                "inputs_needed": self.inputs_needed
                if self.inputs_needed is not None
                else None,
                "outputs_secured": self.outputs_secured
                if self.outputs_secured is not None
                else None,
                "outputs_needed": self.outputs_needed
                if self.outputs_needed is not None
                else None,
                "buy_negotiations": [
                    _.annotation["seller"]
                    for _ in self.running_negotiations
                    if _.annotation["buyer"] == self.id
                ],
                "sell_negotiations": [
                    _.annotation["buyer"]
                    for _ in self.running_negotiations
                    if _.annotation["seller"] == self.id
                ],
                "_balance": self.awi.state.balance,
                "_input_inventory": self.awi.state.inventory[self.awi.my_input_product],
                "_output_inventory": self.awi.state.inventory[
                    self.awi.my_output_product
                ],
            }
        )
        return state


class SkyTradingStrategy(SkyPredictionStrategy, TradingStrategy):
    def init(self):
        super().init()
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        self.outputs_needed[1:] = self.expected_inputs[:-1]

    def _update_needs(self):
        s = self.awi.current_step
        self.inputs_needed[s:-1] = self.expected_outputs[s + 1 :]
        self.outputs_needed[s + 1 :] = self.expected_inputs[s:-1]

    def before_step(self):
        super().before_step()
        self._update_needs()

    def step(self):
        super().step()
        self._update_needs()
        self.today_price_update()

    # Negotiation時の更新
    def today_price_update(self):
        if self.today_purchased > 0:
            self.today_input_price = self.today_input_price * 0.9
        else:
            self.today_input_price = (
                self.awi.catalog_prices[self.awi.my_input_product] * 0.9
            )

        if self.today_sold > 0:
            self.today_output_price = self.today_output_price * 1.1
        elif self.awi.current_step < self.awi.n_steps * 0.6:
            self.today_output_price = (
                self.awi.catalog_prices[self.awi.my_output_product] * 1.1
            )
        elif self.awi.current_step < self.awi.n_steps * 0.9:
            self.today_output_price = INF
        else:
            self.today_output_price = (
                self.awi.catalog_prices[self.awi.my_output_product] * 1.1
            )

        self.today_purchased, self.today_sold = 0, 0

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        # keeps track of the procution slots consumed by signed contracts processed
        consumed = 0
        bought, sold = 0, 0
        s = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            q, t = (
                contract.agreement["quantity"],
                contract.agreement["time"],
            )
            if is_seller:
                self.outputs_secured[t] += q
                sold += 1
            else:
                self.inputs_secured[t] += q
                bought += 1
            if contract.annotation["caller"] == self.id:
                continue
            if is_seller:
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                if input_product >= 0 and t > 0:
                    steps, _ = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    self.inputs_needed[t - 1] += max(1, q)
                continue

            input_product = contract.annotation["product"]
            output_product = input_product + 1

            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                self.outputs_needed[t + 1] += max(1, q)

            self.cal_average_input(bought, s)
            self.update_contract_threshold(sold, bought, s)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        signatures = [None] * len(contracts)

        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["time"],
                (
                    x[0].agreement["unit_price"]
                    - self.output_price[x[0].agreement["time"]]
                )
                if x[0].annotation["seller"] == self.id
                else (
                    self.input_cost[x[0].agreement["time"]]
                    - x[0].agreement["unit_price"]
                ),
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
            ),
        )
        sold, bought = 0, 0
        s = self.awi.current_step
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle. The second
            if t < s and len(contract.issues) == 3:
                continue

            # check threshold
            updated_buy = self.input_cost[t]
            updated_sell = self.output_price[t]
            # check that the gontract has a good price
            if is_seller:
                if s < self.awi.n_steps * 0.5:
                    if (
                        u < updated_sell
                        or u < (self.average_input + self.production_cost) * 1.1
                    ):
                        continue
                elif s < self.awi.n_steps * 0.7:
                    if u < updated_sell or u < (
                        self.average_input + self.production_cost
                    ):
                        continue
                elif s < self.awi.n_steps * 0.9:
                    if u < updated_sell * 0.9 or u < (
                        self.average_input + self.production_cost
                    ):
                        continue
            elif not is_seller:
                if s > self.awi.n_steps * 0.8:
                    continue
                elif u > updated_buy:
                    continue

            if is_seller:
                trange = (s, t - 1)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, _ = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )

            if len(steps) - taken < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                if is_seller:
                    sold += q
                    self.today_sold = self.today_sold + 1
                else:
                    bought += q
                    self.today_purchased = self.today_purchased + 1

        return signatures

    def update_contract_threshold(self, sold_n, bought_n, current_step):
        if self.awi.my_output_product != (self.awi.n_products - 1):
            if sold_n > 0:
                self.output_price[current_step:] = min(
                    self.output_price[current_step] * 1.05, self.output_price[0] * 2
                )
            else:
                self.output_price[current_step:] = max(
                    self.output_price[current_step] * 0.95, self.output_price[0]
                )

        if self.awi.my_input_product != 0:
            if bought_n > 0:
                self.input_cost[current_step:] = max(
                    self.input_cost[current_step] * 0.95, self.input_cost[0] // 2
                )
            else:
                self.input_cost[current_step:] = min(
                    self.input_cost[current_step] * 1.05, self.input_cost[0]
                )

    def cal_average_input(self, bought_n, current_step):
        self.total_input_n += bought_n
        self.total_input_cost += self.input_cost[current_step] * bought_n
        if self.total_input_n != 0:
            self.average_input = self.total_input_cost / self.total_input_n

    def _format(self, c: Contract):
        return (
            f"{f'>' if c.annotation['seller'] == self.id else '<'}"
            f"{c.annotation['buyer'] if c.annotation['seller'] == self.id else c.annotation['seller']}: "
            f"{c.agreement['quantity']} of {c.annotation['product']} @ {c.agreement['unit_price']} on {c.agreement['time']}"
        )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)

        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            s = self.awi.current_step
            if t < self.awi.current_step:
                continue
            # distribute the missing quantity over time
            if contract.annotation["seller"] == self.id:
                # self.outputs_secured[t] -= missing
                if t > s:
                    for tau in range(t - 1, s - 1, -1):
                        if self.inputs_needed[tau] <= 0:
                            continue
                        if self.inputs_needed[tau] >= missing:
                            self.inputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.inputs_needed[tau] = 0
                        missing -= self.inputs_needed[tau]
                        if missing <= 0:
                            break
                if missing > 0:
                    if t < self.awi.n_steps - 1:
                        for tau in range(t + 1, self.awi.n_steps):
                            if self.outputs_secured[tau] <= 0:
                                continue
                            if self.outputs_secured[tau] >= missing:
                                self.outputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.outputs_secured[tau] = 0
                            missing -= self.outputs_secured[tau]
                            if missing <= 0:
                                break

            else:
                if t < self.awi.n_steps - 1:
                    for tau in range(t + 1, self.awi.n_steps):
                        if self.outputs_needed[tau] <= 0:
                            continue
                        if self.outputs_needed[tau] >= missing:
                            self.outputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.outputs_needed[tau] = 0
                        missing -= self.outputs_needed[tau]
                        if missing <= 0:
                            break
                if missing > 0:
                    if t > s:
                        for tau in range(t - 1, s - 1, -1):
                            if self.inputs_secured[tau] <= 0:
                                continue
                            if self.inputs_secured[tau] >= missing:
                                self.inputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.inputs_secured[tau] = 0
                            missing -= self.inputs_secured[tau]
                            if missing <= 0:
                                break
