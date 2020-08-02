import numpy as np
from pprint import pformat
from typing import List, Optional

from negmas import Contract

from scml.scml2020.components.trading import TradingStrategy
from scml.scml2020.components import FixedTradePredictionStrategy, SignAllPossible
from scml.scml2020.common import is_system_agent
from scml.scml2020.common import ANY_LINE
from scml.scml2020.components.prediction import MeanERPStrategy


class ConsumeDrivenPredictionBasedTradingStrategy(
    FixedTradePredictionStrategy, MeanERPStrategy, TradingStrategy
):
    """A trading strategy that uses prediction strategies to manage inputs/outputs needed

    Hooks Into:
        - `init`
        - `on_contracts_finalized`
        - `sign_all_contracts`
        - `on_agent_bankrupt`

    """

    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:] * 0
        self.inputs_needed[1:10] = self.expected_outputs[1:10]
        # for i in range(len(self.expected_outputs)):
        #    if i % 2  == 0:
        #        self.inputs_needed[i] = 0
        # self.inputs_needed = np.array([int(j) for j in self.expected_outputs])
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # self.awi.logdebug_agent(
        #     f"Enter Contracts Finalized:\n"
        #     f"Signed {pformat([self._format(_) for _ in signed])}\n"
        #     f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
        #     f"{pformat(self.internal_state)}"
        # )
        super().on_contracts_finalized(signed, cancelled, rejectors)
        consumed = 0
        for contract in signed:
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                self.outputs_secured[t] += q
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, lines = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    if contract.annotation["caller"] != self.id:
                        # this is a sell contract that I did not expect yet. Update needs accordingly
                        self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.inputs_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                # this is a buy contract that I did not expect yet. Update needs accordingly
                if self.inputs_secured[t] > self.inputs_needed[t]:
                    self.outputs_needed[t + 1] += max(1, q)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
        signatures = [None] * len(contracts)
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
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
            # check that the contract is executable in principle
            if t < s and len(contract.issues) == 3:
                continue
            if is_seller:
                trange = (s, t)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
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
                else:
                    bought += q
        return signatures

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
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                if t > 0:
                    self.inputs_needed[t - 1] -= missing
            else:
                self.inputs_secured[t] += missing
                if t < self.awi.n_steps - 1:
                    self.outputs_needed[t + 1] -= missing
