from typing import List, Optional
from negmas import Contract
from scml.scml2020.components.trading import TradingStrategy
from .nego_strategy import *
import matplotlib.pyplot as plt
from scml.scml2020.components import FixedTradePredictionStrategy, SignAllPossible
from scml.scml2020.common import is_system_agent
from scml.scml2020.common import ANY_LINE
from scml.scml2020.components.prediction import MeanERPStrategy, TradePredictionStrategy
from negmas import Contract, Breach
from typing import Union, Iterable, List, Optional
from sklearn import svm
import pickle
import math
import numpy as np


USE_DATA_COLLECTION = False
TEAM_AGENT_DEBUG_FLAG = False


class avg_data_collect:
    def __init__(
        self,
        expected_inputs,
        expected_outputs,
        input_cost=0,
        output_cost=0,
        number_of_running=10,
    ):
        if not self.load_From_file():
            self.prev_expected_inputs = expected_inputs
            self.expected_inputs = np.zeros(number_of_running)

            self.prev_expected_outputs = expected_outputs
            self.expected_outputs = np.zeros(number_of_running)

            self.prev_input_cost = input_cost
            self.input_cost = input_cost

            self.prev_output_cost = output_cost
            self.output_cost = output_cost

            self.len = 0

    def update_expected(self, new_expected_inputs, new_expected_outputs):
        delta = new_expected_inputs - self.prev_expected_inputs
        self.prev_expected_inputs = new_expected_inputs
        self.expected_inputs += delta

        delta = new_expected_outputs - self.prev_expected_outputs
        self.prev_expected_inputs = new_expected_outputs
        self.expected_outputs += delta

        self.len += 1
        self.save_to_file()

    def save_to_file(self, filename="avg_data.pkl"):
        if USE_DATA_COLLECTION == True:
            pickle.dump(self, open(filename, "wb"))

    def load_From_file(self, filename="avg_data.pkl"):
        if USE_DATA_COLLECTION == False:
            return False
        try:
            obj = pickle.load(open(filename, "rb"))
            if len(obj.prev_expected_inputs) != self.self.awi.n_steps:
                return False
            self.prev_expected_inputs = obj.prev_expected_inputs.copy()
            self.expected_inputs = obj.expected_inputs.copy()

            self.prev_expected_outputs = obj.prev_expected_outputs.copy()
            self.expected_outputs = obj.expected_outputs.copy()

            self.prev_input_cost = obj.prev_input_cost
            self.input_cost = obj.input_cost

            self.prev_output_cost = obj.prev_output_cost
            self.output_cost = obj.output_cost

            self.len = obj.len

            return True
        except:
            return False

    def get_expected_inputs(self):
        data_len = self.len if self.len > 0 else 1
        self.expected_inputs = np.where(
            self.expected_inputs > 0, self.expected_inputs, 5
        )
        return self.expected_inputs // data_len

    def get_expected_outputs(self):
        data_len = self.len if self.len > 0 else 1
        self.expected_outputs = np.where(
            self.expected_outputs > 0, self.expected_outputs, 5
        )
        return self.expected_outputs // data_len


# class TestTradingStrategy(SignAllPossible, TradingStrategy):
#     """The agent reactively responds to contracts for selling by buying and vice versa.
#
#     Hooks Into:
#         - `on_contracts_finalized`
#
#     Remarks:
#         - `Attributes` section describes the attributes that can be used to construct the component (passed to its
#           `__init__` method).
#         - `Provides` section describes the attributes (methods, properties, data-members) made available by this
#           component directly. Note that everything provided by the bases of this components are also available to the
#           agent (Check the `Bases` section above for all the bases of this component).
#         - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
#           or properties/data-members that must exist in the agent that uses this component. These requirement are
#           usually implemented as abstract methods in the component
#         - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
#         - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
#           components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
#           hooked into this way.
#         - `Overrides` section describes the methods this component overrides without calling `super` effectively
#           disallowing any other components after it in the MRO to call this method. Usually methods that do some
#           action (i.e. not starting with `on_`) are overridden this way.
#
#     """
#
#     def on_contracts_finalized(
#         self,
#         signed: List[Contract],
#         cancelled: List[Contract],
#         rejectors: List[List[str]],
#     ) -> None:
#         # call the production strategy
#         super().on_contracts_finalized(signed, cancelled, rejectors)
#         this_step = self.awi.current_step
#         inp, outp = self.awi.my_input_product, self.awi.my_output_product
#         for contract in signed:
#             t, q = contract.agreement["time"], contract.agreement["quantity"]
#             if contract.annotation["caller"] == self.id:
#                 continue
#             if contract.annotation["product"] != outp:
#                 continue
#             is_seller = contract.annotation["seller"] == self.id
#             # find the earliest time I can do anything about this contract
#             if t > self.awi.n_steps - 1 or t < this_step:
#                 continue
#             if is_seller:
#                 # if I am a seller, I will schedule production then buy my needs to produce
#                 steps, _ = self.awi.available_for_production(
#                     repeats=q, step=(this_step + 1, t - 1)
#                 )
#                 if len(steps) < 1:
#                     continue
#                 self.inputs_needed[min(steps)] += q
#                 continue
#
#             # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
#             if inp != contract.annotation["product"]:
#                 continue
#             self.outputs_needed[t] += q


class NewFixedTradePredictionStrategy(TradePredictionStrategy):
    """
    Predicts a fixed amount of trade both for the input and output products.

    Hooks Into:
        - `internal_state`
        - `on_contracts_finalized`

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

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, self.awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

        self.datacollect = avg_data_collect(
            self.expected_inputs,
            self.expected_outputs,
            number_of_running=self.awi.n_steps,
        )

        self.expected_outputs = self.datacollect.get_expected_outputs()
        self.expected_inputs = self.datacollect.get_expected_inputs()

    def trade_prediction_step(self):
        self.debug_print_status("trade_prediction_step", TEAM_AGENT_DEBUG_FLAG)
        pass

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
            }
        )
        return state

    def debug_print_status(self, title="", debug=True):
        if debug:
            print("\n" + "-" * 5 + title + "-" * 5)
            print("step = " + str(self.awi.current_step))
            print(f"agent: {self}")
            print(f"inputs_needed: {self.inputs_needed}")
            print(f"inputs_secured: {self.inputs_secured}")
            print(f"outputs_needed:  {self.outputs_needed}")
            print(f"outputs_secured: {self.outputs_secured}")
            print("-" * len("-" * 5 + title + "-" * 5))
            print("sum buy = " + str(sum(self.inputs_secured)))
            print("sum sel = " + str(sum(self.outputs_secured)))

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if not self._add_trade:
            return
        for contract in signed:
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            if contract.annotation["seller"] == self.id:
                self.expected_outputs[t] += q
            else:
                self.expected_inputs[t] += q

        self.datacollect.update_expected(self.expected_inputs, self.expected_outputs)
        # self.debug_print_status("on_contracts_finalized")


class NewPredictionBasedTradingStrategy(
    NewFixedTradePredictionStrategy, MeanERPStrategy, TradingStrategy
):
    """A trading strategy that uses prediction strategies to manage inputs/outputs needed

    Hooks Into:
        - `init`
        - `on_contracts_finalized`
        - `sign_all_contracts`
        - `on_agent_bankrupt`

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

    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:] * 2
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1] * 0
        # self.outputs_needed[0:3] = self.outputs_needed[0:3]*0

        factor = self.expected_inputs[1] * 2
        ones = np.copy(self.expected_inputs / 5)
        n = len(ones)
        for i in range(len(ones)):
            j = i / n
            ones[i] = factor * 3.5 * (1 - math.tanh(1.6 * j))
        self.baseline_inputs = np.copy(ones)

        self.baseline_outputs = np.copy(self.expected_outputs * 2)
        self.avg_i_p = self.input_cost[0]
        self.avg_i_q = 1
        self.avg_o_p = self.output_price[0]
        self.avg_o_q = 1

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
                self.avg_o_p = (self.avg_o_p * self.avg_o_q + u * q) / (
                    self.avg_o_q + q
                )
                self.avg_o_q += q
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, lines = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    # if contract.annotation["caller"] != self.id:
                    #    # this is a sell contract that I did not expect yet. Update needs accordingly
                    #    self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.inputs_secured[t] += q
            self.avg_i_p = (self.avg_i_p * self.avg_i_q + u * q) / (self.avg_i_q + q)
            self.avg_i_q += q
            # if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
            #    if contract.annotation["caller"] != self.id:
            #        # this is a buy contract that I did not expect yet. Update needs accordingly
            #        self.outputs_needed[t + 1] += max(1, q)

        # self.awi.logdebug_agent(
        #     f"Exit Contracts Finalized:\n{pformat(self.internal_state)}"
        # )

    def calc_expected_inv(self, inputs, outputs, t, step):
        cntr = 0
        for i in range(0, step):
            if outputs[i] > cntr:
                cntr = 0
            else:
                cntr -= outputs[i]
            cntr += inputs[i]

        for i in range(step, t):
            cntr += inputs[i]
            cntr -= outputs[i]

        future_inv = 0
        tobesold = 0
        for i in range(t, len(outputs) - 1):
            if outputs[i] > future_inv:
                tobesold += outputs[i] - future_inv
                future_inv = 0
            else:
                future_inv += inputs[i]
                future_inv -= outputs[i]

        cntr -= tobesold

        return cntr

    def step(self):
        t = self.awi.current_step
        ex_arr = [0] * len(self.inputs_secured)
        if t > 0:
            for i in range(t, len(self.inputs_secured)):
                expected_inventory = self.calc_expected_inv(
                    self.inputs_secured, self.outputs_secured, i, t
                )  # sum(self.inputs_secured[0:i]) - sum(self.outputs_secured[0:i])
                ex_arr[i] = expected_inventory
                if expected_inventory < -self.awi.n_lines / 2:
                    beta = 3
                    gamma = 0
                elif expected_inventory < 0:
                    beta = 3
                    gamma = 0
                else:
                    alpha = expected_inventory / (2 * self.awi.n_lines)
                    if alpha > 1:
                        # got a lot of inventory
                        beta = 0.8 / alpha
                    else:
                        # if alpha < 0.5:
                        #    beta = 1.5
                        # else:
                        beta = 2 - alpha
                    if expected_inventory == 0:
                        gamma = 0.2
                    else:
                        gamma = expected_inventory / self.awi.n_lines

                alpha = expected_inventory / (2 * self.awi.n_lines)
                if expected_inventory < 0:
                    beta = 3
                else:
                    beta = np.exp(1 - alpha)

                # ctrlr = self.buyers[i]
                # if beta < 1:
                #    ctrlr.unit_price = ctrlr.unit_price*1.05
                # else:
                #    ctrlr.unit_price = self.acceptable_unit_price(step, False)

                self.inputs_needed[i] = self.baseline_inputs[i] * beta
                self.outputs_needed[i] = self.baseline_outputs[i] * gamma
        super().step()

    def is_good_price(self, price, step, is_sell):
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if is_sell:
            return False

        return price < self.output_price[step] / 2 - production_cost

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
            if t <= s and len(contract.issues) == 3:
                continue
            if contract.annotation["seller"] == self.id:
                trange = (s, t)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            if is_seller:
                if t < self.awi.n_steps - int(self.awi.n_steps / 10):
                    factor = 1.5
                else:
                    factor = 1
            else:
                factor = 1
            if (factor * needed[t] < secured[t] + q + taken) and (
                not self.is_good_price(u, t, is_seller)
            ):
                continue

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - sold < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                if is_seller:
                    sold += self.predict_quantity(contract)
                else:
                    bought += self.predict_quantity(contract)
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


class PredTestAgent(
    MyNegotiationManager,
    NewPredictionBasedTradingStrategy,
    DemandDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


from collections import defaultdict


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


if __name__ == "__main__":
    for _ in range(10):
        world = SCML2020World(
            **SCML2020World.generate(
                [PredTestAgent, DecentralizingAgent],
                n_steps=10,
                n_processes=3,
                n_agents_per_process=2,
                log_stats_every=1,
            ),
            construct_graphs=True,
        )
        world.run_with_progress()
        world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
        plt.show()
        show_agent_scores(world)
