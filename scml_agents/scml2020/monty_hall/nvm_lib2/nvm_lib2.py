import itertools as it
import json
import pathlib
import pprint
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pulp
from prettytable import PrettyTable
from scipy.stats import binom, geom


class NVMPlan:
    """
    This class represents a solution to the multi-period news-vendor model optimization problem.
    """

    def __init__(self, x: list, y: list, z: list):
        """
        A production plan consists of a buy plan (x) a sell plan (y) and a production plan (z)
        :param x: buy plan, a list of integers where x[i] is the amount of input to buy at t.
        :param y: sell plan, a list of integers where y[i] is the amount of output to sell at t.
        :param z: production plan, a list of integers where z[i] is the amount of inputs to turn to outputs at t.
        """
        assert len(x) == len(y) == len(z)
        self.x = x
        self.y = y
        self.z = z

    def get_buy_plan(self):
        return self.x

    def get_sell_plan(self):
        return self.y

    def get_production_plan(self):
        return self.z

    def get_buy_plan_at(self, t: int):
        return self.x[t]

    def get_sell_plan_at(self, t: int):
        return self.y[t]

    def get_production_plan_at(self, t: int):
        return self.z[t]

    def __str__(self):
        return str(
            [tuple((self.x[t], self.y[t], self.z[t])) for t in range(len(self.x))]
        )

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.x):
            ret = (self.x[self.n], self.y[self.n], self.z[self.n])
            self.n += 1
            return ret
        else:
            raise StopIteration


class NVMLib2:
    def __init__(
        self,
        mpnvp_number_of_periods: int,
        mpnvp_quantities_domain_size: int,
        game_length: int,
        input_product_index: int,
        output_product_index: int,
        num_intermediate_products: int,
        production_cost: float,
        current_inventory: int,
        current_time: int,
    ):
        """
        Initializes the NVMLib.

        :param mpnvp_number_of_periods: planning horizon, an integer.
        :param mpnvp_quantities_domain_size: the size of the quantities domain to consider, an integer.
        :param game_length: the lenght of the game, an integer.
        :param input_product_index: the index of the product the agent consumes, an integer.
        :param output_product_index: the index of the product the agent produces, an integer.
        :param num_intermediate_products: number of intermediate products in the chain, an integer.
        :param production_cost: the unit cost of turning one input into an output, a float.
        """
        # Initialize properties of the game.
        # why are all of these instance attributes? lol
        self.game_length = game_length
        self.input_product_index = input_product_index
        self.output_product_index = output_product_index
        self.num_intermediate_products = num_intermediate_products
        self.mpnvp_production_cost = production_cost
        self.current_time = current_time

        # Initialize properties of the multi-period news-vendor problem (MPNVP) that is going to be reused each time
        self.mpnvp_number_of_periods = mpnvp_number_of_periods
        self.mpnvp_quantities_domain_size = mpnvp_quantities_domain_size
        self.current_inventory = current_inventory

        # Adjust plan length if plan would exceed game length
        if current_time + mpnvp_number_of_periods > game_length:
            self.mpnvp_number_of_periods = game_length - current_time

        # ---------------------- solve plan --------------------------
        T = self.mpnvp_number_of_periods
        # q_max = self.mpnvp_quantities_domain_size
        q_max = 10
        current_inventory = self.current_inventory

        # check data exists
        assert self.check_if_data_exists() is True

        # TODO: fall back plan with self.current_time + 1, 2, etc
        if self.check_if_data_exists() is True:

            # Get the data for quantities
            path = (
                pathlib.Path(__file__).parent
                / "data2"
                / f"dict_qtty_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
            )
            path_string = str(path)
            q_uncertainty_model = NVMLib2.get_json_dict(path_string)
            # quantity distribution for the input product, time step distribution of probability distribution
            Q_inn = q_uncertainty_model["p" + str(self.input_product_index)]
            # quantity distribution for the output product, time step distribution of probability distribution
            Q_out = q_uncertainty_model["p" + str(self.output_product_index)]

            ##print(f"q_uncertainty_model {q_uncertainty_model}")
            ##print(f"Q_inn : {Q_inn}")
            ##print(f"Q_out : {Q_out}")

            # Get the data for prices
            price_path = (
                pathlib.Path(__file__).parent
                / "data2"
                / f"dict_price_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
            )
            price_path_string = str(price_path)
            prices = NVMLib2.get_json_dict(price_path_string)
            p_inn = prices[
                "p" + str(self.input_product_index)
            ]  # price distribution for the input product
            p_out = prices[
                "p" + str(self.output_product_index)
            ]  # price distribution for the output product

            # print(f"QUANTITY PATH READ: {path_string}")
            # print(f"PRICE PATH READ: {price_path_string}")
            # pprint.pprint(f"p_inn: {'p' + str(self.input_product_index)}: {p_inn}")
            # pprint.pprint(f"p_out: {'p' + str(self.output_product_index)}: {p_out}")
            # print(f"CURRENT TIME SELF: {self.current_time}")

            # Compute minima
            inn, out = self.compute_minima(T, q_max, Q_inn, Q_out)

            ##print(f"inn: {inn}")
            ##print(f"out: {out}")

            # Construct ILP
            # inn_vars, model, optimistic, out_vars, t0, time_to_generate_ILP = self.construct_ILP(
            #     T, q_max, inn, out, p_inn, p_out, current_inventory)

            # Solve ILP
            # buy_plan, sell_plan, solve_time, total_time = self.get_solved_plan(T, q_max, model, t0, inn_vars, out_vars)

            # Construct ILP and then Solve in one step
            (
                buy_plan,
                sell_plan,
                solve_time,
                total_time,
                inn_vars,
                model,
                optimistic,
                out_vars,
                t0,
                time_to_generate_ILP,
            ) = self.construct_ILP(T, q_max, inn, out, p_inn, p_out, current_inventory)

            # print(f"Buy Plan: {buy_plan}")
            # print(f"Sell Plan: {sell_plan}")

            # #print Pretty Table
            self.print_pretty_tables(
                T,
                q_max,
                current_inventory,
                inn,
                out,
                p_inn,
                p_out,
                buy_plan,
                sell_plan,
                time_to_generate_ILP,
                solve_time,
                total_time,
                optimistic,
            )

            # Create buy and sell plan instance attributes
            self.buy_plan = buy_plan
            self.sell_plan = sell_plan

        else:
            # print("data nonexistent so using the fall back plan")
            buy_plan = {
                self.current_time: 9,
                self.current_time + 1: 9,
                self.current_time + 2: 9,
                self.current_time + 3: 9,
                self.current_time + 4: 9,
            }
            sell_plan = {
                self.current_time: 5,
                self.current_time + 1: 9,
                self.current_time + 2: 9,
                self.current_time + 3: 9,
                self.current_time + 4: 9,
            }
            self.buy_plan = buy_plan
            self.sell_plan = sell_plan

    def check_if_data_exists(self) -> bool:
        """
        Check if the uncertainty model exists, both for prices and quantity
        :return:
        """
        qtty_path = (
            pathlib.Path(__file__).parent
            / "data2"
            / f"dict_qtty_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
        )
        price_path = (
            pathlib.Path(__file__).parent
            / "data2"
            / f"dict_price_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
        )
        if not qtty_path.is_file():
            raise Exception(
                f"The uncertainty model for quantities could not be found at {qtty_path}"
            )
        if not price_path.is_file():
            raise Exception(
                f"The uncertainty model for prices could not be found at {price_path}"
            )
        return True

    # There might be a bug here when the probability distribution, dict_data, does not add to exactly one.
    @staticmethod
    def compute_min_expectation(dict_data, size) -> dict:
        """
        Compute the expectation of min(y, X) for all values of y in the support of X where X is a discrete random variable.
        This function implements a simple dynamic program which is fully documented in a separate latex document.
        If dict_data is empty, then we assume the random variable X has no support.
        :param dict_data: {x: P(X = x)} where x ranges from 0 to size.
        :param size: the support of random variable is from 0, ..., size.
        :return: a dictionary {y : E[min(y, X)]} for y ranging from 0, ..., size.
        """
        ret = {0: 0.0}
        temp = 1
        for i in range(1, size):
            # The dictionary only stores the values where X has positive probability. All other values are assumed to be zero.
            temp -= dict_data[i - 1] if i - 1 in dict_data else 0.0
            ret[i] = ret[i - 1] + temp
        return ret

    @staticmethod
    def get_json_dict(json_file_name: str) -> dict:
        """
        Return the dictionary stored in the given json file name.
        :param json_file_name:
        :return:
        """
        with open(json_file_name) as JSON:
            return json.load(JSON)

    def generate_synthetic_uncertainty_model(T: int, q_max: int):
        # Random quantities inn
        random_quantities_inn = np.random.randint(0, q_max, (T, q_max))
        normalization_qtt_inn = random_quantities_inn.sum(axis=1)
        Q_inn = {
            t: {
                q: random_quantities_inn[t][q] / normalization_qtt_inn[t]
                for q in range(0, q_max)
            }
            for t in range(0, T)
        }
        # Random quantities out
        random_quantities_out = np.random.randint(0, q_max, (T, q_max))
        normalization_qtt_out = random_quantities_out.sum(axis=1)
        Q_out = {
            t: {
                q: random_quantities_out[t][q] / normalization_qtt_out[t]
                for q in range(0, q_max)
            }
            for t in range(0, T)
        }
        # Random prices
        p_inn = {t: np.random.uniform(7, 12) for t in range(0, T)}
        p_out = {t: np.random.uniform(10, 15) for t in range(0, T)}
        # Sanity check: if the price of outputs is 0 always, then the program should buy and sell nothing.
        # p_out = {t: np.random.uniform(0, 0) for t in range(0, T)}
        # Debug #print
        # pprint.pprint(Q_inn)
        # pprint.pprint(Q_out)
        return Q_inn, Q_out, p_inn, p_out

    def compute_minima(self, T: int, q_max: int, Q_inn: Dict[str, Dict], Q_out):
        t0 = time.time()
        ##print("TEST #print Q_INN[0]: ")
        ##print(f"Q_INN[0]: {Q_inn[str(0)]}")
        inn = {
            t: NVMLib2.compute_min_expectation(Q_inn[str(t)], q_max)
            for t in range(self.current_time, self.current_time + T)
        }
        out = {
            t: NVMLib2.compute_min_expectation(Q_out[str(t)], q_max)
            for t in range(self.current_time, self.current_time + T)
        }
        # print(f'took {time.time() - t0} to generate the minima')
        # Sanity check: the expectation of the minima should be non-negative
        for t, min_data in inn.items():
            for q, the_min in min_data.items():
                assert the_min >= 0
        # Debug #print
        # pprint.pprint(inn)
        return inn, out

    def construct_ILP(
        self,
        T: int,
        q_max: int,
        inn: Dict[int, Dict],
        out: Dict[int, Dict],
        p_inn,
        p_out,
        current_inventory: int,
    ):
        t0 = time.time()
        # Generate the pulp problem.
        model = pulp.LpProblem("Business_Plan_Solver", pulp.LpMaximize)
        # Generate the integer 0/1 decision variables. There are two kinds: inn_vars[t][k] and out_vars[t][k]
        # inn_vars[t][k] == 1 iff in the business plan the agent tries to buy k inputs at time t
        inn_vars = pulp.LpVariable.dicts(
            "inn",
            (
                (t, k)
                for t, k in it.product(
                    range(self.current_time, self.current_time + T), range(0, q_max)
                )
            ),
            lowBound=0,
            upBound=1,
            cat="Integer",
        )
        # out_vars[t][k] == 1 iff in the business plan the agent tries to sell k inputs at time t
        out_vars = pulp.LpVariable.dicts(
            "out",
            (
                (t, k)
                for t, k in it.product(
                    range(self.current_time, self.current_time + T), range(0, q_max)
                )
            ),
            lowBound=0,
            upBound=1,
            cat="Integer",
        )
        # print(f'it took {time.time() - t0 : .4f} to generate dec. vars')
        # Generate the objective function - the total profit of the plan. Profit = revenue - cost
        # Here, revenue is the money received from sales of outputs, and cost is the money used to buy inputs.
        model += pulp.lpSum(
            [
                out_vars[t, k] * out[t][k] * p_out[str(t)]
                - inn_vars[t, k] * inn[t][k] * p_inn[str(t)]
                for t, k in it.product(
                    range(self.current_time, self.current_time + T), range(0, q_max)
                )
            ]
        )
        # print(f'it took {time.time() - t0 : .4f} to generate objective function')
        # Genrate the constraints. Only one quantity can be planned for at each time step for buying or selling.
        for t in range(self.current_time, self.current_time + T):
            model += sum(out_vars[t, k] for k in range(0, q_max)) <= 1
            model += sum(inn_vars[t, k] for k in range(0, q_max)) <= 1
        # print(f'it took {time.time() - t0 : .4f} to generate constraints ')
        # Document here: optimistic == True means no bluffing, otherwise there is bluffing going on
        optimistic = True
        if optimistic:
            # ToDo: add redundant constraint that models: the number of inputs == number of outputs
            # Constraints that ensure there are enough outputs to sell at each time step.
            right_hand_size = current_inventory
            for t in range(self.current_time, self.current_time + T):
                model += (
                    sum(out_vars[t, k] * k for k in range(0, q_max)) <= right_hand_size
                )
                right_hand_size += sum(
                    inn_vars[t, k] * k - out_vars[t, k] * k for k in range(0, q_max)
                )
        else:
            # Constraints that ensure there are enough outputs, in expectation, to sell at each time step.
            right_hand_size = current_inventory
            for t in range(self.current_time, self.current_time + T):
                model += (
                    sum(out_vars[t, k] * out[t][k] for k in range(0, q_max))
                    <= right_hand_size
                )
                right_hand_size += sum(
                    inn_vars[t, k] * inn[t][k] - out_vars[t, k] * out[t][k]
                    for k in range(0, q_max)
                )
        # We assume that the planning starts with no inventory and thus, the agent cannot sell anything at time 0.
        # for k in range(current_inventory+1, q_max):
        #            model += out_vars[0, k] == 0
        time_to_generate_ILP = time.time() - t0
        # print(f'it took {time_to_generate_ILP : .4f} to generate the ILP')
        # return inn_vars, model, optimistic, out_vars, t0, time_to_generate_ILP

        # solve ILP
        t0_solve = time.time()
        model.solve()
        solve_time = time.time() - t0_solve
        # print(f'it took {solve_time : .4f} sec to solve, result = {pulp.LpStatus[model.status]}')
        total_time = time.time() - t0
        # print(f"MODEL OBJECTIVE: {pulp.value(model.objective)}")
        # #print(f'it took {total_time : .4f} sec in total, and has opt profit of {pulp.value(model.objective) : .4f}')
        t0 = time.time()
        buy_plan = {
            t: sum(int(k * inn_vars[t, k].varValue) for k in range(0, q_max))
            for t in range(self.current_time, self.current_time + T)
        }
        sell_plan = {
            t: sum(int(k * out_vars[t, k].varValue) for k in range(0, q_max))
            for t in range(self.current_time, self.current_time + T)
        }
        # print(f'it took {time.time() - t0 : .4f} sec to produce the plan')

        return (
            buy_plan,
            sell_plan,
            solve_time,
            total_time,
            inn_vars,
            model,
            optimistic,
            out_vars,
            t0,
            time_to_generate_ILP,
        )

    def get_solved_plan(self, T, q_max, model, t0, inn_vars, out_vars):
        t0_solve = time.time()
        model.solve()
        solve_time = time.time() - t0_solve
        # print(f'it took {solve_time : .4f} sec to solve, result = {pulp.LpStatus[model.status]}')
        total_time = time.time() - t0
        # print(f"MODEL OBJECTIVE: {pulp.value(model.objective)}")
        ##print(f'it took {total_time : .4f} sec in total, and has opt profit of {pulp.value(model.objective) : .4f}')
        t0 = time.time()
        buy_plan = {
            t: sum(int(k * inn_vars[t, k].varValue) for k in range(0, q_max))
            for t in range(0, T)
        }
        sell_plan = {
            t: sum(int(k * out_vars[t, k].varValue) for k in range(0, q_max))
            for t in range(0, T)
        }
        # print(f'it took {time.time() - t0 : .4f} sec to produce the plan')
        return buy_plan, sell_plan, solve_time, total_time

    def print_pretty_tables(
        self,
        T,
        q_max,
        current_inventory,
        inn,
        out,
        p_inn,
        p_out,
        buy_plan,
        sell_plan,
        time_to_generate_ILP,
        solve_time,
        total_time,
        optimistic,
    ):
        x = PrettyTable()
        x.field_names = (
            ["t"]
            + [t for t in range(self.current_time, self.current_time + T)]
            + ["total"]
        )
        total_buy_qtty = sum(
            buy_plan[t] for t in range(self.current_time, self.current_time + T)
        )
        x.add_row(
            ["B-Q"]
            + [buy_plan[t] for t in range(self.current_time, self.current_time + T)]
            + [total_buy_qtty]
        )
        total_sell_qtty = sum(
            sell_plan[t] for t in range(self.current_time, self.current_time + T)
        )
        x.add_row(
            ["S-Q"]
            + [sell_plan[t] for t in range(self.current_time, self.current_time + T)]
            + [total_sell_qtty]
        )
        x.add_row(
            ["B-P"]
            + [
                str(round(p_inn[str(t)], 2))
                for t in range(self.current_time, self.current_time + T)
            ]
            + ["--"]
        )
        x.add_row(
            ["S-P"]
            + [
                str(round(p_out[str(t)], 2))
                for t in range(self.current_time, self.current_time + T)
            ]
            + ["--"]
        )
        total_exp_buy_qtty = sum(
            inn[t][buy_plan[t]] for t in range(self.current_time, self.current_time + T)
        )
        x.add_row(
            ["B-E"]
            + [
                str(round(inn[t][buy_plan[t]], 2))
                for t in range(self.current_time, self.current_time + T)
            ]
            + [str(round(total_exp_buy_qtty, 2))]
        )
        total_exp_sell_qtty = sum(
            out[t][sell_plan[t]]
            for t in range(self.current_time, self.current_time + T)
        )
        x.add_row(
            ["S-E"]
            + [
                str(round(out[t][sell_plan[t]], 2))
                for t in range(self.current_time, self.current_time + T)
            ]
            + [str(round(total_exp_sell_qtty, 2))]
        )
        # print(x)
        x = PrettyTable()
        x.field_names = ["statistic", "value"]
        x.add_row(["T", T])
        x.add_row(["q_max", q_max])
        x.add_row(["starting inventory", current_inventory])
        x.add_row(
            [
                "total profit",
                f"{sum([out[t][sell_plan[t]] * p_out[str(t)] - inn[t][buy_plan[t]] * p_inn[str(t)] for t in range(self.current_time, self.current_time + T)]) :.4f}",
            ]
        )
        x.add_row(["build time", f"{time_to_generate_ILP : .4f} sec"])
        x.add_row(["solve time", f"{solve_time : .4f} sec"])
        x.add_row(["total time", f"{total_time : .4f} sec"])
        x.add_row(["optimistic", f"{optimistic}"])
        # print(x)

    def create_NVM_plan(self) -> NVMPlan:
        buy_list = []
        sell_list = []
        produce_list = []

        assert (
            self.current_time + self.mpnvp_number_of_periods <= self.game_length
        ), "PLAN EXCEEDS GAME LENGTH, THIS SHOULD NOT BE HAPPENING"

        assert len(self.buy_plan) == len(self.sell_plan)
        for i in range(len(self.buy_plan)):
            buy_list.append(self.buy_plan[i + self.current_time])
            sell_list.append(self.sell_plan[i + self.current_time])
            produce_list.append(0)

        return NVMPlan(buy_list, sell_list, produce_list)
