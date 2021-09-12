import json
import os
import time
from typing import List, Optional, Tuple

import pandas as pd


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


class NVMLib:
    def __init__(
        self,
        mpnvp_number_of_periods: int,
        mpnvp_quantities_domain_size: int,
        game_length: int,
        input_product_index: int,
        output_product_index: int,
        num_intermediate_products: int,
        production_cost: float,
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
        self.game_length = game_length
        self.input_product_index = input_product_index
        self.output_product_index = output_product_index
        self.num_intermediate_products = num_intermediate_products
        self.mpnvp_production_cost = production_cost

        # Initialize properties of the multi-period news-vendor problem (MPNVP) that is going to be reused each time
        self.mpnvp_number_of_periods = mpnvp_number_of_periods
        self.mpnvp_quantities_domain_size = mpnvp_quantities_domain_size
        self.mpnvp_feasible_sols = NVMLib.read_qtty_feasible_domain(
            self.mpnvp_number_of_periods, self.mpnvp_quantities_domain_size
        )

        # Check if the data exists.
        if self.check_if_data_exists():
            # Get the data for quantities
            q_uncertainty_model = NVMLib.get_json_dict(
                f"nvm_lib/data/dict_qtty_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
            )
            self.q_inn_uncertainty_model = q_uncertainty_model[
                "p" + str(self.input_product_index)
            ]
            self.q_out_uncertainty_model = q_uncertainty_model[
                "p" + str(self.output_product_index)
            ]

            # Check that the model is good, i.e., has an entry for all possible times
            for t in range(game_length - 1):
                assert str(t) in self.q_inn_uncertainty_model
                assert str(t) in self.q_out_uncertainty_model
            self.expectations_q_min_inn = {
                t: NVMLib.compute_min_expectation(
                    self.q_inn_uncertainty_model[str(t + 1)],
                    self.mpnvp_quantities_domain_size,
                )
                for t in range(game_length - 1)
            }
            self.expectations_q_min_out = {
                t: NVMLib.compute_min_expectation(
                    self.q_out_uncertainty_model[str(t + 1)],
                    self.mpnvp_quantities_domain_size,
                )
                for t in range(game_length - 1)
            }

            # Compute the expected quantities
            self.q_inn_expected = {
                t: sum(
                    int(i) * p
                    for i, p in self.q_inn_uncertainty_model[str(t + 1)].items()
                )
                for t in range(0, game_length - 1)
            }
            self.q_out_expected = {
                t: sum(
                    int(i) * p
                    for i, p in self.q_out_uncertainty_model[str(t + 1)].items()
                )
                for t in range(0, game_length - 1)
            }

            # Get the data for prices
            prices = NVMLib.get_json_dict(
                f"nvm_lib/data/dict_price_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
            )
            self.prices_inn = prices["p" + str(self.input_product_index)]
            self.prices_out = prices["p" + str(self.output_product_index)]

    @staticmethod
    def compute_min_expectation(dict_data, size) -> dict:
        """
        Compute the expectation of min(y, X) for all values of y in the support of X where X is a discrete random variable. Returns a dictionary.
        This function implements a simple dynamic program which is fully documented in a separate latex document.
        :param dict_data:
        :param size:
        :return:
        """
        # What if we received an empty dictionary? Then we assume the random variable X has no support.
        ret = {"min_0": 0}
        temp = 1
        for i in range(1, size):
            # The dictionary only stores the values where X has positive probability. All other values are assumed to be zero.
            temp -= dict_data[str(i - 1)] if str(i - 1) in dict_data else 0
            ret["min_" + str(i)] = ret["min_" + str(i - 1)] + temp
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

    @staticmethod
    def read_qtty_feasible_domain(
        the_number_of_periods: int, the_quantities_domain_size: int
    ) -> pd.DataFrame:
        """
        Reads the variables associated with a domain for a number of time periods and a size for each of the quantities variables.
        :param the_number_of_periods: planning horizon of the NVM, an integer.
        :param the_quantities_domain_size:
        :return:
        feasible solutions?? --eddy
        """
        # file_location = 'C:/Users/ED2016/Documents/SCML/scml20areyan/new_agent_5_31_20_sell_all_produce_all/nvm_lib/qtty_domain/qtty_domain_t_' + str(
        #    the_number_of_periods) + '_d_' + str(the_quantities_domain_size) + '.zip'
        file_location = (
            "nvm_lib/qtty_domain/qtty_domain_t_"
            + str(the_number_of_periods)
            + "_d_"
            + str(the_quantities_domain_size)
            + ".zip"
        )
        # file_location = '/'.join(__file__.split('/')[:-1]) + '/qtty_domain/qtty_domain_t_' + str(the_number_of_periods) + '_d_' + str(the_quantities_domain_size) + '.zip'
        if not os.path.isfile(file_location):
            raise Exception(
                f"Could not find the file with feasible domain at {file_location}"
            )
        return pd.read_csv(file_location, compression="gzip", sep=",")

    @staticmethod
    def pandas_tuple_to_nvm_plan(sol: pd, number_of_periods: int) -> Optional[NVMPlan]:
        """
        Takes in a pandas dataframe with a solution and turns it into a NVMPlan object.
        :param sol:
        :param number_of_periods:
        :return:
        """
        if sol is None:
            return None
        return NVMPlan(
            [sol[t * 3] for t in range(number_of_periods)],
            [sol[t * 3 + 1] for t in range(number_of_periods)],
            [sol[t * 3 + 2] for t in range(number_of_periods)],
        )

    @staticmethod
    def solve_mpnvm(
        the_feasible_sols: pd.DataFrame,
        the_expectations_q_min_out: dict,
        the_expectations_q_min_in: dict,
        the_prices_out: dict,
        the_prices_in: dict,
        the_production_cost: float,
        verbose: bool = False,
    ):
        """
        Solves the stochastic Multi-Step NewsVendor Problem.
        :param the_feasible_sols: a DataFrame with all the solutions to be checked. The number of columns must be a multiple of 3
        :param the_expectations_q_min_out:
        :param the_expectations_q_min_in:
        :param the_prices_out:
        :param the_prices_in:
        :param the_production_cost:
        :param verbose
        :return:
        """
        assert len(the_feasible_sols.columns) % 3 == 0
        optimal_sol = None
        optimal_sol_revenue = 0.0
        # Solve the MPNVM for each feasible solution.
        positive_solutions = []
        # The time horizon is implicit in the number of columns of the solutions' DataFrame.
        T = int(len(the_feasible_sols.columns) / 3)
        # Loop through each row of the feasible solutions DataFrame. Use itertuples for faster looping.
        t0 = time.time()
        for row in the_feasible_sols.itertuples(index=False):
            # Compute the objective value
            candidate_sol_value = sum(
                the_prices_out[t]
                * the_expectations_q_min_out[t]["min_" + str(row[(t * 3) + 1])]
                - the_prices_in[t]
                * the_expectations_q_min_in[t]["min_" + str(row[t * 3])]
                - the_production_cost * row[(t * 3) + 2]
                for t in range(T)
            )
            # Keep track in case this solution improves on the optimal so far
            if candidate_sol_value > optimal_sol_revenue:
                optimal_sol_revenue = candidate_sol_value
                optimal_sol = row
                if verbose:
                    print(
                        f"\tit took "
                        + format(time.time() - t0, ".4f")
                        + f" seconds to find a better solution: {NVMLib.pandas_tuple_to_nvm_plan(optimal_sol, T)}, "
                        f" revenue = " + format(optimal_sol_revenue, ".4f")
                    )

            # For debugging purposes only, keep track of all solutions with positive objective value.
            if candidate_sol_value > 0:
                positive_solutions.append((row, candidate_sol_value))
        return (
            optimal_sol_revenue,
            NVMLib.pandas_tuple_to_nvm_plan(optimal_sol, T),
            positive_solutions,
        )

    def check_if_data_exists(self) -> bool:
        """
        Check if the uncertainty model exists, both for prices and quantity
        :return:
        """
        qttyFilePath = f"nvm_lib/data/dict_qtty_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
        priceFilePath = f"nvm_lib/data/dict_price_num_intermediate_products_{self.num_intermediate_products}_{self.game_length}.json"
        if not os.path.isfile(qttyFilePath):
            raise Exception(
                f"The uncertainty model for quantities could not be found at {qttyFilePath}"
            )
        if not os.path.isfile(priceFilePath):
            raise Exception(
                f"The uncertainty model for prices could not be found at {priceFilePath}"
            )
        return True

    def get_complete_plan(
        self, current_time: int, verbose: bool = False
    ) -> Optional[NVMPlan]:
        # -> Optional[List[Tuple[int, int, int]]]
        """
        Given a time of the simulation, solves for a plan.
        :param current_time: the current simulation time.
        :param verbose: boolean to indicate if debug prints should be shown or go silent.
        :return: either an object NVMPlan if an optimal plan could be computed or None otherwise.
        """
        end = current_time + self.mpnvp_number_of_periods
        if end >= self.game_length:
            if verbose:
                print(f"Time past the end of game, returning None")
            return None

        # Unlike prices, we assume that for quantities we have a distribution for all possible values within our domain and we check this in the constructor.
        slice_expectations_q_min_out = {
            t - current_time: self.expectations_q_min_out[t]
            for t in range(current_time, end)
        }
        slice_expectations_q_min_inn = {
            t - current_time: self.expectations_q_min_inn[t]
            for t in range(current_time, end)
        }

        # Since we only store prices that are present in the data, we might have some prices missing for some time period. Here we guard against this missing data.
        # TODO: Add catalog prices in the case for which we have no data about prices. Otherwise, having prices of 0.0 is problematic, stuff is given away for free.
        slice_prices_inn = {
            t - current_time: self.prices_inn[str(t)]
            if str(t) in self.prices_inn
            else 0.0
            for t in range(current_time, end)
        }
        slice_prices_out = {
            t - current_time: self.prices_out[str(t)]
            if str(t) in self.prices_out
            else 0.0
            for t in range(current_time, end)
        }

        if verbose:
            print(
                f"\n*Solving MPNVP for number of periods = {self.mpnvp_number_of_periods}, and domain size = {self.mpnvp_quantities_domain_size}"
            )
            t0 = time.time()
        optimal_sol_value, optimal_sol, positive_solutions = NVMLib.solve_mpnvm(
            self.mpnvp_feasible_sols,
            slice_expectations_q_min_out,
            slice_expectations_q_min_inn,
            slice_prices_out,
            slice_prices_inn,
            self.mpnvp_production_cost,
            verbose,
        )
        if verbose:
            print(f"\t\t Done solving MPNVP. Took {time.time() - t0} sec. ")
        return optimal_sol
