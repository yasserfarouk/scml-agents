import multiprocessing
import os
import time

from qtty_feasible_domain_cp import generate_and_save_sol_df, get_zip_location, solve_cp


def generate_domain(
    the_production_capacity: int, number_of_periods: int, quantities_domain_size: int
):
    """
    Generates a single domain with the given parameters.
    :param the_production_capacity:
    :param number_of_periods:
    :param quantities_domain_size:
    :return:
    """
    t0_generation = time.time()
    sols = solve_cp(the_production_capacity, number_of_periods, quantities_domain_size)
    print(
        f"done generating solutions for t = {number_of_periods}, "
        f"Domain(q) = {quantities_domain_size}. "
        f"took " + format(time.time() - t0_generation, ".6f") + " secs. ",
        end="",
    )

    t0_saving_data = time.time()
    generate_and_save_sol_df(number_of_periods, quantities_domain_size, sols)
    print(
        f"done saving compressed dataframe. took "
        + format(time.time() - t0_saving_data, ".6f")
        + " seconds"
    )


def bulk_domain_generation(
    the_production_capacity: int,
    the_lower_bound_number_of_periods: int,
    the_upper_bound_number_of_periods: int,
    the_lower_bound_qtty_domain_size: int,
    the_upper_bound_qtty_domain_size: int,
):
    """

    :param the_production_capacity:
    :param the_lower_bound_number_of_periods:
    :param the_upper_bound_number_of_periods:
    :param the_lower_bound_qtty_domain_size:
    :param the_upper_bound_qtty_domain_size:
    :return:
    """
    # Solve the CP
    for number_of_periods in range(
        the_lower_bound_number_of_periods, the_upper_bound_number_of_periods + 1
    ):
        for quantities_domain_size in range(
            the_lower_bound_qtty_domain_size, the_upper_bound_qtty_domain_size + 1
        ):
            if not os.path.isfile(
                get_zip_location(number_of_periods, quantities_domain_size)
            ):
                generate_domain(
                    the_production_capacity, number_of_periods, quantities_domain_size
                )
            else:
                print(
                    f"Already have the feasible domain for t = {number_of_periods} and Domain(q) = {quantities_domain_size}"
                )


production_capacity = 10


def generate_helper(horizon: int):
    generate_domain(production_capacity, horizon, 11)


# Production capacity is fixed here to 10 as per ANAC 2019.

# horizons = [x for x in range(50,201,1)]
# with multiprocessing.Pool() as pool:
#        pool.map(generate_helper, horizons)
generate_domain(production_capacity, 5, 11)


# Bulk Generates the feasible domain of the NVM optimization problem as saves each possible domain as a compress .csv file.
# This is mostly for experimentation, for actual game play, we only use one such domain, see example below.
"""
lower_bound_number_of_periods = 1
upper_bound_number_of_periods = 6
lower_bound_qtty_domain_size = 2
upper_bound_qtty_domain_size = 20

bulk_domain_generation(production_capacity,
                       lower_bound_number_of_periods,
                       upper_bound_number_of_periods,
                       lower_bound_qtty_domain_size,
                       upper_bound_qtty_domain_size)
"""
