import itertools as it
import pprint
import time

import numpy as np
import pulp
from prettytable import PrettyTable
from scipy.stats import binom, geom


# There might be a bug here when the probability distribution, dict_data, does not add to exactly one.
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


T = 10
q_max = 30
current_inventory = 20

# T = 200
# q_max = 30


def generate_synthetic_uncertainty_model():
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
    # Debug print
    # pprint.pprint(Q_inn)
    # pprint.pprint(Q_out)
    return Q_inn, Q_out, p_inn, p_out


Q_inn, Q_out, p_inn, p_out = generate_synthetic_uncertainty_model()

# print(f"Q_inn : {Q_inn}")
# print(f"Q_out : {Q_out}")


def compute_minima():
    t0 = time.time()
    inn = {t: compute_min_expectation(Q_inn[t], q_max) for t in range(0, T)}
    out = {t: compute_min_expectation(Q_out[t], q_max) for t in range(0, T)}
    print(f"took {time.time() - t0} to generate the minima")
    # Sanity check: the expectation of the minima should be non-negative
    for t, min_data in inn.items():
        for q, the_min in min_data.items():
            assert the_min >= 0
    # Debug print
    # pprint.pprint(inn)
    return inn, out


inn, out = compute_minima()
# print(f"in: {inn}, {out}")
print(f"inn: {inn}")
print(f"out: {out}")


def construct_ILP():
    t0 = time.time()
    # Generate the pulp problem.
    model = pulp.LpProblem("Business_Plan_Solver", pulp.LpMaximize)
    print(model)
    # Generate the integer 0/1 decision variables. There are two kinds: inn_vars[t][k] and out_vars[t][k]
    # inn_vars[t][k] == 1 iff in the business plan the agent tries to buy k inputs at time t
    inn_vars = pulp.LpVariable.dicts(
        "inn",
        ((t, k) for t, k in it.product(range(0, T), range(0, q_max))),
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    # out_vars[t][k] == 1 iff in the business plan the agent tries to sell k inputs at time t
    out_vars = pulp.LpVariable.dicts(
        "out",
        ((t, k) for t, k in it.product(range(0, T), range(0, q_max))),
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    print(f"it took {time.time() - t0 : .4f} to generate dec. vars")
    # Generate the objective function - the total profit of the plan. Profit = revenue - cost
    # Here, revenue is the money received from sales of outputs, and cost is the money used to buy inputs.
    model += pulp.lpSum(
        [
            out_vars[t, k] * out[t][k] * p_out[t]
            - inn_vars[t, k] * inn[t][k] * p_inn[t]
            for t, k in it.product(range(0, T), range(0, q_max))
        ]
    )
    print(f"it took {time.time() - t0 : .4f} to generate objective function")
    # Genrate the constraints. Only one quantity can be planned for at each time step for buying or selling.
    for t in range(0, T):
        model += sum(out_vars[t, k] for k in range(0, q_max)) <= 1
        model += sum(inn_vars[t, k] for k in range(0, q_max)) <= 1
    print(f"it took {time.time() - t0 : .4f} to generate constraints ")
    # Document here: optimistic == True means no bluffing, otherwise there is bluffing going on
    optimistic = True
    if optimistic:
        # ToDo: add redundant constraint that models: the number of inputs == number of outputs
        # Constraints that ensure there are enough outputs to sell at each time step.
        right_hand_size = current_inventory
        for t in range(0, T):
            model += sum(out_vars[t, k] * k for k in range(0, q_max)) <= right_hand_size
            right_hand_size += sum(
                inn_vars[t, k] * k - out_vars[t, k] * k for k in range(0, q_max)
            )
    else:
        # Constraints that ensure there are enough outputs, in expectation, to sell at each time step.
        right_hand_size = current_inventory
        for t in range(0, T):
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
    print(f"it took {time_to_generate_ILP : .4f} to generate the ILP")
    return inn_vars, model, optimistic, out_vars, t0, time_to_generate_ILP


inn_vars, model, optimistic, out_vars, t0, time_to_generate_ILP = construct_ILP()
print(model)


def get_solved_plan():
    t0_solve = time.time()
    model.solve()
    solve_time = time.time() - t0_solve
    print(
        f"it took {solve_time : .4f} sec to solve, result = {pulp.LpStatus[model.status]}"
    )
    total_time = time.time() - t0
    print(
        f"it took {total_time : .4f} sec in total, and has opt profit of {pulp.value(model.objective):.4f}"
    )
    t02 = time.time()
    buy_plan = {
        t: sum(int(k * inn_vars[t, k].varValue) for k in range(0, q_max))
        for t in range(0, T)
    }
    sell_plan = {
        t: sum(int(k * out_vars[t, k].varValue) for k in range(0, q_max))
        for t in range(0, T)
    }
    print(f"it took {time.time() - t02 : .4f} sec to produce the plan")
    return buy_plan, sell_plan, solve_time, total_time


buy_plan, sell_plan, solve_time, total_time = get_solved_plan()


def print_pretty_tables():
    x = PrettyTable()
    x.field_names = ["t"] + [t for t in range(0, T)] + ["total"]
    total_buy_qtty = sum(buy_plan[t] for t in range(0, T))
    x.add_row(["B-Q"] + [buy_plan[t] for t in range(0, T)] + [total_buy_qtty])
    total_sell_qtty = sum(sell_plan[t] for t in range(0, T))
    x.add_row(["S-Q"] + [sell_plan[t] for t in range(0, T)] + [total_sell_qtty])
    x.add_row(["B-P"] + [str(round(p_inn[t], 2)) for t in range(0, T)] + ["--"])
    x.add_row(["S-P"] + [str(round(p_out[t], 2)) for t in range(0, T)] + ["--"])
    total_exp_buy_qtty = sum(inn[t][buy_plan[t]] for t in range(0, T))
    x.add_row(
        ["B-E"]
        + [str(round(inn[t][buy_plan[t]], 2)) for t in range(0, T)]
        + [str(round(total_exp_buy_qtty, 2))]
    )
    total_exp_sell_qtty = sum(out[t][sell_plan[t]] for t in range(0, T))
    x.add_row(
        ["S-E"]
        + [str(round(out[t][sell_plan[t]], 2)) for t in range(0, T)]
        + [str(round(total_exp_sell_qtty, 2))]
    )
    print(x)
    x = PrettyTable()
    x.field_names = ["statistic", "value"]
    x.add_row(["T", T])
    x.add_row(["q_max", q_max])
    x.add_row(["starting inventory", current_inventory])
    x.add_row(
        [
            "total profit",
            f"{sum([out[t][sell_plan[t]] * p_out[t] - inn[t][buy_plan[t]] * p_inn[t] for t in range(0, T)]) :.4f}",
        ]
    )
    x.add_row(["build time", f"{time_to_generate_ILP : .4f} sec"])
    x.add_row(["solve time", f"{solve_time : .4f} sec"])
    x.add_row(["total time", f"{total_time : .4f} sec"])
    x.add_row(["optimistic", f"{optimistic}"])
    print(x)


print_pretty_tables()

print(buy_plan)
print(type(buy_plan))
print(sell_plan)
