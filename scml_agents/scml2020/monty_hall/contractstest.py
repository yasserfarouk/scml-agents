import pprint
import random
import time

import pulp
from click import Tuple
from prettytable import PrettyTable
from traitlets import List

# A contract is a tuple (q, t, p), where q is the quantity, t the time and p the price.
# The following solver assume that the contracts to be signed are received as lists of tuples, e.g., as follows.
# In particular, note this solver assumes that the contracts are separated into two lists: buy and sell.
# Buy contracts are all those contracts the agent is considering to sign for the purpose of buying inputs.
# Sell contracts are all those contracts the agent is considering to sign for the purpose of selling outputs.


def solve_signer(buy_contracts: list, sel_contracts: list, prints: bool = False):
    # For efficiency purposes, we order the contracts by delivery times. But, before we do, we must be able to
    # recover the indices of the contracts as given to the solver, otherwise, we can't map the output to the right contracts.
    buy_contracts = [c + (i,) for i, c in enumerate(buy_contracts)]
    sel_contracts = [c + (i,) for i, c in enumerate(sel_contracts)]
    # pprint.pprint(buy_contracts)
    # pprint.pprint(sel_contracts)

    buy_contracts = sorted(buy_contracts, key=lambda x: x[1])
    sel_contracts = sorted(sel_contracts, key=lambda x: x[1])

    buy_contracts_copy = buy_contracts.copy()
    sel_contracts_copy = sel_contracts.copy()

    # pprint.pprint(buy_contracts_copy)
    # pprint.pprint(sel_contracts_copy)

    def f(buy_contracts, buy_sign_vars, current_sell_time):
        partial_buy_sum = []
        while len(buy_contracts) > 0 and (buy_contracts[0][1] < current_sell_time):
            c = buy_contracts.pop(0)
            partial_buy_sum += [buy_sign_vars[c[3]] * c[0]]
        return partial_buy_sum

    t0 = time.time()
    # Decision variables
    buy_sign_vars = pulp.LpVariable.dicts(
        "buy_sign",
        (i for i, _ in enumerate(buy_contracts)),
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    sel_sign_vars = pulp.LpVariable.dicts(
        "sel_sign",
        (i for i, _ in enumerate(sel_contracts)),
        lowBound=0,
        upBound=1,
        cat="Integer",
    )

    # Generate the pulp problem.
    model = pulp.LpProblem("Contract_Signer_Solver", pulp.LpMaximize)

    # The objective function is profit.
    model += pulp.lpSum(
        [
            sel_contracts[i][0] * sel_contracts[i][2] * sel_sign_vars[s[3]]
            for i, s in enumerate(sel_contracts)
        ]
        + [
            -1.0 * buy_contracts[i][0] * buy_contracts[i][2] * buy_sign_vars[b[3]]
            for i, b in enumerate(buy_contracts)
        ]
    )

    # Now, we construct the constraints.

    current_sell_time = sel_contracts[0][1]
    current_sell_time_sum = []
    partial_sell_sum = []
    partial_buy_sum = []
    result = []
    while len(sel_contracts) > 0:
        s = sel_contracts.pop(0)
        if current_sell_time == s[1]:
            current_sell_time_sum += [sel_sign_vars[s[3]] * s[0]]
        else:
            partial_buy_sum += f(buy_contracts, buy_sign_vars, current_sell_time)
            result += [
                (
                    current_sell_time_sum.copy(),
                    partial_buy_sum.copy(),
                    partial_sell_sum.copy(),
                )
            ]
            partial_sell_sum += current_sell_time_sum
            current_sell_time = s[1]
            current_sell_time_sum = [sel_sign_vars[s[3]] * s[0]]
    partial_buy_sum += f(buy_contracts, buy_sign_vars, current_sell_time)
    result += [
        (current_sell_time_sum.copy(), partial_buy_sum.copy(), partial_sell_sum.copy())
    ]

    time_to_generate_ILP = time.time() - t0

    # if prints:
    #     # Some prints to debug the program.
    #     for a, b, c in result:
    #         print(a)
    #         print(b)
    #         pprint.pprint(c)
    #         print("---------------")

    for l, m, r in result:
        model += sum(l) <= sum(m) - sum(r)

    t0_solve = time.time()
    model.solve()
    print(f"it took { time_to_generate_ILP : .4f} to generate the ILP")
    solve_time = time.time() - t0_solve
    print(
        f"it took { solve_time : .4f} sec to solve, result = {pulp.LpStatus[model.status]}"
    )
    total_time = time.time() - t0
    print(
        f"it took {time_to_generate_ILP + solve_time : .4f} sec in total, and has opt profit of {pulp.value(model.objective):.4f}"
    )

    sign_plan_buy = {
        i: buy_sign_vars[b[3]].varValue for i, b in enumerate(buy_contracts_copy)
    }
    sign_plan_sel = {
        i: sel_sign_vars[s[3]].varValue for i, s in enumerate(sel_contracts_copy)
    }

    for key in sign_plan_buy:
        if sign_plan_buy[key] == None:
            sign_plan_buy[key] = 0

    for key in sign_plan_sel:
        if sign_plan_sel[key] == None:
            sign_plan_sel[key] = 0

    # if prints:
    #     print("Buy Contracts")
    #     for i in sign_plan_buy.keys():
    #         print(f"{i} \t {buy_contracts_copy[i]} \t {int(sign_plan_buy[i])}")
    #
    #     print("Sell Contracts")
    #     for i in sign_plan_sel.keys():
    #         print(f"{i} \t {sel_contracts_copy[i]} \t {int(sign_plan_sel[i])}")

    buy = {}
    for i in sign_plan_buy.keys():
        buy[(buy_contracts_copy[i])[3]] = int(sign_plan_buy[i])

    sell = {}
    for i in sign_plan_sel.keys():
        sell[(sel_contracts_copy[i])[3]] = int(sign_plan_sel[i])

    return buy, sell

    # if prints:
    #     x = PrettyTable()
    #     x.field_names = ['t'] + [t for t in range(0, T)] + ['total']
    #
    #     buy_list = [0 for _ in range(0, T)]
    #     for i in sign_plan_buy.keys():
    #         if int(sign_plan_buy[i]) == 1:
    #             buy_list[buy_contracts_copy[i][1]] = buy_list[buy_contracts_copy[i][1]] + buy_contracts_copy[i][0]
    #
    #     sel_list = [0 for _ in range(0, T)]
    #     for i in sign_plan_sel.keys():
    #         if int(sign_plan_sel[i]) == 1:
    #             sel_list[sel_contracts_copy[i][1]] = sel_list[sel_contracts_copy[i][1]] + sel_contracts_copy[i][0]
    #
    #     x.add_row(['buy'] + buy_list + [sum([b[0] if int(sign_plan_buy[i]) == 1 else 0 for i, b in enumerate(buy_contracts_copy)])])
    #     x.add_row(['sel'] + sel_list + [sum([s[0] if int(sign_plan_sel[i]) == 1 else 0 for i, s in enumerate(sel_contracts_copy)])])
    #     print(x)


buy_contracts = [
    (5, 0, 0.1),
    (1, 3, 1.5),
    (1, 1, 1.75),
    (10, 2, 2.5),
    (8, 7, 9.8),
    (3, 1, 7.25),
    (8, 3, 6.25),
]

sel_contracts = [
    (10, 3, 1.5),
    (5, 1, 0.1),
    (11, 3, 0.25),
    (3, 1, 0.5),
    (9, 8, 11.01),
    (9, 8, 8.01),
]

buy_contracts = [
    (12, 2, 10),
    (14, 2, 10),
    (15, 2, 10),
    (4, 2, 10),
    (15, 4, 10),
    (11, 4, 10),
    (4, 2, 10),
    (13, 2, 10),
    (1, 1, 10),
    (9, 4, 10),
    (18, 3, 10),
    (0, 3, 10),
    (12, 2, 10),
    (13, 1, 10),
    (6, 2, 10),
    (11, 1, 10),
    (7, 3, 10),
    (11, 2, 10),
    (9, 3, 10),
    (14, 1, 10),
    (21, 2, 10),
    (10, 4, 10),
    (1, 4, 10),
    (9, 1, 10),
    (17, 4, 10),
    (6, 1, 10),
    (2, 2, 10),
    (15, 4, 10),
    (19, 5, 10),
    (14, 4, 10),
    (18, 2, 10),
    (9, 3, 10),
    (0, 2, 10),
    (17, 1, 10),
    (14, 2, 10),
    (7, 2, 10),
    (3, 3, 10),
    (1, 5, 10),
    (15, 2, 10),
    (1, 3, 10),
    (13, 4, 10),
    (16, 3, 10),
    (10, 1, 10),
    (11, 2, 10),
    (14, 4, 10),
    (12, 2, 10),
    (19, 4, 10),
    (3, 1, 10),
    (2, 1, 10),
    (5, 3, 10),
    (13, 1, 10),
    (20, 3, 10),
]

sel_contracts = [
    (6, 3, 10),
    (20, 1, 10),
    (21, 3, 10),
    (16, 1, 10),
    (10, 2, 10),
    (10, 2, 10),
    (16, 2, 10),
    (4, 1, 10),
    (19, 2, 10),
    (11, 2, 10),
    (8, 1, 10),
    (10, 2, 10),
    (1, 2, 10),
    (7, 3, 10),
    (17, 4, 10),
    (3, 3, 10),
    (20, 3, 10),
    (22, 2, 10),
    (1, 1, 10),
    (5, 3, 10),
    (22, 4, 10),
    (16, 1, 10),
    (17, 4, 10),
    (7, 2, 10),
    (21, 2, 10),
    (15, 1, 10),
    (7, 2, 10),
    (9, 1, 10),
    (12, 3, 10),
    (16, 3, 10),
    (4, 3, 10),
    (15, 2, 10),
    (13, 3, 10),
    (6, 1, 10),
    (19, 1, 10),
    (22, 1, 10),
    (21, 2, 10),
    (18, 1, 10),
    (22, 2, 10),
    (2, 2, 10),
    (0, 2, 10),
    (10, 2, 10),
    (6, 3, 10),
    (2, 5, 10),
    (3, 3, 10),
    (7, 2, 10),
    (18, 3, 10),
    (3, 4, 10),
    (22, 1, 10),
    (12, 3, 10),
    (8, 5, 10),
    (22, 3, 10),
    (1, 1, 10),
    (14, 1, 10),
]

if __name__ == "__main__":
    x = solve_signer(buy_contracts, sel_contracts, True)
    print(x[0])
    print(x[1])
