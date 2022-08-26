import copy

# =====================
#       Schedule
# =====================


def format_schedule(schedule, max_number_of_steps):
    """
    Format the AWI schedule into dict. Where key is step, value is available day count for that step.
    E.g. {'1': 5, '2': 7}
    """
    schedule = list(schedule)  # convert schedule to list.
    formatted_schedule = {}
    for i in range(0, max_number_of_steps):
        formatted_schedule[i] = schedule.count(i)

    return formatted_schedule


def calculate_produced_quantity(formatted_schedule, step):
    """
    Takes step and formatted schedule, calculates the produced q until that step.
    """
    prod_q = 0

    for i in range(0, step):
        prod_q += 10 - formatted_schedule[i]

    return prod_q


def schedule_dispatch_production(schedule, amount, start_date, finish_date):
    """
    Takes a 'formatted' schedule, and plans production on it.
    For dispatch usage, not scheduling 'actual' production.
    Also returns the 'currently' planned schedule, that contains only this amount of production. For partner usage.
    """
    partner_schedule = {}

    # Check for special keep flag, if it is then leave schedule as it is.
    if start_date != -1:
        for step in range(start_date, finish_date):
            possible_production_amount = schedule[step]
            if possible_production_amount > 0:
                if amount - possible_production_amount > 0:
                    amount = amount - possible_production_amount
                    schedule[step] = 0
                    partner_schedule[step] = possible_production_amount
                else:
                    schedule[step] = possible_production_amount - amount
                    partner_schedule[step] = amount
                    break

    return schedule, partner_schedule


def is_schedule_available(schedule, amount, start_date, finish_date):
    """
    Takes schedule, amount, and dates. Checks whether it is possible to produce 'amount' in time range (start_date, finish_date)
    """
    # Check for special keep flag, if it is then return true since we have produced the amount.
    if start_date == -1:
        return True, {"-1": amount}

    # We will keep the custom schedule that keeps this amount can be produced in closest time.
    closest_schedule = {}

    for step in range(start_date, finish_date):
        possible_production_amount = schedule[step]
        if possible_production_amount > 0:
            if amount - possible_production_amount > 0:
                amount = amount - possible_production_amount
                closest_schedule[step] = possible_production_amount
            else:
                closest_schedule[step] = amount
                return True, closest_schedule

    return False, closest_schedule


def buyer_closest_available_delivery(schedule, amount, start_date, end_date):
    """
    Takes schedule, amount, and start_date. Calculates the buyer's 'closest' delivery day when 'amount' is produced.
    """
    if start_date == -1:
        return start_date + 1

    for i in range(start_date, end_date):
        possible_production_amount = schedule[i]
        if possible_production_amount > 0:
            if amount - possible_production_amount <= 0:
                return (
                    i + 1
                )  # Since production completes at i. We can deliver it closest to (i + 1).
            else:
                amount -= possible_production_amount
    # If cant find closest, send the max instead.
    return -1


def seller_closest_available_delivery(schedule, amount, end_date):
    """
    Takes schedule, amount, and end_date. Calculates the seller's 'closest' delivery day when 'amount' is produced.
    """
    date_range = reversed(range(0, end_date))

    for i in date_range:
        possible_production_amount = schedule[i]
        if possible_production_amount > 0:
            if amount - possible_production_amount <= 0:
                return i  # Since production completes at i, We can deliver at i.
            else:
                amount -= possible_production_amount
    # If cant find closest, send the min instead.
    return -1


def most_available_amount_in_schedule(schedule, start_date, finish_date):
    """
    Takes schedule, and dates. Checks how many product it is possible to produce in time range (start_date, finish_date)
    """
    possible_amount = 0
    for i in range(start_date, finish_date):
        possible_amount += schedule[i]

    return possible_amount


# ============================
#          Balance
# ============================


def is_balance_available(
    balance, production_cost, seller_proposed_min_quantity, seller_proposed_min_price
):
    """Check whether our balance is enough in best case scenario or not. If not, dont bother to negotiate with this agent."""

    if balance >= (
        (seller_proposed_min_quantity + production_cost) * seller_proposed_min_price
    ):
        return True
    else:
        return False


# ============================
#          Dispatch
# ============================


def sort_buyers_by_price(buyers):
    """
    Takes buyers (and offers) as input: dict, where key is agent id and value is offer tuple (q, t, p).
    Sorts in descending order by offer price.
    Returns list of tuples [('contract_id', offer)]
    """
    if len(list(buyers.keys())) > 0:
        buyers_sorted = list(
            buyers.items()
        )  # Convert to tuple. [(('agent', (1, 3, 5)) ('agent2', (3, 5, 8)))]
        buyers_sorted.sort(
            key=lambda x: x[1][2], reverse=True
        )  # Sort tuple by prices. (desceding)
        return buyers_sorted
    else:
        return []


def sort_sellers_by_price(sellers):
    """
    Takes sellers (and offers) as input: dict, where key is agent id and value is offer tuple (q, t, p).
    Sorts in ascending order by offer price.
    Returns list of tuples [('contract_id', offer)]
    """
    if len(list(sellers.keys())) > 0:
        sellers_sorted = list(
            sellers.items()
        )  # Convert to tuple. [(('agent', (1, 3, 5)) ('agent2', (3, 5, 8)))]
        sellers_sorted.sort(
            key=lambda x: x[1][2], reverse=False
        )  # Sort tuple by prices. (ascending)
        return sellers_sorted
    else:
        return []


def sort_negotiators_by_delivery(negotiators):
    """
    Takes negotiators (and offers) as input: dict, where key is agent_id and value is offer tuple (q, t, p).
    Sorts in ascending order by offer delivery date.
    Returns list of tuples [('contract_id', offer)]
    """
    if len(list(negotiators.keys())) > 0:
        negotiators_sorted = list(
            negotiators.items()
        )  # Convert to tuple. [(('agent', (1, 3, 5)) ('agent2', (3, 5, 8)))]
        negotiators_sorted.sort(
            key=lambda x: x[1][1], reverse=False
        )  # Sort tuple by delivery time. (ascending)
        return negotiators_sorted
    else:
        return []


def sort_negotiators_by_descending_delivery(negotiators):
    """
    Takes negotiators (and offers) as input: dict, where key is agent_id and value is offer tuple (q, t, p).
    Sorts in descending order by offer delivery date.
    Returns list of tuples [('contract_id', offer)]
    """
    if len(list(negotiators.keys())) > 0:
        negotiators_sorted = list(
            negotiators.items()
        )  # Convert to tuple. [(('agent', (1, 3, 5)) ('agent2', (3, 5, 8)))]
        negotiators_sorted.sort(
            key=lambda x: x[1][1], reverse=True
        )  # Sort tuple by delivery time. (descending)
        return negotiators_sorted
    else:
        return []


# ============================
#          Contracts
# ============================


def get_contract_buyer_sellers(our_agent_id, contracts):
    """
    Get contracts and return buyer and sellers, with following format
    So that sorting will work.
    {'agentID': (3, 5, 7)}
    """
    buyers = {}
    sellers = {}

    for contract in contracts:
        agreed_offer = contract.agreement
        tuple_agreed_offer = (
            agreed_offer["quantity"],
            agreed_offer["time"],
            agreed_offer["unit_price"],
        )
        # Check whether our agent is a buyer or seller.
        if our_agent_id in contract.annotation["buyer"]:
            sellers[contract.id] = tuple_agreed_offer
        elif our_agent_id in contract.annotation["seller"]:
            buyers[contract.id] = tuple_agreed_offer

    return buyers, sellers


def calculate_contract_quantity(signed_contracts):
    """
    Gets signed contracts: {'agentID': (3, 5, 7)}, and returns the cumulative quantity that are signed.
    """
    signed_q = 0

    for offer in signed_contracts.values():
        signed_q += offer[0]

    return signed_q


# ============================
#             Keep
# ============================


def calculate_current_keep_amount(
    current_step, scheduled_buyer_contracts, current_output_in_inventory
):
    """
    Calculates current keep amount (producted) in the inventory by checking scheduled buyer contracts.
    {buyer_contract: {seller_contract_id: (seller_contract, schedule), seller_contract2_id: (seller_contract2, schedule2) }}
    """
    total_producted_nonkeep_amount = 0
    for buyer_contract_id, seller_contract_details in scheduled_buyer_contracts.items():
        # Check whether buyer is keep or not, we should ignore keep produced quantities.
        if buyer_contract_id != "KEEP":
            for seller_contract_id, (
                seller_contract,
                schedule,
            ) in seller_contract_details.items():
                for step, amount in schedule.items():
                    if int(step) < current_step:
                        total_producted_nonkeep_amount += amount

    return current_output_in_inventory - total_producted_nonkeep_amount


def calculate_total_keep_amount(
    current_step, scheduled_buyer_contracts, current_output_in_inventory
):
    """
    Calculates current keep amount (producted) in the inventory + planned in schedule by checking scheduled buyer contracts.
    {buyer_contract: {seller_contract_id: (seller_contract, schedule), seller_contract2_id: (seller_contract2, schedule2) }}
    Should be called before production given step!
    """
    current_keep_amount = calculate_current_keep_amount(
        current_step, scheduled_buyer_contracts, current_output_in_inventory
    )

    scheduled_keep_quantity = 0
    for buyer_contract_id, seller_contract_details in scheduled_buyer_contracts.items():
        # Check whether buyer is keep or not, we should ignore keep produced quantities.
        if buyer_contract_id == "KEEP":
            for seller_contract_id, (
                seller_contract,
                schedule,
            ) in seller_contract_details.items():
                for step, amount in schedule.items():
                    if int(step) >= current_step:
                        scheduled_keep_quantity += amount

    return current_keep_amount + scheduled_keep_quantity


# ============================
#            PSEUDO
# ============================


def get_unscheduled_total_pseudo_quantity(pseudo_contracts):
    """
    Calculates total amount of product in pseudo contracts.
    """
    total_q = 0
    for contract, offer in pseudo_contracts.items():
        q = offer[0]
        t = offer[1]
        if t != -1:
            total_q += q

    return total_q


def format_pseudo_contracts(pseudo_contracts):
    """
    Format pseudo buyer & sellers so that, find cumulative quantity for each day.
    Returns a dict where key is step, and the value is the quantity.
    """
    formatted_pseudo = {}

    for contract_id, offer in pseudo_contracts.items():
        offer_q = offer[0]
        offer_t = offer[1]

        if offer_t not in formatted_pseudo.keys():
            formatted_pseudo[offer_t] = offer_q
        else:
            formatted_pseudo[offer_t] += offer_q

    return formatted_pseudo


def split(sorted_contracts, nego_quota):
    copy_sorted_c = copy.deepcopy(sorted_contracts)
    k, m = divmod(len(copy_sorted_c), nego_quota)
    return (
        copy_sorted_c[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(nego_quota)
    )


def get_pseudo_seller_contracts_with_quota(
    formatted_schedule,
    seller_pseudo_contracts,
    nego_quota,
    current_step,
    max_buyer_delivery_day,
):
    """
    Splits pseudo contracts into nego_quota pieces, gets the lowest delivery day for that chunk (seller) or highest d.t. for that chunk (buyer).
    Returns list of tuples with (t, q)
    e.g. [(3, 10), (4, 8)]
    """
    req_list = []

    filtered_pseudo_contracts = {}
    for contract_id, offer in seller_pseudo_contracts.items():
        if (
            buyer_closest_available_delivery(
                formatted_schedule, offer[0], offer[1], max_buyer_delivery_day
            )
            > current_step + 1
        ):
            filtered_pseudo_contracts[contract_id] = offer

    sorted_contracts = [
        c[1] for c in sort_negotiators_by_delivery(filtered_pseudo_contracts)
    ]

    splitted_req_contracts = list(split(sorted_contracts, nego_quota))

    for req_contracts in splitted_req_contracts:
        if req_contracts:
            total_q = sum([c[0] for c in req_contracts])
            req_list.append((req_contracts[0][1], total_q))

    return req_list


def get_pseudo_buyer_contracts_with_quota(
    formatted_schedule, buyer_pseudo_contracts, nego_quota
):
    """
    Splits pseudo contracts into nego_quota pieces, gets the lowest delivery day for that chunk (seller) or highest d.t. for that chunk (buyer).
    Returns list of tuples with (t, q)
    e.g. [(3, 10), (4, 8)]
    """
    req_list = []

    filtered_pseudo_contracts = {}
    for contract_id, offer in buyer_pseudo_contracts.items():
        seller_delivery_upper = seller_closest_available_delivery(
            formatted_schedule, offer[0], offer[1]
        )
        if seller_delivery_upper != -1 and seller_delivery_upper < offer[1]:
            filtered_pseudo_contracts[contract_id] = offer

    sorted_contracts = [
        c[1] for c in sort_negotiators_by_delivery(filtered_pseudo_contracts)
    ]

    splitted_req_contracts = list(split(sorted_contracts, nego_quota))

    for req_contracts in splitted_req_contracts:
        if req_contracts:
            total_q = sum([c[0] for c in req_contracts])
            req_list.append((req_contracts[-1][1], total_q))

    return req_list


# ============================
#            Agent
# ============================


def get_agent_reject_rate(agent_id, reject_acceptance_rate, current_step):
    """
    Returns agent reject rate
    """
    total_agent_reject_rate = 0
    total_reject_quantity = 0

    if reject_acceptance_rate[agent_id]["Acceptance"] == 0 or current_step == 0:
        return total_agent_reject_rate
    else:
        for reject_c, (reject_day, reject_quantity) in enumerate(
            reject_acceptance_rate[agent_id]["Reject"].items()
        ):
            # Get rejectance rate of buyer id.
            agent_reject_rate = reject_quantity / (
                reject_acceptance_rate[agent_id]["Acceptance"]
            )
            reject_coefficient = (current_step - reject_day) / (current_step)
            total_agent_reject_rate += (
                (reject_c + 1) * agent_reject_rate * (1 - reject_coefficient)
            )
            total_reject_quantity += reject_quantity

    if total_reject_quantity >= 300:
        return -1
    else:
        return total_agent_reject_rate


def get_negotiable_agent_rate(
    agents, reject_acceptance_rate, current_step, bankrupt_agents
):
    negotiable_agent_count = 0
    if len(agents) != 0:
        for agent in agents:
            if (
                0
                <= get_agent_reject_rate(agent, reject_acceptance_rate, current_step)
                <= 0.75
            ) and (agent not in bankrupt_agents):
                negotiable_agent_count += 1
        return negotiable_agent_count / len(agents)
    else:
        return 0


if __name__ == "__main__":
    pseudo_contracts = {
        "a": (10, 20, 15),
        "b": (5, 7, 9),
        "c": (8, 15, 21),
        "d": (7, 6, 3),
        "e": (4, 8, 9),
    }
    # print(get_pseudo_contracts_with_quota(pseudo_contracts, 3, True))

    pseudo_contracts = {
        "a": (10, 20, 15),
        "b": (5, 7, 9),
        "c": (8, 15, 21),
        "d": (7, 6, 3),
        "e": (4, 8, 9),
        "f": (4, 8, 9),
    }
    # print(get_pseudo_contracts_with_quota(pseudo_contracts, 3, False))

    pseudo_contracts = {"a": (10, 20, 15), "b": (5, 7, 9)}
    # print(get_pseudo_contracts_with_quota(pseudo_contracts, 3, False))

    agent_reject_rate = {"MMM": {"Acceptance": 30, "Reject": {4: 5, 8: 5, 9: 5}}}

    print("Agent Reject Rate: ", get_agent_reject_rate("MMM", agent_reject_rate, 9))
