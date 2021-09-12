import json

import pandas as pd


def get_raw_quantity_uncertainty_model(the_game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Reads the game log data for the given number of intermediate products and returns the raw uncertainty model.
    :param the_game_logs
    :return:
    """
    the_game_logs = the_game_logs.copy()
    the_game_logs = the_game_logs[["time", "product", "quantity"]]
    the_game_logs["count"] = 1
    the_game_logs = the_game_logs.groupby(by=["time", "product", "quantity"]).sum()
    the_game_logs = the_game_logs.reset_index()
    return the_game_logs


def get_prob_dist(raw_uncertainty_model: pd.DataFrame, time: int, product: str) -> dict:
    """
    Given the raw uncertainty model (i.e., the log data), a time and product, return a dictionary {q: mass}, where q is a quantity and mass is
    the probability of seeing that quantity. The probability is w.r.t the log data corresponding to the time and product.
    :param raw_uncertainty_model:
    :param time:
    :param product:
    :return:
    """
    x = raw_uncertainty_model[
        (raw_uncertainty_model["time"] == time)
        & (raw_uncertainty_model["product"] == product)
    ]
    total = x["count"].sum()
    return_dict = {}
    for q in x.quantity.unique():
        # This is a hack to get the value of a column since we knwo at this point x would have a single row. Is there a better way to do this?
        temp = x[x["quantity"] == q]["count"].unique()
        # At this point we should have a single value which is the number of transactions for this time, product, and quantity combination.
        assert len(temp) == 1
        return_dict[int(q)] = temp[0] / total
    # In case there are no transactions in the log, this means that no quantities were traded
    if len(return_dict) == 0:
        return_dict[0] = 1.0
    return return_dict


def save_json_qytt_uncertainty_model(
    json_file_name: str, the_game_logs: pd.DataFrame, game_length: int
):
    """
    Given the raw uncertainty model and a list of products, saves to a json file the dictionary with the uncertainty model ready to be consumed by the MPNVP.
    :param json_file_name
    :param the_game_logs:
    :param game_length:
    :return:
    """
    products = the_game_logs["product"].unique()
    raw_uncertainty_model = get_raw_quantity_uncertainty_model(the_game_logs)
    d = {
        product: {
            t: get_prob_dist(raw_uncertainty_model, t, product)
            for t in range(game_length)
        }
        for product in products
    }
    with open(f"data/{json_file_name}.json", "w") as f:
        json.dump(d, f)


def save_json_price_data(json_file_name: str, the_game_logs: pd.DataFrame):
    """
    Given a json file name and the game logs, save the price uncertainty model as a json file
    :param json_file_name:
    :param the_game_logs:
    :return:
    """
    # Make a copy of the data for safe manipulation in this function.
    the_game_logs = the_game_logs.copy()
    # Compute the weighted price, i.e., the quantity times the price.
    the_game_logs["weighted_price"] = the_game_logs["quantity"] * the_game_logs["price"]
    d = {}
    # For each product, group by time and then computed weighted price, i.e. (\sum_{q, p} q*p) / (sum_{q} q)
    for product in the_game_logs["product"].unique():
        d[product] = {}
        product_data = the_game_logs[the_game_logs["product"] == product].copy()
        product_data_grouped = product_data.groupby(by=["time"]).sum()
        product_data_grouped["avg_weighted_price"] = (
            product_data_grouped["weighted_price"] / product_data_grouped["quantity"]
        )
        product_data_grouped = product_data_grouped.reset_index()
        for index, row in product_data_grouped.iterrows():
            d[product][int(row["time"])] = row["avg_weighted_price"]
    with open(f"data/{json_file_name}.json", "w") as JSON:
        json.dump(d, JSON)


def get_json_dict(json_file_name: str) -> dict:
    """
    Return the dictionary stored in the given json file name.
    :param json_file_name:
    :return:
    """
    with open(json_file_name) as JSON:
        return json.load(JSON)
