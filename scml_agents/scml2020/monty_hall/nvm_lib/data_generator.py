import os

import data
import pandas as pd

game_length = 50
num_intermediate_products = 3

"""
A synthetic example of what the game's log might look like.
The NVM code needs the game log's to be a dataframe where each row has at least
    Time, Product, Quantity, Price
Any other column is ignored.
The interpretation of a row is a transaction that occur between to agents at
simulation Time for Product for Quantity at Price (unit price).
"""
# TODO explain this example.
# dummy_game_log = pd.DataFrame([{'time': 1, 'product': 'p0', 'quantity': 1, 'price': 1110.23},
#                               {'time': 0, 'product': 'p0', 'quantity': 1, 'price': 3.5},
#                               {'time': 2, 'product': 'p1', 'quantity': 1, 'price': 1.23},
#                               {'time': 2, 'product': 'p1', 'quantity': 5, 'price': 1.8},
#                               {'time': 3, 'product': 'p1', 'quantity': 5, 'price': 1.8}],
#                              columns=['time', 'product', 'quantity', 'price'], index=[0, 1, 2, 3, 4])
#
#
## Replace game_logs here with actual game logs.
# game_logs = dummy_game_log

# df = pd.read_csv("/Users/jtsatsaros/Downloads/scml20-master 2/mylogs_2019/num_intermediate_products_3_production_cost_1_n_steps_100_log.csv")
# cols = set(df.columns)
# cols_to_keep = set()
# cols_to_keep.add('time')
# cols_to_keep.add('product')
# cols_to_keep.add('quantity')
# cols_to_keep.add('price')
# drop_cols = cols - cols_to_keep
# new_df = df.drop(drop_cols, axis = 1)

df = pd.read_csv(
    "/Users/jtsatsaros/SCML/newlogs_1/50_steps_3_levels/trial0/signed_contracts.csv"
)
raw_df = df[df.executed_at != -1]


cols = set(raw_df.columns)
cols_to_keep = set()
cols_to_keep.add("delivery_time")
cols_to_keep.add("product_name")
cols_to_keep.add("quantity")
cols_to_keep.add("unit_price")
drop_cols = cols - cols_to_keep
logs = raw_df.drop(drop_cols, axis=1)

game_logs = logs.rename(
    columns={"delivery_time": "time", "product_name": "product", "unit_price": "price"}
)


directory = os.fsencode("/Users/jtsatsaros/SCML/newlogs_1/50_steps_3_levels")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename != "trial0" and filename != ".DS_Store":
        totalpath = (
            "/Users/jtsatsaros/SCML/newlogs_1/50_steps_3_levels/"
            + filename
            + "/signed_contracts.csv"
        )
        df = pd.read_csv(totalpath)
        raw_df = df[df.executed_at != -1]
        cols = set(raw_df.columns)
        cols_to_keep = set()
        cols_to_keep.add("delivery_time")
        cols_to_keep.add("product_name")
        cols_to_keep.add("quantity")
        cols_to_keep.add("unit_price")
        drop_cols = cols - cols_to_keep
        log1 = raw_df.drop(drop_cols, axis=1)
        log2 = log1.rename(
            columns={
                "delivery_time": "time",
                "product_name": "product",
                "unit_price": "price",
            }
        )
        frames = [game_logs, log2]
        game_logs = pd.concat(frames, sort=True)


directory = os.fsencode("/Users/jtsatsaros/SCML/newlogs_2/50_steps_3_levels")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename != ".DS_Store":
        totalpath = (
            "/Users/jtsatsaros/SCML/newlogs_2/50_steps_3_levels/"
            + filename
            + "/signed_contracts.csv"
        )
        df = pd.read_csv(totalpath)
        raw_df = df[df.executed_at != -1]
        cols = set(raw_df.columns)
        cols_to_keep = set()
        cols_to_keep.add("delivery_time")
        cols_to_keep.add("product_name")
        cols_to_keep.add("quantity")
        cols_to_keep.add("unit_price")
        drop_cols = cols - cols_to_keep
        log1 = raw_df.drop(drop_cols, axis=1)
        log2 = log1.rename(
            columns={
                "delivery_time": "time",
                "product_name": "product",
                "unit_price": "price",
            }
        )
        frames = [game_logs, log2]
        game_logs = pd.concat(frames, sort=True)

directory = os.fsencode("/Users/jtsatsaros/SCML/newlogs_3/50_steps_3_levels")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename != ".DS_Store":
        totalpath = (
            "/Users/jtsatsaros/SCML/newlogs_3/50_steps_3_levels/"
            + filename
            + "/signed_contracts.csv"
        )
        df = pd.read_csv(totalpath)
        raw_df = df[df.executed_at != -1]
        cols = set(raw_df.columns)
        cols_to_keep = set()
        cols_to_keep.add("delivery_time")
        cols_to_keep.add("product_name")
        cols_to_keep.add("quantity")
        cols_to_keep.add("unit_price")
        drop_cols = cols - cols_to_keep
        log1 = raw_df.drop(drop_cols, axis=1)
        log2 = log1.rename(
            columns={
                "delivery_time": "time",
                "product_name": "product",
                "unit_price": "price",
            }
        )
        frames = [game_logs, log2]
        game_logs = pd.concat(frames, sort=True)

directory = os.fsencode("/Users/jtsatsaros/SCML/newlogs_4/50_steps_3_levels")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename != ".DS_Store":
        totalpath = (
            "/Users/jtsatsaros/SCML/newlogs_4/50_steps_3_levels/"
            + filename
            + "/signed_contracts.csv"
        )
        df = pd.read_csv(totalpath)
        raw_df = df[df.executed_at != -1]
        cols = set(raw_df.columns)
        cols_to_keep = set()
        cols_to_keep.add("delivery_time")
        cols_to_keep.add("product_name")
        cols_to_keep.add("quantity")
        cols_to_keep.add("unit_price")
        drop_cols = cols - cols_to_keep
        log1 = raw_df.drop(drop_cols, axis=1)
        log2 = log1.rename(
            columns={
                "delivery_time": "time",
                "product_name": "product",
                "unit_price": "price",
            }
        )
        frames = [game_logs, log2]
        game_logs = pd.concat(frames, sort=True)

directory = os.fsencode("/Users/jtsatsaros/SCML/newlogs_5/50_steps_3_levels")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename != ".DS_Store":
        totalpath = (
            "/Users/jtsatsaros/SCML/newlogs_5/50_steps_3_levels/"
            + filename
            + "/signed_contracts.csv"
        )
        df = pd.read_csv(totalpath)
        raw_df = df[df.executed_at != -1]
        cols = set(raw_df.columns)
        cols_to_keep = set()
        cols_to_keep.add("delivery_time")
        cols_to_keep.add("product_name")
        cols_to_keep.add("quantity")
        cols_to_keep.add("unit_price")
        drop_cols = cols - cols_to_keep
        log1 = raw_df.drop(drop_cols, axis=1)
        log2 = log1.rename(
            columns={
                "delivery_time": "time",
                "product_name": "product",
                "unit_price": "price",
            }
        )
        frames = [game_logs, log2]
        game_logs = pd.concat(frames, sort=True)


"""
Save quantity uncertainty model.
The model is saved to json file data/dict_qtty_num_intermediate_products_{num_intermediate_products}.json
The model is read as product -> time -> quantity -> probability of observing the quantity of the product traded at the time.
"""
data.save_json_qytt_uncertainty_model(
    json_file_name=f"dict_qtty_num_intermediate_products_{num_intermediate_products}",
    the_game_logs=game_logs,
    game_length=game_length,
)

"""
Save price uncertainty model.
The model is saved to json file data/dict_price_num_intermediate_products_{num_intermediate_products}.json
The model is read as product -> time -> average price at which the product was traded at the time.
"""
data.save_json_price_data(
    json_file_name=f"dict_price_num_intermediate_products_{num_intermediate_products}",
    the_game_logs=game_logs,
)
