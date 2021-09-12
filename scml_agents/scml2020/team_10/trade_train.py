from random import shuffle

import numpy as np
from agent import *
from hyperparameters import *
from patches import *
from scml import SCML2020World
from tqdm import tqdm
from trade_model import Model, load_trade_model

# def train(world):
#     tags = []
#     good = bad = 0
#     for contract in world.saved_contracts:
#         if contract['breaches'] == '':
#             good += 1
#             tags.append([1, 0])
#         else:
#             bad += 1
#             tags.append([0, 1])
#
#     # TODO: collect more features here to enable predicting better...
#     features = []
#     for contract in world.saved_contracts:
#         feature_vec = []
#         feature_vec = [
#             contract['delivery_time'] - contract['signed_at'],
#             contract['quantity'],
#             contract['unit_price'],
#             contract['product'],
#             contract['n_neg_steps'],
#         ]
#         features.append(feature_vec)
#
#     in_dim = len(features[0])
#     features = np.array(features)
#     tags = np.array(tags)
#
#     import os
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     from keras.models import Sequential
#     from keras.layers import Dense
#
#     # define the keras model
#     model = Sequential()
#     model.add(Dense(10, input_dim=in_dim, activation='relu'))
#     model.add(Dense(5, activation='relu'))
#     model.add(Dense(2, activation='sigmoid'))
#
#     # compile the keras model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # fit the keras model on the dataset
#     model.fit(features, tags, epochs=150, batch_size=10)
#
#     # evaluate the keras model
#     _, accuracy = model.evaluate(features, tags)
#     print('Accuracy: %.2f' % (accuracy*100))
#     print('Naive:',100 * good / (good + bad))
#     print('Finished Training')
#
# train(world)

"""
# TODO: our goal here is to learn online a trade prediction strategy
# we would run some simulation of the world, and try to learn a prediction strategy
# we need to learn 2 networks: inputs needed + outputs needed (for now the secured is just the contracts we've signed)
# our offline data: after the simulation has ended, we know for each agent how many resources it needed for every time step
# this should be a function of:
# 1. the history:
#    - agent's balance history
#    - agent's inventory history
#    - agent's inputs/outputs needed/secured history (can be inferred from inventory history)
#    - agent's breaches' history
#    - number of contracts signed history
# TODO: 2. bulletin board:
#    - financial reports
#    - breach history
#    - bankruptcies history
"""


def get_train_data(world):
    train_data = []
    train_input_tags = []
    train_output_tags = []

    for agent in world.agents.values():
        if agent.id in ["SELLER", "BUYER"]:
            continue
        data = agent.history
        if len(data) > 0:
            train_data.append(np.array(data))

            # 0: balance
            # 1: inventory_input
            # 2: inventory_output
            # 3: productivity
            # 4: needed_input[t-1]
            # 5: needed_input[t]
            # 6: needed_input[t+1]
            # 7: needed_output[t-1]
            # 8: needed_output[t]
            # 9: needed_output[t+1]

            # TODO: fix this to be the actual needed inputs at time t + 1
            input_tag = [x[5] for x in data][1:] + [0]

            # TODO: fix this to be the actual needed outputs at time t + 1
            output_tag = [x[8] for x in data][1:] + [0]

            train_input_tags.append(input_tag)
            train_output_tags.append(output_tag)
    return train_data, train_input_tags, train_output_tags


train_data = []
train_input_tags = []
train_output_tags = []

print("training...")

for i in tqdm(range(TRADE_TRAIN_DATA)):
    world = SCML2020World(
        **SCML2020World.generate(
            agent_types=[NegotiatorAgent, UnicornAgent],
            n_steps=40,
            n_processes=1,  # TODO: 2
        ),
        construct_graphs=False
    )
    SCML2020World.cancelled_contracts = cancelled_contracts
    world.run()

    data, input_tags, output_tags = get_train_data(world)
    train_data += data
    train_input_tags += input_tags
    train_output_tags += output_tags

import torch
from trade_model import MAX_HORIZON


def pytorch_rolling_window(x, window_size=MAX_HORIZON, step_size=1):
    # unfold dimension to make our rolling window
    x = x.unfold(0, window_size, step_size)
    last = x[-1]
    for i in range(window_size - 1, 0, -1):
        y = torch.zeros((1, window_size))
        y[:, :i] = last[window_size - i :]
        x = torch.cat((x, y), 0)
    return x


inputs = torch.from_numpy(np.array(train_data)).float()
input_tags = torch.from_numpy(np.array(train_input_tags)).float()
output_tags = torch.from_numpy(np.array(train_output_tags)).float()

input_tags = [pytorch_rolling_window(inp) for inp in input_tags]
output_tags = [pytorch_rolling_window(out) for out in output_tags]


all_input_data = list(zip(inputs, input_tags))
shuffle(all_input_data)
train_input_data = all_input_data[: -int(len(all_input_data) * TRADE_VALIDATION_SPLIT)]
test_input_data = all_input_data[-int(len(all_input_data) * TRADE_VALIDATION_SPLIT) :]

all_output_data = list(zip(inputs, output_tags))
shuffle(all_output_data)
train_output_data = all_output_data[
    : -int(len(all_output_data) * TRADE_VALIDATION_SPLIT)
]
test_output_data = all_output_data[
    -int(len(all_output_data) * TRADE_VALIDATION_SPLIT) :
]

if TRADE_LOAD_MODEL:
    input_model = load_trade_model(path=TRADE_INPUT_PATH)
    output_model = load_trade_model(path=TRADE_OUTPUT_PATH)
else:
    input_model = Model()
    output_model = Model()

print("learning...")

train_input = False
if train_input:
    train_input, valid_input = input_model.fit(
        train_input_data,
        test_input_data,
        save_model=TRADE_SAVE_MODEL,
        path=TRADE_INPUT_PATH,
    )
    input_model.plot(train_input, valid_input)

train_output = True
if train_output:
    train_output, valid_output = output_model.fit(
        train_output_data,
        test_output_data,
        save_model=TRADE_SAVE_MODEL,
        path=TRADE_OUTPUT_PATH,
    )
    output_model.plot(train_output, valid_output)

print("done")

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Flatten
# from keras.models import load_model

# if _load_model:
#     model = load_model('models/trade_prediction_model.h5')
# else:
#     model = Sequential()
#     model.add(LSTM(hidden_size, return_sequences=True, input_shape=data[0].shape))
#     model.add(LSTM(hidden_size, return_sequences=True))
#     if use_dropout:
#         model.add(Dropout(dropout))
#     model.add(TimeDistributed(Dense(hidden_size // 2, activation='relu')))
#     model.add(TimeDistributed(Dense(hidden_size // 4, activation='relu')))
#     model.add(TimeDistributed(Dense(output_dim)))
#     model.add(Flatten())
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#
# history = model.fit(np.array(data), np.array(tags), epochs=epochs, validation_split=0.1)
#
# import matplotlib.pyplot as plt
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# if _save_model:
#     model.save('models/trade_prediction_model.h5')
