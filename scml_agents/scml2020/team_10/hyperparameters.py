import os
import sys

sys.path.append(os.path.dirname(__file__))

import pathlib

######################## negotiation data ########################
MAX_HORIZON = 5  # TODO: to make > 1, change predict_outcome...

AUX_NEEDED_START = 0
AUX_NEEDED_END = MAX_HORIZON - 1
AUX_PRICE_START = MAX_HORIZON
AUX_PRICE_END = 2 * MAX_HORIZON - 1
OFFER_RELATIVE_TIME = AUX_PRICE_END + 1
OFFER_STATE_STEP = AUX_PRICE_END + 2
OFFER_QUANTITY = AUX_PRICE_END + 3
OFFER_COST = AUX_PRICE_END + 4
OFFER_TIME = AUX_PRICE_END + 5
OFFER_UTILITY = AUX_PRICE_END + 6
RESPONSE_RELATIVE_TIME = AUX_PRICE_END + 7  # 0
RESPONSE_STATE_STEP = AUX_PRICE_END + 8  # 1
RESPONSE_OFFER_QUANTITY = AUX_PRICE_END + 9  # 2
RESPONSE_OFFER_COST = AUX_PRICE_END + 10  # 3
RESPONSE_OFFER_TIME = AUX_PRICE_END + 11  # 4
RESPONSE_UTILITY = AUX_PRICE_END + 12  # 5
###############################################################

######################## utility model ########################
UTILITY_SELL_PATH = pathlib.Path(__file__).parent / "sell_utility_prediction_model.pth"
UTILITY_BUY_PATH = pathlib.Path(__file__).parent / "buy_utility_prediction_model.pth"

UTILITY_INPUT_DIM = OFFER_UTILITY
UTILITY_HIDDEN_SIZE = 100
UTILITY_USE_DROPOUT = True
UTILITY_DROPOUT = 0.5
UTILITY_OUTPUT_DIM = 1  # a utility value
UTILITY_CLIP = 5
UTILITY_EPOCHS = 20
BAD_UTILITY_THRESHOLD = -1000
###############################################################

###################### negotiation model ######################
NEG_SELL_PATH = pathlib.Path(__file__).parent / "sell_neg_prediction_model.pth"
NEG_BUY_PATH = pathlib.Path(__file__).parent / "buy_neg_prediction_model.pth"

NEG_INPUT_DIM = RESPONSE_RELATIVE_TIME
NEG_HIDDEN_SIZE = 100
NEG_USE_DROPOUT = True
NEG_DROPOUT = 0.5
NEG_OUTPUT_DIM = RESPONSE_UTILITY - RESPONSE_RELATIVE_TIME + 1
NEG_CLIP = 5
NEG_EPOCHS = 20
###############################################################

###################### negotiation train ######################
NEG_TRAIN_DATA = 2
NEG_LOAD_MODEL = False
NEG_SAVE_MODEL = False
NEG_VALIDATION_SPLIT = 0.1
NEG_TRAIN_SELLER = True
NEG_TRAIN_BUYER = True
###############################################################

######################## utility train ########################
UTILITY_TRAIN_DATA = 1
UTILITY_LOAD_MODEL = False
UTILITY_SAVE_MODEL = False
UTILITY_VALIDATION_SPLIT = 0.1
UTILITY_TRAIN_SELLER = True
UTILITY_TRAIN_BUYER = True
###############################################################

######################### trade model #########################
TRADE_INPUT_PATH = pathlib.Path(__file__).parent / "trade_prediction_input_model.pth"
TRADE_OUTPUT_PATH = pathlib.Path(__file__).parent / "trade_prediction_output_model.pth"

TRADE_INPUT_DIM = 10
TRADE_HIDDEN_SIZE = 100
TRADE_USE_DROPOUT = True
TRADE_DROPOUT = 0.5
TRADE_OUTPUT_DIM = MAX_HORIZON
TRADE_EPOCHS = 5
###############################################################

######################### trade train #########################
TRADE_TRAIN_DATA = 100
TRADE_LOAD_MODEL = False
TRADE_SAVE_MODEL = False
TRADE_VALIDATION_SPLIT = 0.1
###############################################################
