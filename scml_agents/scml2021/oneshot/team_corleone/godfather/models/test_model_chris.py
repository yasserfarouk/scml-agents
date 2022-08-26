#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error

if __name__ == "__main__":

    #%%
    models_q = {}
    models_p = {}
    for i in range(0, 22, 2):
        models_q[i] = xgb.Booster()
        models_p[i] = xgb.Booster()
        models_p[i].load_model(f"p_model_{i}.json")
        models_q[i].load_model(f"q_model_{i}.json")

    #%%
    HEADERS_LIST = "trace 0 price,trace 0 quant,trace 1 price,trace 1 quant,trace 2 price,trace 2 quant,trace 3 price,trace 3 quant,trace 4 price,trace 4 quant,trace 5 price,trace 5 quant,trace 6 price,trace 6 quant,trace 7 price,trace 7 quant,trace 8 price,trace 8 quant,trace 9 price,trace 9 quant,trace 10 price,trace 10 quant,trace 11 price,trace 11 quant,trace 12 price,trace 12 quant,trace 13 price,trace 13 quant,trace 14 price,trace 14 quant,trace 15 price,trace 15 quant,trace 16 price,trace 16 quant,trace 17 price,trace 17 quant,trace 18 price,trace 18 quant,trace 19 price,trace 19 quant,trace null,trace null.1,day,n_level,n_opp_level,my_layer_size,opp_layer_size,competitiveness,ufun_param_0,ufun_param_1,ufun_param_2,ufun_param_3,ufun_param_4,ufun_param_5,ufun_param_6,ufun_param_7,ufun_param_8,ufun_param_9,empirical_distr_q_0,empirical_distr_q_1,empirical_distr_q_2,empirical_distr_q_3,empirical_distr_q_4,empirical_distr_q_5,empirical_distr_q_6,empirical_distr_q_7,empirical_distr_q_8,empirical_distr_q_9,empirical_distr_q_10,empirical_distr_p"

    for _ in range(10000):
        l1 = "outcome_price,outcome_quant,trace 0 price,trace 0 quant,trace 1 price,trace 1 quant,trace 2 price,trace 2 quant,trace 3 price,trace 3 quant,trace 4 price,trace 4 quant,trace 5 price,trace 5 quant,trace 6 price,trace 6 quant,trace 7 price,trace 7 quant,trace 8 price,trace 8 quant,trace 9 price,trace 9 quant,trace 10 price,trace 10 quant,trace 11 price,trace 11 quant,trace 12 price,trace 12 quant,trace 13 price,trace 13 quant,trace 14 price,trace 14 quant,trace 15 price,trace 15 quant,trace 16 price,trace 16 quant,trace 17 price,trace 17 quant,trace 18 price,trace 18 quant,trace 19 price,trace 19 quant,trace null,trace null.1,day,n_level,n_opp_level,my_layer_size,opp_layer_size,competitiveness,ufun_param_0,ufun_param_1,ufun_param_2,ufun_param_3,ufun_param_4,ufun_param_5,ufun_param_6,ufun_param_7,ufun_param_8,ufun_param_9,empirical_distr_q_0,empirical_distr_q_1,empirical_distr_q_2,empirical_distr_q_3,empirical_distr_q_4,empirical_distr_q_5,empirical_distr_q_6,empirical_distr_q_7,empirical_distr_q_8,empirical_distr_q_9,empirical_distr_q_10,empirical_distr_p"
        l1 = l1.split(",")
        l2 = [1] * 70
        cols = HEADERS_LIST.split(",")
        row = pd.DataFrame([l2], columns=cols)
        row = row[models_q[0].feature_names]
        df = xgb.DMatrix(row)
        models_q[0].predict(df)

    #%%
