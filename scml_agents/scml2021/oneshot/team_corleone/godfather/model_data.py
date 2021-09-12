import copy
import warnings
from datetime import datetime
from typing import List

from scml import *
from scml.oneshot import *

from .bilat_ufun import *

# from .model import *
from .model_empirical import ModelEmpirical
from .negotiation_history import BilateralHistory, WorldInfo
from .offer import Offer
from .spaces import *
from .strategy import Strategy, StrategyGoldfishParetoAspiration

NOTHING_VAL = -99


def calc_world_cols(day_idx, agent):
    my_level = agent.awi.level
    opp_level = (agent.awi.level + 1) % 2
    my_layer_size = len(agent.awi.all_consumers[my_level])
    opp_layer_size = len(agent.awi.all_consumers[opp_level])
    comp = (
        agent.awi.exogenous_contract_summary[0][0]
        - agent.awi.exogenous_contract_summary[-1][0]
    )

    headers = [
        "day",
        "n_level",
        "n_opp_level",
        "my_layer_size",
        "opp_layer_size",
        "competitiveness",
    ]
    row = [day_idx, my_level, opp_level, my_layer_size, opp_layer_size, comp]
    return row, headers


def calc_trace_cols(hist: BilateralHistory, len_trace: int):
    headers = []
    hist_data = []
    trace = hist.moves[:len_trace]
    for i in range(41):
        headers.append(f"trace {i} price")
        headers.append(f"trace {i} quant")
        if i < len_trace and type(trace[i][1]) == Offer:
            offer: Offer = trace[i][1]  # type: ignore
            hist_data.append(offer.price)
            hist_data.append(offer.quantity)
        else:
            hist_data.append(NOTHING_VAL)
            hist_data.append(NOTHING_VAL)
    return hist_data, headers


def calc_empirical_cols(prev_negs: List[BilateralHistory], offer_space, prior_bias=0):
    dummy_negotiation = BilateralHistory(
        "", OutcomeSpace(offer_space), WorldInfo(-1, -1, -1, -1)
    )

    model_empirical = ModelEmpirical(
        "", strategy_self=StrategyGoldfishParetoAspiration, prior_bias=prior_bias
    )
    distr = model_empirical(
        BilatUFunUniform(offer_space),  # dummy ufun
        prev_negs if prev_negs else [dummy_negotiation],  # need at least one in there
        enable_realistic_checks=False,
    )  # bc of dummy ufun
    q_probs, est_p = distr.marginalize()

    headers = [f"empirical_distr_q_{i}" for i in range(len(q_probs))] + [
        "empirical_distr_p"
    ]
    row = q_probs + [est_p]
    return row, headers

    # empirical_distr_q_probs = [0.0] * 10
    # empirical_distr_p = 0.0
    # for prev_neg in prev_negs:
    #     outc = Offer(0, 0)
    #     try:
    #         outc = prev_neg.outcome()
    #     except Exception as e:
    #         warnings.warn("Couldn't find outcome offer")
    #     empirical_distr_q_probs[outc.quantity] += 1
    #     empirical_distr_p += outc.price
    # empirical_distr_p /= (len(prev_negs) or 1)
    # z = sum(empirical_distr_q_probs)
    # empirical_distr_q_probs = [i / z for i in empirical_distr_q_probs]


def calc_outcome_cols(hist: BilateralHistory):
    outcome_cols = []
    try:
        outcome_offer = hist.outcome()
    except Exception as e:
        warnings.warn("Couldn't find outcome offer")
        outcome_offer = Offer(0, 0)
    outcome_cols.append(outcome_offer.price)
    outcome_cols.append(outcome_offer.quantity)
    headers = ["outcome_price", "outcome_quant"]
    return outcome_cols, headers


def calc_ufun_cols(ufun_params):
    row = ufun_params
    headers = [f"ufun_param_{i}" for i in range(len(row))]
    return row, headers


def insert_rows_for_agent_pairing(
    rows_by_trace_idx,
    headers_by_trace_idx,
    hists,
    opp_type,
    offer_space,
    world_data,
    world_headers,
):
    hist = hists[-1]
    for (
        prediction_point
    ) in hist.prediction_points:  # -2 because the last one is always empty
        len_trace = prediction_point["trace_idx"]
        if len_trace >= 50:
            warnings.warn("trace longer than 50" + str(list(enumerate(hist.moves))))
            continue

        outcome_cols, outcome_headers = calc_outcome_cols(hist)
        trace_cols, trace_headers = calc_trace_cols(hist, len_trace)
        ufun_cols, ufun_headers = calc_ufun_cols(prediction_point["ufun_params"])
        empirical_cols, empirical_headers = calc_empirical_cols(hists, offer_space)

        data, head = outcome_cols, outcome_headers
        data.extend(trace_cols)
        head.extend(trace_headers)
        data.extend(copy.copy(world_data))
        head.extend(world_headers)
        data.extend(ufun_cols)
        head.extend(ufun_headers)
        data.extend(empirical_cols)
        head.extend(empirical_headers)
        data.extend([opp_type])
        head.extend(["opp_type"])
        rows_by_trace_idx[len_trace].append(data)
        headers_by_trace_idx[len_trace] = head


def construct_test_cols(
    day_idx, agent, histories, opp_type, offer_space, len_trace, ufun_params
):
    """For use in Godfather"""
    hist = histories[-1]
    trace_cols, trace_headers = calc_trace_cols(hist, len_trace)
    world_data, world_headers = calc_world_cols(day_idx, agent)
    ufun_cols, ufun_headers = calc_ufun_cols(ufun_params)
    empirical_cols, empirical_headers = calc_empirical_cols(
        histories, offer_space, prior_bias=10
    )  # TODO remove later

    data, head = trace_cols, trace_headers
    data.extend(world_data)
    head.extend(world_headers)
    data.extend(ufun_cols)
    head.extend(ufun_headers)
    data.extend(empirical_cols)
    head.extend(empirical_headers)
    # data.extend([opp_type]) # TODO add later
    # head.extend(['opp_type'])
    return data, head
