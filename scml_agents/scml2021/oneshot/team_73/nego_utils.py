from statistics import mean
from typing import List

from negmas import SAONMI

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2


def param_normalization(params: list):
    params_sum = sum(params)
    return [_ / params_sum for _ in params]


def t(step, n_steps):
    return (step + 1) / n_steps


def shorten_name(name: str):
    return name.split("-")[0]


def opponent_agreements(nmi: SAONMI, is_selling: bool, success_contracts: list) -> list:
    """指定された相手との合意（contract）を返す"""
    if is_selling:
        opponent_name = nmi.annotation["buyer"]
        success_agreements = [
            _ for _ in success_contracts if _.partners[0] == opponent_name
        ]
    else:
        opponent_name = nmi.annotation["seller"]
        success_agreements = [
            _ for _ in success_contracts if _.partners[1] == opponent_name
        ]

    return success_agreements


def worst_opp_acc_price(
    nmi: SAONMI, is_selling: bool, success_contracts: list
) -> float:
    """
    指定された相手との合意の中で，相手にとって最も良い価格を返す．
    合意がない場合は，0かinfを返す．
    :param nmi:
    :param is_selling:
    :param success_contracts:
    :return worst_opp_acc_price:
    """
    success_agreements = opponent_agreements(nmi, is_selling, success_contracts)

    if is_selling:
        price = min(
            [_.agreement["unit_price"] for _ in success_agreements] + [float("inf")]
        )
    else:
        price = max([_.agreement["unit_price"] for _ in success_agreements] + [0])

    return price


def TF_sign(x: bool):
    """Trueなら1，Falseなら-1を返す"""
    if x:
        return 1
    else:
        return -1


def opponent_rank(opponent_names: List[str], is_selling: bool, success_contract: list):
    """相手を合意価格によって順位付け"""
    rank = {}
    if is_selling:
        for name in opponent_names:
            agreements = [
                _.agreement["unit_price"]
                for _ in success_contract
                if _.partners[0] == name
            ]
            rank[name] = mean(agreements) if agreements else 0
        sorted(rank.items(), key=lambda x: x[1], reverse=True)
    else:
        for name in opponent_names:
            agreements = [
                _.agreement["unit_price"]
                for _ in success_contract
                if _.partners[1] == name
            ]
            rank[name] = mean(agreements) if agreements else float("inf")
        sorted(rank.items(), key=lambda x: x[1], reverse=False)

    return rank


def price_comparison(is_selling: bool, x: float, y: float) -> bool:
    """
    与えられた2つの価格x,yのうち，xの方がエージェントにとって良い場合，Trueを返す．
    :param is_selling:
    :param x:
    :param y:
    :return: True or False
    """
    if is_selling:
        return x >= y
    else:
        return x <= y
