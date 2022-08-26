from statistics import mean

import numpy as np
import pandas as pd
from negmas import SAONMI
from typing import List, Dict

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2
INF = 1000
ON = True

__all__ = [
    "print_log",
    "param_normalization",
    "t",
    "shorten_name",
    "type_name",
    "opponent_agreements",
    "worst_opp_acc_price_",
    "worst_opp_acc_price",
    "TF_sign",
    "opponent_rank",
    "price_comparison",
    "todays_contract",
    "opp_name_from_contract",
    "opp_name_from_ami",
    "price_normalization",
    "price_inverse_normalization",
    "calculate_price_with_slack",
]


def print_log(names, values, on=ON):
    if on:
        if type(names) == str:
            pass  # print(f"{names}:{values}")
        if type(names) == list:
            for name, value in dict(zip(names, values)).items():
                pass  # print(f"{name}:{value}", end=' ')
            pass  # print()


def param_normalization(params: list):
    params_sum = sum(params)
    return [_ / params_sum for _ in params]


def t(step, n_steps):
    return (step + 1) / n_steps


def shorten_name(name: str):
    return name.split("-")[0]


def type_name(name: str):
    return name[2:6]


def opponent_agreements(ami: SAONMI, is_selling: bool, success_contracts: list) -> list:
    """指定された相手との合意（contract）を返す"""
    if is_selling:
        opponent_name = ami.annotation["buyer"]
        success_agreements = [
            _ for _ in success_contracts if _.partners[0] == opponent_name
        ]
    else:
        opponent_name = ami.annotation["seller"]
        success_agreements = [
            _ for _ in success_contracts if _.partners[1] == opponent_name
        ]

    return success_agreements


def worst_opp_acc_price_(
    ami: SAONMI, is_selling: bool, success_contracts: list
) -> float:
    """
    〜旧版〜
    指定された相手との合意の中で，相手にとって最も良い価格を返す．
    合意がない場合は，0かinfを返す．
    """
    success_agreements = opponent_agreements(ami, is_selling, success_contracts)

    if is_selling:
        price = min([_.agreement["unit_price"] for _ in success_agreements] + [INF])
    else:
        price = max([_.agreement["unit_price"] for _ in success_agreements] + [0])

    return price


def worst_opp_acc_price(
    ami: SAONMI, is_selling: bool, success_contracts: list
) -> float:
    """
    指定された相手との合意の中で，相手にとって最も良い価格を返す．
    合意がない場合は，0かinfを返す．
    :param ami:
    :param is_selling:
    :param success_contracts:
    :return worst_opp_acc_price:
    """
    success_agreements = opponent_agreements(ami, is_selling, success_contracts)

    if not success_agreements:
        # listが空の場合は，最も悪い価格を返す
        if is_selling:
            price = 0
        else:
            price = INF
    else:
        # 合意がある時
        if is_selling:
            price = min([_.agreement["unit_price"] for _ in success_agreements] + [INF])
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


def todays_contract(step: int, success_contract: list) -> list:
    return [_ for _ in success_contract if _.agreement["time"] == step]


def opp_name_from_contract(is_selling: bool, contract) -> str:
    if is_selling:
        return contract.partners[0]
    else:
        return contract.partners[1]


def opp_name_from_ami(is_selling: bool, ami: SAONMI):
    name = ami.annotation["buyer"] if is_selling else ami.annotation["seller"]
    return name


def price_normalization(is_selling: bool, price: float, up_range: dict):
    """
    与えられた価格を売り手か買い手かにしたがって，正規化する
    売り手の場合，最高値を0，最安値を1とする
    買い手の場合，最安値を0，最高値を1とする

    :param is_selling: 売り手か買い手か
    :param price: 正規化の対象となる価格
    :param up_range: マーケットにおける価格の最大値・最小値
    :return: 正規化した価格[0,1] or NaN
    """

    if price == np.nan:
        return np.nan

    mx, mn = up_range["max_price"], up_range["min_price"]
    if is_selling:
        result = (price - mx) / (mn - mx)
    else:
        result = (price - mn) / (mx - mn)

    result = 0 if result < 0 else result
    result = 1 if result > 1 else result
    return result


def price_inverse_normalization(
    is_selling: bool, normalized_value: float, up_range: dict
):
    """
    与えられた正規化値を売り手か買い手かにしたがって，正規化の逆変換を行う

    :param normalized_value: 標準化の対象となる価格
    :param is_selling: 売り手か買い手か
    :param up_range: マーケットにおける価格の最大値・最小値
    :return: 逆変換した価格 or NaN
    """

    if normalized_value == np.nan:
        return np.nan

    mx, mn = up_range["max_price"], up_range["min_price"]
    if is_selling:
        result = mx - (mx - mn) * normalized_value
    else:
        result = mn + (mx - mn) * normalized_value

    result = mn if result < mn else result
    result = mx if result > mx else result

    return result


def calculate_price_with_slack(
    is_selling: bool, price: float, slack: float, up_range: dict
):
    """
    与えられた価格 price をスラック変数 slack に基づいて計算する
    価格帯 up_range の最大値or最小値の差を用いる

    :param is_selling: 売り手か買い手か
    :param price: 対象となる価格
    :param slack: スラック変数
    :param up_range: 価格帯の最大値及び最小値
    :return: 計算結果
    """

    rng = up_range["max_price"] - up_range["min_price"]
    if is_selling:
        if up_range["max_price"] == price:
            result = up_range["max_price"]
        else:
            result = price - slack * rng
    else:
        if up_range["min_price"] == price:
            result = up_range["min_price"]
        else:
            result = price + slack * rng

    return result
