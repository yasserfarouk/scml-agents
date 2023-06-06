import math
import random
from itertools import combinations
from typing import *

# from .negotiation_control_strategy import *
# from .trading_strategy import *
import numpy as np
from negmas import AspirationNegotiator, LinearUtilityFunction, MappingUtilityFunction
from scml.scml2020 import (
    NO_COMMAND,
    QUANTITY,
    TIME,
    UNIT_PRICE,
    DecentralizingAgent,
    SCML2020Agent,
)
from scml.scml2020.components.negotiation import IndependentNegotiationsManager
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    SupplyDrivenProductionStrategy,
    TradeDrivenProductionStrategy,
)
from scml.scml2020.components.trading import PredictionBasedTradingStrategy

from .agents_std import Prot11std


class Prot11(Prot11std):
    my_friends = dict()

    def init(self):
        super().init()
        self.my_friends[self.id] = dict(
            level=self.awi.my_input_product,
            cost=self.cost,
            step_input_quantities=self.step_input_quantities,
            step_output_quantities=self.step_output_quantities,
        )

    """def init_(self):
        super().init_()
        self.same_level_friends = {id:info for id,info in self.my_friends.items() if info['level']==self.awi.my_input_product} # {id1:info1, id2:info2, ...}
        pass # print(list(self.same_level_friends.keys()))"""

    def sign_all_contracts(self, contracts: List["Contract"]) -> List[Optional[str]]:
        signed = [None] * len(contracts)
        mysigned_step_input_quantities = [0] * self.awi.n_steps
        mysigned_step_output_quantities = [0] * self.awi.n_steps
        output_contracts_by_step = [[] for _ in range(self.awi.n_steps)]

        for i, contract in enumerate(contracts):
            # print(contract)
            t, p, q, is_seller = (
                contract.agreement["time"],
                contract.agreement["unit_price"],
                contract.agreement["quantity"],
                contract.annotation["seller"] == self.id,
            )
            partner = (
                contract.annotation["buyer"]
                if is_seller
                else contract.annotation["seller"]
            )
            self.opp_n_contracts[is_seller][partner] += 1

            if t < self.awi.current_step or self.awi.n_steps - 1 < t:
                continue

            # 売りの契約は一旦リストにまとめておく
            if is_seller:
                output_contracts_by_step[t].append(contract)
            # フレンドの買いの契約に優先的に署名．各日の入荷量が工場の生産ライン数を超えないようにする．最終日は生産品の在庫コスト-生産コストより安井契約があれば署名
            elif partner in self.my_friends.keys():
                # print(self.id,partner,self.my_friends.keys())
                if (
                    self.step_input_quantities[t]
                    + mysigned_step_input_quantities[t]
                    + q
                    <= self.awi.n_lines
                ):
                    if self.awi.current_step < self.awi.n_steps - 1:
                        signed[i] = self.id
                        mysigned_step_input_quantities[t] += q
                    elif (
                        p
                        < self.awi.trading_prices[self.awi.my_output_product] / 2
                        - self.cost
                    ):
                        signed[i] = self.id
                        mysigned_step_input_quantities[t] += q

        # フレンドの以外の買いの契約に署名．各日の入荷量が工場の生産ライン数を超えないようにする．最終日は生産品の在庫コスト-生産コストより安井契約があれば署名
        for i, contract in enumerate(contracts):
            t, p, q, is_seller = (
                contract.agreement["time"],
                contract.agreement["unit_price"],
                contract.agreement["quantity"],
                contract.annotation["seller"] == self.id,
            )
            partner = (
                contract.annotation["buyer"]
                if is_seller
                else contract.annotation["seller"]
            )
            if (not is_seller) and (partner not in self.my_friends.keys()):
                if (
                    self.step_input_quantities[t]
                    + mysigned_step_input_quantities[t]
                    + q
                    <= self.awi.n_lines
                ):
                    if partner != "SELLER":
                        if self.awi.current_step < self.awi.n_steps - 1:
                            signed[i] = self.id
                            mysigned_step_input_quantities[t] += q
                        elif (
                            p
                            < self.awi.trading_prices[self.awi.my_output_product] / 2
                            - self.cost
                        ):
                            signed[i] = self.id
                            mysigned_step_input_quantities[t] += q
                    elif (
                        self.awi.current_step < self.awi.n_steps / 2
                        or sum(self.step_input_quantities)
                        + sum(mysigned_step_input_quantities)
                        - sum(self.step_output_quantities)
                        < self.max_inventory
                    ):
                        # print(partner,sum(self.step_input_quantities)+sum(mysigned_step_input_quantities)-sum(self.step_output_quantities),self.max_inventory)
                        if self.awi.current_step < self.awi.n_steps - 1:
                            signed[i] = self.id
                            mysigned_step_input_quantities[t] += q
                        elif (
                            p
                            < self.awi.trading_prices[self.awi.my_output_product] / 2
                            - self.cost
                        ):
                            signed[i] = self.id
                            mysigned_step_input_quantities[t] += q
                    # else:
                    #    print(partner,sum(self.step_input_quantities)+sum(mysigned_step_input_quantities)-sum(self.step_output_quantities),self.max_inventory)

        # 売りの契約に署名．署名済み(finarize済)の契約により確定している生産品在庫を可能な限り全て売る

        total_output = sum(self.step_output_quantities)  # 最終日までの総出荷予定量

        for t in range(self.awi.current_step, self.awi.n_steps):
            target_quantity = (
                sum(self.step_input_quantities[:t]) - total_output
            )  # t-1日目までの総入荷量-総出荷(予定)量 (製品の生産は契約の実行より後のフェーズのため入荷品を生産して使えるのは翌日から)

            contract_combinations = []
            for l in range(len(output_contracts_by_step[t])):
                contract_combinations += list(
                    combinations(output_contracts_by_step[t], l + 1)
                )

            candidates = []
            while candidates == [] and target_quantity > 0:
                for contract_combination in contract_combinations:
                    # print(self.awi.current_step,t,[contract.agreement["quantity"] for contract in contract_combination])
                    if (
                        sum(
                            [
                                contract.agreement["quantity"]
                                for contract in contract_combination
                            ]
                        )
                        == target_quantity
                    ):
                        candidates.append(contract_combination)
                target_quantity -= 1

            if candidates == []:
                continue

            random.shuffle(candidates)

            signed_combination = candidates[0]  # 後でもうちょっとちゃんと選ぶように改良する
            # 署名する売りの契約の組み合わせの候補(candidates)から，相手の署名率*契約単価を最大化する契約の組み合わせを選択
            # val = sum([contract.agreement["unit_price"]*self.opp_sign_rates[1][contract.annotation['buyer']] for contract in signed_combination])
            max_up = max(
                [
                    max([contract.agreement["unit_price"] for contract in candidate])
                    for candidate in candidates
                ]
            )
            # val = sum([contract.agreement["unit_price"]/max_up + self.opp_sign_rates[1][contract.annotation['buyer']] for contract in signed_combination])
            # val = sum([(1-self.awi.current_step/(self.awi.n_steps-1))*contract.agreement["unit_price"]/max_up + (self.awi.current_step/(self.awi.n_steps-1))*self.opp_sign_rates[1][contract.annotation['buyer']] for contract in signed_combination])
            val = sum(
                [
                    (
                        0.5
                        - 0.5
                        * np.sin(
                            (self.awi.current_step / (self.awi.n_steps - 1) - 0.5)
                            * np.pi
                        )
                    )
                    * contract.agreement["unit_price"]
                    / max_up
                    + (
                        0.5
                        + 0.5
                        * np.sin(
                            (self.awi.current_step / (self.awi.n_steps - 1) - 0.5)
                            * np.pi
                        )
                    )
                    * self.opp_sign_rates[1][contract.annotation["buyer"]]
                    for contract in signed_combination
                ]
            )

            for candidate in candidates[1:]:
                # candidate_val = sum([contract.agreement["unit_price"]*self.opp_sign_rates[1][contract.annotation['buyer']] for contract in candidate]) # 合意単価と相手の署名率の積で価値を算出
                # candidate_val = sum([contract.agreement["unit_price"]/max_up + self.opp_sign_rates[1][contract.annotation['buyer']] for contract in candidate]) # 合意単価/最高合意単価と相手の署名率の和で価値を算出
                # candidate_val = sum([(1-self.awi.current_step/(self.awi.n_steps-1))*contract.agreement["unit_price"]/max_up + (self.awi.current_step/(self.awi.n_steps-1))*self.opp_sign_rates[1][contract.annotation['buyer']] for contract in candidate]) # 合意単価/最高合意単価と相手の署名率の，シミュレーションの終盤ほど署名率重視になるように重み付けした線形和で価値を算出
                candidate_val = sum(
                    [
                        (
                            0.5
                            - 0.5
                            * np.sin(
                                (self.awi.current_step / (self.awi.n_steps - 1) - 0.5)
                                * np.pi
                            )
                        )
                        * contract.agreement["unit_price"]
                        / max_up
                        + (
                            0.5
                            + 0.5
                            * np.sin(
                                (self.awi.current_step / (self.awi.n_steps - 1) - 0.5)
                                * np.pi
                            )
                        )
                        * self.opp_sign_rates[1][contract.annotation["buyer"]]
                        for contract in candidate
                    ]
                )  # 合意単価/最高合意単価と相手の署名率の，シミュレーションの終盤ほど署名率重視になるように重み付けした線形和で価値を算出

                # 現在の契約の組み合わせの候補にフレンドが含まれる場合，新たな組み合わせ候補にフレンドが含まれ，かつ評価値が高い場合のみ更新
                if (
                    sum(
                        [
                            contract.annotation["buyer"] in self.my_friends
                            for contract in signed_combination
                        ]
                    )
                    != 0
                ):
                    if (
                        sum(
                            [
                                contract.annotation["buyer"] in self.my_friends
                                for contract in candidate
                            ]
                        )
                        != 0
                    ):
                        if candidate_val > val:
                            signed_combination = candidate
                            val = candidate_val
                # 現在の契約の組み合わせの候補にフレンドが含まれない場合，
                else:
                    # 新たな組み合わせ候補にフレンドが含まれるか，
                    if (
                        sum(
                            [
                                contract.annotation["buyer"] in self.my_friends
                                for contract in candidate
                            ]
                        )
                        != 0
                    ):
                        signed_combination = candidate
                        val = candidate_val
                    # 新たな組み合わせ候補にもフレンドが含まれないが評価値が高い場合に更新
                    elif candidate_val > val:
                        signed_combination = candidate
                        val = candidate_val

            for i, contract in enumerate(contracts):
                t, p, q, is_seller = (
                    contract.agreement["time"],
                    contract.agreement["unit_price"],
                    contract.agreement["quantity"],
                    contract.annotation["seller"] == self.id,
                )
                if contract in signed_combination:
                    signed[i] = self.id
                    mysigned_step_output_quantities[t] += q
                    total_output += q

        return signed

    def on_contracts_finalized(
        self,
        signed: List["Contract"],
        cancelled: List["Contract"],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        self.my_friends[self.id]["step_input_quantities"] = self.step_input_quantities
        self.my_friends[self.id]["step_output_quantities"] = self.step_output_quantities

    def step(self):
        super().step()
        if self.awi.current_step == self.awi.n_steps - 1:
            del self.my_friends[self.id]

    """def acceptable_unit_price(self, step: int, sell: bool) -> int:
        acceptable_unit_price = super().acceptable_unit_price(step,sell)
        same_level_friends = {id:info for id,info in self.my_friends.items() if info['level']==self.awi.my_input_product}
        if len(same_level_friends)>=2 and not sell:
            if self.awi.current_step < self.awi.n_steps/4:
                return self.awi.trading_prices[self.awi.my_input_product]
            elif self.awi.current_step < self.awi.n_steps/2:
                return min(self.awi.trading_prices[self.awi.my_input_product],self.average_prices[0])
            elif self.awi.current_step < self.awi.n_steps-2:
                return self.average_prices[1]-min([info['cost'] for info in same_level_friends.values()])
            else:
                return self.awi.trading_prices[self.awi.my_output_product]/2-self.cost"""

    def target_quantity(self, step: int, sell: bool) -> int:
        target_q = super().target_quantity(step, sell)
        if not sell:
            same_level_friends = {
                id: info
                for id, info in self.my_friends.items()
                if info["level"] == self.awi.my_input_product
            }  # {id1:info1, id2:info2, ...}
            # print(self.awi.current_step,self.id,self.my_friends.keys())
            # print(same_level_friends.keys())
            if len(list(same_level_friends.keys())) >= 2:
                same_level_friends = sorted(
                    same_level_friends.items(), key=lambda x: x[1]["cost"]
                )  # [(id1,info1), (id2,info2), ...]
                # print(same_level_friends)
                rank = 0
                for id, _ in same_level_friends:
                    if id == self.id:
                        break
                    rank += 1
                target_q = max(1, self.awi.n_lines // (rank + 1))
        return target_q