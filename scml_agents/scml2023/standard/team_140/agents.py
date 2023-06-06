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

__all__ = ["AgentVSC"]

MAX_INVENTORY = 50
MIN_INVENTORY = 5

"""
- 入荷した製品を(入荷品の平均単価+生産コスト)以上で販売するエージェント
- 入荷品を一つ以上獲得するまでは自身が売り手となる交渉には参加しない
- (入荷品の平均単価+生産コスト)が出荷品の市場価格を超える場合
    - 入荷品の市場価格 < 入荷品の平均単価：入荷のための交渉のprice_range=[1,(出荷品の市場価格-生産コスト)]
    - 
"""


class AgentVSC(SCML2020Agent):
    my_friends = dict()

    def init(self):
        self.average_prices = [
            self.awi.catalog_prices[self.awi.my_input_product],
            self.awi.catalog_prices[self.awi.my_output_product],
        ]
        self.step_input_quantities = [0] * self.awi.n_steps
        self.step_output_quantities = [0] * self.awi.n_steps
        self.cost = self.awi.profile.costs[0][self.awi.my_input_product]
        self.opp_sign_rates = [
            {partner: 1.0 for partner in self.awi.my_suppliers},
            {partner: 1.0 for partner in self.awi.my_consumers},
        ]
        self.opp_n_contracts = [
            {partner: 0 for partner in self.awi.my_suppliers},
            {partner: 0 for partner in self.awi.my_consumers},
        ]
        self.opp_n_signed_contracts = [
            {partner: 0 for partner in self.awi.my_suppliers},
            {partner: 0 for partner in self.awi.my_consumers},
        ]
        self.inventory = 0
        self.max_inventory = MAX_INVENTORY
        self.min_inventory = MIN_INVENTORY
        self.p_min_for_selling = self.awi.catalog_prices[self.awi.my_output_product]
        self.p_max_for_buying = self.awi.catalog_prices[self.awi.my_input_product]

        self.my_friends[self.id] = dict(
            level=self.awi.my_input_product,
            cost=self.cost,
            step_input_quantities=self.step_input_quantities,
            step_output_quantities=self.step_output_quantities,
        )

        # print(f'{self.id} catalog proces:{self.awi.catalog_prices[self.awi.my_input_product:self.awi.my_output_product+1]}, production cost:{self.cost}')

    def sign_all_contracts(self, contracts: List["Contract"]) -> List[Optional[str]]:
        signed = [None] * len(contracts)
        mysigned_step_input_quantities = [0] * self.awi.n_steps
        mysigned_step_output_quantities = [0] * self.awi.n_steps
        input_contructs_with_index = []
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
            # フレンドの買いの契約に優先的に署名．各日の入荷量が工場の生産ライン数を超えないようにする．最終日は生産品の在庫コスト-生産コストより安い契約があれば署名
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
            else:
                input_contructs_with_index.append([i, contract])

        # フレンドの以外の買いの契約に署名．各日の入荷量が工場の生産ライン数を超えないようにする．最終日は生産品の在庫コスト-生産コストより安井契約があれば署名
        input_contructs_with_index = sorted(
            input_contructs_with_index, key=lambda x: x[1].agreement["unit_price"]
        )
        for i, contract in input_contructs_with_index:
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
                if p > self.average_prices[1] - self.cost:
                    break

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
                        sum(self.step_input_quantities)
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

        # print(f"{self.id} step:{self.awi.current_step}  signed(is_caller,is_seller,(q,t,p)):{[(contract.annotation['caller']==self.id, contract.annotation['seller']==self.id, (contract.agreement['quantity'],contract.agreement['time'],contract.agreement['unit_price'])) for contract in signed]}")

        for contract in signed:
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
            self.opp_n_signed_contracts[is_seller][partner] += 1

            # 平均価格を更新
            if is_seller:
                old_total_q = sum(self.step_output_quantities)
                self.step_output_quantities[t] += q
            else:
                old_total_q = sum(self.step_input_quantities)
                self.step_input_quantities[t] += q
            if q > 0:
                self.average_prices[is_seller] = (
                    self.average_prices[is_seller] * old_total_q + p * q
                ) / (old_total_q + q)

        for contract, rejector in zip(cancelled, rejectors):
            # rejectorが自分だけなら相手は署名をしているので，opp_n_signed_contractsをインクリメント
            if rejector == [self.id]:
                is_seller = contract.annotation["seller"] == self.id
                partner = (
                    contract.annotation["buyer"]
                    if is_seller
                    else contract.annotation["seller"]
                )
                self.opp_n_signed_contracts[is_seller][partner] += 1

        # 相手の署名率を更新
        for partner in self.awi.my_suppliers:
            if self.opp_n_contracts[0][partner] != 0:
                self.opp_sign_rates[0][partner] = (
                    self.opp_n_signed_contracts[0][partner]
                    / self.opp_n_contracts[0][partner]
                )
        for partner in self.awi.my_consumers:
            if self.opp_n_contracts[1][partner] != 0:
                self.opp_sign_rates[1][partner] = (
                    self.opp_n_signed_contracts[1][partner]
                    / self.opp_n_contracts[1][partner]
                )

        self.my_friends[self.id]["step_input_quantities"] = self.step_input_quantities
        self.my_friends[self.id]["step_output_quantities"] = self.step_output_quantities

        # inventoryとp_max_for_buyingを更新
        pre_inventory = self.inventory
        self.inventory = sum(self.step_input_quantities) - sum(
            self.step_output_quantities
        )
        # 自分の在庫が増えたら買いの交渉への参加条件の価格を強気に(0.8倍に)
        if pre_inventory > self.inventory and self.inventory >= self.min_inventory:
            self.p_max_for_buying = max(
                0.8 * self.p_max_for_buying,
                1
                if self.awi.my_input_product == 0
                else min(
                    self.awi.catalog_prices[self.awi.my_input_product - 1],
                    self.awi.trading_prices[self.awi.my_input_product - 1],
                ),
            )
        # 自分の在庫が減ったら買いの交渉への参加条件の価格を弱気に(1.25倍に)
        elif pre_inventory < self.inventory and self.inventory <= self.max_inventory:
            # self.p_max_for_buying = min(1.25*self.p_max_for_buying,min(self.average_prices[1],self.awi.catalog_prices[self.awi.my_output_product],self.awi.trading_prices[self.awi.my_output_product])-self.cost)
            self.p_max_for_buying = min(
                1.1 * self.p_max_for_buying,
                min(
                    self.average_prices[1],
                    self.awi.catalog_prices[self.awi.my_output_product],
                    self.awi.trading_prices[self.awi.my_output_product],
                )
                - self.cost,
            )

    def before_step(self):
        super().before_step()

        # print(self.awi.my_suppliers,self.awi.my_consumers)

        if MAX_INVENTORY < self.awi.n_steps:
            if self.awi.current_step <= self.awi.n_steps - MAX_INVENTORY:
                self.max_inventory -= 1
        else:
            if self.awi.current_step <= self.awi.n_steps - MAX_INVENTORY / 5:
                self.max_inventory -= 5  # 在庫がmax_inventory個を超えたら買いの交渉には参加しない

        # Request Buying

        if self.awi.current_step == self.awi.n_steps - 2:
            # 最終日1日前には，最終日に実行される買いの契約についての交渉をリクエスト(実際に交渉が行われるのは最終日)

            prange = (
                0,
                self.awi.trading_prices[self.awi.my_output_product] / 2 - self.cost,
            )  # 在庫価値-生産コスト 以下の価格でのみ購入
            if prange[0] < prange[1]:
                qrange = (1, self.awi.n_lines)
                t = self.awi.n_steps - 1
                negotiators = [
                    AspirationNegotiator(
                        ufun=self.create_ufun_for_request(
                            is_seller=False, prange=prange, trange=(t, t), issues=None
                        )
                    )
                    for _ in self.awi.my_suppliers
                ]
                self.awi.request_negotiations(
                    is_buy=True,
                    product=self.awi.my_input_product,
                    unit_price=prange,
                    quantity=qrange,
                    time=t,
                    negotiators=negotiators,
                )
        elif self.awi.current_step < self.awi.n_steps - 2:
            # if sum(self.step_output_quantities)-sum(self.step_input_quantities)<self.max_inventory:
            # 最終日より2日以上前には，直近3日間に実行される買いの契約についての交渉をリクエスト
            prange = (0, self.p_max_for_buying)
            qrange = (1, self.awi.n_lines)
            trange = (
                self.awi.current_step + 1,
                min(self.awi.current_step + 3, self.awi.n_steps - 2),
            )
            negotiators = [
                AspirationNegotiator(
                    ufun=self.create_ufun_for_request(
                        is_seller=False, prange=prange, trange=trange, issues=None
                    )
                )
                for _ in self.awi.my_suppliers
            ]
            if trange[0] <= trange[1]:
                self.awi.request_negotiations(
                    is_buy=True,
                    product=self.awi.my_input_product,
                    quantity=qrange,
                    unit_price=prange,
                    time=trange,
                    negotiators=negotiators,
                )

        # Request Selling

        # (t-1日目までに入荷される原材料の総和)-(生産品の総予定出荷量)>0 → t日目の売りをリクエスト
        # t=self.awi.current_step+1で上記が成り立たなければ，(t日目までに入荷される原材料の総和)-(生産品の総予定出荷量)>0 → t+1日目の売りをリクエスト
        t = self.awi.current_step + 1
        q = sum(self.step_input_quantities[:t]) - sum(self.step_output_quantities)
        while q <= 0 and t < self.awi.n_steps - 1:
            q = sum(self.step_input_quantities[: t + 1]) - sum(
                self.step_output_quantities
            )
            t += 1
        if q <= 0:
            # 最終日(t==self.awi.n_steps-1)にq==0ならもう売れる分ない
            return None
        # p_min = self.average_prices[0]+self.cost
        # p_min = max(self.average_prices[0]+self.cost,self.awi.trading_prices[self.awi.my_output_product],self.average_prices[1])

        # p_min_for_sellingを更新
        if self.awi.current_step < self.awi.n_steps / 4:
            self.p_min_for_selling = self.awi.trading_prices[self.awi.my_output_product]
        elif self.awi.current_step < self.awi.n_steps / 2:
            self.p_min_for_selling = max(
                self.awi.trading_prices[self.awi.my_output_product],
                self.average_prices[1],
            )
        elif self.awi.current_step < self.awi.n_steps - 2:
            # 後半は材料費+生産コストまで妥協する
            self.p_min_for_selling = self.average_prices[0] + self.cost
        else:
            # 最終日の交渉では在庫コスト(市場価格の半分)まで妥協する
            self.p_min_for_selling = (
                self.awi.trading_prices[self.awi.my_output_product] / 2 + 1
            )

        prange = (self.p_min_for_selling, 4 * self.p_min_for_selling)
        qrange = (q // len(self.awi.my_consumers), q)
        negotiators = [
            AspirationNegotiator(
                ufun=self.create_ufun_for_request(
                    is_seller=True, prange=prange, trange=(t, t), issues=None
                )
            )
            for _ in self.awi.my_consumers
        ]
        if self.awi.current_step < self.awi.n_steps - 1:
            self.awi.request_negotiations(
                is_buy=False,
                product=self.awi.my_output_product,
                quantity=qrange,
                unit_price=prange,
                time=t,
                negotiators=negotiators,
            )

    def step(self):
        super().step()

        # produce everything I can
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        self.awi.set_commands(commands)

        if self.awi.current_step == 0:
            same_level_friends_costs = {
                id: info["cost"]
                for id, info in self.my_friends.items()
                if info["level"] == self.awi.my_input_product
            }  # {id1:info1, id2:info2, ...}
            self.p_max_for_buying = self.p_max_for_buying - (
                self.cost - min(same_level_friends_costs.values())
            )
        if self.awi.current_step == self.awi.n_steps - 1:
            del self.my_friends[self.id]
            """print(f'{self.id} step:{self.awi.current_step} cp:{self.awi.catalog_prices[self.awi.my_input_product:self.awi.my_output_product+1]} tp:{self.awi.trading_prices[self.awi.my_input_product:self.awi.my_output_product+1]} ap:{self.average_prices}')
            pass # print(f'         suppliers:{self.awi.my_suppliers} consumers:{self.awi.my_consumers}')
            pass # print(f'         input:{self.step_input_quantities}({sum(self.step_input_quantities)})')
            pass # print(f'         output:{self.step_output_quantities}({sum(self.step_output_quantities)})')"""

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "NegotiatorMechanismInterface",
    ) -> Optional["Negotiator"]:
        is_seller = annotation["seller"] == self.id
        # do not engage in negotiations that obviouly have bad prices for me
        # 提示された価格の選択肢が全て市場価格より悪いなら，その交渉には参加しない

        if is_seller and (
            issues[TIME].max_value <= self.awi.current_step
            or self.awi.n_steps <= issues[TIME].min_value
            or issues[UNIT_PRICE].max_value < self.p_min_for_selling
        ):
            return None
        if not is_seller and (
            issues[TIME].max_value <= self.awi.current_step
            or self.awi.n_steps <= issues[TIME].min_value
            or self.p_max_for_buying < issues[UNIT_PRICE].min_value
        ):
            return None
        ufun = self.create_ufun_for_respond(is_seller, issues=issues)
        return AspirationNegotiator(ufun=ufun)

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            return max(
                0,
                sum(self.step_input_quantities[:step])
                - sum(self.step_output_quantities),
            )
        else:
            return self.awi.n_lines

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        return self.p_min_for_selling if sell else self.p_max_for_buying

    def create_ufun_for_request(self, is_seller, prange, trange, issues):
        if is_seller:
            return MappingUtilityFunction(
                lambda x: -1000
                if x[TIME] < trange[0]
                or trange[1] < x[TIME]
                or x[UNIT_PRICE] < prange[0]
                else x[UNIT_PRICE],
                reserved_value=0.0,
                issues=issues,
            )
        return MappingUtilityFunction(
            lambda x: -1000
            if x[TIME] < trange[0] or trange[1] < x[TIME] or prange[1] < x[UNIT_PRICE]
            else prange[1] - x[UNIT_PRICE],
            reserved_value=0.0,
            issues=issues,
        )

    """def create_ufun_for_respond(self, is_seller: bool, issues, outcomes=None) -> "UtilityFunction":
        if is_seller:
            return MappingUtilityFunction(
                lambda x: -10000 if x[TIME]<self.awi.current_step or self.awi.n_steps-1<x[TIME] or x[QUANTITY]>self.target_quantity(x[TIME],is_seller) or x[UNIT_PRICE]<self.p_min_for_selling
                else (
                    self.awi.current_step/(self.awi.n_steps-1)*(x[QUANTITY]-issues[QUANTITY].min_value)/(issues[QUANTITY].max_value-issues[QUANTITY].min_value) + (x[UNIT_PRICE]-issues[UNIT_PRICE].min_value)/(issues[UNIT_PRICE].max_value-issues[UNIT_PRICE].min_value)
                    if issues!=None else x[UNIT_PRICE]
                ), reserved_value=0.0, issues=issues
            )
        else:
            return MappingUtilityFunction(
                lambda x: -10000 if x[TIME]<self.awi.current_step or self.awi.n_steps-1<x[TIME] or x[QUANTITY]>self.target_quantity(x[TIME],is_seller) or self.p_max_for_buying<x[UNIT_PRICE]
                else (
                    -0.25*(x[TIME]-issues[TIME].min_value)/(issues[TIME].max_value-issues[TIME].min_value) - (x[UNIT_PRICE]-issues[UNIT_PRICE].min_value)/(issues[UNIT_PRICE].max_value-issues[UNIT_PRICE].min_value)
                    if issues!=None else -x[UNIT_PRICE]
                ), reserved_value=0.0,issues=issues
            )"""

    def create_ufun_for_respond(
        self, is_seller: bool, issues, outcomes=None
    ) -> "UtilityFunction":
        if is_seller:
            return MappingUtilityFunction(
                lambda x: -1000
                if x[TIME] <= self.awi.current_step
                or self.awi.n_steps - 1 < x[TIME]
                or x[UNIT_PRICE] < self.p_min_for_selling
                else (
                    x[UNIT_PRICE]
                    if issues == None
                    else (
                        x[UNIT_PRICE]
                        if issues[QUANTITY].max_value == issues[QUANTITY].min_value
                        else (
                            x[QUANTITY]
                            if issues[UNIT_PRICE].max_value
                            == issues[UNIT_PRICE].min_value
                            else self.awi.current_step
                            / (self.awi.n_steps - 1)
                            * (x[QUANTITY] - issues[QUANTITY].min_value)
                            / (issues[QUANTITY].max_value - issues[QUANTITY].min_value)
                            + (x[UNIT_PRICE] - issues[UNIT_PRICE].min_value)
                            / (
                                issues[UNIT_PRICE].max_value
                                - issues[UNIT_PRICE].min_value
                            )
                        )
                    )
                ),
                reserved_value=0.0,
                issues=issues,
            )
        else:
            return MappingUtilityFunction(
                lambda x: -1000
                if x[TIME] <= self.awi.current_step
                or self.awi.n_steps - 1 < x[TIME]
                or self.p_max_for_buying < x[UNIT_PRICE]
                else (
                    self.p_max_for_buying - x[UNIT_PRICE]
                    if issues == None
                    else (
                        self.p_max_for_buying - x[UNIT_PRICE]
                        if issues[TIME].max_value == issues[TIME].min_value
                        else (
                            issues[TIME].max_value - x[TIME]
                            if issues[UNIT_PRICE].max_value
                            == issues[UNIT_PRICE].min_value
                            else 0.25
                            * (issues[TIME].max_value - x[TIME])
                            / (issues[TIME].max_value - issues[TIME].min_value)
                            + (issues[UNIT_PRICE].max_value - x[UNIT_PRICE])
                            / (
                                issues[UNIT_PRICE].max_value
                                - issues[UNIT_PRICE].min_value
                            )
                        )
                    )
                ),
                reserved_value=0.0,
                issues=issues,
            )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List["Contract"],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        for contract in contracts:
            q, t, p = (
                contract.agreement["quantity"],
                contract.agreement["time"],
                contract.agreement["unit_price"],
            )
            if t >= self.awi.current_step:
                if contract.annotation["buyer"] == self.id:
                    old_total_q = sum(self.step_input_quantities)
                    self.step_input_quantities[t] -= q
                    self.average_prices[0] = (
                        (self.average_prices[0] * old_total_q - p * q)
                        / sum(self.step_input_quantities)
                        if sum(self.step_input_quantities) > 0
                        else self.awi.catalog_prices[self.awi.my_input_product]
                    )
                    self.my_friends[self.id][
                        "step_input_quantities"
                    ] = self.step_input_quantities
                elif contract.annotation["seller"] == self.id:
                    old_total_q = sum(self.step_output_quantities)
                    self.step_output_quantities[t] -= q
                    self.average_prices[1] = (
                        (self.average_prices[1] * old_total_q - p * q)
                        / sum(self.step_output_quantities)
                        if sum(self.step_output_quantities) > 0
                        else self.awi.catalog_prices[self.awi.my_output_product]
                    )
                    self.my_friends[self.id][
                        "step_output_quantities"
                    ] = self.step_output_quantities
