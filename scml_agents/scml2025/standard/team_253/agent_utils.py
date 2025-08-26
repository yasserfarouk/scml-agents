from itertools import chain, combinations
import numpy as np

from scml.std import StdAWI

class AgentUtils:
    @staticmethod
    def powerset(iterable):
        """
        冪集合を生成する
        powerset([1, 2, 3]) --> () (1,) (2,) (3,) (1, 2) (1, 3) (2, 3) (1, 2, 3)
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
    @staticmethod
    def weighted_average(values: list, weights: list = None) -> float:
        """
        重み付き平均を計算する
        weightsがNoneの場合は直近の値に高い重みがつくような指数重み(0.9^t)を用いる
        """
        if not values:
            return 0.0
        if weights is None:
            n = len(values)
            weights = [0.9 ** i for i in range(n)]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    
    @staticmethod
    def needs(awi: StdAWI, partner, time, productivity):
        """
        エージェントがpartnerに対してtime時点で必要とする数量を返す
        """
        # エージェントの役割に応じて対象ニーズを計算
        if AgentUtils.is_seller(awi, partner):
            # 相手がsellerのため，自分はbuyerとして原材料ニーズを計算
            base_needs = awi.needed_supplies
            inventory = awi.current_inventory_input
        elif AgentUtils.is_buyer(awi, partner):
            # 相手がbuyerのため，自分はsellerとして販売ニーズを計算
            base_needs = awi.needed_sales
            inventory = awi.current_inventory_output
        else:
            return 0
        
        # 中間エージェントの場合は生産能力を反映
        if awi.is_middle_level:
            base_needs = max(base_needs, int(awi.n_lines * productivity))
        
        # 在庫の差し引いた実質的なニーズを計算
        remaining_need = max(base_needs - inventory, 0)

        # 残りステップ数を考慮して，1ステップあたりの必要量を調整
        remaining_steps = max(awi.n_steps - time, 1)

        return remaining_need // remaining_steps
    
    @staticmethod
    def is_seller(awi: StdAWI, partner_id: str) -> bool:
        return partner_id in awi.my_suppliers
    
    @staticmethod
    def is_buyer(awi: StdAWI, partner_id: str) -> bool:
        return partner_id in awi.my_consumers