from scml.std import StdAWI
import random
from itertools import repeat
from collections import Counter

from .partner_scorer import PartnerScorer
from .agent_utils import AgentUtils as au

class SupplyDemandAllocator:
    def __init__(self, awi:StdAWI, scorer: PartnerScorer):
        self.awi = awi
        self.scorer = scorer

    def _distribute(self, quantity: int, n_negotiators: int) -> list[int]:
        """quantityの量をn_negotiators人に分配する"""
        if quantity < n_negotiators:
            lst = [0] * (n_negotiators - quantity) + [1] * quantity
            random.shuffle(lst)
            return lst
        if quantity == n_negotiators:
            return [1] * n_negotiators
        r = Counter(random.choices(range(n_negotiators), k=quantity))
        return [r.get(_, 0) for _ in range(n_negotiators)]
    
    def _distribute_with_score(self, quantity: int, partners: list[str]) -> list[int]:
        """スコアに基づいて分売戦略を立てる"""
        scores = [self.scorer.get_weighted_score(pid) for pid in partners]
        total = sum(scores)
        if total == 0:
            return self._distribute(quantity, len(partners))
        
        raw_allocation = [quantity * s / total for s in scores]
        int_allocation = [int(x) for x in raw_allocation]
        remaining = quantity - sum(int_allocation)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in range(remaining):
            int_allocation[sorted_indices[i]] += 1

        return int_allocation

    def _distribute_equally_limited(self, quantity: int, partners: list[str], max_per_partner: int) -> list[int]:
        """
        与えられたquantityを全パートナーに可能な限り平等に分配する．
        ただし1人あたりmax_per_partnerを超えない．
        戻り値は quantity のリスト（partner順）．
        """
        n = len(partners)
        if n == 0 or quantity <= 0:
            return [0] * n

        base = min(quantity // n, max_per_partner)
        result = [base] * n
        remainder = quantity - base * n

        # 余りを1個ずつ，max_per_partnerに達していない人に追加
        for i in range(n):
            if remainder == 0:
                break
            if result[i] < max_per_partner:
                result[i] += 1
                remainder -= 1

        return result

    def distribute_todays_needs(self, productivity: float = 1.0, ptoday: float = 1.0, partners=None):
        if partners is None:
            return None
        
        self.productivity = productivity
        self.ptoday = ptoday
        
        response = dict(zip(partners, repeat(0)))
        for is_partner, edge_needs in (
            (au.is_seller, self.awi.needed_supplies),
            (au.is_buyer, self.awi.needed_sales),
        ):
            # 在庫を加味した needs 計算
            if is_partner == au.is_seller:
                base_needs = self.awi.needed_supplies
                inventory = self.awi.current_inventory_input
            elif is_partner == au.is_buyer:
                base_needs = self.awi.needed_sales
                inventory = self.awi.current_inventory_output
            else:
                base_needs = 0
                inventory = 0

            if self.awi.is_middle_level:
                base_needs = max(base_needs, int(self.awi.n_lines * self.productivity))

            # 実質ニーズを計算
            needs = int(min(max(base_needs - inventory, 0), self.awi.n_lines * self.productivity))
            # special rules: step > 2 & 自分が購入する & (lv.1 or lv.2) -> 原材料を確保したいのでneedsを無理やり変更
            if self.awi.current_step < 2 and is_partner == au.is_seller and self.awi.level > 0:
                # needs = int(self.awi.n_lines * 2.0)
                if self.awi.current_inventory_input >= self.awi.n_lines * 1.8:
                    pass
                else:
                    needs = min(self.awi.n_lines * 2, len(self.awi.my_suppliers) * 4)
            elif is_partner == au.is_seller and self.awi.level > 0:
                if self.awi.current_inventory_input > self.awi.n_lines * 3:
                    needs = 0
            elif is_partner == au.is_seller and self.awi.current_inventory_input <= self.awi.n_lines and self.awi.total_supplies_from(self.awi.current_step) < self.awi.n_lines * 2:
                needs = min(self.awi.n_lines * 2 - self.awi.current_inventory_input, len(self.awi.my_suppliers) * 4)
            if is_partner == au.is_seller and self.awi.total_supplies_until(self.awi.current_step + 2) + self.awi.current_inventory_input >= self.awi.n_lines * 1.8:
                needs = 0

            active_partners = [_ for _ in partners if is_partner(self.awi, _)]
            if not active_partners or needs < 1:
                continue

            scored_partners = self.scorer.sorted_partners()
            if not scored_partners:
                scored_partners = partners
            active_partners = [p for p in scored_partners if p in active_partners]
            active_partners = active_partners[: max(1, int(self.ptoday * len(active_partners)))]

            n_partners = len(active_partners)

            if needs <= 0 or n_partners <= 0:
                continue

            if needs < n_partners:
                active_partners = active_partners[:needs]
                n_partners = len(active_partners)
            
            if self.awi.current_step < 2 and is_partner == au.is_seller and self.awi.level > 0:
                max_per_partner = int(self.awi.n_lines * self.productivity)
                allocation = self._distribute_equally_limited(needs, active_partners, max_per_partner)
            else:
                allocation = self._distribute_with_score(needs, active_partners)

            response |= dict(zip(active_partners, allocation))
        return response
    
    def has_supplier_distribution(self, distribution: dict[str, int]) -> bool:
        """
        distribution の中で supplier に割り当てられた量の合計が 0 でない場合 True を返す．
        """
        total_supply = sum(
            qty for pid, qty in distribution.items()
            if au.is_seller(self.awi, pid)
        )
        return total_supply == 0