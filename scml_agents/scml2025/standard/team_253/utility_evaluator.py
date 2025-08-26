from scml.std import StdAWI
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from .agent_utils import AgentUtils as au
from .supply_demand_allocator import SupplyDemandAllocator
from .record_manager import RecordManager

class UtilityEvaluator:
    def __init__(self, awi: StdAWI, distribution: SupplyDemandAllocator, record: RecordManager):
        self.awi = awi
        self.todays_distribution = distribution
        self.record = record
        self.nmis = {}

    def set_nmis(self, nmis):
        self.nmis = nmis
                
    def quantity_utility(self, partner_id: str, quantity: int) -> float:
        # 交渉相手が分配の対象でない場合，0.0
        if partner_id not in self.todays_distribution:
            return 0.0
        
        # 交渉相手に分配数がない場合，0.0
        expected = self.todays_distribution.get(partner_id, 0)
        if expected == 0:
            return 0.0
        
        # special rules: step < 2 & 自分が販売する & (lv.1 or lv.2) -> いい効用を返す
        if self.awi.current_step < 2 and au.is_buyer(self.awi, partner_id) and self.awi.level > 0:
            if quantity >= expected:
                # 相手の提案量が分配量よりも多いが，許容できる範囲
                if quantity < self.awi.n_lines * 0.7:
                    return 1.0
                # 許容できないほど大きい場合，普通の効用を計算し返す
                else:
                    pass
            else:
                return quantity / expected
        
        base_util = min(quantity, expected) / max(quantity, expected)

        if au.is_buyer(self.awi, partner_id):
            available = self.awi.current_inventory_input
            stock_ratio = min(1.0, available / max(quantity, 1))
            adjust_util = base_util * stock_ratio

            contracted = self.record.get_contracted_sales_current_step()
            capacity = self.awi.n_lines * 0.7
            projected_total = contracted + quantity

            if capacity > 0:
                if projected_total <= capacity:
                    ratio = 1.0
                else:
                    overflow = projected_total - capacity
                    decay_strength = (overflow / (capacity + 1e-6)) ** 2
                    ratio = max(0.001, 1.0 - decay_strength)
                    adjust_util *= ratio
            
            return min(adjust_util, 1.0)

        return base_util
    
    def time_utility(self, partner_id: str, time: int) -> float:
        now = self.awi.current_step # 現在のステップ数
        max_delay = self.awi.n_steps - now # 発注から納品までで取りうるステップ数の最大値
        lead_time = time - now # 相手が提案してきた発注から納品までのステップ数

        if max_delay <= 0:
            return 0.0

        return max(0.0, 1 - (lead_time / max_delay))
    
    def price_utility(self, partner_id: str, price: int) -> float:
        price_issue = self.nmis[partner_id].issues[UNIT_PRICE]
        min_price = price_issue.min_value
        max_price = price_issue.max_value
        price_range = max(max_price - min_price, 1)

        if au.is_buyer(self.awi, partner_id):
            utility = (price - min_price) / price_range
        else:
            utility = (max_price - price) / price_range
        
        return max(0.0, min(utility, 1.0))
    
    def ufun(self, partner_id, offer: tuple[int, int, int]):
        if partner_id not in self.nmis:
            return 0.0

        quantity_util = self.quantity_utility(partner_id, offer[QUANTITY])
        time_util = self.time_utility(partner_id, offer[TIME])
        price_util = self.price_utility(partner_id, offer[UNIT_PRICE])

        if self.awi.current_step < 2 and au.is_seller(self.awi, partner_id) and self.awi.level > 0:
            w_q, w_t, w_p = 0.3, 0.4, 0.2
        else:
            w_q, w_t, w_p = 0.6, 0.2, 0.2
        return w_q * quantity_util + w_t * time_util + w_p * price_util