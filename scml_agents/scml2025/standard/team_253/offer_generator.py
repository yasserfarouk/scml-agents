from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from .agent_utils import AgentUtils as au
from .utility_evaluator import UtilityEvaluator

class OfferGenerator:
    def __init__(self, awi, distribution, ue: UtilityEvaluator):
        self.awi = awi
        self.todays_distribution = distribution
        self.ue = ue
        self.nmis = {}
    
    def set_nmis(self, nmis):
        self.nmis = nmis

    def get_best_quantity(self, partner_id: str, pid_distribution: int) -> float:
        """自分にとって最良な数量を返す"""
        issue = self.nmis[partner_id].issues[QUANTITY]
        max_quantity = issue.max_value
        min_quantity = issue.min_value
        return max(min_quantity, min(max_quantity, pid_distribution))

    def get_best_time(self, partner_id: str) -> float:
        """自分にとって最良な納期を返す"""
        now = self.awi.current_step
        issue = self.nmis[partner_id].issues[TIME]
        best_time = issue.min_value
        best_score = -1

        for t in range(now + 1, issue.max_value + 1):
            score = self.ue.time_utility(partner_id, t)
            if score > best_score:
                best_score = score
                best_time = t
        return max(issue.min_value, min(issue.max_value, best_time))

    def get_best_price(self, partner_id: str) -> float:
        """自分にとって最良な単価を返す"""
        issue = self.nmis[partner_id].issues[UNIT_PRICE]
        best_price = issue.max_value
        best_score = -1

        for p in range(issue.min_value, issue.max_value + 1):
            score = self.ue.price_utility(partner_id, p)
            if score > best_score:
                best_score = score
                best_price = p
        
        if au.is_buyer(self.awi, partner_id) == True:
            catalog_price = self.awi.catalog_prices[self.awi.level + 1]
        elif au.is_seller(self.awi, partner_id) == True:
            catalog_price = self.awi.catalog_prices[self.awi.level]
        return int((best_price + catalog_price) / 2)
    
    def generate_base_offer(self, partner_id: str, threshold: float) -> tuple:
        """
        自身のベストオファーと受け入れ可能な最低限のオファーを比較し，
        現時点で受け入れ可能かつ最大効用を目指す base offer を構築する．
        BUYER ならより安く/早く/多く，SELLER ならより高く/遅く/少なくを選ぶ．
        """
        # Step 1: ベストオファーの生成（効用最大）
        quantity_best = self.get_best_quantity(partner_id, self.todays_distribution.get(partner_id, 0) if self.todays_distribution else 0)
        time_best = self.get_best_time(partner_id)
        price_best = self.get_best_price(partner_id)
        best_offer = (quantity_best, time_best, price_best)

        # Step 2: 受け入れ可能な最低限のオファーの探索
        acceptable_offer = self.find_acceptable_offer(partner_id, threshold)

        # Step 3: ベースオファーの構築
        base_offer = []
        for i, (b, a) in enumerate(zip(best_offer, acceptable_offer)):
            if i == QUANTITY:
                base_offer.append(min(b, a))
            elif i == TIME:
                time_val = min(b, a)
                if self.awi.current_step == 0 and time_val == 0:
                    time_val = 1
                base_offer.append(time_val)
            elif i == UNIT_PRICE:
                base_offer.append(min(b, a) if au.is_buyer(self.awi, partner_id) else max(b, a))
        return tuple(base_offer)
    
    def find_acceptable_offer(self, partner_id: str, threshold: float) -> tuple:
        """
        閾値 just above を満たすような最小効用変化のオファーを返す．
        """
        acceptable_offer = None
        min_margin = float("inf")
        issues = self.nmis[partner_id].issues

        for q in range(issues[QUANTITY].min_value, issues[QUANTITY].max_value + 1):
            for t in range(issues[TIME].min_value, issues[TIME].max_value + 1):
                for p in range(issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value + 1):
                    offer = (q, t, p)
                    score = self.ue.ufun(partner_id, offer)
                    margin = score - threshold
                    if margin >= 0 and margin < min_margin:
                        min_margin = margin
                        acceptable_offer = offer

        if acceptable_offer is None:
            acceptable_offer = (
                issues[QUANTITY].min_value,
                issues[TIME].min_value,
                issues[UNIT_PRICE].max_value if au.is_buyer(self.awi, partner_id) else issues[UNIT_PRICE].min_value
            )
        return acceptable_offer
        
    def find_least_concession_offer(self, partner_id: str, base_offer: tuple) -> tuple:
        """
        base_offer から 1つの issue を変更し，効用が base_offer 以下かつ最小減少のオファーを返す．
        """
        best_offer = base_offer
        min_drop = float("inf")
        base_score = self.ue.ufun(partner_id, base_offer)
        issues = self.nmis[partner_id].issues

        for i in range(3):  # 各 issue: quantity, time, price
            for val in range(issues[i].min_value, issues[i].max_value + 1):
                if i == TIME and self.awi.current_step == 0 and val == 0:
                    continue
                if val == base_offer[i]:
                    continue
                new_offer = list(base_offer)
                new_offer[i] = val
                new_offer = tuple(new_offer)
                score = self.ue.ufun(partner_id, new_offer)
                drop = base_score - score
                if 0 <= drop < min_drop:
                    min_drop = drop
                    best_offer = new_offer
        return best_offer
