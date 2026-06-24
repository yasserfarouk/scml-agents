from __future__ import annotations

import random
from collections import Counter
from itertools import repeat

# required for typing
from negmas import SAOResponse, ResponseType
from numpy.random import choice

from scml.std import StdSyncAgent, TIME, QUANTITY, UNIT_PRICE

__all__ = ["Std"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes q values over n bins with at least one item per bin assuming q > n"""
    if q <= 0 or n <= 0:
        return []
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


class Std(StdSyncAgent):
    """
    ANAC 2026 SCML Standard Track 向けエージェント
    """

    def __init__(self, *args, ptoday=0.75, target_productivity=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptoday = ptoday
        self._productivity = target_productivity
        self._my_catalog_prices = []

    def init(self):
        """
        シミュレーションの初期化時に呼ばれるメソッド
        エラーを回避するための正しいプロパティへのアクセス方法を実装
        """
        super().init()
        try:
            if hasattr(self.awi, "catalog_prices"):
                for item_id in range(len(self.awi.catalog_prices)):
                    pass
        except Exception as e:
            print(f"[Std] Initialization error: {e}")
            if hasattr(self.awi, "log_error"):
                self.awi.log_error(f"[Std] Initialization error: {e}")

    def first_proposals(self):
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

        first_dict = dict()
        future_supplie_partner = []
        future_consume_partner = []

        for k, q in distribution.items():
            if q > 0:
                first_dict[k] = (q, s, self.best_price(k))
            elif self.is_supplier(k):
                future_supplie_partner.append(k)
            elif self.is_consumer(k):
                future_consume_partner.append(k)

        response = dict()
        response |= first_dict
        response |= self.future_supplie_offer(future_supplie_partner)
        response |= self.future_consume_offer(future_consume_partner)

        return response

    def counter_all(self, offers, states):
        response = dict()
        awi = self.awi

        for all_partners in [self.awi.my_suppliers, self.awi.my_consumers]:
            if not all_partners:
                continue

            is_buying = self.is_supplier(all_partners[0])
            
            # 今日必要な量を計算
            day_production = self.awi.n_lines * self._productivity
            needs = 0
            if is_buying:
                needs = int(max(0, day_production - awi.current_inventory_input - awi.total_supplies_at(awi.current_step)))
            else:
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = int(max(0, min(self.awi.n_lines, day_production + awi.current_inventory_input) - awi.total_sales_at(awi.current_step)))

            partners = {_ for _ in all_partners if _ in offers.keys()}
            current_step_offers = dict()
            future_step_offers = dict()

            for p in partners:
                offer = offers[p]
                if offer is None:
                    continue
                if self.is_valid_price(offer[UNIT_PRICE], p):
                    if offer[TIME] == self.awi.current_step:
                        current_step_offers[p] = offer
                    else:
                        future_step_offers[p] = offer

            # 将来のオファー処理
            duplicate_list = [0 for _ in range(awi.n_steps + 1)]
            for p, offer in future_step_offers.items():
                step = offer[TIME]
                if step < awi.n_steps:
                    if offer[QUANTITY] + duplicate_list[step] <= self.needs_at(step, p):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                        duplicate_list[step] += offer[QUANTITY]

            # Greedyアルゴリズムによる当日のオファー評価
            sorted_partners = sorted(
                current_step_offers.keys(),
                key=lambda x: current_step_offers[x][UNIT_PRICE],
                reverse=not is_buying
            )

            current_accumulated = 0
            for p in sorted_partners:
                offer = current_step_offers[p]
                q = offer[QUANTITY]
                
                if needs <= 0:
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    continue

                if current_accumulated + q <= needs + (self.awi.n_lines * 0.1): 
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    current_accumulated += q
                else:
                    rem = int(max(0, needs - current_accumulated))
                    if rem > 0:
                        response[p] = SAOResponse(ResponseType.REJECT_OFFER, (rem, self.awi.current_step, self.price(p)))
                        current_accumulated += rem
                    else:
                        response[p] = SAOResponse(ResponseType.REJECT_OFFER, None)

            # 不足分の再要求
            if current_accumulated < needs:
                remaining_needs = int(needs - current_accumulated)
                other_partners = [p for p in all_partners if p not in response.keys() and p in self.negotiators.keys()]
                
                if other_partners and remaining_needs > 0:
                    distribution = self.distribute_todays_supplie_consume_needs(other_partners, remaining_needs)
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, self.price(k)))

        return response

    def is_valid_price(self, price, partner):
        nmi = self.get_nmi(partner)
        issues = nmi.issues if nmi else (self.awi.current_output_issues if self.is_consumer(partner) else self.awi.current_input_issues)
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        
        if self.is_consumer(partner):
            return price >= minp
        elif self.is_supplier(partner):
            return price <= maxp
        return False

    def needs_at(self, step, partner):
        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        if self.is_supplier(partner):
            return int(max(0, day_production - awi.current_inventory_input - awi.total_supplies_at(step)))
        elif self.is_consumer(partner):
            return int(max(0, min(self.awi.n_lines, day_production + awi.current_inventory_input) - awi.total_sales_at(step)))
        return 0

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        if partners is None:
            partners = list(self.negotiators.keys())

        response = dict(zip(partners, repeat(0)))
        supplie_partners = [x for x in partners if self.is_supplier(x)]
        consume_partners = [x for x in partners if self.is_consumer(x)]

        awi = self.awi
        day_production = self.awi.n_lines * self._productivity
        supplie_needs = int(max(0, day_production - awi.current_inventory_input - awi.total_supplies_at(awi.current_step)))
        consume_needs = int(max(0, min(self.awi.n_lines, day_production + awi.current_inventory_input) - awi.total_sales_at(awi.current_step)))

        if supplie_partners and supplie_needs > 0:
            response |= self.distribute_todays_supplie_consume_needs(supplie_partners, supplie_needs)

        if consume_partners and consume_needs > 0 and awi.total_sales_at(awi.current_step) <= self.awi.n_lines:
            response |= self.distribute_todays_supplie_consume_needs(consume_partners, consume_needs)

        return response

    def distribute_todays_supplie_consume_needs(self, partners, needs) -> dict[str, int]:
        response = dict(zip(partners, repeat(0)))
        if not partners or needs <= 0:
            return response
            
        random.shuffle(partners)
        target_partners = partners[: max(1, int(self._ptoday * len(partners)))]
        n_partners = len(target_partners)

        if needs < n_partners:
            target_partners = random.sample(target_partners, max(1, needs))
            n_partners = len(target_partners)

        response |= dict(zip(target_partners, distribute(needs, n_partners)))
        return response

    def future_supplie_offer(self, partner_list):
        if not partner_list: return {}
        response = dict()
        awi = self.awi
        s = awi.current_step
        p = awi.n_lines * self._productivity

        n_total = len(partner_list)
        p1 = partner_list[: int(n_total * 0.5)]
        p2 = partner_list[int(n_total * 0.5) : int(n_total * 0.8)]
        p3 = partner_list[int(n_total * 0.8) :]

        for offset, plist in enumerate([p1, p2, p3], start=1):
            if s + offset < awi.n_steps and plist:
                needs = int(max(0, (p - awi.current_inventory_input - awi.total_supplies_at(s + offset)) / 3))
                if needs > 0:
                    dist = dict(zip(plist, distribute(needs, len(plist))))
                    for k, q in dist.items():
                        if q > 0:
                            response[k] = (q, s + offset, self.best_price(k))
        return response

    def future_consume_offer(self, partner_list):
        if not partner_list: return {}
        response = dict()
        awi = self.awi
        s = awi.current_step
        p = awi.n_lines * self._productivity

        n_total = len(partner_list)
        p1 = partner_list[: int(n_total * 0.5)]
        p2 = partner_list[int(n_total * 0.5) : int(n_total * 0.8)]
        p3 = partner_list[int(n_total * 0.8) :]

        for offset, plist in enumerate([p1, p2, p3], start=1):
            if s + offset < awi.n_steps and awi.total_sales_at(s + offset) <= awi.n_lines and plist:
                needs = int(max(0, min(awi.n_lines, p + awi.current_inventory_input) - awi.total_sales_at(s + offset)) / 3)
                if needs > 0:
                    dist = dict(zip(plist, distribute(needs, len(plist))))
                    for k, q in dist.items():
                        if q > 0:
                            response[k] = (q, s + offset, self.best_price(k))
        return response

    def best_price(self, partner):
        nmi = self.get_nmi(partner)
        issues = nmi.issues if nmi else (self.awi.current_output_issues if self.is_consumer(partner) else self.awi.current_input_issues)
        pmin, pmax = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner):
        nmi = self.get_nmi(partner)
        issues = nmi.issues if nmi else (self.awi.current_output_issues if self.is_consumer(partner) else self.awi.current_input_issues)
        minp, maxp = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value

        progress = self.awi.current_step / max(1, self.awi.n_steps)
        
        if self.is_consumer(partner):
            return int(maxp - (maxp - minp) * (0.1 + 0.6 * progress))
        else:
            return int(minp + (maxp - minp) * (0.1 + 0.6 * progress))

if __name__ == "__main__":
    import sys
    from scml.std.helpers.runner import run
    run([Std], sys.argv[1] if len(sys.argv) > 1 else "std")