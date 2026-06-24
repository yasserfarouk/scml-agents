#!/usr/bin/env python
"""
Submitted to ANAC 2024 SCML (OneShot track)
Authors: [Masaki Yamashita / team_yamashita]
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import chain, combinations
from typing import Any

import numpy as np
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType
from scml.oneshot import OneShotAWI, OneShotSyncAgent, QUANTITY, TIME, UNIT_PRICE

# ==========================================
# ヘルパー関数群
# ==========================================

def distribute(q: int, n: int) -> list[int]:
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    r = Counter(np.random.choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# ==========================================
# エージェント本体
# ==========================================

class YamashitaAgent(OneShotSyncAgent):

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        return {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response = dict()
        
        for needs, all_partners, issues in [
            (self.awi.needed_supplies, self.awi.my_suppliers, self.awi.current_input_issues),
            (self.awi.needed_sales, self.awi.my_consumers, self.awi.current_output_issues),
        ]:
            if not issues:
                continue
                
            price = issues[UNIT_PRICE].rand()
            partners = {_ for _ in all_partners if _ in offers.keys()}
            if not partners:
                continue

            all_offers = sum(offers[p][QUANTITY] for p in partners)
            if all_offers <= needs:
                partner_ids = partners
                response.update({k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k]) for k in partner_ids})

            plist = list(powerset(partners))
            best_under_diff, best_over_diff, best_under_indx, best_over_indx = float("inf"), float("inf"), -1, -1
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                over_diff, under_diff = float('inf'), float('inf')
                if offered == needs:
                    best_over_diff = 0
                    best_over_indx = i
                    break
                if not self.awi.is_first_level:
                    if offered > needs:
                        over_diff = (offered - needs)
                    elif needs > offered:
                        under_diff = (needs - offered)
                else:
                    if offered > needs:
                        over_diff = float('inf')
                    elif needs > offered:
                        under_diff = (needs - offered)
                if over_diff < best_over_diff:
                    best_over_diff, best_over_indx = over_diff, i
                if under_diff < best_under_diff:
                    best_under_diff, best_under_indx = under_diff,i
                

            current_steps = [_.relative_time for _ in states.values()]
            th_over = self._current_threshold_over(min(current_steps) if current_steps else 0.0)
            th_under = self._current_threshold_under(min(current_steps) if current_steps else 0.0)
            shortfall_penalty = self.awi.current_shortfall_penalty
            disposal_cost = self.awi.current_disposal_cost
            x = 0.5
            y = 3
            alpha_slope = (x - y) / 0.2
            disposal_weight = alpha_slope * disposal_cost + y
            beta_slope = (y - x) / 0.8
            penalty_weight = beta_slope * (shortfall_penalty - 0.2) + x

            ratio = all_offers / needs
            if 1.0 < ratio < 3.0:
                expedite = 10 * np.exp(-1.151 * (ratio - 1.0))
            else:
                expedite = 1

            
            if best_over_diff < th_over * penalty_weight * disposal_weight * expedite:
                partner_ids = plist[best_over_indx]
                others = list(partners.difference(partner_ids))
                response.update({k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k]) for k in partner_ids})
                response.update({k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others})
                continue
            if best_under_diff < th_under * expedite:
                partner_ids = plist[best_under_indx]
                others = list(partners.difference(partner_ids))
                response.update({k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k]) for k in partner_ids})
                if others:
                    dist_list = distribute(int(best_under_diff), len(others))
                    for k, q in zip(others, dist_list):
                        if q == 0:
                            response[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                        else:
                            response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                continue
            distribution = self.distribute_needs()
            for k in partners:
                if k not in distribution: continue
                q = distribution[k]
                if q == 0:
                    response[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                else:
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
        
        return response

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        pass

    def before_step(self):
        pass

    def step(self):
            super().step()
    pass        

    # =====================
    # Internal Methods
    # =====================

    def distribute_needs(self) -> dict[str, int]:
        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            if not partner_ids:
                continue
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * len(partner_ids))))
                continue
            dist.update(dict(zip(partner_ids, distribute(needs, len(partner_ids)))))
        return dist

    def _current_threshold_over(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return (mn + (mx - mn) * (r**4))
    
    def _current_threshold_under(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return (mn + (mx - mn) * (1 - r**4)) + mx / 2
    

    def _step_and_price(self, best_price=False):
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = self.awi.current_output_issues if seller else self.awi.current_input_issues
        if not issues:
            return s, 0
        pmin, pmax = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(self, partners: list[str], annotation: dict[str, Any], mechanism: OneShotAWI, state: SAOState) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:
        pass

if __name__ == "__main__":
    import sys
    from .helpers.runner import run
    run([YamashitaAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")