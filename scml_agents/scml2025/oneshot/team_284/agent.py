from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.oneshot import *
from scml.std import *
from scml.oneshot.common import is_system_agent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

import numpy
import random

__all__ = ["AlmostEqualAgent"]

def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=True,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] = [],
) -> list[int]:
    
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        # 誰に多く配分するか
        # 1. 各binに最低1個ずつ配分
        lst = [1] * n
        remaining = q - n
        
        # 2. 残りを1個ずつ順番に配分
        idx = 0
        while remaining > 0:
            if lst[idx] < mx:
                lst[idx] += 1
                remaining -= 1
            idx = (idx + 1) % n
        
        # 3. concentrated_idxの順序で値を直接割り当て
        result = [0] * n
        assigned_indices = set()
        
        # concentrated_idxの順序でlstの値を割り当て
        for i, target_idx in enumerate(concentrated_idx):
            if i < len(lst) and target_idx < n:
                result[target_idx] = lst[i]
                assigned_indices.add(target_idx)
        
        return result

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]

def powerset(iterable):
    """冪集合"""
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

class AlmostEqualAgent(OneShotSyncAgent):
    def __init__(
        self,
        *args,
        equal: bool = True,
        over_buying: float = 0.2,
        quantity_price_balance: float = 0.85,
        buyer_QP_score_threshold: float = 0.6,  # 買い手の意思決定閾値
        seller_QP_score_threshold: float = 0.6,  # 売り手の意思決定閾値
        **kwargs,
    ):
        self.equal_distribution = equal
        self.over_buying = over_buying
        self.quantity_price_balance = quantity_price_balance
        self.buyer_QP_score_threshold = buyer_QP_score_threshold
        self.seller_QP_score_threshold = seller_QP_score_threshold
        # self.check_needs = 0
        super().__init__(*args, **kwargs)
        
    def init(self):
        self.total_agreed_quantity = {k:0 for k in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers)}
        self.is_seller = self.awi.my_suppliers == ["SELLER"]
        return super().init()    
    
    def distribute_needs(
        self, t: float, mx: int|None = None, equal: bool|None = None, allow_zero: bool|None = None,
        concentrated: bool = False, concentrated_ids: list[str] = []
    ) -> dict[str, int]:
        """Distributes my needs equally over all my partners"""

        if equal is None: equal = True  
        if allow_zero is None: allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partners,n_partners = [],0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # 買い手の場合のみ必要数量より多めに買う
            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)
            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            adjusted_needs,
                            n_partners,
                            mx = mx,
                            equal = equal,
                            concentrated = concentrated,
                            allow_zero = allow_zero,
                            concentrated_idx = concentrated_idx 
                        ),
                    )
                )
            )
        return dist
        
    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]

    def first_proposals(self):
        # distribute my needs equally over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        my_negotiators = [p for p in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers) if p in self.negotiators.keys()]
        if self.awi.current_step > max(50,self.awi.n_steps * 0.5) and self.awi.level == 0 and len(my_negotiators) > 0:
            concentrated_ids = sorted(my_negotiators, key=lambda x:self.total_agreed_quantity[x], reverse=True)
            distribution = self.distribute_needs(t=0, mx=3, equal=True, allow_zero=False, concentrated=True, concentrated_ids=concentrated_ids)
        elif self.awi.current_step > max(100,self.awi.n_steps * 0.75) and self.awi.level == 1 and len(my_negotiators) > 0:
            concentrated_ids = sorted(my_negotiators, key=lambda x:self.total_agreed_quantity[x], reverse=True)
            distribution = self.distribute_needs(t=0, mx=3, equal=True, allow_zero=False, concentrated=True, concentrated_ids=concentrated_ids)
        else: 
            distribution = self.distribute_needs(t=0, mx=3,allow_zero=False,equal=True)
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d
        
    def counter_all(self, offers, states):
        response = dict()
        future_partners = {k for k, v in offers.items() if v[TIME] != self.awi.current_step}
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers
                
            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None) if not self.awi.allow_zero_quantity
                else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, price))
            )
            
            # 交渉相手の提案の組み合わせを評価
            plist = list(powerset(partners))[::-1]
            
            # 最適な組み合わせを選択するためのカウンター
            best_QP_score, best_quantity, best_indx = float("-inf"), 0, -1
            
            # 買い手の場合のみ必要数量を調整
            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)
            
            # 各組み合わせを評価する際に実際の必要数を使用
            for i, partner_ids in enumerate(plist):
                offered_quantity = sum(offers[p][QUANTITY] for p in partner_ids)
                quantity_diff = offered_quantity - needs  
                
                # コスト計算
                total_price = sum(offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in partner_ids)
                # ペナルティ
                penalty = 0
                if is_selling:  # 売り手
                    if offered_quantity > needs:  # 過剰販売
                        penalty += (offered_quantity - needs) * self.awi.current_shortfall_penalty
                    if offered_quantity < needs:  # 不足
                        penalty += (needs - offered_quantity) * self.awi.current_disposal_cost
                else:  # 買い手
                    if offered_quantity < needs:  # 不足
                        penalty += (needs - offered_quantity) * self.awi.current_shortfall_penalty
                    if offered_quantity > needs:  # 過剰購入
                        penalty += (offered_quantity - needs) * self.awi.current_disposal_cost
                
                # Pスコアを計算
                P_score = (total_price - penalty) if is_selling else -(total_price - penalty)
                # 正規化
                normalized_P_score = 0.5 + (P_score / 1000) if P_score > 0 else 0.5 - (abs(P_score) / 1000)
                normalized_P_score = max(0, min(1, normalized_P_score))
            
                normalized_quantity = 1.0 - abs(quantity_diff) / needs if needs != 0 else 0
                
                #QPスコアを計算
                QP_score = (
                    self.quantity_price_balance * normalized_quantity
                    + (1 - self.quantity_price_balance) * normalized_P_score
                )

                # 最良の組み合わせを更新
                if QP_score > best_QP_score:
                    best_QP_score = QP_score
                    best_quantity = quantity_diff
                    best_indx = i
                elif QP_score == best_QP_score:
                    # 同じスコアの場合、売り手は過剰を、買い手は不足を避ける
                    if (is_selling and quantity_diff < best_quantity) or \
                       (not is_selling and quantity_diff > best_quantity and quantity_diff <= 0):
                        best_quantity = quantity_diff
                        best_indx = i
            
            # 選択のための閾値の決定（売り手と買い手で異なる）
            QP_score_threshold = self.seller_QP_score_threshold if is_selling else self.buyer_QP_score_threshold
            
            # 交渉時間の経過に応じて閾値を調整（時間経過で閾値を下げる）
            relative_time = min(state.relative_time for state in states.values())
            adjusted_threshold = QP_score_threshold * (1 - 0.5 * relative_time)
            
            # 意思決定: スコアが閾値を超えた場合、提案を受け入れる
            if best_indx >= 0 and best_QP_score >= adjusted_threshold:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                
                # 不足数量がある場合、他の相手に対して追加提案
                if best_quantity < 0 and len(others) > 0:
                    s, price = self._step_and_price(best_price=False)
                    
                    # 不足数量を計算
                    shortage = -best_quantity
                    
                    if self.awi.current_step > max(50,self.awi.n_steps * 0.5) and self.awi.level == 0:
                        concentrated_ids = sorted(others, key=lambda x:self.total_agreed_quantity[x], reverse=True)
                        concentrated_idx = []
                        for p in concentrated_ids:  # concentrated_idsの順序で処理
                            if p in others:
                                idx = others.index(p)
                                concentrated_idx.append(idx)
                        distribution = dict(zip(
                            others,
                            distribute(
                                shortage,
                                len(others),
                                mx = 3,
                                equal = True,
                                concentrated = True,
                                concentrated_idx = concentrated_idx,
                                allow_zero = False
                            )
                        ))
                    elif self.awi.current_step > max(100,self.awi.n_steps * 0.75) and self.awi.level == 1:
                        concentrated_ids = sorted(others, key=lambda x:self.total_agreed_quantity[x], reverse=True)
                        concentrated_idx = []
                        for p in concentrated_ids:  # concentrated_idsの順序で処理
                            if p in others:
                                idx = others.index(p)
                                concentrated_idx.append(idx)
                        distribution = dict(zip(
                            others,
                            distribute(
                                shortage,
                                len(others),
                                mx = 3,
                                equal = True,
                                concentrated = True,
                                concentrated_idx = concentrated_idx,
                                allow_zero = False
                            )
                        ))
                    else:
                        distribution = dict(zip(
                            others,
                            distribute(shortage, len(others), mx = 3, equal = True,allow_zero = False)
                        ))
                    response.update({
                        k:(
                            unneeded_response if q==0
                            else SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                        ) for k,q in distribution.items()
                    })
                
                continue
                
            
            t = min(_.relative_time for _ in states.values())
            
            partners = partners.union(future_partners)
            partners = list(partners)
            
            # 買い手の場合のみ必要数量を調整
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)
            
            if self.awi.current_step > max(50,self.awi.n_steps * 0.5) and self.awi.level == 0 and len(partners) > 0:
                concentrated_ids = sorted(partners, key=lambda x:self.total_agreed_quantity[x], reverse=True)
                concentrated_idx = []
                for p in concentrated_ids:  # concentrated_idsの順序で処理
                    if p in partners:
                        idx = partners.index(p)
                        concentrated_idx.append(idx)
                distribution = dict(zip(
                    partners,
                    distribute(
                        adjusted_needs,
                        len(partners),
                        mx = 3,
                        equal = True,
                        concentrated = True,
                        concentrated_idx = concentrated_idx, 
                        allow_zero = False
                    )
                ))
            elif self.awi.current_step > max(100,self.awi.n_steps * 0.75) and self.awi.level == 1 and len(partners) > 0:
                concentrated_ids = sorted(partners, key=lambda x:self.total_agreed_quantity[x], reverse=True)
                concentrated_idx = []
                for p in concentrated_ids:  # concentrated_idsの順序で処理
                    if p in partners:
                        idx = partners.index(p)
                        concentrated_idx.append(idx)
                distribution = dict(zip(
                    partners,
                    distribute(
                        adjusted_needs,
                        len(partners),
                        mx = 3,
                        equal = True,
                        concentrated = True,
                        concentrated_idx = concentrated_idx, 
                        allow_zero = False
                    )
                ))
            else:
                distribution = dict(zip(
                    partners,
                    distribute(adjusted_needs, len(partners), mx = 3, equal = True, allow_zero = False)
                ))

            response.update({
                k: (
                    unneeded_response if q == 0
                    else SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))
                )for k, q in distribution.items()
            })
        return response
        
    def _step_and_price(self, best_price=False):
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        
        if self.awi.level == 0:  # 売り手
            if self.awi.current_step > max(50, self.awi.n_steps * 0.5):
                price = pmax if random.random() < 0.5 else pmin
            else:
                price = pmax if random.random() < 0.5 else pmin
        elif self.awi.level == 1:  # 買い手
            if self.awi.current_step > max(100, self.awi.n_steps * 0.75):
                price = pmax if random.random() < 0.5 else pmin
            else:
                price = pmax if random.random() < 0.5 else pmin
        else:
            # デフォルトはランダム
            price = random.randint(pmin, pmax)

        return s, price


   
            
