from __future__ import annotations

import random
from itertools import chain, combinations

# required for typing
from negmas import Contract, ResponseType, SAOResponse

# required for development
from scml.oneshot import *
from scml.std import *

__all__ = ["Oneshot"]


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] = [],
) -> list[int]:
    """Distributes q values over n bins."""
    from collections import Counter
    from numpy.random import choice

    q, n = int(q), int(n)

    if n <= 0:
        return []

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q <= 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        
        # 集中させる対象を確保しつつシャッフル
        concentrated_lst = sorted(lst, reverse=True)[: len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i, x in zip(concentrated_idx, concentrated_lst):
            lst.insert(i, x)
        return lst

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
    """Generates the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class Oneshot(OneShotSyncAgent):
    """
    An advanced OneShot Agent designed for ANAC 2026.
    Combines cautious partner selection, dynamic Boulware pricing, and scalable offer resolution.
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        overmismatch_max_selling: float = 0.0,
        overmismatch_max_buying: float = 0.3,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.2,
        concession_exp: float = 3.0, # 追加: 価格譲歩の強さ (Boulware)
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        self.concession_exp = concession_exp
        super().__init__(*args, **kwargs)

    def init(self):
        self.overordering_max = (
            self.overordering_max_selling
            if self.awi.my_suppliers == ["SELLER"]
            else self.overordering_max_buying
        )
        self.overmismatch_max_selling *= self.awi.n_lines
        self.overmismatch_max_buying *= self.awi.n_lines
        self.undermismatch_min_selling *= self.awi.n_lines
        self.undermismatch_min_buying *= self.awi.n_lines

        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        return super().init()

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]

    def _get_price_for_step(self, t: float, is_seller: bool, issues: tuple) -> int:
        """
        改良点: ランダムではなく、時間(t)に基づく譲歩(Concession)で価格を決定する。
        序盤は強気、終盤に向かって譲歩するBoulware戦略を採用。
        """
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        
        # t は 0.0 ~ 1.0 の相対時間
        concession_rate = t ** self.concession_exp
        
        if is_seller:
            # 売り手: 序盤は高く(pmax)、終盤に安く(pmin)
            price = pmax - (pmax - pmin) * concession_rate
        else:
            # 買い手: 序盤は安く(pmin)、終盤に高く(pmax)
            price = pmin + (pmax - pmin) * concession_rate
            
        return int(price)

    def first_proposals(self):
        # Initial proposals with the best possible price (t=0)
        s = self.awi.current_step
        is_seller = self.awi.is_first_level
        issues = self.awi.current_output_issues if is_seller else self.awi.current_input_issues
        price = self._get_price_for_step(0.0, is_seller, issues)

        my_negotiators, not_negotiators = [], []
        target_group = self.awi.my_consumers if self.awi.my_suppliers == ["SELLER"] else self.awi.my_suppliers
        needed_quantity = self.awi.needed_sales if self.awi.my_suppliers == ["SELLER"] else self.awi.needed_supplies

        for k in target_group:
            # 半分経過後、取引実績がゼロの相手とは交渉しない (慎重な選択)
            if self.awi.is_bankrupt(k) or (
                self.awi.current_step > min(self.awi.n_steps * 0.5, 50)
                and self.total_agreed_quantity[k] == 0
            ):
                not_negotiators.append(k)
            else:
                my_negotiators.append(k)
                
        offering_quantity = (
            int(needed_quantity * (1 + self._overordering_fraction(0)))
            if len(my_negotiators) > 1
            else needed_quantity
        )

        d = {}
        if len(my_negotiators) > 0:
            if self.awi.current_step > self.awi.n_steps * 0.5:
                # 信頼できるトップ実績のエージェントに集中
                concentrated_ids = sorted(
                    my_negotiators,
                    key=lambda x: self.total_agreed_quantity.get(x, 0),
                    reverse=True,
                )[:1]
                concentrated_idx = [
                    i for i, k in enumerate(my_negotiators) if k in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        my_negotiators,
                        distribute(
                            offering_quantity,
                            len(my_negotiators),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            else:
                distribution = dict(
                    zip(
                        my_negotiators,
                        distribute(offering_quantity, len(my_negotiators)),
                    )
                )

            d |= {
                k: (q, s, price) if q > 0 or self.awi.allow_zero_quantity else None
                for k, q in distribution.items()
            }
            
        d |= {k: None for k in not_negotiators}
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
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
            if needs <= 0 and not self.awi.allow_zero_quantity:
                 continue
                 
            active_partners = [_ for _ in all_partners if _ in offers.keys()]
            if not active_partners:
                continue

            # 現在の相対時間(交渉の残り時間指標)
            current_rel_time = min(state.relative_time for state in states.values())
            is_selling = all_partners == self.awi.my_consumers
            
            # 時間に応じた適正価格の算出 (逆提案用)
            price = self._get_price_for_step(current_rel_time, is_selling, issues)

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            best_partner_ids = tuple()
            best_diff = float("inf")

            # 改良点: 相手が多い場合のタイムアウト(O(2^N))回避
            # 相手が10人を超える場合は、価格が有利な順にソートして貪欲法(Greedy)で選ぶ
            if len(active_partners) > 10:
                sorted_partners = sorted(
                    active_partners, 
                    key=lambda p: offers[p][UNIT_PRICE], 
                    reverse=is_selling # 売り手なら高い順、買い手なら安い順
                )
                accumulated_q = 0
                greedy_selected = []
                for p in sorted_partners:
                    if accumulated_q < needs:
                        greedy_selected.append(p)
                        accumulated_q += offers[p][QUANTITY]
                
                best_partner_ids = tuple(greedy_selected)
                best_diff = accumulated_q - needs

            else:
                # 10人以下なら安全に全探索(Powerset)を実行
                plist = list(powerset(active_partners))[::-1]
                plus_best_diff, plus_best_indx = float("inf"), -1
                minus_best_diff, minus_best_indx = -float("inf"), -1
                
                for i, partner_ids in enumerate(plist):
                    offered = sum(offers[p][QUANTITY] for p in partner_ids)
                    diff = offered - needs
                    
                    if diff >= 0:  # 過剰
                        if diff < plus_best_diff:
                            plus_best_diff, plus_best_indx = diff, i
                        elif diff == plus_best_diff:
                            # 同じ数量なら価格が有利な方を選ぶ
                            p_sum_current = sum(offers[p][UNIT_PRICE] for p in partner_ids)
                            p_sum_best = sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx])
                            if (is_selling and p_sum_current > p_sum_best) or (not is_selling and p_sum_current < p_sum_best):
                                plus_best_diff, plus_best_indx = diff, i
                    if diff <= 0:  # 不足
                        if diff > minus_best_diff:
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == minus_best_diff:
                            if diff < 0 and len(partner_ids) < len(plist[minus_best_indx]):
                                minus_best_diff, minus_best_indx = diff, i
                            elif diff == 0 or len(partner_ids) == len(plist[minus_best_indx]):
                                p_sum_current = sum(offers[p][UNIT_PRICE] for p in partner_ids)
                                p_sum_best = sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx])
                                if (is_selling and p_sum_current > p_sum_best) or (not is_selling and p_sum_current < p_sum_best):
                                    minus_best_diff, minus_best_indx = diff, i

                th_min, th_max = self._allowed_mismatch(current_rel_time, is_selling)
                
                if th_min <= minus_best_diff or plus_best_diff <= th_max:
                    if th_min <= minus_best_diff and plus_best_diff <= th_max:
                        if -minus_best_diff == plus_best_diff:
                            best_diff, best_indx = (minus_best_diff, minus_best_indx) if is_selling else (plus_best_diff, plus_best_indx)
                        elif -minus_best_diff < plus_best_diff:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else:
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif minus_best_diff < th_min and plus_best_diff <= th_max:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                    else:
                        best_diff, best_indx = minus_best_diff, minus_best_indx
                    
                    best_partner_ids = plist[best_indx]
                else:
                    # 許容範囲内に収まる提案がない場合はベストを更新せずスルー（全てRejectになる）
                    best_partner_ids = tuple()
                    best_diff = -needs

            # 採用と拒絶の振り分け
            partner_ids = best_partner_ids
            others = list(set(active_partners).difference(partner_ids).union(future_partners))
            
            # 受諾するオファーを登録
            for k in partner_ids:
                response[k] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                
            # 受諾しなかった相手への対応（Reject & End or Counter Offer）
            if best_diff < 0 and len(others) > 0:  # 不足している場合は残りの相手に逆提案
                s = self.awi.current_step
                offering_quanitity = (
                    int(-best_diff * (1 + self._overordering_fraction(current_rel_time)))
                    if len(others) > 1 else -best_diff
                )
                
                if self.awi.current_step > self.awi.n_steps * 0.5:
                    concentrated_ids = sorted(
                        others, key=lambda x: self.total_agreed_quantity.get(x, 0), reverse=True
                    )[:1]
                    concentrated_idx = [i for i, p in enumerate(others) if p in concentrated_ids]
                    distribution = dict(
                        zip(others, distribute(offering_quanitity, len(others), mx=self.awi.n_lines, concentrated=True, concentrated_idx=concentrated_idx))
                    )
                else:
                    distribution = dict(
                        zip(others, distribute(offering_quanitity, len(others), mx=self.awi.n_lines))
                    )
                    
                for k, q in distribution.items():
                    if q == 0:
                        response[k] = unneeded_response
                    else:
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, (q, s, price))
            else:
                # 数量が満たされている場合は残りの交渉を終了
                for k in others:
                    response[k] = unneeded_response

        return response

    def _allowed_mismatch(self, r: float, is_selling: bool):
        """Calculates allowed mismatch bounds based on relative time."""
        undermismatch_min = self.undermismatch_min_selling if is_selling else self.undermismatch_min_buying
        overmismatch_max = self.overmismatch_max_selling if is_selling else self.overmismatch_max_buying
        return undermismatch_min * ((1 - r) ** self.mismatch_exp), overmismatch_max * (r ** (1 / self.mismatch_exp))

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)