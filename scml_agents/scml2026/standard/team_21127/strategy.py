#!/usr/bin/env python
"""
**Submitted to ANAC 2026 SCML (Std track)**
*Authors* Rinon Asanuma <asanuma@katfuji.lab.tuat.ac.jp>

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2026 SCML.
"""

from __future__ import annotations

import random
from itertools import chain, combinations

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from scml.std import QUANTITY, TIME, UNIT_PRICE, StdAWI, StdSyncAgent

__all__ = ["ShimijimiShijimi"]


def distribute(total_needs: int, partner_count: int) -> list[int]:
    """必要量（total_needs）をパートナーの人数（partner_count）にできるだけ均等に分配する"""
    if partner_count == 0:
        return []
    if total_needs == partner_count:
        return [1] * partner_count

    base_quantity = total_needs // partner_count
    remainder_quantity = total_needs % partner_count
    
    return [base_quantity + 1] * remainder_quantity + [base_quantity] * (partner_count - remainder_quantity)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class ShimijimiShijimi(StdSyncAgent):
    def __init__(
        self,
        *args,
        threshold: float | None = None,
        productivity: float = 0.7,
        consumer_price_ratio: float = 0.7,  # 販売時の価格比率
        supplier_price_ratio: float = 1.2,  # 仕入時の価格比率
        future_offer_ratio: float = 0.4,    # 未来の必要量に対するオファー割合
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._threshold = threshold if threshold is not None else 1
        self._productivity = productivity
        self.consumer_price_ratio = consumer_price_ratio
        self.supplier_price_ratio = supplier_price_ratio
        self.future_offer_ratio = future_offer_ratio
        
        # 確保済み量帳簿
        self._secured_sales: dict[int, int] = {}
        self._secured_supplies: dict[int, int] = {}

    def before_step(self):
        super().before_step()
        self._secured_sales.clear()
        self._secured_supplies.clear()

    def step(self):
        super().step()
        self._threshold = self.awi.n_lines * 0.1

    def first_proposals(self) -> dict[str, Outcome | None]:
        # 現在アクティブな交渉相手だけに絞る
        active_suppliers = [p for p in self.awi.my_suppliers if p in self.negotiators]
        active_consumers = [p for p in self.awi.my_consumers if p in self.negotiators]

        response = {}
        response |= self._generate_offers_for_side(active_suppliers, is_supplier=True)
        response |= self._generate_offers_for_side(active_consumers, is_supplier=False)
        return response

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response = dict()
        
        # 供給(True)側と販売(False)側の両方を順番に処理
        for is_supplier, all_partners in [(True, self.awi.my_suppliers), (False, self.awi.my_consumers)]:
            if not all_partners:
                continue
                
            needs = self._needs(is_supplier, self.awi.current_step)
            partners = {p for p in all_partners if p in offers.keys()}
            
            current_offers, future_offers = self._categorize_offers(partners, offers)
            self._process_future_offers(future_offers, is_supplier, response)
            
            accepted_partners = self._find_best_current_combination(
                set(current_offers.keys()), current_offers, needs
            )
            
            # 最適な組み合わせをACCEPT
            if accepted_partners and needs > 0:
                for p in accepted_partners:
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, current_offers[p])
            
            # ACCEPTされなかった残りの全員に対して、カウンター（7:2:1分配）を一括生成
            unaccepted = [p for p in all_partners if p not in response and p in self.negotiators]
            if unaccepted:
                counters = self._generate_offers_for_side(unaccepted, is_supplier)
                for p, offer in counters.items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        return response

    # ==========================================
    # オファー生成のコアロジック
    # ==========================================

    def _divide_partners(self, partners: list[str]) -> tuple[list[str], list[str], list[str]]:
        """全員を1回シャッフルし、当日(70%)・1日後(20%)・2日後(10%)にシンプルにスライスする"""
        if not partners:
            return [], [], []
            
        shuffled = partners[:]
        random.shuffle(shuffled)
        
        n = len(shuffled)
        idx1 = int(n * 0.7)
        idx2 = int(n * 0.9)
        
        return shuffled[:idx1], shuffled[idx1:idx2], shuffled[idx2:]

    def _generate_offers_for_side(self, partners: list[str], is_supplier: bool) -> dict[str, Outcome]:
        """リストを受け取り、7:2:1に分割してすべてのオファーを生成する"""
        today, future1, future2 = self._divide_partners(partners)
        
        offers = {}
        offers |= self._create_offers_for_group(today, is_supplier, offset=0)
        offers |= self._create_offers_for_group(future1, is_supplier, offset=1)
        offers |= self._create_offers_for_group(future2, is_supplier, offset=2)
        
        return offers

    def _create_offers_for_group(self, group: list[str], is_supplier: bool, offset: int) -> dict[str, Outcome]:
        """与えられたグループに必要量を分配してオファーを作る。ここで初めて価格順に並べる"""
        if not group:
            return {}
            
        step = self.awi.current_step + offset
        if step >= self.awi.n_steps:
            return {}
            
        total_needs = self._needs(is_supplier, step)
        needs = total_needs if offset == 0 else int(total_needs * self.future_offer_ratio)
        
        if needs <= 0:
            return {}
            
        # 実際にオファーを出す直前に、価格が有利な順に並べる
        group = sorted(group, key=lambda x: self.best_price(x), reverse=not is_supplier)
        
        # 当日分で人数より必要量が少ない場合、価格が良い順から絞る
        if offset == 0 and needs < len(group):
            group = group[:needs]
            
        offers = {}
        distribution = distribute(needs, len(group))
        for k, q in zip(group, distribution):
            if q > 0:
                # 当日は交渉余地を持たせた price、未来は best_price を使用
                price = self.price(k) if offset == 0 else self.best_price(k)
                offers[k] = (q, step, price)
                
        return offers

    # ==========================================
    # 状態管理・判定ロジック
    # ==========================================

    def _needs(self, is_supplier: bool, target_time: int) -> int:
        """在庫と帳簿を加味して、指定ステップの必要量を計算"""
        awi = self.awi
        day_production = awi.n_lines * self._productivity

        if is_supplier:
            total_supplies = awi.total_supplies_at(target_time) + self._secured_supplies.get(target_time, 0)
            return int(day_production - awi.current_inventory_input - total_supplies)
            
        else:
            total_sales = awi.total_sales_at(target_time) + self._secured_sales.get(target_time, 0)
            if total_sales <= awi.n_lines:
                return int(
                    max(0, min(awi.n_lines, day_production + awi.current_inventory_input) - total_sales)
                )
        return 0

    def _categorize_offers(self, partners: set[str], offers: dict[str, Outcome]) -> tuple[dict, dict]:
        current_offers, future_offers = {}, {}
        for p in partners:
            if offers[p] is None:
                continue
            if self.is_valid_price(offers[p][UNIT_PRICE], p):
                if offers[p][TIME] == self.awi.current_step:
                    current_offers[p] = offers[p]
                else:
                    future_offers[p] = offers[p]
        return current_offers, future_offers

    def _process_future_offers(self, future_offers: dict[str, Outcome], is_supplier: bool, response: dict[str, SAOResponse]):
        duplicate_list = [0 for _ in range(self.awi.n_steps)]
        for p, offer in future_offers.items():
            step = offer[TIME]
            if step <= self.awi.n_steps:
                if offer[QUANTITY] + duplicate_list[step - 1] <= self._needs(is_supplier, step):
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    duplicate_list[step - 1] += offer[QUANTITY]

    def _find_best_current_combination(self, current_partners: set[str], offers: dict[str, Outcome], needs: int) -> list[str]:
        partner_list = list(current_partners)
        if not partner_list:
            return []

        # layers[r][total_quantity] keeps the earliest index tuple in the original powerset order.
        layers = [{0: ()}] + [dict() for _ in partner_list]

        for i, p in enumerate(partner_list):
            q = offers[p][QUANTITY]

            for r in range(i, -1, -1):
                for total, idxs in list(layers[r].items()):
                    new_total = total + q
                    new_idxs = idxs + (i,)

                    old = layers[r + 1].get(new_total)
                    if old is None or new_idxs < old:
                        layers[r + 1][new_total] = new_idxs

        best_by_total = {}
        for r in range(len(partner_list) + 1):
            for total, idxs in layers[r].items():
                if total not in best_by_total:
                    best_by_total[total] = idxs

        best_plus_total = None
        if needs > 0:
            for total in best_by_total:
                if total >= needs:
                    if best_plus_total is None or total < best_plus_total:
                        best_plus_total = total

        best_minus_total = None
        for total in best_by_total:
            if 0 < total < needs:
                if best_minus_total is None or total > best_minus_total:
                    best_minus_total = total

        if best_plus_total is not None and best_plus_total - needs <= self._threshold:
            return [partner_list[i] for i in best_by_total[best_plus_total]]

        if best_minus_total is not None:
            return [partner_list[i] for i in best_by_total[best_minus_total]]

        return partner_list

    # ▼ nmi取得時のエラーを防ぐための安全なラッパー
    def _safe_get_issues(self, partner: str):
        if partner in self.negotiators:
            try:
                nmi = self.get_nmi(partner)
                if nmi is not None:
                    return nmi.issues
            except Exception:
                pass
        return self.awi.current_input_issues if self.is_supplier(partner) else self.awi.current_output_issues

    def is_valid_price(self, price: float, partner: str) -> bool:
        issues = self._safe_get_issues(partner)
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        
        if self.is_supplier(partner):
            return price <= maxp
        else:
            return price >= minp

    def is_supplier(self, partner: str) -> bool:
        return partner in self.awi.my_suppliers

    def best_price(self, partner: str) -> int:
        issues = self._safe_get_issues(partner)
        pmin, pmax = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value
        return pmin if self.is_supplier(partner) else pmax

    def price(self, partner: str) -> int:
        issues = self._safe_get_issues(partner)
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value

        if self.is_supplier(partner):
            return int(min(minp * self.supplier_price_ratio, maxp))
        else:
            return int(max(maxp * self.consumer_price_ratio, minp))
        
    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        super().on_negotiation_success(contract, mechanism)
        agreement = contract.agreement
        if agreement is None:
            return
            
        quantity = int(agreement.get(QUANTITY, 0))
        time = int(agreement.get(TIME, self.awi.current_step))
        
        partners = [p for p in contract.partners if p != self.id]
        if partners and not self.is_supplier(partners[0]):
            self._secured_sales[time] = self._secured_sales.get(time, 0) + quantity
        elif partners:
            self._secured_supplies[time] = self._secured_supplies.get(time, 0) + quantity


if __name__ == "__main__":
    import sys
    from .helpers.runner import run
    run([ShimijimiShijimi], sys.argv[1] if len(sys.argv) > 1 else "std")