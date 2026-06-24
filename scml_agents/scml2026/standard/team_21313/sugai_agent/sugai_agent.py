#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations

# required for typing
from typing import Any

import random
import math

# required for development
from scml.std import StdAWI, StdSyncAgent, QUANTITY, TIME, UNIT_PRICE

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType


class SugaiAgent(StdSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `StdUFun` in the docs for more details).
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        proposals: dict[str, Outcome | None] = {}

        for selling in (True, False):
            partners = self._side_partners(selling)
            if not partners:
                continue

            need = self.needed_sales if selling else self.needed_supplies
            if need <= 0:
                for p in partners:
                    proposals[p] = None
                continue
 
            total_trust = sum(self._trust(p) for p in partners) or 1.0
 
            for p in partners:
                trust = self._trust(p)
                weight = trust / total_trust
 
                avg_q = min(self._avg_quantity(p), need)
                q = max(1, min(need, int(round(0.5 * need * weight + 0.5 * avg_q))))
 
                price = self._opening_price(selling, trust)
                proposals[p] = (q, self._delivery_time(selling), float(price))
 
        # 全 negotiator を必ず網羅（不要な側は None=交渉打ち切り）
        for p in self.negotiators:
            proposals.setdefault(p, None)
 
        return proposals

    def counter_all(
        self,
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
    ) -> dict[str, SAOResponse]:
        # 売り側・買い側を別々に処理する
        sell_offers = {p: o for p, o in offers.items() if self._selling(p)}
        buy_offers = {p: o for p, o in offers.items() if not self._selling(p)}
 
        responses: dict[str, SAOResponse] = {}
        responses.update(self._counter_side(sell_offers, states, selling=True))
        responses.update(self._counter_side(buy_offers, states, selling=False))
        return responses

    def _counter_side(
        self,
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
        selling: bool,
    ) -> dict[str, SAOResponse]:
        if not offers:
            return {}
 
        need = self.needed_sales if selling else self.needed_supplies
 
        # この方向はもう不要 → すべて終了
        if need <= 0:
            return {p: SAOResponse(ResponseType.END_NEGOTIATION, None) for p in offers}
 
        # 代表時刻（受理閾値の緩和に使用）
        t = max(
            (states[p].relative_time for p in offers if p in states),
            default=0.0,
        )
        # 序盤は高利益オファーのみ受理、終盤は緩和して必要量を確保
        threshold = 0.6 * (1.0 - t)
 
        # --- 評価: 順位付けは ufun、受理の絶対基準は正規化価格効用 ---
        scored = []
        for p, o in offers.items():
            pu = self._price_utility(selling, float(o[UNIT_PRICE]))
            scored.append((p, o, float(self.ufun(o)), pu, int(o[QUANTITY])))
        scored.sort(key=lambda x: x[2], reverse=True)
 
        # --- 受理: 閾値を超えるものを「必要量を超えない範囲」で貪欲選択 ---
        accepted: set[str] = set()
        secured = 0
        for p, o, u, pu, q in scored:
            if secured >= need:
                break
            if pu < threshold:
                continue
            if secured + q <= need:  # オーバーシュートさせない
                accepted.add(p)
                secured += q
 
        remaining = max(0, need - secured)
 
        out: dict[str, SAOResponse] = {}
        for p, o, u, pu, q in scored:
            st = states.get(p)
            rt = st.relative_time if st is not None else t
 
            if p in accepted:
                out[p] = SAOResponse(ResponseType.ACCEPT_OFFER, o)
                continue
 
            # 必要量を満たしたら残りは終了（過剰契約防止）
            if remaining <= 0:
                out[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
 
            # 見込みの薄い相手は早めに切る
            _, trials = self.accept_rate.get(p, [0, 0])
            if trials > 5 and self._trust(p) < 0.2 and rt > 0.5:
                out[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
 
            # 残り必要量と相手提示量の小さい方を要求（相手の納品日を尊重）
            cq = max(1, min(remaining, q))
            price = self._counter_price(selling, p, float(o[UNIT_PRICE]), rt)
            out[p] = SAOResponse(
                ResponseType.REJECT_OFFER, (cq, int(o[TIME]), float(price))
            )
 
        return out

    # =====================
    # Pricing helpers
    # =====================

    def _opening_price(self, selling: bool, trust: float) -> float:
        """初手価格: 自分に有利な端から小さく譲歩。信頼できる相手には少し歩み寄る。"""
        if selling:
            return self.sell_max - (self.sell_max - self.sell_min) * (0.1 + 0.2 * trust)
        return self.buy_min + (self.buy_max - self.buy_min) * (0.1 + 0.2 * trust)
 
    def _target_price(self, selling: bool, t: float) -> float:
        """時間とともに不利な端へ譲歩する目標価格。"""
        if selling:  # 売り: 高く始めて下げる
            return self.sell_max - (self.sell_max - self.sell_min) * (0.1 + 0.7 * t)
        # 買い: 安く始めて上げる
        return self.buy_min + (self.buy_max - self.buy_min) * (0.1 + 0.7 * t)
 
    def _counter_price(self, selling: bool, partner: str, offer_price: float, t: float) -> float:
        trust = self._trust(partner)
        target = self._target_price(selling, t)
        alpha = 0.8 if t > 0.8 else (0.3 + 0.5 * trust)  # 相手提示価格への寄せ具合
        price = alpha * offer_price + (1.0 - alpha) * target
        if selling:
            return float(min(self.sell_max, max(self.sell_min, price)))
        return float(min(self.buy_max, max(self.buy_min, price)))
 
    def _price_utility(self, selling: bool, price: float) -> float:
        """価格の自分にとっての良さを 0-1 に正規化（方向別）。"""
        if selling:
            lo, hi = self.sell_min, self.sell_max
            if hi - lo < 1e-9:
                return 0.5
            return max(0.0, min(1.0, (price - lo) / (hi - lo)))
        lo, hi = self.buy_min, self.buy_max
        if hi - lo < 1e-9:
            return 0.5
        return max(0.0, min(1.0, (hi - price) / (hi - lo)))
 
    def _delivery_time(self, selling: bool) -> int:
        issues = self.awi.current_output_issues if selling else self.awi.current_input_issues
        try:
            return int(issues[TIME].min_value)
        except Exception:
            return 0

    # =====================
    # Side / partner helpers
    # =====================

    def _selling(self, partner: str) -> bool:
        """この相手に対して自分は売り手か（consumer 相手なら売り）。"""
        if partner in self.awi.my_consumers:
            return True
        if partner in self.awi.my_suppliers:
            return False
        return self.awi.is_first_level
 
    def _side_partners(self, selling: bool) -> list[str]:
        return [p for p in self.negotiators if self._selling(p) == selling]

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

        self.accept_rate: dict[str, list[int]] = {}
        self.partner_stats: dict[str, dict[str, Any]] = {}
        self.buy_min = self.buy_max = 0.0
        self.sell_min = self.sell_max = 0.0
        self.needed_supplies = 0
        self.needed_sales = 0

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""

        self.buy_min, self.buy_max = self._safe_price(self.awi.current_input_issues)
        self.sell_min, self.sell_max = self._safe_price(self.awi.current_output_issues)
 
        # 両側を常に取得（不要な側は 0 になるので中間層も自然に扱える）
        self.needed_supplies = max(0, int(self.awi.needed_supplies))
        self.needed_sales = max(0, int(self.awi.needed_sales))
 
    def step(self):
        pass

    @staticmethod
    def _safe_price(issues) -> tuple[float, float]:
        try:
            ip = issues[UNIT_PRICE]
            return float(ip.min_value), float(ip.max_value)
        except Exception:
            return 0.0, 1.0

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ):
        for partner in partners:
            self.accept_rate.setdefault(partner, [0, 0])
            self.accept_rate[partner][1] += 1

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI):
        partner = (
            contract.partners[0]
            if contract.partners[0] != self.id
            else contract.partners[1]
        )
 
        self.accept_rate.setdefault(partner, [0, 0])
        self.accept_rate[partner][0] += 1
        self.accept_rate[partner][1] += 1
 
        q = int(contract.agreement["quantity"])
        # 成立した契約の分だけ、対応する側の必要量を減らす（過剰契約防止）
        if partner in self.awi.my_consumers:
            self.needed_sales = max(0, self.needed_sales - q)
        else:
            self.needed_supplies = max(0, self.needed_supplies - q)
 
        stats = self._partner_stat(partner)
        stats["contracts"] += 1
        stats["total_quantity"] += q

    # =====================
    # Statistics helpers
    # =====================

    def _partner_stat(self, partner: str) -> dict[str, Any]:
        if partner not in self.partner_stats:
            self.partner_stats[partner] = {"contracts": 0, "total_quantity": 0}
        return self.partner_stats[partner]
 
    def _trust(self, partner: str) -> float:
        """その相手と合意に至れた経験的確率（ラプラス平滑化）。"""
        s, t = self.accept_rate.get(partner, [0, 0])
        return (s + 1) / (t + 2)
 
    def _avg_quantity(self, partner: str) -> float:
        stats = self._partner_stat(partner)
        if stats["contracts"] == 0:
            return 1
        return stats["total_quantity"] / stats["contracts"]
 
 
if __name__ == "__main__":
    import sys
 
    from .helpers.runner import run
 
    run([SugaiAgent], sys.argv[1] if len(sys.argv) > 1 else "std")

