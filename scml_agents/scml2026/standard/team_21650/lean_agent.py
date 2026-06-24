"""
LeanAgent: scml の inventory bug を踏まえて「在庫を抱えない」戦略を取る agent.

原則:
  qin (今日の input) == qout (今日の output) を維持。

位置別戦略:
  A0 (is_first_level): forced exogenous supply は自動的に来る。それ "だけ" を売り切る。
      supply_needs = 0, consume_needs = forced_qin
  A_last (is_last_level): forced exogenous sales がある。それを賄うだけ仕入れる。
      supply_needs = forced_qout, consume_needs = 0
  middle: **MyAgent (AS0) のデフォルト動作**に丸投げ (daily_throughput_target 等 AS0 全機能を使う)

MyAgent(AS0) ベースで継承。middle は super() に委譲。A0/A_last は lean override。
"""
from __future__ import annotations

import random
from itertools import repeat
from typing import Any

from negmas import ResponseType, SAOResponse

from scml.std import QUANTITY, TIME, UNIT_PRICE
from .sbd_agent import _AS0Base
from .penguinagent import distribute, quantity_subset_dp


class LeanAgent(_AS0Base):
    """在庫を抱えない戦略 (バグあり scml 用). middle は AS0 default, A0 は lean.
    A_last はデフォルト Option C (AS0 default) だが、サブクラスがフラグで lean に切替可能."""

    # サブクラス制御フラグ (default は現 SBD と同じ動作)
    _a_last_use_lean: bool = False        # True なら A_last も lean path を使う
    _a_last_strict_cap: bool = False      # True なら A_last の overmismatch を 0 強制

    def __init__(self, *args, productivity: float = 0.7, **kwargs):
        super().__init__(*args, productivity=productivity, **kwargs)

    # ----------------------------
    # 位置別 needs 計算
    # ----------------------------
    def _lean_needs(self) -> tuple[int, int]:
        awi = self.awi
        s = awi.current_step
        forced_qin = int(awi.current_exogenous_input_quantity or 0)
        forced_qout = int(awi.current_exogenous_output_quantity or 0)

        if awi.is_first_level:
            supply_needs = 0
            consume_needs = max(0, forced_qin - int(awi.total_sales_at(s)))
        elif awi.is_last_level:
            supply_needs = max(0, forced_qout - int(awi.total_supplies_at(s)))
            consume_needs = 0
        else:
            # middle: AS0 default — daily_throughput_target を使う
            target = int(self.daily_throughput_target())
            supply_needs = max(0, target - int(awi.current_inventory_input) - int(awi.total_supplies_at(s)))
            consume_needs = max(0, min(awi.n_lines, target + int(awi.current_inventory_input)) - int(awi.total_sales_at(s)))

        return supply_needs, consume_needs

    # ----------------------------
    # distribute_todays_needs: middle と A_last は AS0 default, A0 のみ lean
    # ----------------------------
    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        # middle は常に AS0 default. A_last はフラグで切替.
        if self.awi.is_middle_level:
            return super().distribute_todays_needs(partners)
        if self.awi.is_last_level and not self._a_last_use_lean:
            return super().distribute_todays_needs(partners)

        # A0 (and A_last if flag): lean logic
        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))
        suppliers = [p for p in partners if self.is_supplier(p)]
        consumers = [p for p in partners if self.is_consumer(p)]
        supply_needs, consume_needs = self._lean_needs()

        if suppliers and supply_needs > 0:
            selected = self.select_partners_by_performance(suppliers)
            response |= self._distribute_to_partners(selected, supply_needs)

        awi = self.awi
        if (
            consumers and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= awi.n_lines
        ):
            selected = self.select_partners_by_performance(consumers)
            response |= self._distribute_to_partners(selected, consume_needs)

        return response

    # ----------------------------
    # counter_all: middle と A_last は AS0 default, A0 のみ lean
    # ----------------------------
    def counter_all(self, offers, states):
        if self.awi.is_middle_level:
            return super().counter_all(offers, states)
        if self.awi.is_last_level and not self._a_last_use_lean:
            return super().counter_all(offers, states)

        # A0 (and A_last if flag): lean logic
        response: dict[str, SAOResponse] = {}
        awi = self.awi
        s = awi.current_step

        supply_needs, consume_needs = self._lean_needs()

        for needs, all_partners, issues, is_buying in [
            (supply_needs, awi.my_suppliers, awi.current_input_issues, True),
            (consume_needs, awi.my_consumers, awi.current_output_issues, False),
        ]:
            partners = {p for p in all_partners if p in offers}

            current_step_offers: dict[str, Any] = {}
            future_step_offers: dict[str, Any] = {}
            for p in partners:
                if offers[p] is None:
                    continue
                if offers[p][TIME] == s and self.is_valid_price(offers[p][UNIT_PRICE], p):
                    current_step_offers[p] = offers[p]
                elif offers[p][TIME] != s and self.is_valid_price(offers[p][UNIT_PRICE], p):
                    future_step_offers[p] = offers[p]

            all_current_step_partners = set(current_step_offers.keys())

            # 将来契約は需要に合えば受諾
            duplicate_list = [0] * awi.n_steps
            for p, x in future_step_offers.items():
                step = x[TIME]
                if step <= awi.n_steps:
                    if x[QUANTITY] + duplicate_list[step - 1] <= self.needs_at(step, p):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, x)
                        duplicate_list[step - 1] += x[QUANTITY]

            # 今日の数量合わせは全partnerをDPで評価
            best_plus, best_plus_diff, best_minus, _ = quantity_subset_dp(
                current_step_offers, needs
            )

            has_accept = True
            # A_last strict cap: 過剰仕入 (plus side) 絶対 NG (一致のみ OK)
            strict_cap_active = (
                self.awi.is_last_level
                and self._a_last_strict_cap
                and is_buying
            )
            if strict_cap_active:
                if best_plus_diff == 0 and best_plus:
                    partner_ids = best_plus
                elif best_minus:
                    partner_ids = best_minus
                else:
                    has_accept = False
                    partner_ids = tuple()
            elif best_plus_diff <= self._threshold and best_plus:
                partner_ids = best_plus
            elif best_minus:
                partner_ids = best_minus
            else:
                has_accept = False
                partner_ids = tuple()

            flag = 0
            if has_accept and needs > 0:
                others = [
                    p
                    for p in all_current_step_partners
                    if p not in partner_ids and p not in response
                ]
                response.update({
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, current_step_offers[k])
                    for k in partner_ids
                })
                others_s = [x for x in others if self.is_supplier(x)]
                others_c = [x for x in others if self.is_consumer(x)]
                for k, x in self.future_supplie_offer(others_s).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(others_c).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                flag = 1

            if flag != 1:
                other_partners = {
                    p for p in all_partners
                    if p not in response and p in self.negotiators
                }
                if needs > 0:
                    distribution = self.distribute_todays_needs(other_partners)
                    future_s, future_c = [], []
                    for k, q in distribution.items():
                        if q > 0:
                            response[k] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (q, s, self.price(k)),
                            )
                        elif self.is_supplier(k):
                            future_s.append(k)
                        elif self.is_consumer(k):
                            future_c.append(k)
                    for k, x in self.future_supplie_offer(future_s).items():
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                    for k, x in self.future_consume_offer(future_c).items():
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                else:
                    future_s = [p for p in other_partners if self.is_supplier(p)]
                    future_c = [p for p in other_partners if self.is_consumer(p)]
                    for k, x in self.future_supplie_offer(future_s).items():
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                    for k, x in self.future_consume_offer(future_c).items():
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    # Option C: A_last は完全に AS0 default に丸投げ (lean strategy が本番に合わない)
    # A0 のみ lean を維持．future_supplie_offer / future_consume_offer / needs_at すべて parent.

    def needs_at(self, step: int, partner: str) -> int:
        awi = self.awi
        s = awi.current_step
        # middle: parent (AS0) default
        if awi.is_middle_level:
            return super().needs_at(step, partner)
        # A_last: flag で切替
        if awi.is_last_level:
            if not self._a_last_use_lean:
                return super().needs_at(step, partner)
            # A_last lean: 今日も未来も forced_qout 上限 (over-buy 防止)
            est_forced_qout = int(awi.current_exogenous_output_quantity or 0)
            if self.is_supplier(partner):
                return max(0, est_forced_qout - int(awi.total_supplies_at(step)))
            return 0
        # A0: 今日のみ lean, 未来は parent
        if step == s:
            supply_needs, consume_needs = self._lean_needs()
            return supply_needs if self.is_supplier(partner) else consume_needs
        return super().needs_at(step, partner)

    def price(self, partner):
        # MyAgent には price() がない (AS0 は smart_price)。PENGUIN 流の price を実装。
        nmi = self.get_nmi(partner)
        if nmi is None:
            if self.is_consumer(partner):
                issues = self.awi.current_output_issues
            else:
                issues = self.awi.current_input_issues
        else:
            issues = nmi.issues
        pissue = issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return max(maxp * 0.7, minp)
        return min(minp * 1.2, maxp)
