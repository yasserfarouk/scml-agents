#!/usr/bin/env python
from __future__ import annotations

from typing import Any

from scml.std import StdAWI, StdSyncAgent
from scml.oneshot.common import *

from negmas import Contract, SAOResponse, SAOState, ResponseType

from itertools import repeat
import random

# def distribute(self, q: int, n: int, mx: int = None, **kwargs) -> list[int]:
#     # mxが設定されていれば、分配の最大制限として扱う
#     if mx is not None:
#         q = min(q, mx)
#     # それ以外は元のロジック
#     if q < n:
#         lst = [0] * (n - q) + [1] * q
#         random.shuffle(lst)
#         return lst
#     if q == n:
#        return [1] * n
#     # q > nの場合は余剰分をランダムに分配
#     r = Counter(choice(n, q - n))
#     return (r.get(_, 0) + 1 for _ in range(n))

# def powerset(iterable):
#     # 与えられたイテラブルの部分集合を生成
#     s = list(iterable)
#     return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

__all__ = ["UltraSuperMiracleSoraFinalAgentZ"]


class UltraSuperMiracleSoraFinalAgentZ(StdSyncAgent):
    def max_safe_sales_today(self):
        # 本日安全に販売できる最大数量を計算（在庫＋入荷分、ライン数上限あり）
        inv = self.awi.current_inventory_input
        incoming = self.awi.total_supplies_at(self.awi.current_step)
        return int(min(self.awi.n_lines, inv + incoming))

    def __init__(self, *args, threshold=None, ptoday=0.75, productivity=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1
        self._threshold = threshold  # 需要と供給の許容誤差（閾値）
        self._ptoday = ptoday  # 今日の需要を分配するパートナー割合
        self._productivity = productivity  # 生産性パラメータ
        # HACK: デフォルトの生産レベル
        self.production_level = 0.5
        self.future_concession = 0.1  # 将来提案時の譲歩率
        from collections import defaultdict

        # 交渉履歴記録（成功・失敗回数や平均価格など）
        self.negotiation_history = defaultdict(lambda: {"success": 0, "fail": 0})
        self.p_max_for_buying = None  # 最大購入価格（動的に決定される）
        self.future_selling_contracts = []  # 今後の販売契約（buyerとの確定契約）を保持
        # 互換性のためprojected_sales, max_inventory, cumulative_inputを追加
        self.projected_sales = 0
        self.max_inventory = 0
        self.cumulative_input = 0

    def first_proposals(self):
        # 複合的な提案ロジック・動的閾値・販売戦略を統合
        offers = {}
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )
        # --- 仕入れ（サプライヤーへの提案） ---
        if self.awi.my_consumers == ["BUYER"]:
            # 序盤の戦略調整・動的閾値
            todays_input_needed = (
                self.awi.current_exogenous_output_quantity
                - (
                    min(self.awi.current_inventory_input, self.awi.n_lines)
                    + self.awi.current_inventory_output
                )
                + int(
                    self.awi.n_lines
                    * (1 - (self.awi.current_step + 1) / self.awi.n_steps)
                )
            )
        else:
            if sum(self.awi.current_inventory) > self.max_inventory:
                todays_input_needed = 0
            else:
                todays_input_needed = max(
                    self.awi.n_lines - sum(self.awi.current_inventory), 0
                )
        # 戦略的な分配ロジックを利用
        scores = self.get_partner_scores()
        distribution = self.distribute_history_based(
            todays_input_needed, self.awi.my_suppliers, scores
        )
        offers = {
            k: (q, self.awi.current_step, self.best_price(k)) if q > 0 else unneeded
            for k, q in distribution.items()
        }
        # --- 販売（消費者への提案） ---
        remained_consumers = self.awi.my_consumers.copy()
        secured_output = sum(
            [
                contract.agreement["quantity"]
                for contract in self.future_selling_contracts
            ]
        )
        for t in range(self.awi.current_step, self.awi.n_steps):
            if self.awi.my_consumers == ["BUYER"]:
                break
            todays_output_needed = max(self.awi.needed_sales - secured_output, 0)
            if todays_output_needed <= self.awi.n_lines or t == self.awi.n_steps - 1:
                distribution = dict(
                    zip(
                        remained_consumers,
                        distribute(todays_output_needed, len(remained_consumers)),
                    )
                )
            else:
                todays_output_needed = self.awi.n_lines
                concentrated_ids = sorted(
                    remained_consumers,
                    key=lambda x: self.total_agreed_quantity[x],
                    reverse=True,
                )
                distribution = dict(
                    zip(
                        remained_consumers,
                        distribute(
                            todays_output_needed,
                            len(remained_consumers),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=[
                                i
                                for i, p in enumerate(remained_consumers)
                                if p in concentrated_ids
                            ],
                            allow_zero=True,
                        ),
                    )
                )
            offers |= {
                k: (q, t, self.best_price(k)) if q > 0 else unneeded
                for k, q in distribution.items()
            }
            remained_consumers = [
                k
                for k in remained_consumers
                if k not in distribution.keys() or distribution[k] <= 0
            ]
            secured_output += todays_output_needed
            if len(remained_consumers) == 0 or todays_output_needed == 0:
                break
        offers |= {k: unneeded for k in remained_consumers}
        return offers

    def counter_all(self, offers, states):
        today_offers = {
            k: v for k, v in offers.items() if v[TIME] == self.awi.current_step
        }

        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )
        response = {}

        # 供給先がBUYERの場合は組合せによる数量重視
        if self.awi.my_consumers == ["BUYER"]:
            # Accept/generate responses for buying only during early steps (retain advanced logic from myagent2)
            if self.awi.current_step < int(self.awi.n_steps * 0.3):
                valid_suppliers = [
                    _ for _ in self.awi.my_suppliers if _ in offers.keys()
                ]
                today_partners = [
                    _ for _ in self.awi.my_suppliers if _ in today_offers.keys()
                ]
                today_partners = set(today_partners)
                plist = list(powerset(today_partners))[::-1]
                price_good_plist = [
                    ps
                    for ps in plist
                    if len(ps) > 0
                    and max([offers[p][UNIT_PRICE] for p in ps])
                    * self.awi.current_exogenous_output_quantity
                    < self.awi.current_exogenous_output_price
                    - self.awi.profile.cost * self.awi.current_exogenous_output_quantity
                ]
                if len(price_good_plist) > 0:
                    plist = price_good_plist
                plus_best_diff, plus_best_indx = float("inf"), -1
                minus_best_diff, minus_best_indx = -float("inf"), -1
                best_diff, best_indx = float("inf"), -1
                todays_input_needed = self.awi.current_exogenous_output_quantity - (
                    min(self.awi.current_inventory_input, self.awi.n_lines)
                    + self.awi.current_inventory_output
                )

                for i, partner_ids in enumerate(plist):
                    offered = sum(offers[p][QUANTITY] for p in partner_ids)
                    diff = offered - todays_input_needed
                    if diff >= 0:  # 必要以上の量のとき
                        if diff < plus_best_diff:
                            plus_best_diff, plus_best_indx = diff, i
                        elif diff == plus_best_diff:
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                    if diff <= 0:  # 必要量に満たないとき
                        if diff > minus_best_diff:
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == minus_best_diff:
                            if (
                                diff < 0
                                and len(partner_ids) < len(plist[minus_best_indx])
                            ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                                minus_best_diff, minus_best_indx = diff, i
                            elif diff == 0 or (
                                diff < 0
                                and len(partner_ids) == len(plist[minus_best_indx])
                            ):
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) < sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i

                r = min(state.relative_time for state in states.values())
                mismatch_margin = int(
                    self.awi.n_lines * (1 - r)
                )  # 譲歩に応じて許容範囲を調整
                th_min, th_max = -mismatch_margin, mismatch_margin
                if th_min <= minus_best_diff or plus_best_diff <= th_max:
                    if plus_best_diff <= th_max:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                    else:
                        best_diff, best_indx = minus_best_diff, minus_best_indx

                    response |= {
                        p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                        for p in plist[best_indx]
                    }

                    remained_suppliers = set(valid_suppliers).difference(
                        plist[best_indx]
                    )
                    if best_diff < 0 and len(remained_suppliers) > 0:
                        concentrated_ids = sorted(
                            remained_suppliers,
                            key=lambda x: self.total_agreed_quantity[x],
                            reverse=True,
                        )
                        if (
                            len(concentrated_ids) > 0
                            and self.awi.current_step > self.awi.n_steps * 0.5
                        ):
                            scores = self.get_partner_scores()
                            distribution = self.distribute_history_based(
                                -best_diff, list(remained_suppliers), scores
                            )
                        else:
                            scores = self.get_partner_scores()
                            distribution = self.distribute_history_based(
                                -best_diff, list(remained_suppliers), scores
                            )

                        response |= {
                            k: SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    q,
                                    self.awi.current_step,
                                    self.buy_price(
                                        states[k].relative_time,
                                        self.awi.current_input_issues[
                                            UNIT_PRICE
                                        ].min_value,
                                        self.p_max_for_buying,
                                        True,
                                    ),
                                ),
                            )
                            if q > 0
                            else unneeded_response
                            for k, q in distribution.items()
                        }
            else:
                response |= {
                    partner_id: unneeded_response
                    for partner_id in offers.keys()
                    if partner_id in self.awi.my_suppliers
                }
        # 供給先がBUYER以外の場合は安ければ買い、高ければ買わない
        # 最終日は(供給先がBUYERのとき以外は)買わない
        elif self.awi.current_step == self.awi.n_steps - 1:
            response |= {
                partner_id: unneeded_response
                for partner_id in offers.keys()
                if partner_id in self.awi.my_suppliers
            }
        else:
            # Advanced proposal logic (from myagent2): strategic, dynamic threshold, productivity
            buying_offers = {
                partner_id: offer
                for partner_id, offer in offers.items()
                if partner_id in self.awi.my_suppliers
            }
            if sum(self.awi.current_inventory) > self.max_inventory:
                response |= {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for k in buying_offers.keys()
                }
            else:
                input_secured = 0
                for partner_id, offer in sorted(
                    buying_offers.items(), key=lambda x: x[1][UNIT_PRICE]
                ):
                    if offer[UNIT_PRICE] <= self.buy_price(
                        states[partner_id].relative_time,
                        self.awi.current_input_issues[UNIT_PRICE].min_value,
                        self.p_max_for_buying,
                    ):
                        if offer[TIME] == self.awi.current_step:
                            response[partner_id] = SAOResponse(
                                ResponseType.ACCEPT_OFFER, offer
                            )
                        else:
                            response[partner_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    offer[QUANTITY],
                                    self.awi.current_step,
                                    offer[UNIT_PRICE],
                                ),
                            )
                    else:
                        response[partner_id] = (
                            SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    offer[QUANTITY],
                                    self.awi.current_step,
                                    self.buy_price(
                                        states[partner_id].relative_time,
                                        self.awi.current_input_issues[
                                            UNIT_PRICE
                                        ].min_value,
                                        self.p_max_for_buying,
                                    ),
                                ),
                            )
                            if offer[QUANTITY] > 0
                            else unneeded_response
                        )
                    input_secured += offer[QUANTITY]
                    if (
                        sum(self.awi.current_inventory) + input_secured
                        > self.max_inventory
                    ):
                        break

        # consumersとの交渉は、とにかく(前日までに入荷した)在庫を売り切れる組み合わせを重視する、在庫を超える量の契約は結ばない
        selling_offers = {
            partner_id: offer
            for partner_id, offer in offers.items()
            if partner_id in self.awi.my_consumers
        }
        plist = list(powerset(selling_offers.keys()))[::-1]
        plist = [
            ps
            for ps in plist
            if sum(
                [
                    self.awi.current_step <= offers[p][TIME] < self.awi.n_steps
                    for p in ps
                ]
            )
            == len(ps)
        ]
        best_diff, best_indx = float("inf"), -1
        secured_output = sum(
            [
                contract.agreement["quantity"]
                for contract in self.future_selling_contracts
            ]
        )
        todays_output_needed = min(
            max(self.awi.needed_sales - secured_output, 0), self.awi.n_lines
        )
        for i, partner_ids in enumerate(plist):
            offered = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = offered - todays_output_needed
            if -best_diff < diff <= 0 or (
                -diff == best_diff
                and sum(offers[p][UNIT_PRICE] for p in plist[best_indx])
                < sum(offers[p][UNIT_PRICE] for p in partner_ids)
            ):
                best_diff, best_indx = -diff, i
        response |= {
            p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
            for p in plist[best_indx]
        }
        remained_consumers = list(set(selling_offers).difference(plist[best_indx]))
        remained_output_needs = best_diff
        for t in range(self.awi.current_step, self.awi.n_steps):
            if len(remained_consumers) == 0:
                break
            if remained_output_needs == 0:
                response |= {k: unneeded_response for k in remained_consumers}
                break
            tmp_output_needs = (
                min(
                    remained_output_needs,
                    self.awi.n_lines - (todays_output_needed - best_diff),
                )
                if t == self.awi.current_step
                else min(remained_output_needs, self.awi.n_lines)
            )
            if tmp_output_needs == 0:
                continue
            concentrated_ids = sorted(
                remained_consumers,
                key=lambda x: self.total_agreed_quantity[x],
                reverse=True,
            )
            scores = self.get_partner_scores()
            distribution = self.distribute_history_based(
                tmp_output_needs, list(remained_consumers), scores
            )
            mn = (
                max(
                    self.awi.current_exogenous_input_price
                    / self.awi.current_exogenous_input_quantity
                    + self.awi.profile.cost
                    + 1,
                    self.awi.current_output_issues[UNIT_PRICE].min_value,
                )
                if self.awi.current_exogenous_input_quantity > 0
                else max(
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + self.awi.profile.cost
                    + 1,
                    self.awi.current_output_issues[UNIT_PRICE].min_value,
                )
            )
            mx = mn + (self.awi.current_output_issues[UNIT_PRICE].max_value - mn) / 2
            response |= {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (q, t, self.sell_price(states[k].relative_time, mn, mx)),
                )
                if q > 0
                else unneeded_response
                for k, q in distribution.items()
            }
            remained_output_needs -= tmp_output_needs
            remained_consumers = list(
                set(remained_consumers).difference(
                    {k for k, q in distribution.items() if q > 0}
                )
            )

        # print(f"day {self.awi.current_step} {self.id}")
        # print("Exogenous Contracts:",self.awi.current_exogenous_input_quantity,self.awi.current_exogenous_output_quantity)
        # print("inventories:",self.awi.current_inventory)
        # print("future selling contracts:",[contract.agreement["quantity"] for contract in self.future_selling_contracts])
        # print("Input Responses:",{k:v for k,v in response.items() if k in self.awi.my_suppliers})
        # print("Output Responses:",{k:v for k,v in response.items() if k in self.awi.my_consumers})

        return response

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        if partners is None:
            partners = self.negotiators.keys()

        response = dict(zip(partners, repeat(0)))
        for is_partner, edge_needs in (
            (self.is_supplier, self.awi.needed_supplies),
            (self.is_consumer, self.awi.needed_sales),
        ):
            # 中間層の場合は生産性で需要補正
            needs = (
                max(edge_needs, int(self.awi.n_lines * self._productivity))
                if self.awi.is_middle_level
                else edge_needs
            )

            # 終盤の仕入れ抑制: 残り3ステップ以内なら仕入れを半分に
            if (
                self.awi.current_step >= self.awi.n_steps - 3
                and is_partner == self.is_supplier
            ):
                needs = int(needs * 0.5)

            # 役割に応じたアクティブパートナー抽出
            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            random.shuffle(active_partners)
            # 本日分を分配するパートナー数を調整
            active_partners = active_partners[
                : max(1, int(self._ptoday * len(active_partners)))
            ]

            if needs <= 0 or len(active_partners) <= 0:
                continue

            if needs < len(active_partners):
                active_partners = random.sample(
                    active_partners, random.randint(1, needs)
                )

            # パートナーごとのスコアに基づき分配
            scores = self.get_partner_scores()
            response |= self.distribute_history_based(needs, active_partners, scores)

        return response

    def get_partner_scores(self) -> dict[str, float]:
        # 各パートナーの成功率・平均価格からスコアを算出
        scores = {}
        for partner, record in self.negotiation_history.items():
            s = record["success"]
            f = record["fail"]
            total = s + f
            avg_price = record.get("avg_price", 1.0)
            if total == 0:
                success_rate = 0.5
            else:
                success_rate = s / total
            if self.is_supplier(partner):
                # 仕入れ時は安いほど高評価
                price_score = 1 / (avg_price + 1e-3)
            else:
                # 販売時は高いほど高評価
                price_score = avg_price
            scores[partner] = success_rate * price_score
        return scores

    def distribute_history_based(
        self, q: int, partners: list[str], scores: dict[str, float]
    ) -> dict[str, int]:
        # 交渉履歴に基づきパートナーへ需要を重み付き分配
        from numpy import array
        from numpy.random import multinomial

        if not partners or q <= 0:
            return {p: 0 for p in partners}

        weights = array([max(scores.get(p, 1e-3), 1e-3) for p in partners])
        weights /= weights.sum()

        allocation = multinomial(q, weights)
        return dict(zip(partners, allocation.tolist()))

    # def sample_future_offer(self, partner):
    #     # 将来タイミングでのランダムな提案を生成（交渉継続のためのREJECT提案）
    #     nmi = self.get_nmi(partner)
    #     outcome = nmi.random_outcome()
    #     t = outcome[TIME]
    #     if t == self.awi.current_step:
    #         mn = max(nmi.issues[TIME].min_value, self.awi.current_step + 1)
    #         mx = max(nmi.issues[TIME].max_value, self.awi.current_step + 1)
    #         if mx <= mn:
    #             return SAOResponse(ResponseType.END_NEGOTIATION, None)
    #         t = random.randint(mn, mx)
    #     return SAOResponse(
    #         ResponseType.REJECT_OFFER, (outcome[QUANTITY], t, self.price(partner, 0.5))
    #     )

    def is_needed(self, partner, offer):
        # 提案された数量がニーズを満たすか判定
        if offer is None:
            return False
        return offer[QUANTITY] <= self._needs(partner, offer[TIME])

    def is_good_price(self, partner, offer, state):
        # 提案価格が許容範囲か判定（譲歩戦略含む）
        if offer is None:
            return False
        nmi = self.get_nmi(partner)
        if not nmi:
            return False
        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value
        r = state.relative_time
        if offer[TIME] > self.awi.current_step:
            r *= self.future_concession
        if self.is_consumer(partner):
            return offer[UNIT_PRICE] >= minp + (1 - r) * (maxp - minp)
        return -offer[UNIT_PRICE] >= -minp + (1 - r) * (minp - maxp)

    def respond(self, negotiator_id, state, source=""):
        # 1件ごとの交渉に対する応答（承諾/拒否）
        offer = state.current_offer

        # 納期が遠すぎる提案は拒否
        if offer and offer[TIME] - self.awi.current_step > 7:
            return ResponseType.REJECT_OFFER

        # 自分がBUYERかどうか判定
        if self.is_supplier(negotiator_id):  # 自分がBUYER
            step_limit = int(self.awi.n_steps * 0.15)
            # 入力在庫が十分高い場合のみ仕入れ制限を適用
            if (
                self.awi.current_step >= step_limit
                and self.awi.current_inventory_input > 10
            ):
                return ResponseType.REJECT_OFFER

        return (
            ResponseType.ACCEPT_OFFER
            if self.is_needed(negotiator_id, offer)
            and self.is_good_price(negotiator_id, offer, state)
            else ResponseType.REJECT_OFFER
        )

    def propose(self, negotiator_id: str, state):
        # 自分がBUYERの場合、ステップ序盤のみ提案
        if self.is_supplier(negotiator_id):
            step_limit = int(self.awi.n_steps * 0.15)
            if self.awi.current_step >= step_limit:
                return None
        return self.good_offer(negotiator_id, state)

    def good_offer(self, partner, state):
        # 数量・価格・納期の高度な提案ロジック
        nmi = self.get_nmi(partner)
        if not nmi:
            return None
        issues = nmi.issues
        qissue = issues[QUANTITY]
        t_list = sorted(list(issues[TIME].all))
        for t in t_list:
            # 納期が遠すぎる場合はスキップ
            if abs(t - self.awi.current_step) > 7:
                continue
            needed = self._needs(partner, t)
            if needed <= 0:
                continue
            r = state.relative_time
            if t > self.awi.current_step:
                r *= self.future_concession
            # セラーの場合は進むほど譲歩しやすく
            if self.is_consumer(partner):
                step_r = self.awi.current_step / max(1, self.awi.n_steps)
                r = min(1.0, r + step_r * 0.5)
            quantity = max(min(needed, qissue.max_value), qissue.min_value)
            price = self.price(partner, r)
            return (quantity, t, price)
        return None

    def _needs(self, partner, t):
        # 指定パートナー・納期に対する必要数量を計算
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines

        if self.is_consumer(partner):
            # 販売側（出荷）では、確実に生産できる数量を超えないよう制限
            safe_limit = self.max_safe_sales_today()
            future_steps = max(0, t - self.awi.current_step)
            total_needs += self.production_level * self.awi.n_lines * future_steps
            total_needs -= self.awi.total_sales_until(t)
            total_needs = min(total_needs, safe_limit)
        else:
            # 在庫が少ない場合は将来仕入れ需要をブースト
            projected_demand = self.projected_sales + int(self.awi.n_lines * 0.5)
            current_supply = self.awi.total_supplies_between(t, self.awi.n_steps - 1)
            total_needs = max(0, projected_demand - current_supply)
        return int(total_needs)

    def is_consumer(self, partner):
        # パートナーが消費先かどうか
        return partner in self.awi.my_consumers

    def is_supplier(self, partner):
        # パートナーが仕入先かどうか
        return partner in self.awi.my_suppliers

    def price(self, partner, relative_time: float):
        # 交渉進捗・履歴を加味した譲歩価格戦略（best_price の機能を統合）
        nmi = self.get_nmi(partner)
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        r = relative_time

        if self.is_consumer(partner):
            setep_r = self.awi.current_step / max(1, self.awi.n_steps)
            r = min(1.0, r + setep_r * 0.5)

        record = self.negotiation_history.get(partner, {})
        avg_price = record.get("avg_price", None)

        if avg_price is not None:
            # rに応じてavg_priceに徐々に寄せる
            target = max(min(avg_price, maxp), minp)
            if self.is_consumer(partner):
                return int((1 - r) * target + r * minp + 0.5)
            else:
                return int((1 - r) * target + r * maxp + 0.5)
        else:
            # フォールバック: best_priceの代替ロジック
            return minp if self.is_supplier(partner) else maxp

    def buy_price(self, relative_time, min_price, max_price, aggressive=False):
        r = relative_time
        if aggressive:
            r = r**0.5  # 譲歩を速める
        return int((1 - r) * max_price + r * min_price + 0.5)

    def sell_price(self, relative_time, min_price, max_price):
        r = relative_time
        return int((1 - r) * min_price + r * max_price + 0.5)

    def best_price(self, partner):
        return self.price(partner, relative_time=0.0)

    @property
    def total_agreed_quantity(self):
        return {
            partner: record.get("total_quantity", 0)
            for partner, record in self.negotiation_history.items()
        }

    def init(self):
        # エージェント-世界インターフェース初期化後に一度だけ呼ばれる
        self.cumulative_input = 0
        self.max_inventory = (
            self.awi.n_lines * self._productivity
        )  # 最大在庫数（例として2倍ライン数）
        self.projected_sales = 0

    def before_step(self):
        # 毎生産ステップ（1日）の冒頭で呼ばれる
        if self.p_max_for_buying is None:
            try:
                issue = self.awi.current_input_issues[UNIT_PRICE]
                self.p_max_for_buying = issue.max_value
            except:
                self.p_max_for_buying = 100  # フォールバック値

        # 過去3日間の販売実績に基づいて販売予測を更新
        history_steps = 3
        current_step = self.awi.current_step
        sales_data = [
            self.awi.total_sales_at(current_step - i - 1)
            for i in range(min(history_steps, current_step))
        ]
        avg_sales = (
            sum(sales_data) / len(sales_data) if sales_data else self.awi.n_lines * 0.5
        )

        # 在庫状況による補正（在庫多い場合は販売予測を抑制）
        inventory_factor = max(
            0.0, 1.0 - (self.awi.current_inventory_input / (self.awi.n_lines * 3.0))
        )

        # 残りステップ数による減衰
        remaining = max(1, self.awi.n_steps - self.awi.current_step)
        decay = min(1.0, remaining / self.awi.n_steps)
        # 販売予測値を更新（在庫補正・減衰を反映、最低1）
        self.projected_sales = max(
            1, int((avg_sales + self.awi.n_lines * 0.3) * inventory_factor * decay)
        )

        # # ペナルティ状況に応じて生産性を調整
        # penalty_level = getattr(self, "prev_inventory_penalized", 0)
        # if penalty_level > self.awi.n_lines * 0.5:
        #     self._productivity *= 0.8  # 生産性を少し下げる
        #     self._productivity = max(0.3, self._productivity)  # 下限を設ける
        # else:
        #     self._productivity *= 1.05  # 生産性を少し戻す
        #     self._productivity = min(1.0, self._productivity)  # 上限を設ける

    def step(self):
        # その日の実績（不足ペナルティ・在庫コスト等）を記録
        self.prev_shortfall = self.awi.current_shortfall_penalty
        self.prev_storage_cost = self.awi.current_storage_cost
        self.prev_inventory_penalized = self.awi.current_inventory_input
        # self.print_customer_summary()

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """エージェントが参加する交渉が合意せず終了した時に呼ばれる"""
        # 交渉失敗時の履歴更新
        partner = [p for p in partners if p != self.id][0]
        self.negotiation_history[partner]["fail"] += 1

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        """エージェントが参加する交渉が合意で終了した時に呼ばれる"""
        # 交渉成立時の履歴更新（累計価格・数量・平均価格など）
        role = "SELLER" if contract.annotation.get("seller") == self.id else "BUYER"
        partner = [p for p in contract.partners if p != self.id][0]
        self.negotiation_history[partner]["success"] += 1
        agreement = contract.agreement
        incoming_quantity = contract.agreement.get("quantity")
        time = contract.agreement.get("time")
        if role == "BUYER":
            self.cumulative_input += incoming_quantity
        input_inv, output_inv = self.awi.current_inventory

        pass  # print(f"[CONTRACT] Step={self.awi.current_step:<2}, Partner={partner:<7}, Role={role:<6}, input_inventory={input_inv:<2}, time={time:<2},quantity={incoming_quantity:<2}, total_quantity={self.cumulative_input}")
        if partner not in self.negotiators:
            pass  # print(f"[NOTICE] Unexpected contract with non-negotiator partner: {partner}")

    # def print_customer_summary(self):
    #     """
    #     交渉相手ごとの成功/失敗回数と平均価格をCSVファイルとして保存する（顧客リスト風）
    #     """
    #     import csv
    #     import os
    #
    #     # 保存先ディレクトリのパスを修正
    #     step_dir = f"step{self.awi.current_step}"
    #     log_dir = os.path.join("myagent", "customerlog", step_dir)
    #     os.makedirs(log_dir, exist_ok=True)
    #
    #     filename = f"customer_summary_step{self.awi.current_step}4.csv"
    #     filepath = os.path.join(log_dir, filename)
    #
    #     with open(filepath, mode="w", newline="", encoding="utf-8") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["partner_id", "success", "fail", "avg_price", "total_quantity"])
    #         for partner, record in self.negotiation_history.items():
    #             success = record.get("success", 0)
    #             fail = record.get("fail", 0)
    #             avg_price = record.get("avg_price", "N/A")
    #             total_quantity = record.get("total_quantity", 0)
    #             writer.writerow([partner, success, fail, avg_price, total_quantity])


if __name__ == "__main__":
    import sys
    from .helpers.runner import run

    run([UltraSuperMiracleSoraFinalAgentZ], sys.argv[1] if len(sys.argv) > 1 else "std")
