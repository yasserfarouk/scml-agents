#!/usr/bin/env python
# first_proposals score-based allocation version

from __future__ import annotations

from collections import defaultdict
from typing import Literal
import math

from negmas import *
from scml.std import *

from dataclasses import dataclass, field
from typing import Any


__all__ = ["BaseAgeAgeAgent"]

@dataclass
class OfferDecisionResult:
    accepted_responses: dict[str, SAOResponse] = field(default_factory=dict)
    counter_buy_offers: dict[str, Any] = field(default_factory=dict)
    counter_sell_offers: dict[str, Any] = field(default_factory=dict)

class BaseAgeAgeAgent(StdSyncAgent):
    QUANTITY_AVG_DISCOUNT_RATE = 0.2 # 取引量の加重平均の割引率
    PRICE_AVG_DISCOUNT_RATE = 0.2
    AVG_CONCESSION_DISCOUNT_RATE = 0.2
    AVG_DECREASE_ON_FAULT = 0.5 # 取引に失敗したときに加重平均をどれくらい減らすか

    NEAR_DELIVERY_WINDOW = 1
    FAR_DELIVERY_WINDOW = 5
    
    MIN_PROFIT = 0

    avg_sell_price: float
    avg_buy_price: float
    partner_avg_proposal_quantity: float

    partner_weighted_avg_quantity: dict[str, float]
    partner_weighted_avg_price: dict[str, float]

    exo_input_q: int
    exo_output_q: int

    partner_history: dict[str, list[int]]
    success_rate: dict[str, float]

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.partner_weighted_avg_quantity = defaultdict(float)
        self.partner_weighted_avg_price = defaultdict(float)
        self.partner_avg_proposal_quantity = defaultdict(float)

        # partner_history[partner] = [契約成功回数, 契約失敗回数]
        self.partner_history = defaultdict(lambda: [0, 0])

        # 履歴がない相手の初期成功率は 0.3 とする
        self.success_rate = defaultdict(lambda: 0.7)

        self.avg_buy_price = 0.0
        self.avg_sell_price = 0.0

        self.last_offers = {}

        self.long_concession_avg = defaultdict(
            lambda: {
                QUANTITY: 0.0,
                UNIT_PRICE: 0.0,
            }
        )

        self.short_concession_avg = defaultdict(
            lambda: {
                QUANTITY: 0.0,
                UNIT_PRICE: 0.0,
            }
        )

        self.last_round_seen = defaultdict(lambda: -1)
        self.last_ufun_extra_check = {}

    def step(self):
        awi = self.awi
        
        if awi.current_step == 0:
            partners = self.negotiators.keys()
            self.init_partner_avg_quantity(partners)
            self.init_partner_avg_price(partners)
            self.init_partner_history(partners)

        input_q = awi.current_exogenous_input_quantity
        output_q = awi.current_exogenous_output_quantity
        input_total_price = awi.current_exogenous_input_price
        output_total_price = awi.current_exogenous_output_price

        if input_q > 0:
            input_unit_price = input_total_price / input_q
            self.update_partner_avg_quantity("exogenous_input", input_q)
            self.update_partner_avg_price("exogenous_input", input_unit_price)
            return

        if output_q > 0:
            output_unit_price = output_total_price / output_q
            self.update_partner_avg_quantity("exogenous_output", output_q)
            self.update_partner_avg_price("exogenous_output", output_unit_price)
            return

    def on_negotiation_success(self, contract, mechanism):
        partner = next(p for p in contract.partners if p != self.id)

        agreement = contract.agreement

        quantity = agreement["quantity"]
        unit_price = agreement["unit_price"]

        # 加重平均の計算
        self.update_partner_avg_quantity(partner, quantity)

        # 平均取引価格の更新
        self.update_partner_avg_price(partner, unit_price)

        # 成功回数と成功率の更新
        self.update_partner_success_rate(partner, succeeded=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # 契約が成立しなかった交渉相手の取引量の加重平均を減らす
        partner = next(p for p in partners if p != self.id)
        current_quantity = self.partner_weighted_avg_quantity[partner]
        self.partner_weighted_avg_quantity[partner] = max(
            1,
            current_quantity - self.AVG_DECREASE_ON_FAULT
        )

        # 失敗回数と成功率の更新
        self.update_partner_success_rate(partner, succeeded=False)

    def first_proposals(self):
        """
        初回提案を作成する。

        方針:
        1. 当日の必要量がある場合は、get_score が高い相手から平均取引量で埋める。
        2. 平均取引量だけで埋まらない場合は、平均取引量 + 2 を上限として再配分する。
        3. それでも埋まらない場合は、無理に増やさず作れた分だけ提案する。
        4. 当日分で使わなかった相手は、将来契約用に回す。
        5. 当日の必要量がない側は、全員を将来契約用に回す。
        """
        response = {}
        current_step = self.awi.current_step

        remaining_suppliers = list(self.awi.my_suppliers)
        remaining_consumers = list(self.awi.my_consumers)

        # まずは当日の必要量を確認する
        buy_needs, sell_needs = self.get_needs(current_step)

        # 当日分: スコア順に必要量を埋める
        today_buy_offers, remaining_suppliers = self.make_today_offers_by_score(
            partners=remaining_suppliers,
            need=buy_needs,
            step=current_step,
        )

        today_sell_offers, remaining_consumers = self.make_today_offers_by_score(
            partners=remaining_consumers,
            need=sell_needs,
            step=current_step,
        )

        response |= today_buy_offers
        response |= today_sell_offers

        # 将来分: 当日に使わなかった相手をスコア基準ナップサックに回す
        future_start_step = current_step + 1

        if future_start_step >= self.awi.n_steps:
            return response

        future_buy_offers = self.make_base_offers_for_partners(
            remaining_suppliers,
            future_start_step,
        )

        future_sell_offers = self.make_base_offers_for_partners(
            remaining_consumers,
            future_start_step,
        )

        response |= self.assign_delivery_steps_by_score_knapsack(
            future_buy_offers,
            "buy_offer",
            future_start_step,
        )

        response |= self.assign_delivery_steps_by_score_knapsack(
            future_sell_offers,
            "sell_offer",
            future_start_step,
        )

        return response

    def make_today_offers_by_score(self, partners, need: int, step: int):
        """
        当日の必要量を、get_score が高い相手から順に埋める。

        1回目は平均取引量で配分する。
        足りない場合は、平均取引量 + 2 を上限としてもう一度配分する。
        """
        partners = list(partners)

        # 当日の必要量がない場合は、全員を将来契約に回す
        if need <= 0 or not partners:
            return {}, partners

        sorted_partners = sorted(
            partners,
            key=lambda p: self.get_score(p, step),
            reverse=True,
        )

        def build_offers(extra_quantity: int):
            offers = {}
            total_quantity = 0
            used_partners = set()

            for partner in sorted_partners:
                quantity = self.get_avg_offer_quantity(
                    partner,
                    extra_quantity=extra_quantity,
                )

                if quantity <= 0:
                    continue

                offers[partner] = (
                    quantity,
                    step,
                    self.get_valid_price(partner),
                )

                total_quantity += quantity
                used_partners.add(partner)

                if total_quantity >= need:
                    break

            return offers, total_quantity, used_partners

        # 1回目: 平均取引量で埋める
        offers, total_quantity, used_partners = build_offers(extra_quantity=0)

        if total_quantity >= need:
            remaining_partners = [
                p for p in sorted_partners
                if p not in used_partners
            ]
            return offers, remaining_partners

        # 2回目: 平均取引量 + 2 まで増やして再配分する
        offers, total_quantity, used_partners = build_offers(extra_quantity=2)

        if total_quantity >= need:
            remaining_partners = [
                p for p in sorted_partners
                if p not in used_partners
            ]
            return offers, remaining_partners

        # それでも埋まらない場合はあきらめる。
        # ただし、作れた分のオファーはそのまま出す。
        return offers, []

    def make_base_offers_for_partners(self, partners, step: int):
        """
        将来契約に回すためのベースオファーを作る。
        納期は assign_delivery_steps_by_score_knapsack 側で決め直す。
        """
        offers = {}

        for partner in partners:
            quantity = self.get_avg_offer_quantity(partner)

            if quantity <= 0:
                continue

            offers[partner] = (
                quantity,
                step,
                self.get_valid_price(partner),
            )

        return offers

    def get_avg_offer_quantity(self, partner, extra_quantity: int = 0) -> int:
        """
        相手の平均取引量をもとに、こちらから提示する取引量を決める。
        extra_quantity=2 の場合は、平均取引量 + 2 まで増やす。
        """
        quantity_issue = self.get_quantity_issue(partner)

        quantity = math.ceil(
            self.partner_weighted_avg_quantity[partner]
        ) + extra_quantity

        quantity = max(
            quantity_issue.min_value,
            min(quantity_issue.max_value, quantity),
        )

        return int(quantity)

    def get_quantity_issue(self, partner):
        if partner in self.awi.my_suppliers:
            return self.awi.current_input_issues[QUANTITY]
        else:
            return self.awi.current_output_issues[QUANTITY]

    def assign_delivery_steps_by_score_knapsack(
        self,
        offers,
        mode: Literal["buy_offer", "sell_offer"],
        step: int,
    ):
        """
        将来契約用。
        納期が近い step から順に、get_score(partner, step) が高い相手を
        ナップサックで選んで割り当てる。
        """
        response = {}

        if step > self.awi.n_steps - 1:
            return response

        if not offers:
            return response

        if mode == "buy_offer":
            needs, _ = self.get_needs(step)
        elif mode == "sell_offer":
            _, needs = self.get_needs(step)

            # 既存 first_proposals で sell 側だけ多めに見ていた設計を残す
            needs = int(needs * 1.5)
        else:
            return response

        # この step に必要量がないなら、次の step へ回す
        if needs <= 0:
            return self.assign_delivery_steps_by_score_knapsack(
                offers,
                mode,
                step + 1,
            )

        scores = {
            partner: self.get_score(partner, step)
            for partner in offers
        }

        _, selected_partners = solve_knapsack_for_scml_offers_by_score(
            offers,
            needs,
            scores,
        )

        # どのオファーも capacity に入らない場合、全員が将来に回され続けて
        # 結局オファーが消えるため、最もスコアが高い相手を1人だけ選ぶ。
        if not selected_partners:
            selected_partners = [
                max(
                    offers.keys(),
                    key=lambda p: scores.get(p, 0.0),
                )
            ]

        remaining_offers = offers.copy()

        for partner in selected_partners:
            response[partner] = (
                remaining_offers[partner][QUANTITY],
                step,
                remaining_offers[partner][UNIT_PRICE],
            )

            remaining_offers.pop(partner)

        if remaining_offers:
            response |= self.assign_delivery_steps_by_score_knapsack(
                remaining_offers,
                mode,
                step + 1,
            )

        return response

    def counter_all(self, offers, states):
        self.update_concession_stats(offers, states)

        response = defaultdict(
            lambda: SAOResponse(ResponseType.END_NEGOTIATION, None)
        )

        buy_offers, sell_offers = (
            self.split_offers_by_partner(offers)
        )

        # 平均提案量更新
        for partner, offer in offers.items():
            if self.partner_avg_proposal_quantity[partner] == 0:
                self.partner_avg_proposal_quantity[partner] = offer[QUANTITY]
                continue

            rate = 0.2
            self.partner_avg_proposal_quantity[partner] = (
                (1 - rate) * self.partner_avg_proposal_quantity[partner]
                 + rate * offer[QUANTITY]
            )

        # 納期ごとに必要な量の契約を結ぶ
        offer_decision_result = self.select_offers_by_delivery_step(buy_offers, sell_offers, states)

        response |= offer_decision_result.accepted_responses

        counter_buy_offers = offer_decision_result.counter_buy_offers
        counter_sell_offers = offer_decision_result.counter_sell_offers

        # 未採用オファーは、元オファーの quantity / time / price を使わず、
        # partner だけを使ってこちらの理想オファーを作り直す。
        response |= self.make_counter_responses_by_score_fill(
            counter_buy_offers,
            "buy_offer",
            states,
        )

        response |= self.make_counter_responses_by_score_fill(
            counter_sell_offers,
            "sell_offer",
            states,
        )

        return response
    
    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """
        Returns:
            エージェントIDをキー、取引量を値とする辞書
        """
        if partners is None:
            partners = self.negotiators.keys()
        
        response = {}
        total_seller_weight = 0
        total_buyer_weight = 0

        buy_needs, sell_needs = self.get_needs(None)

        for partner in partners:
            if partner in self.awi.my_suppliers:
                response[partner] = min(
                    max(
                        math.ceil(self.partner_avg_proposal_quantity[partner]),
                        math.ceil(buy_needs / len(self.awi.my_suppliers)),
                    ),
                    5
                )
            else:
                response[partner] = min(
                    max(
                        math.ceil(self.partner_avg_proposal_quantity[partner]),
                        math.ceil(sell_needs / len(self.awi.my_consumers)),
                    ),
                    5
                )

        return response

    def select_offers_by_delivery_step(self, buy_offers, sell_offers, states):
        result = OfferDecisionResult()

        # 納期ごとにオファーを分ける
        sorted_buy_offers = group_offers_by_delivery_time(buy_offers)
        sorted_sell_offers = group_offers_by_delivery_time(sell_offers)
        
        # 納期ごとにオファーの受諾判断
        for i in range(self.awi.current_step, self.awi.n_steps):
            buy_offer_dict = sorted_buy_offers.get(i, {})
            sell_offer_dict = sorted_sell_offers.get(i, {})

            offer_decision_result = self.select_offers_at_step(
                buy_offer_dict,
                sell_offer_dict,
                step=i,
                states=states,
            )

            result.accepted_responses |= offer_decision_result.accepted_responses
            result.counter_buy_offers |= offer_decision_result.counter_buy_offers
            result.counter_sell_offers |= offer_decision_result.counter_sell_offers
            
        return result
    
    def select_offers_at_step(self, buy_offer_dict, sell_offer_dict, step, states):
        result = OfferDecisionResult()

        buy_offers = buy_offer_dict.copy()
        sell_offers = sell_offer_dict.copy()

        counter_buy_offers = {}
        counter_sell_offers = {}

        if len(buy_offer_dict) == 0 and len(sell_offer_dict) == 0:
            return result

        # ===========
        # 納期が近いときの最低利益価格チェック
        # ===========

        for partner, offer in buy_offer_dict.items():
            if not (
                self.awi.current_step
                <= step
                < self.awi.current_step + self.FAR_DELIVERY_WINDOW
            ):
                break

            if not self.is_min_profit_price(partner, offer[UNIT_PRICE]):
                buy_offers.pop(partner, None)
                counter_buy_offers[partner] = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_min_profit_price(partner),
                )

        for partner, offer in sell_offer_dict.items():
            if not (
                self.awi.current_step
                <= step
                < self.awi.current_step + self.FAR_DELIVERY_WINDOW
            ):
                break

            if not self.is_min_profit_price(partner, offer[UNIT_PRICE]):
                sell_offers.pop(partner, None)
                counter_sell_offers[partner] = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_min_profit_price(partner),
                )

        # ===========
        # 納期が遠いときの期待値チェック
        # ===========

        for partner, offer in buy_offer_dict.items():
            if not (self.awi.current_step + self.FAR_DELIVERY_WINDOW <= step):
                break

            if len(self.awi.my_suppliers) == 0:
                continue

            total_expected_value = 0

            for supplier in self.awi.my_suppliers:
                total_expected_value += self.get_expected_value(supplier)

            expected_value = self.get_expected_value(partner)
            avg_expected_value = total_expected_value / len(self.awi.my_suppliers)

            if expected_value < avg_expected_value:
                buy_offers.pop(partner, None)

                state = states.get(partner)
                current_round = state.step if state is not None else 0

                counter_buy_offers[partner] = self.make_adjusted_offer(
                    partner,
                    offer,
                    current_round,
                )

        for partner, offer in sell_offer_dict.items():
            if not (self.awi.current_step + self.FAR_DELIVERY_WINDOW <= step):
                break

            if len(self.awi.my_consumers) == 0:
                continue

            total_expected_value = 0

            for consumer in self.awi.my_consumers:
                total_expected_value += self.get_expected_value(consumer)

            expected_value = self.get_expected_value(partner)
            avg_expected_value = total_expected_value / len(self.awi.my_consumers)

            if expected_value < avg_expected_value:
                sell_offers.pop(partner, None)

                state = states.get(partner)
                current_round = state.step if state is not None else 0

                counter_sell_offers[partner] = self.make_adjusted_offer(
                    partner,
                    offer,
                    current_round,
                )

        # ===========
        # 必要量計算
        # ===========

        target_buy_quantity, target_sell_quantity = (
            self.calculate_target_quantities_at_step(
                buy_offers,
                sell_offers,
                step,
            )
        )

        # ===========
        # ナップサックで受諾候補を選ぶ
        # ===========

        _, selected_supplier = solve_knapsack_for_scml_offers(
            buy_offers,
            target_buy_quantity,
            "low",
        )

        _, selected_consumer = solve_knapsack_for_scml_offers(
            sell_offers,
            target_sell_quantity,
            "high",
        )

        # ===========
        # ナップサック後に、容量超過してでも追加受諾した方が得かを ufun で確認
        # ===========

        selected_supplier, selected_consumer = self.improve_selected_offers_by_ufun(
            buy_offers,
            sell_offers,
            selected_supplier,
            selected_consumer,
            step,
        )

        # ===========
        # ufun後の最終選択に基づいて、選ばれなかった相手をカウンター対象にする
        # ===========

        selected_supplier_set = set(selected_supplier)
        selected_consumer_set = set(selected_consumer)

        for partner, offer in buy_offers.items():
            if partner not in selected_supplier_set:
                counter_buy_offers[partner] = offer

        for partner, offer in sell_offers.items():
            if partner not in selected_consumer_set:
                counter_sell_offers[partner] = offer

        # ===========
        # 受諾応答を作成
        # ===========

        for partner in selected_supplier:
            result.accepted_responses[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER,
                None,
            )

        for partner in selected_consumer:
            result.accepted_responses[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER,
                None,
            )

        result.counter_buy_offers |= counter_buy_offers
        result.counter_sell_offers |= counter_sell_offers

        return result
    
    def calculate_target_quantities_at_step(
        self,
        buy_offer_dict,
        sell_offer_dict,
        step,
    ):
        """
        counter_all受諾判断に使う必要量計算

        Returns:
            target_buy_quantity, target_sell_quantity
        """
        total_buy_offer_quantity = get_total_offer_quantity(buy_offer_dict)
        total_sell_offer_quantity = get_total_offer_quantity(sell_offer_dict)

        contract_supply = self.awi.total_supplies_at(step)
        contract_sales = self.awi.total_sales_at(step)

        offer_supply = total_buy_offer_quantity
        offer_sales = total_sell_offer_quantity

        inventory = self.awi.current_inventory_input
        n_lines = self.awi.n_lines

        input_q = self.awi.current_exogenous_input_quantity
        output_q = self.awi.current_exogenous_output_quantity

        is_near_delivery = (
            self.awi.current_step
            <= step
            < self.awi.current_step + self.NEAR_DELIVERY_WINDOW
        )

        if is_near_delivery:
            
            if input_q > 0 or self.partner_weighted_avg_quantity["exogenous_input"] > 0:
                target_quantity = min(
                    inventory + input_q,
                    n_lines,
                ) - contract_sales
                
                return 0, target_quantity
            
            elif output_q > 0 or self.partner_weighted_avg_quantity["exogenous_output"] > 0:
                target_quantity = min(
                    max(
                        output_q,
                        int(self.awi.n_lines * 0.5),
                    ),
                    n_lines,
                ) - inventory - contract_supply

                return target_quantity, 0

            target_quantity = min(
                contract_sales + offer_sales,
                contract_supply + offer_supply + inventory,
                n_lines,
            )

            target_buy_quantity = max(
                0,
                target_quantity - contract_supply - inventory,
            )

            target_sell_quantity = max(
                0,
                target_quantity - contract_sales,
            )

            return target_buy_quantity, target_sell_quantity
        
        else:
            if input_q > 0 or self.partner_weighted_avg_quantity["exogenous_input"] > 0:
                target_quantity = int(self.awi.n_lines * 0.5)
                
                return 0, target_quantity
            
            elif output_q > 0 or self.partner_weighted_avg_quantity["exogenous_output"] > 0:
                if inventory > self.awi.n_lines:
                    target_quantity = 0
                else:
                    target_quantity = int(self.awi.n_lines * 0.5)

                return target_quantity, 0
            
            target_buy_quantity = max(
                0,
                contract_sales - contract_supply - inventory,
            )

            target_sell_quantity = max(
                0,
                self.awi.n_lines - contract_sales,
            )

            return target_buy_quantity, target_sell_quantity
       
    def make_counter_responses_by_score_fill(
        self,
        counter_offers,
        mode: Literal["buy_offer", "sell_offer"],
        states,
    ):
        """
        counter_all 用。

        ナップサックで選ばれなかったオファーに対して、
        元オファーの quantity / time / price は使わず、
        partner だけを使ってこちらの理想オファーを作り直す。

        方針:
        1. 納期が近い step から順に見る。
        2. その step の必要量を確認する。
        3. get_score が高い相手から順に必要量を埋める。
        4. 数量は get_avg_offer_quantity で作り直す。
        5. 価格は get_valid_price で作り直す。
        6. 最後に make_adjusted_offer で譲歩量を反映する。
        """
        response = {}

        if not counter_offers:
            return response

        remaining_partners = list(counter_offers.keys())

        for step in range(self.awi.current_step, self.awi.n_steps):
            if not remaining_partners:
                break

            if mode == "buy_offer":
                need, _ = self.get_needs(step)
            elif mode == "sell_offer":
                _, need = self.get_needs(step)
            else:
                return response

            if need <= 0:
                continue

            # first_proposals と同じ考え方で、
            # スコアが高い相手から必要量を埋める。
            offers_at_step, remaining_partners = self.make_today_offers_by_score(
                partners=remaining_partners,
                need=need,
                step=step,
            )

            for partner, offer in offers_at_step.items():
                state = states.get(partner)
                current_round = state.step if state is not None else 0

                # make_today_offers_by_score で作った数量・納期を使い、
                # 価格は current_round 付きで作り直す。
                base_offer = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_valid_price(
                        partner,
                        current_round=current_round,
                    ),
                )

                new_offer = self.make_adjusted_offer(
                    partner,
                    base_offer,
                    current_round,
                )

                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    new_offer,
                )

        return response

    def check_offer_price(self, offers, states):
        price_acceptable_offers = {}
        price_adjusted_offers = {}

        for partner, offer in offers.items():
            state = states.get(partner)

            # 適正価格よりも利益が出ない価格になっていた場合、修正してカウンターオファー
            if not self.is_valid_price(partner, offer[UNIT_PRICE]):
                new_offer = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_valid_price(partner, current_round=state.step)
                )
                
                price_adjusted_offers[partner] = new_offer

                continue
            
            # 適正価格のオファーはそのまま返す
            price_acceptable_offers[partner] = offer

        return price_acceptable_offers, price_adjusted_offers
        
    def get_needs(self, step=None):
        """
        当日の必要量を求めるメソッド

        Returns:
            buy_needs, sell_needs
        """
        awi = self.awi

        if step is None:
            step = awi.current_step

        inventory = awi.current_inventory_input
        contract_supply = awi.total_supplies_at(step)
        contract_sales = awi.total_sales_at(step)
        n_lines = awi.n_lines

        input_q = awi.current_exogenous_input_quantity
        output_q = awi.current_exogenous_output_quantity

        # 通常時の基本必要量
        buy_needs = int(
            max(
                0,
                contract_sales
                - inventory
                - contract_supply
                + (n_lines - contract_sales) * 0.7,
            )
        )

        sell_needs = int(
            max(
                0,
                n_lines - contract_sales,
            )
        )

        is_near_delivery = (
            self.awi.current_step
            <= step
            < self.awi.current_step + self.NEAR_DELIVERY_WINDOW
        )

        if is_near_delivery:
            if input_q > 0 or self.partner_weighted_avg_quantity["exogenous_input"] > 0:
                sell_needs = min(
                    inventory + input_q,
                    n_lines,
                ) - contract_sales
            
            elif output_q > 0 or self.partner_weighted_avg_quantity["exogenous_output"] > 0:
                buy_needs = min(
                    max(
                        output_q,
                        int(self.awi.n_lines * 0.5),
                    ),
                    n_lines,
                ) - inventory - contract_supply
        else:
            if input_q > 0 or self.partner_weighted_avg_quantity["exogenous_input"] > 0:
                sell_needs = int(self.awi.n_lines * 0.5) - contract_sales
            elif output_q > 0 or self.partner_weighted_avg_quantity["exogenous_output"] > 0:
                if inventory > self.awi.n_lines:
                    buy_needs = 0
                else:
                    buy_needs = int(self.awi.n_lines * 0.5) - contract_supply

        return buy_needs, sell_needs

    def update_concession_stats(self, offers, states):
        alpha = self.AVG_CONCESSION_DISCOUNT_RATE

        for partner, offer in offers.items():
            state = states.get(partner)
            if state is None:
                continue

            current_round = state.step

            # 新しい交渉が始まったら短期履歴をリセット
            if current_round == 0:
                self.short_concession_avg[partner] = {
                    QUANTITY: 0.0,
                    UNIT_PRICE: 0.0,
                }
                self.last_offers[partner] = offer
                self.last_round_seen[partner] = current_round
                continue

            self.last_round_seen[partner] = current_round

            # 前回オファーがなければ保存だけ
            if partner not in self.last_offers:
                self.last_offers[partner] = offer
                continue

            last_offer = self.last_offers[partner]

            for issue in (QUANTITY, UNIT_PRICE):
                change = abs(offer[issue] - last_offer[issue])

                # 長期平均譲歩量
                self.long_concession_avg[partner][issue] = (
                    (1 - alpha) * self.long_concession_avg[partner][issue]
                    + alpha * change
                )

                # 短期平均譲歩量
                self.short_concession_avg[partner][issue] = (
                    (1 - alpha) * self.short_concession_avg[partner][issue]
                    + alpha * change
                )

            self.last_offers[partner] = offer

    def get_issue_range(self, partner, issue):
        if issue == QUANTITY:
            if partner in self.awi.my_suppliers:
                issue_obj = self.awi.current_input_issues[QUANTITY]
            else:
                issue_obj = self.awi.current_output_issues[QUANTITY]

        elif issue == UNIT_PRICE:
            issue_obj = self.get_price_issue(partner)

        else:
            return 1.0

        return max(1.0, issue_obj.max_value - issue_obj.min_value)

    def calc_issue_weights(self, partner, current_round, max_round=20):
        eps = 1e-9

        long_scores = {}
        short_scores = {}

        for issue in (QUANTITY, UNIT_PRICE):
            issue_range = self.get_issue_range(partner, issue)

            long_scores[issue] = (
                self.long_concession_avg[partner][issue]
                / (issue_range + eps)
            )

            short_scores[issue] = (
                self.short_concession_avg[partner][issue]
                / (issue_range + eps)
            )

        long_sum = sum(long_scores.values()) + eps
        short_sum = sum(short_scores.values()) + eps

        w_long = {
            issue: long_scores[issue] / long_sum
            for issue in (QUANTITY, UNIT_PRICE)
        }

        w_short = {
            issue: short_scores[issue] / short_sum
            for issue in (QUANTITY, UNIT_PRICE)
        }

        if max_round <= 1:
            rho = 1.0
        else:
            rho = (current_round - 1) / (max_round - 1)

        rho = max(0.0, min(1.0, rho))

        # μは必ず0〜1にする
        mu_min = 0.5
        mu_max = 0.8
        k = 3.0

        mu = mu_min + (mu_max - mu_min) * (
            (1 - math.exp(-k * rho)) / (1 - math.exp(-k))
        )

        weights = {
            issue: (1 - mu) * w_long[issue] + mu * w_short[issue]
            for issue in (QUANTITY, UNIT_PRICE)
        }

        return weights

    def make_adjusted_offer(self, partner, offer, current_round):
        eps = 1e-9

        q = offer[QUANTITY]
        t = offer[TIME]
        p = offer[UNIT_PRICE]

        weights = self.calc_issue_weights(partner, current_round)

        w_q = weights[QUANTITY]
        w_p = weights[UNIT_PRICE]

        # 期待値差
        # 同じ側の相手の平均期待値 E_avg と、この相手の期待値 E_i の差を使う
        if partner in self.awi.my_suppliers:
            same_side_partners = self.awi.my_suppliers
        else:
            same_side_partners = self.awi.my_consumers

        if len(same_side_partners) == 0:
            expectation_gap = 0.0
        else:
            avg_expected_value = sum(
                self.get_expected_value(p)
                for p in same_side_partners
            ) / len(same_side_partners)

            partner_expected_value = self.get_expected_value(partner)

            # E_avg - E_i
            expectation_gap = avg_expected_value - partner_expected_value

        # 成功確率 S
        # 0 だと調整量が大きくなりすぎるため、下限を置く
        S = max(0.1, self.success_rate[partner])

        # 平均利益 m
        # 0以下だと量の調整が暴れるので下限を置く
        m = max(1.0, abs(self.avg_sell_price - self.avg_buy_price))

        # 価格調整
        price_adjust = w_p * expectation_gap / (S * max(1, q) + eps)

        if partner in self.awi.my_suppliers:
            # 相手が売り手、自分が買い手 → 価格を下げたい
            new_p = p - price_adjust
        else:
            # 相手が買い手、自分が売り手 → 価格を上げたい
            new_p = p + price_adjust

        # 量調整
        quantity_adjust = w_q * expectation_gap / (S * m + 1.0)

        new_q = q + quantity_adjust

        # issue範囲に収める
        if partner in self.awi.my_suppliers:
            q_issue = self.awi.current_input_issues[QUANTITY]
            p_issue = self.awi.current_input_issues[UNIT_PRICE]
        else:
            q_issue = self.awi.current_output_issues[QUANTITY]
            p_issue = self.awi.current_output_issues[UNIT_PRICE]

        new_q = int(round(new_q))
        new_p = int(round(new_p))

        new_q = max(q_issue.min_value, min(q_issue.max_value, new_q))
        new_p = max(p_issue.min_value, min(p_issue.max_value, new_p))

        return (
            new_q,
            t,
            new_p,
        )

    def init_partner_avg_quantity(self, partners) -> None:
        """
        交渉パートナーの取引量の初期値をセット
        初期値は、必要量を人数で分割
        """
        buy_needs, sell_needs = self.get_needs(0)
        sell_needs = int(sell_needs * 1.5)

        for partner in partners:
            self.partner_weighted_avg_quantity[partner] = (
                math.ceil(buy_needs / len(self.awi.my_suppliers))
                if partner in self.awi.my_suppliers
                else math.ceil(sell_needs / len(self.awi.my_consumers))
            )
    def update_partner_avg_quantity(self, partner, quantity):
        current_quantity = self.partner_weighted_avg_quantity[partner]
        next_quantity = quantity

        self.partner_weighted_avg_quantity[partner] = (
            (1-self.QUANTITY_AVG_DISCOUNT_RATE) * current_quantity + self.QUANTITY_AVG_DISCOUNT_RATE * next_quantity
        )

    def init_partner_avg_price(self, partners) -> None:
        """
        平均取引価格の初期値として、市場価格を設定
        """
        market_prices = self.awi.trading_prices
        
        input_market_price = market_prices[self.awi.my_input_product]
        output_market_price = market_prices[self.awi.my_output_product]

        for partner in partners:            
            if partner in self.awi.my_suppliers:
                self.partner_weighted_avg_price[partner] = input_market_price
                self.avg_buy_price = self.partner_weighted_avg_price[partner]
            else:
                self.partner_weighted_avg_price[partner] = output_market_price
                self.avg_sell_price = self.partner_weighted_avg_price[partner]

    
    def update_partner_avg_price(self, partner, price):
        """
        エージェントごとの平均取引価格と全取引の平均取引価格の更新
        """
        self.partner_weighted_avg_price[partner] = (
            (1 - self.PRICE_AVG_DISCOUNT_RATE)
            * self.partner_weighted_avg_price[partner]
            + self.PRICE_AVG_DISCOUNT_RATE
            * price
        )

        if partner in self.awi.my_suppliers or partner == "exogenous_input":
            self.avg_buy_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_buy_price + self.PRICE_AVG_DISCOUNT_RATE * price
        elif partner in self.awi.my_consumers or  partner == "exogenous_output":
            self.avg_sell_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_sell_price + self.PRICE_AVG_DISCOUNT_RATE * price
        
    def is_valid_price(self, partner, price):
        """
        オファーの価格が、十分利益の出るものになっているか判定する。
        """
        valid_price = self.get_valid_price(partner, 100)

        if partner in self.awi.my_suppliers:
            return price <= valid_price

        if partner in self.awi.my_consumers:
            return price >= valid_price

        return False

    def get_valid_price(self, partner, current_round=0):
        price_issue = self.get_price_issue(partner)

        total_success_rate = 0
        for p in self.awi.my_suppliers if partner in self.awi.my_suppliers else self.awi.my_consumers:
            total_success_rate += self.success_rate[p]
        
        avg_success_rate = total_success_rate / len(self.awi.my_suppliers) if partner in self.awi.my_suppliers else total_success_rate / len(self.awi.my_consumers)
        max_partner_bonus = 6
        max_market_bonus = 3

        partner_score = max_partner_bonus * (self.success_rate[partner] - 0.4) * 2

        market_score = 0

        if partner in self.awi.my_suppliers:
            market_score = max(0, max_market_bonus * (avg_success_rate - 0.2) * 2)

        if partner in self.awi.my_suppliers:
            price = self.partner_weighted_avg_price[partner] - partner_score - market_score
        else:
            price = self.partner_weighted_avg_price[partner] + partner_score + market_score

        price = min(price_issue.max_value, max(price_issue.min_value, round(price)))

        return price
        
    def is_min_profit_price(self, partner, price):
        """
        オファーの価格が、最低利益を満たすものになっているか判定する。
        """
        min_profit_price = self.get_min_profit_price(partner)

        if partner in self.awi.my_suppliers:
            return price <= min_profit_price

        if partner in self.awi.my_consumers:
            return price >= min_profit_price

        return False

    def get_min_profit_price(self, partner):
        """
        MIN_PROFIT を確保できる最低限の価格を返す。
        """
        price_issue = self.get_price_issue(partner)

        if partner in self.awi.my_suppliers:
            return max(
                price_issue.min_value,
                min(
                    price_issue.max_value,
                    int(self.avg_sell_price - self.MIN_PROFIT),
                ),
            )
        else:
            return min(
                price_issue.max_value,
                max(
                    price_issue.min_value,
                    int(self.avg_buy_price + self.MIN_PROFIT),
                ),
            )

    def get_expected_value(self, partner):
        """
        相手ごとの期待値を計算する。
        E_i = 成功率 × 1単位あたり利益 × 平均取引量
        """
        success_rate = self.success_rate[partner]

        if partner in self.awi.my_suppliers:
            profit = (
                self.avg_sell_price
                - self.partner_weighted_avg_price[partner]
            )
        else:
            profit = (
                self.partner_weighted_avg_price[partner]
                - self.avg_buy_price
            )

        return (
            success_rate
            * profit
            * self.partner_weighted_avg_quantity[partner]
        )

    def calc_ufun_info_from_partner_offers(self, selected_offers):
        """
        partner -> offer の辞書を self.ufun で評価する。
        """
        offers = tuple(selected_offers.values())
        outputs = tuple(
            partner in self.awi.my_consumers
            for partner in selected_offers.keys()
        )

        return self.ufun.from_offers(
            offers,
            outputs,
            return_info=True,
            ignore_signed_contracts=False,
        )

    def make_selected_offer_dict(
        self,
        buy_offer_dict,
        sell_offer_dict,
        selected_supplier,
        selected_consumer,
    ):
        selected_offers = {}

        for partner in selected_supplier:
            if partner in buy_offer_dict:
                selected_offers[partner] = buy_offer_dict[partner]

        for partner in selected_consumer:
            if partner in sell_offer_dict:
                selected_offers[partner] = sell_offer_dict[partner]

        return selected_offers

    def improve_selected_offers_by_ufun(
        self,
        buy_offer_dict,
        sell_offer_dict,
        selected_supplier,
        selected_consumer,
        step,
    ):
        """
        ナップサックで選んだ集合に対して、
        未選択オファーを1つだけ追加した場合の利益を self.ufun で比較する。

        利益が改善するなら、最も利益が高くなる追加オファーを1つだけ採用する。
        """

        selected_supplier = list(selected_supplier)
        selected_consumer = list(selected_consumer)

        # self.ufun は基本的に現在 step 用。
        # 未来納期の評価には使わない方が安全。
        if step != self.awi.current_step:
            return selected_supplier, selected_consumer

        base_offers = self.make_selected_offer_dict(
            buy_offer_dict,
            sell_offer_dict,
            selected_supplier,
            selected_consumer,
        )

        base_info = self.calc_ufun_info_from_partner_offers(base_offers)
        base_profit = base_info.utility

        best_profit = base_profit
        best_partner = None
        best_side = None
        best_info = base_info

        candidate_logs = []
        eps = 1e-6

        # 未選択の買いオファーを1つ追加して評価
        for partner, offer in buy_offer_dict.items():
            if partner in selected_supplier:
                continue

            candidate_offers = base_offers.copy()
            candidate_offers[partner] = offer

            info = self.calc_ufun_info_from_partner_offers(candidate_offers)
            profit = info.utility

            candidate_logs.append({
                "partner": partner,
                "side": "buy",
                "offer": offer,
                "profit": profit,
                "diff": profit - base_profit,
                "shortfall": getattr(info, "shortfall_quantity", None),
                "remaining": getattr(info, "remaining_quantity", None),
            })

            if profit > best_profit + eps:
                best_profit = profit
                best_partner = partner
                best_side = "buy"
                best_info = info

        # 未選択の売りオファーを1つ追加して評価
        for partner, offer in sell_offer_dict.items():
            if partner in selected_consumer:
                continue

            candidate_offers = base_offers.copy()
            candidate_offers[partner] = offer

            info = self.calc_ufun_info_from_partner_offers(candidate_offers)
            profit = info.utility

            candidate_logs.append({
                "partner": partner,
                "side": "sell",
                "offer": offer,
                "profit": profit,
                "diff": profit - base_profit,
                "shortfall": getattr(info, "shortfall_quantity", None),
                "remaining": getattr(info, "remaining_quantity", None),
            })

            if profit > best_profit + eps:
                best_profit = profit
                best_partner = partner
                best_side = "sell"
                best_info = info

        self.last_ufun_extra_check = {
            "step": step,
            "base_profit": base_profit,
            "base_shortfall": getattr(base_info, "shortfall_quantity", None),
            "base_remaining": getattr(base_info, "remaining_quantity", None),
            "best_partner": best_partner,
            "best_side": best_side,
            "best_profit": best_profit,
            "best_diff": best_profit - base_profit,
            "best_shortfall": getattr(best_info, "shortfall_quantity", None),
            "best_remaining": getattr(best_info, "remaining_quantity", None),
            "candidates": candidate_logs,
        }

        # 利益が改善するなら1つだけ追加採用
        if best_partner is not None:
            if best_side == "buy":
                selected_supplier.append(best_partner)
            elif best_side == "sell":
                selected_consumer.append(best_partner)

        return selected_supplier, selected_consumer

    def get_price_issue(self, partner):
        if partner in self.awi.my_suppliers:
            return self.awi.current_input_issues[UNIT_PRICE]
        else:
            return self.awi.current_output_issues[UNIT_PRICE]
    
    def split_offers_by_partner(self, offers):
        """
        Returns:
            buy_offers, sell_offers
        """
        buy_offers = {}
        sell_offers = {}

        for partner, offer in offers.items():
            if partner in self.awi.my_suppliers:
                buy_offers[partner] = offer

            elif partner in self.awi.my_consumers:
                sell_offers[partner] = offer
            
            else:
                continue

        return buy_offers, sell_offers
    
    def get_score(self, partner, step):
        """
        相手のスコアを計算する。

        仕入れ側は安い価格ほど高スコア、販売側は高い価格ほど高スコアにする。
        step が進むほど価格スコアの比重を高める。
        """
        mu_min = 0.4
        mu_max = 0.8
        k = 3.0

        denominator = max(
            1,
            self.awi.n_steps - 1 - self.awi.current_step,
        )

        rho = (step - self.awi.current_step) / denominator
        rho = max(0.0, min(1.0, rho))

        mu = mu_min + (mu_max - mu_min) * (
            (1 - math.exp(-k * rho)) / (1 - math.exp(-k))
        )

        price_issue = self.get_price_issue(partner)
        price = self.get_valid_price(partner, 0)

        price_range = max(
            1,
            price_issue.max_value - price_issue.min_value,
        )

        price_normalization = (
            price - price_issue.min_value
        ) / price_range

        if partner in self.awi.my_suppliers:
            # 自分が買う側なので、安いほど良い
            price_score = 1.0 - price_normalization
        else:
            # 自分が売る側なので、高いほど良い
            price_score = price_normalization

        success_score = self.success_rate[partner]

        return mu * price_score + (1 - mu) * success_score

    def init_partner_history(self, partners) -> None:
        """
        交渉相手ごとの成功・失敗回数と成功率を初期化する。
        partner_history[partner] = [成功回数, 失敗回数]
        """
        for partner in partners:
            _ = self.partner_history[partner]
            _ = self.success_rate[partner]

    def update_partner_success_rate(self, partner: str, succeeded: bool) -> None:
        """
        交渉結果に基づいて成功回数・失敗回数・成功率を更新する。
        """
        history = self.partner_history[partner]

        if succeeded:
            history[0] += 1
        else:
            history[1] += 1

        success_count, failure_count = history
        total_count = success_count + failure_count

        if total_count == 0:
            self.success_rate[partner] = 0.3
        else:
            self.success_rate[partner] = success_count / total_count
 

def solve_knapsack_for_scml_offers_by_score(
    offers: dict[str, tuple[int, int, int | float]],
    capacity: int,
    scores: dict[str, float],
) -> tuple[float, list[str]]:
    """
    スコアが高い相手を優先するナップサック。

    value = quantity * score

    量を埋めることと、スコアの高い相手を優先することを同時に見る。
    """
    if capacity <= 0 or not offers:
        return 0.0, []

    partners = list(offers.keys())
    n = len(partners)

    dp = [
        [0.0 for _ in range(capacity + 1)]
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        partner = partners[i - 1]
        offer = offers[partner]

        quantity = int(offer[QUANTITY])
        score = max(0.0, scores.get(partner, 0.0))
        value = quantity * score

        for q in range(capacity + 1):
            dp[i][q] = dp[i - 1][q]

            if quantity <= q:
                dp[i][q] = max(
                    dp[i][q],
                    dp[i - 1][q - quantity] + value,
                )

    selected_partners = []
    q = capacity

    for i in range(n, 0, -1):
        if dp[i][q] != dp[i - 1][q]:
            partner = partners[i - 1]
            selected_partners.append(partner)

            quantity = int(offers[partner][QUANTITY])
            q -= quantity

    selected_partners.reverse()

    return dp[n][capacity], selected_partners

def solve_knapsack_for_scml_offers(
    offers: dict[str, tuple[int, int, int | float]],
    capacity: int,
    price_mode: Literal["high", "low"] = "high",
    max_unit_price: int | float | None = None,
) -> tuple[int | float, list[str]]:

    if capacity <= 0 or not offers:
        return 0, []

    if price_mode not in ("high", "low"):
        raise ValueError('price_mode must be "high" or "low"')

    partners = list(offers.keys())
    n = len(partners)

    if price_mode == "low" and max_unit_price is None:
        max_unit_price = max(offer[UNIT_PRICE] for offer in offers.values())

    def calc_value(offer: tuple[int, int, int]) -> int:
        quantity = offer[QUANTITY]
        unit_price = offer[UNIT_PRICE]

        if price_mode == "high":
            unit_value = unit_price
        else:
            # 安いほど価値が高い。
            # +1 しないと、全員同価格のとき価値0になって誰も選ばれない。
            unit_value = max_unit_price - unit_price + 1
            unit_value = max(1, unit_value)

        return quantity * unit_value

    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        partner = partners[i - 1]
        offer = offers[partner]

        quantity = offer[QUANTITY]
        value = calc_value(offer)

        for q in range(capacity + 1):
            dp[i][q] = dp[i - 1][q]

            if quantity <= q:
                dp[i][q] = max(
                    dp[i][q],
                    dp[i - 1][q - quantity] + value,
                )

    selected_partners = []
    q = capacity

    for i in range(n, 0, -1):
        if dp[i][q] != dp[i - 1][q]:
            partner = partners[i - 1]
            selected_partners.append(partner)

            quantity = offers[partner][QUANTITY]
            q -= quantity

    selected_partners.reverse()

    return dp[n][capacity], selected_partners

def group_offers_by_delivery_time(
    offers: dict[str, Outcome],
) -> dict[int, dict[str, Outcome]]:
    """
    オファーを納期ごとにグループ化する。

    Args:
        offers:
            エージェント名をキー、オファーを値に持つ辞書。
            例: {"agentA": (quantity, time, unit_price)}

    Returns:
        納期をキー、その納期のオファー集合を値に持つ辞書。
        例:
        {
            3: {"agentA": (5, 3, 20)},
            4: {"agentB": (2, 4, 18), "agentC": (1, 4, 19)}
        }
    """
    offers_by_time: dict[int, dict[str, Outcome]] = defaultdict(dict)

    for partner, offer in sorted(offers.items(), key=lambda item: item[1][TIME]):
        delivery_time = offer[TIME]
        offers_by_time[delivery_time][partner] = offer

    return dict(offers_by_time)

def get_total_offer_quantity(offers):
    """
    オファー集合から、取引量の合計値を返す
    """
    total_quantity = 0
    for _, offer in offers.items():
        total_quantity += offer[QUANTITY]
    
    return total_quantity