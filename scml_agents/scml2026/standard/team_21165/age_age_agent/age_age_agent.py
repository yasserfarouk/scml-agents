from __future__ import annotations

from itertools import combinations
from collections import defaultdict
from typing import Literal
import math

from negmas import *
from scml.std import *

from dataclasses import dataclass, field
from typing import Any

from .base_age_age_agent import BaseAgeAgeAgent


class AgeAgeAgent(BaseAgeAgeAgent):
    def init(self):
        super().init()
        self.MIN_PROFIT = self.awi.profile.cost

        # 1step保管するのにかかる在庫コスト
        self.INVENTORY_COST_PER_UNIT_PER_STEP = 1

    def counter_all(self, offers, states):
        self.update_concession_stats(offers, states)

        response = defaultdict(
            lambda: SAOResponse(ResponseType.END_NEGOTIATION, None)
        )

        # 価格チェック
        valid_offers = {}

        for partner, offer in offers.items():
            if self.is_min_profit_price(partner, offer[UNIT_PRICE]):
                valid_offers[partner] = offer
            else:
                response[partner] = SAOResponse(
                    ResponseType.END_NEGOTIATION,
                    None,
                )

        buy_offers, sell_offers = self.split_offers_by_partner(valid_offers)

        # 納期ごとにオファーを分ける
        sorted_buy_offers = group_offers_by_delivery_time(buy_offers)
        sorted_sell_offers = group_offers_by_delivery_time(sell_offers)

        # 納期ごとに契約量を合わせる
        matched_buy, matched_sell = self.match_quantities_by_delivery_time(
            sorted_buy_offers,
            sorted_sell_offers,
        )

        # 選ばれた買いオファーを受諾する
        for partner, _ in matched_buy:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER,
                None,
            )

        # 選ばれた売りオファーを受諾する
        for partner, _ in matched_sell:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER,
                None,
            )

        # 余った契約を取り出す
        matched_buy_partners = {
            partner
            for partner, _ in matched_buy
        }

        matched_sell_partners = {
            partner
            for partner, _ in matched_sell
        }

        unmatched_buy = {
            partner: offer
            for partner, offer in buy_offers.items()
            if partner not in matched_buy_partners
        }

        unmatched_sell = {
            partner: offer
            for partner, offer in sell_offers.items()
            if partner not in matched_sell_partners
        }

        # 今は余った契約にはカウンターを出さず、デフォルトで END_NEGOTIATION にする
        # あとで first_proposals と同じ方法でカウンターを作るならここに追加する

        return response

    def is_exogenous_mode(self) -> bool:
        """
        外生契約を考慮すべき状態かどうかを判定する。

        条件:
        - 現在 step に外生入力がある
        - 過去に外生入力の平均取引量が記録されている
        - 現在 step に外生出力がある
        - 過去に外生出力の平均取引量が記録されている
        """

        input_q = self.awi.current_exogenous_input_quantity
        output_q = self.awi.current_exogenous_output_quantity

        return (
            input_q > 0
            or self.partner_weighted_avg_quantity["exogenous_input"] > 0
            or output_q > 0
            or self.partner_weighted_avg_quantity["exogenous_output"] > 0
        )

    def calculate_target_quantities(
        self,
        buy_offer_dict,
        sell_offer_dict,
        step,
    ):
        """
        AgeAgeAgent と同じ考え方で、外生契約時の必要取引量を計算する。

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

        # =========================
        # 納期が近い場合
        # =========================
        if is_near_delivery:
            # 外生入力がある場合:
            # 入力はすでに入ってくるので、追加で買うより売りを増やしたい
            if (
                input_q > 0
                or self.partner_weighted_avg_quantity["exogenous_input"] > 0
            ):
                target_sell_quantity = (
                    min(
                        inventory + input_q,
                        n_lines,
                    )
                    - contract_sales
                )

                return 0, max(0, int(target_sell_quantity))

            # 外生出力がある場合:
            # 出力契約を満たすために、入力を買いたい
            elif (
                output_q > 0
                or self.partner_weighted_avg_quantity["exogenous_output"] > 0
            ):
                target_buy_quantity = (
                    min(
                        max(
                            output_q,
                            int(self.awi.n_lines * 0.5),
                        ),
                        n_lines,
                    )
                    - inventory
                    - contract_supply
                )

                return max(0, int(target_buy_quantity)), 0

            # 通常時:
            # 仕入れ・販売・在庫・生産能力のバランスを見る
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

            return int(target_buy_quantity), int(target_sell_quantity)

        # =========================
        # 納期が遠い場合
        # =========================
        else:
            # 外生入力の傾向がある場合:
            # 将来も入力が入る想定なので、売り側を半分程度確保する
            if (
                input_q > 0
                or self.partner_weighted_avg_quantity["exogenous_input"] > 0
            ):
                target_sell_quantity = (
                    int(self.awi.n_lines * 0.5)
                    - contract_sales
                )

                return 0, max(0, int(target_sell_quantity))

            # 外生出力の傾向がある場合:
            # 将来の出力に備えて、買い側を半分程度確保する
            elif (
                output_q > 0
                or self.partner_weighted_avg_quantity["exogenous_output"] > 0
            ):
                if inventory > self.awi.n_lines:
                    target_buy_quantity = 0
                else:
                    target_buy_quantity = (
                        int(self.awi.n_lines * 0.5)
                        - contract_supply
                    )

                return max(0, int(target_buy_quantity)), 0

            # 通常時:
            # 既存売契約を満たすための買いと、余った生産能力分の売りを見る
            target_buy_quantity = max(
                0,
                contract_sales - contract_supply - inventory,
            )

            target_sell_quantity = max(
                0,
                self.awi.n_lines - contract_sales,
            )

            return int(target_buy_quantity), int(target_sell_quantity)

    def match_quantities_by_delivery_time(self, buy_offers, sell_offers):
        """
        納期ごとに仕入れと販売の契約量を合わせる。

        通常時:
            V2 と同じように、在庫不足を避けながら買い・売りを合わせる。

        外生契約時:
            必要取引量は AgeAgeAgent と同じ式で計算する。
            ただし、受諾するオファーの選び方は V2 と同じく、
            部分集合を作って match_quantities で選ぶ。
        """

        matched_buy: list[tuple[str, Any]] = []
        matched_sell: list[tuple[str, Any]] = []

        # 同じオファーを複数回選ばないために記録する
        used_buy_partners = set()
        used_sell_partners = set()

        # 外生契約を考慮すべき状態かどうか
        exogenous_mode = self.is_exogenous_mode()

        # 買いオファーは1step後の売りにも使えるので、step + 1 も処理対象に入れる
        all_times = sorted(
            set(buy_offers)
            | set(sell_offers)
            | {
                step + 1
                for step in buy_offers
                if step + 1 < self.awi.n_steps
            }
        )

        # 納期ごとに処理
        for step in all_times:
            # 現在stepの売りに対して、1step前の買いも候補に入れる
            prev_buys = buy_offers.get(step - 1, {})
            current_buys = buy_offers.get(step, {})

            buys = {}

            # 1step前の買いオファーを追加
            for partner, offer in prev_buys.items():
                if partner not in used_buy_partners:
                    buys[partner] = offer

            # 現在stepの買いオファーを追加
            for partner, offer in current_buys.items():
                if partner not in used_buy_partners:
                    buys[partner] = offer

            # 売り側は現在stepのオファーだけを見る
            sells = {
                partner: offer
                for partner, offer in sell_offers.get(step, {}).items()
                if partner not in used_sell_partners
            }

            # 現在の在庫量を取得する
            inventory = (
                self.awi.total_supplies_at(step) + self.awi.current_inventory_input
                if step == self.awi.current_step
                else self.awi.total_supplies_at(step)
            )

            # オファーの部分集合を作成
            buy_subsets = get_subsets(buys, "buy")
            sell_subsets = get_subsets(sells, "sell")

            # =========================
            # 外生契約時の必要量設定
            # =========================
            if exogenous_mode:
                target_buy_quantity, target_sell_quantity = (
                    self.calculate_target_quantities(
                        buys,
                        sells,
                        step,
                    )
                )

                # 外生契約時は、AgeAgeAgent と同じ必要量を目標にする
                buy_max_quantity = target_buy_quantity
                sell_max_quantity = target_sell_quantity

                # match_quantities は在庫差で評価するので、
                # 目標買い量・目標売り量から、目標在庫量を逆算する
                inventory_required = (
                    inventory
                    + target_buy_quantity
                    - target_sell_quantity
                )

            # =========================
            # 通常時の必要量設定
            # =========================
            else:
                # 通常時は、既存売契約を満たせる在庫量を目標にする
                inventory_required = self.awi.total_sales_at(step)

                # 買いは将来需要を見越して多めに受けてもよい
                buy_max_quantity = math.inf

                # 売りは1日の生産能力を超えないようにする
                sell_max_quantity = max(
                    0,
                    self.awi.n_lines - self.awi.total_sales_at(step),
                )

            # 1step前の買い + 現在stepの買い と、現在stepの売りを合わせる
            best_buy_offers, best_sell_offers, diff = match_quantities(
                inventory,
                inventory_required,
                buy_subsets,
                sell_subsets,
                buy_max_quantity,
                sell_max_quantity,
                target_step=step,
                min_profit_per_unit=self.MIN_PROFIT,
                inventory_cost_per_unit_per_step=(
                    self.INVENTORY_COST_PER_UNIT_PER_STEP
                ),
                estimated_buy_unit_price=self.avg_buy_price,
                estimated_sell_unit_price=self.avg_sell_price,
            )

            matched_buy.extend(best_buy_offers.items())
            matched_sell.extend(best_sell_offers.items())

            # 選ばれたオファーは次のstep以降では使わない
            used_buy_partners.update(best_buy_offers.keys())
            used_sell_partners.update(best_sell_offers.keys())

        return matched_buy, matched_sell


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

    for partner, offer in sorted(
        offers.items(),
        key=lambda item: item[1][TIME],
    ):
        delivery_time = offer[TIME]
        offers_by_time[delivery_time][partner] = offer

    return dict(offers_by_time)


def get_subsets(
    offers: dict[str, Any],
    side: Literal["sell", "buy"] = "sell",
    include_empty: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    オファー集合から、作れる取引量ごとの最良の部分集合を返す。

    同じ取引量を作れる部分集合が複数ある場合、
    - sell の場合: 売上が高いものを採用
    - buy の場合: 支払い額が低いものを採用
    """

    items = list(offers.items())

    best_subsets: dict[int, dict[str, Any]] = {}
    best_scores: dict[int, float] = {}

    start = 0 if include_empty else 1

    for r in range(start, len(items) + 1):
        for subset_items in combinations(items, r):
            subset = dict(subset_items)

            total_quantity = sum(
                offer[QUANTITY]
                for offer in subset.values()
            )

            total_money = sum(
                offer[QUANTITY] * offer[UNIT_PRICE]
                for offer in subset.values()
            )

            # sellなら売上が高いほど良い
            # buyなら支払いが低いほど良いので、マイナスにする
            if side == "sell":
                score = total_money
            else:
                score = -total_money

            if (
                total_quantity not in best_scores
                or score > best_scores[total_quantity]
            ):
                best_scores[total_quantity] = score
                best_subsets[total_quantity] = subset

    return best_subsets


def match_quantities(
    inventory: float,
    inventory_required: float,
    buy_subsets: dict[int, dict[str, Any]],
    sell_subsets: dict[int, dict[str, Any]],
    buy_max_quantity: float,
    sell_max_quantity: float,
    target_step: int,
    min_profit_per_unit: float,
    inventory_cost_per_unit_per_step: float = 1.0,
    estimated_buy_unit_price: float = 0.0,
    estimated_sell_unit_price: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """
    在庫差が最小になる買い・売り部分集合を選ぶ。

    通常の買い・売りが両方ある場合:
        売上 - 仕入れ - 在庫コスト で利益を見る。

    外生入力で売りだけを受ける場合:
        足りない買いコストを avg_buy_price で見積もる。

    外生出力で買いだけを受ける場合:
        将来売れる価値を avg_sell_price で見積もる。
    """

    best_buy: dict[str, Any] = {}
    best_sell: dict[str, Any] = {}
    best_abs_diff: float = math.inf
    best_profit: float = -math.inf
    best_diff_signed: float = 0.0

    if 0 not in buy_subsets:
        buy_subsets = dict(buy_subsets)
        buy_subsets[0] = {}

    if 0 not in sell_subsets:
        sell_subsets = dict(sell_subsets)
        sell_subsets[0] = {}

    for b_qty, b_subset in buy_subsets.items():
        if b_qty > buy_max_quantity:
            continue

        for s_qty, s_subset in sell_subsets.items():
            if s_qty > sell_max_quantity:
                continue

            net_inventory = inventory + b_qty - s_qty
            diff = net_inventory - inventory_required

            # 在庫不足になる組み合わせは選ばない
            if diff < 0:
                continue

            total_buy_cost = sum(
                offer[QUANTITY] * offer[UNIT_PRICE]
                for offer in b_subset.values()
            )

            total_sell_revenue = sum(
                offer[QUANTITY] * offer[UNIT_PRICE]
                for offer in s_subset.values()
            )

            # 前日以前の買いを使う場合の在庫コスト
            inventory_cost = sum(
                offer[QUANTITY]
                * max(0, target_step - offer[TIME])
                * inventory_cost_per_unit_per_step
                for offer in b_subset.values()
            )

            # 買いの方が多い場合は、余った買いを将来売れるものとして見積もる
            uncovered_buy_quantity = max(0, b_qty - s_qty)
            estimated_future_sell_revenue = (
                uncovered_buy_quantity
                * estimated_sell_unit_price
            )

            # 売りの方が多い場合は、足りない買いコストを平均買値で見積もる
            uncovered_sell_quantity = max(0, s_qty - b_qty)
            estimated_missing_buy_cost = (
                uncovered_sell_quantity
                * estimated_buy_unit_price
            )

            profit = (
                total_sell_revenue
                + estimated_future_sell_revenue
                - total_buy_cost
                - estimated_missing_buy_cost
                - inventory_cost
            )

            # 買いだけ・売りだけの外生契約も見るため、max(b_qty, s_qty) を使う
            required_profit = min_profit_per_unit * max(b_qty, s_qty)

            # 生産コスト込みで利益が出ない組み合わせは選ばない
            if profit < required_profit:
                continue

            abs_diff = abs(diff)

            if abs_diff < best_abs_diff or (
                abs_diff == best_abs_diff and profit > best_profit
            ):
                best_abs_diff = abs_diff
                best_profit = profit
                best_buy = b_subset
                best_sell = s_subset
                best_diff_signed = diff

    return best_buy, best_sell, best_diff_signed


def get_total_offer_quantity(offers):
    """
    オファー集合から、取引量の合計値を返す。
    """
    total_quantity = 0

    for _, offer in offers.items():
        total_quantity += offer[QUANTITY]

    return total_quantity