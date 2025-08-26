#!/usr/bin/env python
# type: ignore
"""
LitaAgentP: An agent based on PenguinAgent's strategy for SCML 2024,
updated for scml==0.11.3 and Python 3.12.
LitaAgentP: 一个基于 PenguinAgent 策略的代理，为 SCML 2024 设计，
已更新以兼容 scml==0.11.3 和 Python 3.12。
"""

from __future__ import annotations

import random
from itertools import chain, combinations, repeat
from collections import Counter
from numpy.random import choice  # type: ignore
import json  # 新增: 用于加载配置文件
import os  # 新增: 用于文件路径操作

# SCML and NegMAS imports for scml==0.11.3
# SCML 和 NegMAS 导入 (针对 scml==0.11.3)
from scml.std import QUANTITY, TIME, UNIT_PRICE  # Standard issue names / 标准问题名称
from scml.std import StdSyncAgent
from negmas import SAOResponse, ResponseType
from negmas import Outcome  # Offers are Outcomes (dictionaries) / 报价是 Outcome 类型 (字典)

# For typing (though type: ignore is used, good practice for future)
# 类型提示 (尽管使用了 type: ignore, 但对未来是好习惯)
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set
from negmas.common import MechanismState

__all__ = ["LitaAgentP"]


# Utility functions (copied and adapted from PenguinAgent)
# 实用函数 (从 PenguinAgent 复制并适配)
def distribute(q: int, n: int) -> list[int]:
    """
    Distributes q items into n bins.
    If q < n, some bins get 0, others get 1.
    If q = n, all bins get 1.
    If q > n, all bins get at least 1, and the remainder (q-n) is distributed.

    将 q 个物品分配到 n 个箱子中。
    如果 q < n，一些箱子得到 0，另一些得到 1。
    如果 q = n，所有箱子都得到 1。
    如果 q > n，所有箱子至少得到 1，剩余的 (q-n) 个物品会被分配。
    """
    if n <= 0:  # Guard for no bins / 防止没有箱子的情况
        return []
    if q < 0:  # Guard for negative quantity / 防止数量为负
        q = 0

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    # q > n: Each of n bins gets 1, distribute remaining q-n items
    # q > n: 每个箱子先得到 1 个，然后分配剩余的 q-n 个物品
    r = Counter(choice(range(n), q - n))  # choice needs a range for the first arg / choice 的第一个参数需要一个范围
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    返回可迭代对象的所有子集 (幂集)。
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class LitaAgentP(StdSyncAgent):
    """
    An agent that distributes today's needs randomly over a percentage of its partners
    and samples future offers randomly. Based on PenguinAgent strategy.
    Updated for scml==0.11.3.

    一个将其当日需求随机分配给一定比例伙伴，并随机抽样未来报价的代理。
    基于 PenguinAgent 策略，已更新以兼容 scml==0.11.3。
    """

    def __init__(self, *args, threshold_factor=0.1, ptoday=0.70, productivity=0.7, **kwargs):
        """
        Args:
            threshold_factor (float): Factor to determine the threshold for accepting combined offers
                                      that slightly exceed current needs.
                                      用于确定接受略微超过当前需求的组合报价的阈值的因子。
            ptoday (float): Percentage of partners to negotiate with for today's needs.
                            为满足当日需求而与之谈判的伙伴的百分比。
            productivity (float): Agent's production efficiency (output per line or input per final product).
                                  代理的生产效率 (每条生产线的产出或每个最终产品的投入)。
        """
        super().__init__(*args, **kwargs)

        # --- 从配置文件加载 ptoday ---
        # 假设配置文件 agent_config.json 在项目根目录
        # LitaAgentP.py 在 litaagent_std/ 子目录中
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'agent_config.json')

        loaded_ptoday = ptoday  # 默认为参数传入的值

        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    config_data = json.load(f)
                if 'ptoday' in config_data:
                    loaded_ptoday = float(config_data['ptoday'])
                    if self.id:  # self.id 可能在 super().__init__ 后才完全可用
                        print(
                            f"LitaAgentP ({self.id if hasattr(self, 'id') else 'N/A'}): Loaded ptoday={loaded_ptoday} from {config_file_path}")
                else:
                    if self.id:
                        print(
                            f"LitaAgentP ({self.id if hasattr(self, 'id') else 'N/A'}): 'ptoday' not found in {config_file_path}. Using default/argument: {loaded_ptoday}")
            except (json.JSONDecodeError, FileNotFoundError, TypeError, ValueError) as e:
                if self.id:
                    print(
                        f"LitaAgentP ({self.id if hasattr(self, 'id') else 'N/A'}): Error loading config {config_file_path}: {e}. Using default/argument: {loaded_ptoday}")
        else:
            if self.id:
                print(
                    f"LitaAgentP ({self.id if hasattr(self, 'id') else 'N/A'}): Config file {config_file_path} not found. Using default/argument: {loaded_ptoday}")
        # --- 加载结束 ---

        self._threshold_factor = threshold_factor
        self._threshold = 1  # Initial placeholder, will be updated in step() by awi / 初始占位符，将在 step() 中由 awi 更新
        self._ptoday = loaded_ptoday  # 使用加载的或默认的 ptoday / Percentage of partners to negotiate with for today's needs / 为满足当日需求而与之谈判的伙伴的百分比
        self._productivity = productivity  # Agent's production efficiency / 代理的生产效率

    def step(self):
        """
        Called at each simulation step. Updates the acceptance threshold.
        在每个模拟步骤调用。更新接受阈值。
        """
        super().step()
        if self.awi:  # awi (Agent World Interface) provides simulation state / awi 提供模拟状态
            self._threshold = self.awi.n_lines * self._threshold_factor

    def _create_offer(self, quantity: int, time: int, unit_price: int | float) -> Outcome | None:
        """
        Helper to create a valid offer dictionary (Outcome).
        SCML contracts usually require positive quantities.

        创建有效报价字典 (Outcome) 的辅助函数。
        SCML 合约通常要求正数量。
        """
        if quantity <= 0:
            return None
        return {QUANTITY: quantity, TIME: time, UNIT_PRICE: unit_price}

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Generates initial proposals to negotiators.
        Distributes today's needs/capacity and proposes future offers.

        生成给谈判者的初始报价。
        分配当日的需求/能力，并提出未来报价。
        """
        proposals: dict[str, Outcome | None] = {}
        if not self.awi:
            return proposals

        current_step = self.awi.current_step
        # Determine quantities needed/wanted for today with various partners
        # 确定今天与各个伙伴需要的/想要的数量
        todays_distribution = self.distribute_todays_needs()

        future_supply_partners: list[str] = []  # Partners I might buy from in the future / 我未来可能从其购买的伙伴
        future_consume_partners: list[str] = []  # Partners I might sell to in the future / 我未来可能向其销售的伙伴

        for partner_id, quantity_needed in todays_distribution.items():
            if quantity_needed > 0:  # If a quantity is assigned for today / 如果今天分配了数量
                price = self.best_price(partner_id)  # Get my ideal price / 获取我的理想价格
                if price is not None:
                    proposals[partner_id] = self._create_offer(quantity_needed, current_step, price)
            # If no quantity for today, consider for future offers
            # 如果今天没有数量，则考虑未来报价
            elif self.is_supplier(partner_id):  # Partner is a supplier to me / 伙伴是我的供应商
                future_supply_partners.append(partner_id)
            elif self.is_consumer(partner_id):  # Partner is a consumer of my goods / 伙伴是我的客户
                future_consume_partners.append(partner_id)

        # Add future proposals
        # 添加未来报价
        proposals.update(self.future_supply_offer(future_supply_partners))
        proposals.update(self.future_consume_offer(future_consume_partners))

        return proposals

    def counter_all(self, offers: dict[str, Outcome | None], states: dict[str, MechanismState]) -> dict[
        str, SAOResponse]:
        """
        Responds to all incoming offers. This is the core negotiation logic.
        It processes buying and selling sides, prioritizes future offers,
        then handles current step offers using a combinatorial approach.

        响应所有收到的报价。这是核心谈判逻辑。
        它处理买卖双方，优先处理未来报价，然后使用组合方法处理当前步骤的报价。
        """
        responses: dict[str, SAOResponse] = {}
        if not self.awi:
            return responses

        # Process buying (supplies) and selling (sales) sides independently
        # 独立处理购买 (供应) 和销售 (销售) 两方面
        for is_selling_mode in [False, True]:  # False for buying, True for selling / False 代表购买, True 代表销售

            all_partners_on_this_side = self.awi.my_suppliers if not is_selling_mode else self.awi.my_consumers
            if not all_partners_on_this_side:
                continue

            daily_production = self.awi.n_lines * self._productivity
            needs_today = 0  # How much to buy/sell today / 今天需要购买/销售多少

            if not is_selling_mode:  # Buying supplies / 购买供应品
                needs_today = int(
                    daily_production
                    - self.awi.current_inventory_input
                    - self.awi.total_supplies_at(self.awi.current_step)
                )
            else:  # Selling products / 销售产品
                # Original Penguin condition: if total_sales_at(current_step) <= n_lines
                # This logic aims to calculate remaining sales capacity for today.
                # 原始 Penguin 条件: 如果当日总销售额 <= 生产线数量
                # 这个逻辑旨在计算当日剩余的销售能力。
                if self.awi.total_sales_at(self.awi.current_step) <= self.awi.n_lines:
                    needs_today = int(
                        max(0, min(self.awi.n_lines, daily_production + self.awi.current_inventory_input)
                            - self.awi.total_sales_at(self.awi.current_step)
                            )
                    )
            needs_today = max(0, needs_today)

            active_partners_this_side = {
                p for p in all_partners_on_this_side if p in offers.keys() and p not in responses.keys()
            }

            current_step_offers_this_side: dict[str, Outcome] = {}  # Offers for today / 今日报价
            future_step_offers_this_side: dict[str, Outcome] = {}  # Offers for future days / 未来日期报价

            for p_id in active_partners_this_side:
                offer = offers[p_id]
                if offer is None:  # No offer from this partner, or negotiation ended by them / 该伙伴无报价，或对方结束谈判
                    responses[p_id] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    continue

                # Validate offer structure
                # 验证报价结构
                if not (isinstance(offer, dict) and
                        all(k in offer for k in (TIME, UNIT_PRICE, QUANTITY))):
                    responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, None)  # Invalid offer structure / 无效报价结构
                    continue

                # Check if the price is "valid" by Penguin's permissive logic
                # 根据 Penguin 的宽松逻辑检查价格是否“有效”
                if self.is_valid_price(offer[UNIT_PRICE], p_id):
                    if offer[TIME] == self.awi.current_step:
                        current_step_offers_this_side[p_id] = offer
                    elif offer[TIME] > self.awi.current_step and offer[
                        TIME] < self.awi.n_steps:  # Valid future offer / 有效的未来报价
                        future_step_offers_this_side[p_id] = offer
                    else:  # Offer for past or too far future / 过去或太遥远的未来报价
                        responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, None)
                else:  # Price not "valid" by Penguin's definition / 价格根据 Penguin 定义“无效”
                    my_price = self.price(p_id)  # My concessional price / 我的让步价格
                    counter_offer = self._create_offer(offer[QUANTITY], offer[TIME],
                                                       my_price) if my_price is not None else None
                    responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)

            # Handle future offers first, accounting for cumulative acceptances within this round
            # 首先处理未来报价，并考虑本轮中已累积接受的数量
            # (step, is_for_my_supply_chain) -> quantity. is_for_my_supply_chain is True if I'm buying.
            # (步骤, 是否为我的供应链) -> 数量。如果我正在购买，则 is_for_my_supply_chain 为 True。
            accepted_qty_future_step_this_round: Counter[Tuple[int, bool]] = Counter()

            for p_id, offer_data in future_step_offers_this_side.items():
                if p_id in responses:  # Already responded to this partner (e.g., due to invalid price earlier) / 已响应过此伙伴 (例如，由于之前的无效价格)
                    continue

                step, quantity_offered = offer_data[TIME], offer_data[QUANTITY]

                is_future_offer_for_my_purchase = self.is_supplier(
                    p_id)  # True if I'm buying from p_id / 如果我从 p_id 购买则为 True

                # Total need for that future step (considering contracts signed *before* this round)
                # 该未来步骤的总需求 (考虑在本轮 *之前* 签订的合同)
                current_total_need_at_future = self.needs_at(step, p_id)

                # Subtract what's already accepted *in this round* for that specific future step & type
                # 减去在本轮中已为该特定未来步骤和类型接受的数量
                already_accepted_this_round = accepted_qty_future_step_this_round[
                    (step, is_future_offer_for_my_purchase)]
                remaining_need_for_this_offer = current_total_need_at_future - already_accepted_this_round

                if quantity_offered > 0 and quantity_offered <= remaining_need_for_this_offer:
                    responses[p_id] = SAOResponse(ResponseType.ACCEPT_OFFER, offer_data)
                    accepted_qty_future_step_this_round[(step, is_future_offer_for_my_purchase)] += quantity_offered
                else:
                    my_concessional_price = self.price(p_id)
                    counter = None
                    if my_concessional_price is not None:
                        proposable_q = max(0, remaining_need_for_this_offer)  # How much I can still take / 我还能接受多少
                        if proposable_q > 0:  # If I still need some, counter for that amount / 如果我还需要一些，则以该数量还价
                            counter = self._create_offer(proposable_q, step, my_concessional_price)
                        elif quantity_offered > 0:  # I have no need, but they offered. Counter their Q at my P. / 我没有需求，但他们报价了。以他们的数量和我的价格还价。
                            counter = self._create_offer(quantity_offered, step, my_concessional_price)
                    responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, counter)

            # Handle current step offers for this side using combinatorics
            # 使用组合方法处理此方面的当前步骤报价
            partners_for_current_combinatorics = {
                p_id for p_id, offer in current_step_offers_this_side.items() if p_id not in responses
            }

            chosen_combination_idx = -1  # Index in all_combinations list / 在 all_combinations 列表中的索引
            accept_partner_ids_current_step: Tuple[str, ...] = tuple()

            if needs_today > 0 and partners_for_current_combinatorics:
                partner_list_powerset = list(partners_for_current_combinatorics)
                all_combinations = list(powerset(partner_list_powerset))  # All subsets of partners / 所有伙伴子集

                best_plus_diff, best_idx_plus = float(
                    "inf"), -1  # Best combination if sum_quantity >= needs_today / 如果总数量 >= 今日需求 的最佳组合
                best_minus_diff, best_idx_minus = float(
                    "inf"), -1  # Best combination if sum_quantity < needs_today / 如果总数量 < 今日需求 的最佳组合

                for i, partner_id_tuple in enumerate(all_combinations):
                    if not partner_id_tuple: continue  # Skip empty set / 跳过空集

                    offered_sum = sum(current_step_offers_this_side[p_id][QUANTITY] for p_id in partner_id_tuple)
                    diff = abs(offered_sum - needs_today)

                    if offered_sum >= needs_today:
                        if diff < best_plus_diff: best_plus_diff, best_idx_plus = diff, i
                    else:  # offered_sum < needs_today
                        if diff < best_minus_diff: best_minus_diff, best_idx_minus = diff, i

                # Prioritize combinations that meet or exceed needs_today, if within threshold
                # 优先选择满足或超过今日需求且在阈值内的组合
                if best_idx_plus != -1 and best_plus_diff <= self._threshold:
                    chosen_combination_idx = best_idx_plus
                elif best_idx_minus != -1:  # Otherwise, take the best one that is less than needs_today / 否则，选择小于今日需求的最佳组合
                    chosen_combination_idx = best_idx_minus

                if chosen_combination_idx != -1:
                    accept_partner_ids_current_step = all_combinations[chosen_combination_idx]
                    for p_id in partner_list_powerset:
                        if p_id in responses: continue  # Already handled (e.g. future offer accepted/rejected) / 已处理 (例如，未来报价已接受/拒绝)
                        if p_id in accept_partner_ids_current_step:
                            responses[p_id] = SAOResponse(ResponseType.ACCEPT_OFFER,
                                                          current_step_offers_this_side[p_id])
                        else:  # Rejected from current step combination, propose future to them / 从当前步骤组合中拒绝，向他们提议未来报价
                            future_prop = None
                            if self.is_supplier(p_id):  # I buy from p_id / 我从 p_id 购买
                                temp_offers = self.future_supply_offer([p_id])
                                if p_id in temp_offers: future_prop = temp_offers[p_id]
                            elif self.is_consumer(p_id):  # I sell to p_id / 我向 p_id 销售
                                temp_offers = self.future_consume_offer([p_id])
                                if p_id in temp_offers: future_prop = temp_offers[p_id]
                            responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, future_prop)

            # Fallback: If no current-step combination was accepted, and still have needs_today.
            # Penguin's "if flag != 1" logic.
            # 后备方案：如果没有接受当前步骤的组合，并且仍然有今日需求。
            # Penguin 的 "if flag != 1" 逻辑。
            current_step_combo_was_accepted = (
                        chosen_combination_idx != -1 and needs_today > 0 and accept_partner_ids_current_step)

            if not current_step_combo_was_accepted and needs_today > 0:
                partners_for_fallback = {
                    p_id for p_id in all_partners_on_this_side
                    if p_id in self.negotiators and p_id in offers and p_id not in responses
                }
                if partners_for_fallback:
                    dist_fallback = self.distribute_todays_needs(list(partners_for_fallback))
                    fb_future_supply, fb_future_consume = [], []

                    for p_id, q_assigned in dist_fallback.items():
                        if p_id not in partners_for_fallback or p_id in responses: continue
                        if q_assigned > 0:
                            price = self.price(p_id)  # Use concessional price / 使用让步价格
                            counter = self._create_offer(q_assigned, self.awi.current_step, price) if price else None
                            responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                        else:  # If not assigned a quantity for today, consider for future / 如果今天未分配数量，则考虑未来
                            if self.is_supplier(p_id):
                                fb_future_supply.append(p_id)
                            elif self.is_consumer(p_id):
                                fb_future_consume.append(p_id)

                    # Propose future offers to those not covered by fallback's today distribution
                    # 向后备方案中当日分配未覆盖的伙伴提议未来报价
                    for p_id, offer_d in self.future_supply_offer(fb_future_supply).items():
                        if p_id in partners_for_fallback and p_id not in responses:
                            responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, offer_d)
                    for p_id, offer_d in self.future_consume_offer(fb_future_consume).items():
                        if p_id in partners_for_fallback and p_id not in responses:
                            responses[p_id] = SAOResponse(ResponseType.REJECT_OFFER, offer_d)

        # Final pass: any partner in `offers` not yet in `responses` gets END_NEGOTIATION
        # (e.g., if they were filtered out early or no logic path covered them)
        # 最后一遍：`offers` 中任何尚未在 `responses` 中的伙伴都将收到 END_NEGOTIATION
        # (例如，如果他们很早就被过滤掉或没有逻辑路径覆盖他们)
        for p_id in offers.keys():
            if p_id not in responses:
                responses[p_id] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return responses

    def is_valid_price(self, price: float | int, partner_id: str) -> bool:
        """
        Checks if a price is 'valid' according to Penguin's original (permissive) logic.
        If I am buying, any price <= max_price_of_issue is "valid".
        If I am selling, any price >= min_price_of_issue is "valid".

        根据 Penguin 原始的 (宽松) 逻辑检查价格是否“有效”。
        如果我购买，任何价格 <= 问题价格区间的最大值 则“有效”。
        如果我销售，任何价格 >= 问题价格区间的最小值 则“有效”。
        """
        if not self.awi: return False
        nmi = self.get_nmi(partner_id)  # NegotiatorMechanismInterface
        if not nmi or not nmi.issues or UNIT_PRICE not in nmi.issues:
            return False

        price_issue = nmi.issues[UNIT_PRICE]
        # Assuming price_issue has min_value and max_value (standard for numeric issues)
        # 假设 price_issue 具有 min_value 和 max_value (数字问题的标准配置)
        min_p = getattr(price_issue, 'min_value', float('-inf'))
        max_p = getattr(price_issue, 'max_value', float('inf'))

        if self.is_consumer(
                partner_id):  # Partner is my consumer (I sell to them). `price` is what they offer to pay me.
            # 伙伴是我的客户 (我卖给他们)。`price` 是他们愿意支付给我的价格。
            return price >= min_p  # Penguin: accept if their buy price is not less than absolute min. / Penguin: 如果他们的购买价格不低于绝对最小值，则接受。
        elif self.is_supplier(
                partner_id):  # Partner is my supplier (I buy from them). `price` is what they offer to sell at.
            # 伙伴是我的供应商 (我从他们那里购买)。`price` 是他们出售的价格。
            return price <= max_p  # Penguin: accept if their sell price is not more than absolute max. / Penguin: 如果他们的销售价格不高于绝对最大值，则接受。
        return False

    def needs_at(self, step: int, partner_id: str) -> int:
        """
        Calculates my remaining need (to buy or sell) for a given future step with a partner type.
        This is used to evaluate future offers.

        计算在给定的未来步骤中，我与某种类型的伙伴之间剩余的需求 (购买或销售)。
        这用于评估未来报价。
        """
        if not self.awi or step < 0 or step >= self.awi.n_steps: return 0

        needed_quantity = 0
        daily_production = self.awi.n_lines * self._productivity

        if self.is_supplier(partner_id):  # Partner is my supplier, I need to buy from them (input materials)
            # 伙伴是我的供应商，我需要从他们那里购买 (输入材料)
            needed_quantity = int(
                daily_production
                - self.awi.current_inventory_input  # What I have now (Penguin used this for future too) / 我现在拥有的 (Penguin 也将其用于未来)
                - self.awi.total_supplies_at(step)  # Already contracted supplies for that future step / 该未来步骤已签约的供应量
            )
        elif self.is_consumer(partner_id):  # Partner is my consumer, I need to sell to them (output products)
            # 伙伴是我的客户，我需要卖给他们 (输出产品)
            available_to_sell = daily_production + self.awi.current_inventory_input  # Penguin's logic for available amount / Penguin 计算可用量的逻辑
            potential_sales_capacity = min(self.awi.n_lines, available_to_sell)  # Capped by number of lines / 受生产线数量限制
            needed_quantity = int(
                potential_sales_capacity
                - self.awi.total_sales_at(step)  # Already contracted sales for that step / 该步骤已签约的销售量
            )
        return max(0, needed_quantity)

    def is_consumer(self, partner_id: str) -> bool:
        """
        True if partner_id is a consumer of my products (I am a supplier to them).
        如果 partner_id 是我产品的消费者 (我是他们的供应商)，则为 True。
        """
        if not self.awi: return False
        return partner_id in self.awi.my_consumers

    def is_supplier(self, partner_id: str) -> bool:
        """
        True if partner_id is a supplier of products to me (I am a consumer from them).
        如果 partner_id 是我的产品供应商 (我是他们的消费者)，则为 True。
        """
        if not self.awi: return False
        return partner_id in self.awi.my_suppliers

    def distribute_todays_needs(self, partners: Optional[list[str]] = None) -> dict[str, int]:
        """
        Distributes my urgent (today's) needs/capacity over specified or all partners.
        Uses `_distribute_among_subset` to pick a random subset based on `_ptoday`.

        将我紧急的 (今日的) 需求/能力分配给指定的或所有伙伴。
        使用 `_distribute_among_subset` 根据 `_ptoday` 选择一个随机子集。
        """
        if not self.awi: return {}

        target_partners = list(partners) if partners is not None else list(self.negotiators.keys())
        if not target_partners: return {}

        response: dict[str, int] = dict(zip(target_partners, repeat(0)))  # Initialize all to 0 / 全部初始化为 0

        supply_partners_dist: list[str] = [p for p in target_partners if
                                           self.is_supplier(p)]  # I buy from these / 我从这些伙伴购买
        consume_partners_dist: list[str] = [p for p in target_partners if
                                            self.is_consumer(p)]  # I sell to these / 我向这些伙伴销售

        current_step = self.awi.current_step
        daily_production = self.awi.n_lines * self._productivity

        # Calculate net supplies needed today
        # 计算今天所需的净供应量
        supplies_needed_today = max(0, int(
            daily_production - self.awi.current_inventory_input - self.awi.total_supplies_at(current_step)
        ))
        # Calculate net sales capacity for today
        # 计算今天的净销售能力
        sales_capacity_today = max(0, int(
            min(self.awi.n_lines, daily_production + self.awi.current_inventory_input) - self.awi.total_sales_at(
                current_step)
        ))

        if supply_partners_dist and supplies_needed_today > 0:
            dist = self._distribute_among_subset(supply_partners_dist, supplies_needed_today)
            response.update(dist)

        if consume_partners_dist and sales_capacity_today > 0:
            # Penguin's original condition: and awi.total_sales_at(awi.current_step) <= self.awi.n_lines
            # This is implicitly handled if sales_capacity_today is calculated considering n_lines cap.
            # Penguin 的原始条件: 并且当日总销售额 <= 生产线数量
            # 如果 sales_capacity_today 的计算考虑了生产线数量上限，则此条件已隐式处理。
            dist = self._distribute_among_subset(consume_partners_dist, sales_capacity_today)
            response.update(dist)
        return response

    def _distribute_among_subset(self, Wpartners: list[str], total_quantity: int) -> dict[str, int]:
        """
        Helper to distribute a total quantity among a random subset of Wpartners.
        The size of the subset is determined by `_ptoday`.
        If `total_quantity` is less than selected partners, further reduces partners.

        辅助函数，用于将总数量分配给 Wpartners 的一个随机子集。
        子集的大小由 `_ptoday` 决定。
        如果 `total_quantity` 小于选定的伙伴数量，则进一步减少伙伴数量。
        """
        if not Wpartners or total_quantity <= 0: return {}

        shuffled_partners = random.sample(Wpartners, len(Wpartners))  # Shuffle for randomness / 打乱以实现随机性
        # 使用 self._ptoday 来决定选择多少伙伴
        num_to_select = max(1, int(self._ptoday * len(shuffled_partners)))  # Number of partners to select / 选择的伙伴数量
        selected_partners = shuffled_partners[:num_to_select]

        n_selected = len(selected_partners)
        # If fewer items than selected partners, further reduce partners to at most `total_quantity`
        # to ensure each gets at least 1 if possible, or some get 0 if q < n_selected_final.
        # 如果物品数量少于选定的伙伴数量，则进一步将伙伴数量减少到最多 `total_quantity`，
        # 以确保如果可能，每个伙伴至少得到1个，或者如果 q < 最终选定的伙伴数，则某些伙伴得到0。
        if total_quantity < n_selected and total_quantity > 0:  # total_quantity > 0 确保不会因 total_quantity=0 而进入此逻辑
            num_final = random.randint(1,
                                       total_quantity)  # Select between 1 and total_quantity partners / 选择 1 到 total_quantity 个伙伴
            selected_partners = random.sample(selected_partners,
                                              num_final)  # Randomly pick from the already selected subset / 从已选子集中随机挑选
            n_selected = len(selected_partners)

        if n_selected == 0 and total_quantity > 0 and Wpartners:  # 如果没有选出伙伴但仍有需求且有可选伙伴，则至少选一个
            selected_partners = random.sample(Wpartners, 1)
            n_selected = 1

        if n_selected > 0:
            quantities_list = distribute(total_quantity, n_selected)  # Use the utility to distribute / 使用实用函数进行分配
            return dict(zip(selected_partners, quantities_list))
        return {}

    def _generate_future_offers_for_group(
            self, partners_group: list[str], future_step: int, total_need_at_future_step: int, is_buying: bool
    ) -> dict[str, Outcome | None]:
        """
        Helper for future_supply_offer and future_consume_offer.
        Distributes `total_need_at_future_step` among `partners_group` for `future_step`.

        `future_supply_offer` 和 `future_consume_offer` 的辅助函数。
        将 `total_need_at_future_step` 分配给 `partners_group` 用于 `future_step`。
        """
        proposals: dict[str, Outcome | None] = {}
        if not partners_group or total_need_at_future_step <= 0:
            return proposals

        quantities = distribute(total_need_at_future_step, len(partners_group))
        for partner_id, q in zip(partners_group, quantities):
            if q > 0:
                price = self.best_price(partner_id)  # Use my best price for initial future offers / 对初始未来报价使用我的最优价格
                if price is not None:
                    proposals[partner_id] = self._create_offer(q, future_step, price)
        return proposals

    def future_supply_offer(self, partners_list: list[str]) -> dict[str, Outcome | None]:
        """
        Generates future supply offers (I buy from these partners).
        Divides partners into 3 groups for future steps s+1, s+2, s+3.
        Calculates 1/3rd of projected need for each group.

        生成未来供应报价 (我从这些伙伴处购买)。
        将伙伴分为3组，用于未来步骤 s+1, s+2, s+3。
        为每组计算预计需求的1/3。
        """
        proposals: dict[str, Outcome | None] = {}
        if not self.awi or not partners_list: return proposals

        s, n_steps = self.awi.current_step, self.awi.n_steps
        len_p = len(partners_list)
        # Divide partners into three rough groups for future steps
        # 将伙伴大致分为三组用于未来步骤
        p_groups = [partners_list[:int(len_p * 0.5)], partners_list[int(len_p * 0.5):int(len_p * 0.8)],
                    partners_list[int(len_p * 0.8):]]

        # Penguin used current_inventory_input for future needs calculation.
        # It also divided the calculated need by 3 for each future step's group.
        # Penguin 使用 current_inventory_input 计算未来需求。
        # 它还将计算出的需求除以3，分配给每个未来步骤的组。
        base_production_need = self.awi.n_lines * self._productivity - self.awi.current_inventory_input

        for i, step_offset in enumerate([1, 2, 3]):  # For s+1, s+2, s+3 / 对于 s+1, s+2, s+3
            future_s = s + step_offset
            if future_s < n_steps and p_groups[
                i]:  # If valid future step and partners exist for this group / 如果是有效的未来步骤且该组存在伙伴
                # Total need for supplies at future_s, then take 1/3rd (Penguin's heuristic)
                # future_s 的总供应需求，然后取1/3 (Penguin 的启发式方法)
                need_at_fut_s_total = base_production_need - self.awi.total_supplies_at(future_s)
                need_for_this_group = max(0, int(need_at_fut_s_total / 3.0))  # Distribute 1/3 of the need / 分配1/3的需求
                proposals.update(
                    self._generate_future_offers_for_group(p_groups[i], future_s, need_for_this_group, True))
        return proposals

    def future_consume_offer(self, partners_list: list[str]) -> dict[str, Outcome | None]:
        """
        Generates future consume offers (I sell to these partners).
        Similar logic to `future_supply_offer` but for selling.

        生成未来消费报价 (我向这些伙伴销售)。
        逻辑与 `future_supply_offer` 类似，但用于销售。
        """
        proposals: dict[str, Outcome | None] = {}
        if not self.awi or not partners_list: return proposals

        s, n_steps = self.awi.current_step, self.awi.n_steps
        len_p = len(partners_list)
        p_groups = [partners_list[:int(len_p * 0.5)], partners_list[int(len_p * 0.5):int(len_p * 0.8)],
                    partners_list[int(len_p * 0.8):]]

        # Penguin used current_inventory_input for future sales capacity.
        # Penguin 使用 current_inventory_input 计算未来销售能力。
        base_available_to_sell = self.awi.n_lines * self._productivity + self.awi.current_inventory_input

        for i, step_offset in enumerate([1, 2, 3]):
            future_s = s + step_offset
            if future_s < n_steps and p_groups[i]:
                # Total sales capacity at future_s, then take 1/3rd
                # future_s 的总销售能力，然后取1/3
                sales_cap_at_fut_s_total = min(self.awi.n_lines, base_available_to_sell) - self.awi.total_sales_at(
                    future_s)
                cap_for_this_group = max(0, int(sales_cap_at_fut_s_total / 3.0))

                # Penguin's condition: awi.total_sales_at(future_s) <= self.awi.n_lines
                # This is implicitly handled if sales_cap_at_fut_s_total is correctly calculated.
                # Penguin 的条件: future_s 的总销售额 <= 生产线数量
                # 如果 sales_cap_at_fut_s_total 计算正确，则此条件已隐式处理。
                if self.awi.total_sales_at(
                        future_s) <= self.awi.n_lines:  # Keep explicit check if vital / 如果至关重要则保留显式检查
                    proposals.update(
                        self._generate_future_offers_for_group(p_groups[i], future_s, cap_for_this_group, False))
        return proposals

    def best_price(self, partner_id: str) -> float | int | None:
        """
        My most preferred price with this partner (min when buying, max when selling).
        Used for initial proposals.

        我与此伙伴最偏好的价格 (购买时为最小值，销售时为最大值)。
        用于初始报价。
        """
        if not self.awi: return None
        nmi = self.get_nmi(partner_id)
        if not nmi or not nmi.issues: return None
        try:
            if nmi.issues[UNIT_PRICE] is None: return None
        except (KeyError, TypeError):
            return None

        price_issue = nmi.issues[UNIT_PRICE]
        min_p = getattr(price_issue, 'min_value', None)
        max_p = getattr(price_issue, 'max_value', None)
        if min_p is None or max_p is None: return None

        if self.is_supplier(partner_id):  # Partner supplies to me (I buy) / 伙伴向我供应 (我购买)
            return min_p
        elif self.is_consumer(partner_id):  # Partner consumes from me (I sell) / 伙伴从我处消费 (我销售)
            return max_p
        return None

    def price(self, partner_id: str) -> float | int | None:
        """
        A concessional price for countering or urgent needs.
        If buying, offers a bit higher than my best (min_p * 1.2).
        If selling, offers a bit lower than my best (max_p * 0.7).

        用于还价或紧急需求的让步价格。
        如果购买，出价比我的最优价格略高 (min_p * 1.2)。
        如果销售，出价比我的最优价格略低 (max_p * 0.7)。
        """
        if not self.awi: return None
        nmi = self.get_nmi(partner_id)
        if not nmi or not nmi.issues: return None
        try:
            if nmi.issues[UNIT_PRICE] is None: return None
        except (KeyError, TypeError):
            return None

        price_issue = nmi.issues[UNIT_PRICE]
        min_p = getattr(price_issue, 'min_value', None)
        max_p = getattr(price_issue, 'max_value', None)
        if min_p is None or max_p is None: return None

        if self.is_supplier(partner_id):  # Partner supplies to me (I buy). My best is min_p. Concede higher.
            # 伙伴向我供应 (我购买)。我的最优价是 min_p。让步到更高价格。
            return min(min_p * 1.2, max_p)  # Concede up to 20% more, capped by max_p / 最多让步20%，但不超过 max_p
        elif self.is_consumer(partner_id):  # Partner consumes from me (I sell). My best is max_p. Concede lower.
            # 伙伴从我处消费 (我销售)。我的最优价是 max_p。让步到更低价格。
            return max(max_p * 0.7, min_p)  # Concede down to 30% less, floored by min_p / 最多让步30%，但不低于 min_p
        return None


if __name__ == "__main__":
    # This section is for running the agent directly.
    # In SCML competitions, agents are usually run by the tournament platform.
    # To run locally, you might use `anaclan run` or a custom script
    # that sets up a world and runs agents.
    # 此部分用于直接运行代理。
    # 在 SCML 竞赛中，代理通常由锦标赛平台运行。
    # 要在本地运行，您可以使用 `anaclan run` 或设置世界并运行代理的自定义脚本。

    # Example (conceptual, actual runner might differ for scml 0.11.3):
    # 示例 (概念性的，scml 0.11.3 的实际运行程序可能不同):
    # from scml.std.runner import run_std_agent_in_tournament
    # run_std_agent_in_tournament(LitaAgentP, "LitaAgentP", "std")

    # For now, just a pass, as the primary goal is the agent class.
    # 目前只是一个 pass，因为主要目标是代理类本身。
    pass # print(f"{LitaAgentP.__name__} class is defined. To run, use SCML simulation tools.")
    pass # print(f"{LitaAgentP.__name__} 类已定义。要运行，请使用 SCML 模拟工具。")
    pass