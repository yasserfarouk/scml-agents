#!/usr/bin/env python
"""
LitaAgent — SCML2025  Standard 赛道谈判代理
===========================================

本类在官方 StdRLAgent 骨架 (scml.std.StdRLAgent) 的基础上，
完整实现了 first_proposals与 counter_all策略逻辑，
同时保留LitaAgent 已有的强化学习框架（模型加载 / ObservationManager /
ActionManager 等）。

这样做的目标是分两阶段逐步演进：
1. 本阶段 — 复制去年的获胜者 *PenguinAgent* 的规则策略，保证代理具有
   即战力。
2. 下一阶段 — 在不破坏接口的前提下，将规则策略替换为强化学习
   / 需求预测等智能模型的输出。

This class is based on the official StdRLAgent skeleton (scml.std.
It fully implements the first_proposals and counter_all policy logic.
At the same time, it retains LitaAgent's existing reinforcement learning framework (model loading / ObservationManager / ActionManager, etc.).
ActionManager, etc.).

The goal is to evolve in two phases:
1. **This phase** - Replicate the rule-based strategy of last year's winner, *PenguinAgent*, to ensure that the agent has
   immediate power.
2. **Next phase** - Replace the rule strategy with reinforcement learning without breaking the interface.
   / demand forecasting and other intelligent modeling outputs without breaking the interface.
"""

from __future__ import annotations

# =============== 基础 & 框架依赖 ===============
from typing import Any, Iterable, Dict, List, Tuple
import random
import os

from scml import RandDistOneShotAgent
# from tensorflow.python.eager.execute import must_record_gradient

from . import inventory_manager_n
from itertools import chain, combinations
from collections import Counter

from scml.std import (
    StdAWI,
    TIME,
    QUANTITY,
    UNIT_PRICE,
    StdSyncAgent,
)
from negmas import SAOState, SAOResponse, Outcome, Contract, ResponseType


# numpy 的随机分配在 distribute() 中使用；仅在运行时导入以避免训练阶段缺少依赖
from numpy.random import choice as np_choice  # type: ignore

__all__ = ["LitaAgentN"]

from .inventory_manager_n import IMContract, MaterialType, IMContractType


# ---------------------------------------------------------------------------
# 通用辅助工具
# General helper functions
# ---------------------------------------------------------------------------


def _distribute(q: int, n: int) -> List[int]:
    """
    将 ``q`` 个单位随机分配到 ``n`` 个桶中，保证 **每桶至少 1**（q>=n），
    若 ``q < n``，则前 ``q`` 个桶获得 1，其余 0。
    Randomly assign ``q`` units to ``n`` buckets, ensuring **at least 1** per bucket (q>=n).
    If ``q < n``, the first ``q`` buckets get 1 and the rest 0.
    """
    if n <= 0:
        return []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n

    r = Counter(np_choice(n, q - n))
    return [r.get(i, 0) + 1 for i in range(n)]


def _powerset(iterable: Iterable[Any]):
    """
    返回 ``iterable`` 的幂集（所有子集）。用于穷举报价组合。
    Returns the power set of ``iterable`` (all subsets). Used to exhaustively enumerate combinations of quotes.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# ---------------------------------------------------------------------------
# 代理实现 Agent implementation
# ---------------------------------------------------------------------------


class LitaAgentN(StdSyncAgent):
    # =====================
    # Initialization
    # =====================

    def __init__(self, *args, ptoday: float = 0.70, **kwargs):
        """初始化。

        Parameters
        ----------
        ptoday : float, optional
            当期报价时选取的伙伴比例（默认 70%），与 PenguinAgent `_ptoday` 对应。

        RL相关的东西暂时全部注释
        """
        # --------- 1. RL 相关: 模型路径 / 加载 ---------

        """
        base_name = MODEL_PATH.name
        self.paths = [
            MODEL_PATH.parent / f"{base_name}_supplier",
            MODEL_PATH.parent / f"{base_name}_consumer",
        ]
        models = tuple(model_wrapper(TrainingAlgorithm.load(p)) for p in self.paths)

        # 生成上下文并创建 Observation / Action Manager
        contexts = (make_context(as_supplier=True), make_context(as_supplier=False))
        kwargs.update(
            dict(
                models=models,
                observation_managers=(
                    MyObservationManager(context=contexts[0]),
                    MyObservationManager(context=contexts[1]),
                ),
                action_managers=(
                    FlexibleActionManager(context=contexts[0]),
                    FlexibleActionManager(context=contexts[1]),
                ),
            )
        )   
        """

        # --------- 3. Penguin 策略相关参数 ---------
        self._ptoday: float = (
            ptoday  # 当期挑选伙伴比例 Proportion of partners selected today
        )
        # 生产效率：此处沿用 RLAgent 的 production_level，故不额外定义 _productivity
        # 每日超量接受阈值将在 step() 中根据 future_concession 动态计算
        self._threshold: float = 1.0  # 占位，step() 时更新

        # 记录当前库存管理器
        # Record the current inventory manager
        self.im = None  # initialize in init()

        # 记录当前库存不足量
        # Record the current inventory shortfall
        self.today_insufficient = None
        self.total_insufficient = None

        # 记录今天已签约的，且今天就要交割的交易量，用于保证最低购买
        # Record deals signed today that deliver today to ensure minimum purchasing
        self.today_signed_sale = None
        self.today_signed_supply = None

        # 设置一个fallback agent TODO： 做一个好点的fallback agent 现在用RandDistOneShotAgent
        # Set up a fallback agent. TODO: create a better one; currently using RandDistOneShotAgent
        # self._fallback_agent = RandDistOneShotAgent()
        self._fallback_type = RandDistOneShotAgent

        self.future_concession = None
        # --------- 2. 调用父类：保留原 production_level / future_concession ---------
        super().__init__(*args, **kwargs)
        # super().__init__(*args, production_level=0.25, future_concession=0.10, **kwargs)

    # ---------------------------------------------------------------------
    # 常用快捷判断 & 价格工具
    # Utility checks and pricing helpers
    # ---------------------------------------------------------------------

    # 伙伴角色判断 ---------------------------------------------------------
    # Partner role checks -------------------------------------------------
    def _is_supplier(self, partner: str) -> bool:
        return partner in self.awi.my_suppliers

    def _is_consumer(self, partner: str) -> bool:
        return partner in self.awi.my_consumers

    # 价格工具 -------------------------------------------------------------
    # Pricing utilities ----------------------------------------------------
    def _best_price(self, partner: str) -> float:
        """对 *自己* 最有利的价格：买入 ⇒ 最低价；卖出 ⇒ 最高价"""
        # Price most favorable to us: buy at minimum, sell at maximum
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        return pmin if self._is_supplier(partner) else pmax

    def _concession_price(self, partner: str) -> float:
        """在 **还价** 时的小幅让步价格：
        * 若我是买家（对方为供应商），则在最低价基础上 *上浮* 20%
        * 若我是卖家（对方为顾客），      在最高价基础上 *下调* 30%
        该比例与 PenguinAgent 一致。"""
        # Concession price when countering:
        # * If I am the buyer, raise 20% above the minimum price
        # * If I am the seller, cut 30% off the maximum price
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        if self._is_consumer(partner):  # 我卖货，对方买
            return max(pmax * 0.7, pmin)
        else:  # 我买货，对方卖
            return min(pmin * 1.2, pmax)

    def _is_valid_price(self, price: float, partner: str) -> bool:
        """过滤法外价格：
        * 对顾客：必须 >= 最低价
        * 对供应商：必须 <= 最高价
        TODO: 要从Inventory Manager 中获取成本价，阻拦低于成本价的交易（最好再加入一个机会成本计算器）
        """
        issue = self.get_nmi(partner).issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        if self._is_consumer(partner):
            return price >= pmin
        elif self._is_supplier(partner):
            return price <= pmax
        return False

    # ---------------------------------------------------------------------
    # 当期需求 & 随机分配工具 —— 这是处理 *今天* 的需求
    # Current-day demand distribution tools
    # ---------------------------------------------------------------------

    def _day_production(self) -> float:
        """根据生产线数量与 production_level 估计当日产量。"""
        # return self.awi.n_lines * self.production_level
        # 这里使用IM的值
        return self.im.get_max_possible_production(self.awi.current_step)

    def _needs_today(self) -> Tuple[int, int]:
        """返回 (待采购原料量, 可销售产品量)。如为 0 则无需该方向谈判。"""
        awi = self.awi
        prod = self._day_production()
        # 1) 采购需求（向供应商买）
        # Purchase demand toward suppliers
        # TODO: 这里有问题，这里的购买量是简单的根据最大产量*production_level，应该优化为IM的值
        # buy_need = int(max(0.0, prod - awi.current_inventory_input - awi.total_supplies_at(awi.current_step)))
        buy_need = int(
            self.total_insufficient
            * 1.1  # 我们使用总提不足量的1.1倍来作为采购需求，姑且
        )  # Using 110% of total shortage as buy need for now
        # 2) 销售需求（向顾客卖）
        # Sales demand toward consumers
        # 仅当当前累计销售未超过产能上限才继续卖
        # TODO: 之后最好是接入IM，现在暂且用awi的值
        if awi.total_sales_at(awi.current_step) < awi.n_lines:
            sell_need = int(
                max(
                    0.0,
                    min(awi.n_lines, prod + awi.current_inventory_input)
                    - awi.total_sales_at(awi.current_step),
                )
            )
        else:
            sell_need = 0
        return buy_need, sell_need

    def _distribute_todays_needs(
        self, partners: Iterable[str] | None = None
    ) -> Dict[str, int]:
        """随机将今日需求分配给一部分伙伴（按 _ptoday 比例）。"""
        # 暂且先这样
        # For now we'll keep it simple
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)

        # 初始化：默认所有伙伴分配量 0
        # Initialize distribution: all partners get 0 by default
        response: Dict[str, int] = {p: 0 for p in partners}

        # 分类伙伴
        # Categorize partners
        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]

        buy_need, sell_need = self._needs_today()

        # --- 1) 分配采购需求给供应商 ---
        # Allocate purchase needs to suppliers
        if suppliers and buy_need > 0:
            response.update(self._distribute_to_partners(suppliers, buy_need))

        # --- 2) 分配销售需求给顾客 ---
        # Allocate sales needs to consumers
        if consumers and sell_need > 0:
            # 由于计算需求时已经做过了限制，所以这里不需要再判断了
            response.update(self._distribute_to_partners(consumers, sell_need))

        return response

    def _distribute_to_partners(
        self, partners: List[str], needs: int
    ) -> Dict[str, int]:
        """核心分配算法：随机打乱伙伴，取前 ``ptoday`` 比例分配 ``needs``。"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}

        random.shuffle(partners)
        # 取前 ptoday 比例（至少 1 个）
        # Take the first ptoday proportion (at least one partner)
        k = max(1, int(len(partners) * self._ptoday))
        chosen = partners[:k]

        # 若需求小于伙伴数，随机抽 subset
        # If needs are less than partners, randomly sample a subset
        if needs < len(chosen):
            chosen = random.sample(chosen, random.randint(1, needs))

        quantities = _distribute(needs, len(chosen))
        distribution = dict(zip(chosen, quantities))
        # 其余伙伴分配量为 0
        # All other partners receive 0
        return {p: distribution.get(p, 0) for p in partners}

    # ---------------------------------------------------------------------
    # 未来报价生成工具 —— 这是处理 *未来* 的需求
    # Helper to generate offers for future needs
    # ---------------------------------------------------------------------

    def _future_supplie_offer(self, partners: List[str]) -> Dict[str, Outcome]:
        """给供应商伙伴生成未来 1/2/3 步的采购报价。"""
        # TODO:将报价延伸至未来的1...horizon天，而不是3天
        response: Dict[str, Outcome] = {}
        if not partners:
            return response

        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        prod = self._day_production()
        # 按 50% / 30% / 20% 划分伙伴列表
        step1, step2, step3 = self._split_partners(partners)

        def _needs(step: int) -> int:
            # return int((prod - awi.current_inventory_input - awi.total_supplies_at(step)) / 3)
            # 这里不使用PA的定额生产逻辑，改为按照IM中的总insufficient
            return int(self.im.get_total_insufficient(step))

        for offset, plist in zip((1, 2, 3), (step1, step2, step3)):
            step = s + offset
            if step >= n or not plist:
                continue
            need = _needs(step)
            if need <= 0:
                continue
            distribution = _distribute(need, len(plist))
            for p, q in zip(plist, distribution):
                if q > 0:
                    response[p] = (q, step, self._best_price(p))
        return response

    def _future_consume_offer(self, partners: List[str]) -> Dict[str, Outcome]:
        """给顾客伙伴生成未来 1/2/3 步的销售报价。"""
        response: Dict[str, Outcome] = {}
        if not partners:
            return response

        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        prod = self._day_production()
        step1, step2, step3 = self._split_partners(partners)

        def _needs(step: int) -> int:
            return int(
                # TODO：这里要改成根据IM的值来计算
                # max(0.0, min(awi.n_lines, prod + awi.current_inventory_input) - awi.total_sales_at(step)) / 3
                # 去IM获取最大的可能产值
                self.im.get_max_possible_production(step)
            )

        for offset, plist in zip((1, 2, 3), (step1, step2, step3)):
            step = s + offset
            if step >= n or not plist:
                continue
            # 若未来该日销量已达产能，则不报价
            if awi.total_sales_at(step) > awi.n_lines:
                continue
            need = _needs(step)
            if need <= 0:
                continue
            distribution = _distribute(need, len(plist))
            for p, q in zip(plist, distribution):
                if q > 0:
                    response[p] = (q, step, self._best_price(p))
        return response

    @staticmethod
    def _split_partners(partners: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """把伙伴按 50% / 30% / 20% 划分 3 组。"""
        n = len(partners)
        return (
            partners[: int(n * 0.5)],
            partners[int(n * 0.5) : int(n * 0.8)],
            partners[int(n * 0.8) :],
        )

    # ---------------------------------------------------------------------
    # 需求量辅助 (for future acceptance check)
    # ---------------------------------------------------------------------

    def _needs_at(self, step: int, partner: str) -> int:
        """计算在未来 ``step`` 日仍需与 ``partner`` 方向交易的数量。"""
        # TODO: 这里要改成根据IM的值来计算
        prod = self._day_production()
        awi = self.awi
        if self._is_supplier(partner):  # 采购
            # return int(prod - awi.current_inventory_input - awi.total_supplies_at(step))
            return int(self.im.get_total_insufficient(step))
        else:  # 销售
            return int(
                # max(0.0, min(awi.n_lines, prod + awi.current_inventory_input) - awi.total_sales_at(step))
                self.im.get_max_possible_production(step)
            )

    # =====================
    # Negotiation Callbacks
    # =====================

    # --------- 主要回调 1: first_proposals ---------
    def first_proposals(self) -> Dict[str, Outcome | None]:
        """生成首轮报价。逻辑同 PenguinAgent。

        Returns
        -------
        dict
            {partner_id: Outcome}，如返回空字典则终止全部谈判。
        """
        partners = list(self.negotiators.keys())
        if not partners:
            return {}

        s = self.awi.current_step  # 当前步骤 (day index)

        # 1. 计算并分配当期需求
        # Step 1: compute and distribute today's requirements
        distribution = self._distribute_todays_needs(partners)

        # 2. 构建当前报价 & 未来伙伴列表
        # Step 2: build today's offers and a list of future partners
        first_dict: Dict[str, Outcome] = {}
        future_suppliers: List[str] = []
        future_consumers: List[str] = []

        for p, q in distribution.items():
            if q > 0:
                first_dict[p] = (q, s, self._best_price(p))
            elif self._is_supplier(p):
                future_suppliers.append(p)
            elif self._is_consumer(p):
                future_consumers.append(p)

        # 3. 生成未来报价
        # Step 3: generate offers for future periods
        response: Dict[str, Outcome] = {}
        response.update(first_dict)
        response.update(self._future_supplie_offer(future_suppliers))
        response.update(self._future_consume_offer(future_consumers))

        return response

    # --------- 主要回调 2: counter_all ---------
    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """对所有正在谈判的伙伴做出回应 (Accept / Reject + new offer)。"""

        responses: Dict[str, SAOResponse] = {}
        awi = self.awi

        # 分别处理 [采购] 与 [销售] 两条边
        # Handle supplier and consumer sides separately
        for all_partners, is_supplier_side in [
            (awi.my_suppliers, True),
            (awi.my_consumers, False),
        ]:
            if not all_partners:
                continue

            # ------------ 1. 计算当前需求量 ------------
            if is_supplier_side:  # 我买, 对方卖
                # I'm buying, partner selling
                needs = self.im.get_total_insufficient(self.awi.current_step)
                # needs = int(prod - awi.current_inventory_input - awi.total_supplies_at(awi.current_step))
            else:  # 我卖, 对方买
                # I'm selling, partner buying
                """
                if awi.total_sales_at(awi.current_step) < awi.n_lines:
                    needs = int(
                        max(0.0, min(awi.n_lines, prod + awi.current_inventory_input) - awi.total_sales_at(awi.current_step))
                    )
                else:
                    needs = 0
                """
                needs = prod = self._day_production()  # 计算到今天为止的最大产量
                # Maximum producible quantity up to today

            # ------------ 2. 拆分类别报价 ------------
            # Break down offers by category
            # 将报价氛围分为当前报价与未来报价，然后过滤掉不符合系统条件的报价
            # Split offers into current and future ones and filter out invalid bids
            current_offers: Dict[str, Outcome] = {}
            future_offers: Dict[str, Outcome] = {}
            active_partners = {
                p for p in all_partners if p in offers and offers[p] is not None
            }

            for p in active_partners:
                offer = offers[p]
                if not self._is_valid_price(offer[UNIT_PRICE], p):
                    continue  # 跳过法外价格 TODO：跳过不合规价格的功能沿用了 需要在之后再实现这个功能（因为LitaAgent合规价格是和日期相关的）
                if offer[TIME] == awi.current_step:
                    current_offers[p] = offer
                else:
                    future_offers[p] = offer
            # TODO：改到这里了 这里其实还相当于啥也没做，除了排序了一下
            # TODO: from here on nothing much happens except sorting
            # ----------- 2-a. 优先锁定今天必须购买的数量 ------------
            # 2-a. Prioritize locking in the quantity that must be purchased today
            must_buy = self.today_insufficient
            purchased_today = 0
            # 将offer根据价格由低到高排序
            current_offers = dict(
                sorted(current_offers.items(), key=lambda x: x[1][UNIT_PRICE])
            )

            for p, offer in current_offers.items():
                if os.path.exists("env.test"):
                    print(
                        f"代理{self.id}当前收到来自{p}的报价：单位价格：{offer[UNIT_PRICE]}, 数量：{offer[QUANTITY]}, 交割日期：{offer[TIME]}, 交易类型：{IMContractType.SUPPLY if p in awi.my_suppliers else IMContractType.DEMAND}"
                    )
                # 判断价格和罚款比起来哪个比较贵 如果罚款比较贵，则准备counter offer
                penalty = self.awi.current_shortfall_penalty
                if offer[UNIT_PRICE] > penalty:
                    # TODO：改进这个consession_price
                    counter_offer = (
                        offer[QUANTITY],
                        offer[TIME],
                        self._concession_price(p),
                    )
            # ------------ 3. 先锁定所有合理的未来报价 ------------
            # Step 3: lock in all reasonable future offers first

            duplicate_quantity = [0] * awi.n_steps  # 防重计数表
            # Table to avoid double counting
            for p, offer in future_offers.items():
                step = offer[TIME]
                if step >= awi.n_steps:
                    continue
                if offer[QUANTITY] + duplicate_quantity[step - 1] <= self._needs_at(
                    step, p
                ):
                    responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    duplicate_quantity[step - 1] += offer[QUANTITY]

            # ------------ 4. 对当前报价做组合选择 ------------
            best_plus_idx = best_minus_idx = -1
            best_plus_diff = best_minus_diff = float("inf")
            ps_list = list(_powerset(current_offers.keys()))

            for idx, subset in enumerate(ps_list):
                if not subset:
                    continue
                total = sum(current_offers[p][QUANTITY] for p in subset)
                diff = abs(total - needs)
                if total >= needs:  # 超出
                    if diff < best_plus_diff:
                        best_plus_diff, best_plus_idx = diff, idx
                else:  # 未达
                    if diff < best_minus_diff:
                        best_minus_diff, best_minus_idx = diff, idx

            # 阈值 = future_concession * 产能
            # self._threshold = self.future_concession * awi.n_lines
            # TODO：阈值 = 总需求不足量 * fc 总需求不足量基于订单交付日，由im计算
            self._threshold = max(
                0,
                self.im.get_total_insufficient(self.awi.current_step)
                * self.future_concession,
            )
            accepted_subset = None
            if best_plus_idx != -1 and best_plus_diff <= self._threshold and needs > 0:
                accepted_subset = ps_list[best_plus_idx]
            elif best_minus_idx != -1 and needs > 0:
                accepted_subset = ps_list[best_minus_idx]

            # ------------ 5. 如果找到组合则接受，否则准备还价 ------------
            handled_partners = set(responses.keys())  # 已经在未来报价被接受的伙伴
            # Partners whose future offers were already accepted

            if accepted_subset:
                # (a) 接受组合内报价
                for p in accepted_subset:
                    responses[p] = SAOResponse(
                        ResponseType.ACCEPT_OFFER, current_offers[p]
                    )
                handled_partners.update(accepted_subset)

            # 取出其余需要处理的伙伴（当前对我报价但尚未回应）
            remaining_partners = [
                p for p in current_offers if p not in handled_partners
            ]

            # (b) 对 remaining_partners 进行还价
            if remaining_partners or (not accepted_subset and needs > 0):
                self._make_counter_offers(remaining_partners, responses, needs)

        return responses

    # ------------------------------------------------------------------
    # 还价生成 (私有)
    # Counter-offer generation (internal)
    # ------------------------------------------------------------------

    def _make_counter_offers(
        self, partners: List[str], responses: Dict[str, SAOResponse], needs: int
    ) -> None:
        """针对 *partners* 生成新的报价 (Reject+counter / Future)。"""
        if not partners:
            return

        # 重新分配剩余需求（可能 < 0）
        # Redistribute remaining needs (may be negative)
        distribution = self._distribute_todays_needs(partners)
        future_suppliers: List[str] = []
        future_consumers: List[str] = []

        for p, q in distribution.items():
            if q > 0:
                counter_offer = (q, self.awi.current_step, self._concession_price(p))
                responses[p] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            else:
                if self._is_supplier(p):
                    future_suppliers.append(p)
                elif self._is_consumer(p):
                    future_consumers.append(p)

        # 对零分配伙伴：生成未来报价并作为拒绝伴随提议
        # For partners assigned zero today, generate future offers and attach them to rejections
        for p, offer in self._future_supplie_offer(future_suppliers).items():
            responses[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        for p, offer in self._future_consume_offer(future_consumers).items():
            responses[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)

    # =====================
    # Time‑Driven Callbacks (可按需扩展)
    # =====================

    def init(self):
        """
        raw_storage_cost: float,
        product_storage_cost: float,
        processing_cost: float,
        # 可选：日生产能力，若不指定则视为无限 Unlimited if not specified
        daily_production_capacity: Optional[float] = None,
        # 可选：最大仿真天数，默认100天 Should be initialized while the instance is created
        max_day: int = 100,
        """

        # 记录角色（买/卖）以加速判断
        # Record roles (buyer/seller) to speed up checks
        self.is_buyer_negotiator = not self.awi.is_first_level
        self.is_seller_negotiator = not self.awi.is_last_level

        self.im = inventory_manager_n.InventoryManager(
            raw_storage_cost=self.awi.current_storage_cost,
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=0,  # TODO: How can I get the processing cost?
            daily_production_capacity=self.awi.n_lines,
            max_day=self.awi.n_steps,
        )
        """世界接口初始化后调用。此处可做自定义准备。"""

    def before_step(self):
        """每天 **开始** 调用，可在此同步内部状态。"""
        # insufficant inventory, for negotiation
        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)
        self.today_signed_supply = 0
        self.today_signed_sale = 0
        self.future_concession = (
            self.awi.current_shortfall_penalty / self.awi.current_storage_cost
        )

    def step(self):
        """每天 **结束** 调用。这里更新超量阈值等。"""
        # self._threshold = self.future_concession * self.awi.n_lines
        # TODO: 我总觉得这里的future_concession算法有点问题，但反正是AI推荐的，就先不管了
        self.future_concession = (
            self.awi.current_shortfall_penalty / self.awi.current_storage_cost
        )
        self._threshold = max(
            0,
            self.im.get_max_possible_production(self.awi.current_step)
            * self.future_concession,
        )
        # 1. 更新库存管理器
        # Update the inventory manager
        self.im.receive_materials()
        self.im.plan_production(self.awi.current_step + self.awi.horizon)
        self.im.execute_production(self.awi.current_step)
        self.im.deliver_products()
        self.im.update_day()

    # ==============================
    # Negotiation Control Callbacks
    # ==============================

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """谈判失败时触发。当前不展开，可用于收集数据。"""

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        """谈判成功时触发，可打印日志或更新内部模型。"""

        # 1. 将协议加入库存管理器
        if os.path.exists("env.test"):
            print(
                f"{self.id}准备调用库存管理器的add_transaction方法，合同ID：{contract.id}, 交易伙伴：{contract.partners}, 交易类型：{IMContractType.SUPPLY if contract.partners in self.awi.my_suppliers else IMContractType.DEMAND}, 数量: {contract.agreement['quantity']}, 单价: {contract.agreement['unit_price']}, 交割日期: {contract.agreement['time']}, RAW: {contract}"
            )
        self.im.add_transaction(
            contract=IMContract(
                contract_id=contract.id,
                partner_id=contract.partners,
                type=IMContractType.SUPPLY
                if contract.partners in self.awi.my_suppliers
                else IMContractType.DEMAND,
                quantity=contract.agreement["quantity"],
                price=contract.agreement["unit_price"],
                delivery_time=contract.agreement["time"],
                bankruptcy_risk=0,
                material_type=MaterialType.RAW
                if contract.partners in self.awi.my_suppliers
                else MaterialType.PRODUCT,
            )
        )
        # 1-a 更新今天已完成的，且今天就要交割的交易
        if (contract.partners in self.awi.my_suppliers) and (
            contract.issues[TIME] == self.awi.current_step
        ):
            self.today_signed_supply += contract.issues[QUANTITY]
        elif (contract.partners in self.awi.my_consumers) and (
            contract.issues[TIME] == self.awi.current_step
        ):
            self.today_signed_sale += contract.issues[QUANTITY]
        # 2. 库存管理器重新安排生产和采购（库存管理器内部执行）
        # Inventory manager reschedules production and purchasing internally
        pass


# ----------------------------------------------------------------------------
# CLI 入口：用于本地测试 (与官方 runner 兼容)
# ----------------------------------------------------------------------------
"""
别管这个倒霉代码 这是AI干的
# Ignore this messy code, it was generated by the AI
if __name__ == "__main__":
    import sys
    from scml.helpers.runner import run  # type: ignore

    run([LitaAgentN], sys.argv[1] if len(sys.argv) > 1 else "std")
"""
