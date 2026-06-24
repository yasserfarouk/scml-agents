#!/usr/bin/env python
"""
**Submitted to ANAC 2026 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

Step 1 ベースライン: ANAC 2025 SCML Std 優勝エージェント (AS0) の戦略を移植。
後続ステップで以下を順次置き換え予定:
  - データ収集の足場 (partner_log, tp_history, my_exog_history)
  - Layer 4: Boulware 時間譲歩 (smart_price 置換)
  - Layer 3: Appetite-based allocation (partner_success_rate 置換)
  - Layer 1: 市場ビュー + 中間層在庫バッファ戦略

AS0 戦略の柱:
  1. 動的な交渉キャップ τₜ (在庫圧力 × 残時間圧力)
  2. パートナー成功率ベースの選別
  3. 50/30/20 の Forward Contracting + 均等配分
  4. プロフィットトレンドで Aggressive/Conservative モード切替
  5. 信頼度ベースの smart_price
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import chain, combinations, repeat
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState
from numpy.random import choice
from scml.std import QUANTITY, TIME, UNIT_PRICE, StdAWI, StdSyncAgent

__all__ = ["SBD"]


@dataclass
class NegoRecord:
    """1 回の交渉結果を後段の戦略 (appetite 推定など) のために保存する。"""

    day: int
    is_consumer: bool  # 相手が自分の consumer (= 自分が売る側) なら True
    success: bool
    rounds_used: int = 0
    rounds_total: int = 0
    agreed_q: int = 0
    agreed_p: float = 0.0
    agreed_t: int = 0


def distribute(q: int, n: int) -> list[int]:
    """q を n ビンに分配。q < n のときは一部ゼロ。"""
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    """空集合を含む全ての部分集合をイテレート。"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class _AS0Base(StdSyncAgent):
    """内部用: AS0 (ANAC 2025 SCML Std 優勝) の戦略移植。
    lean_agent.py が継承元として使用 (公開しない)。
    """

    def __init__(
        self,
        *args,
        threshold: int | None = None,
        ptoday: float = 0.70,
        productivity: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._threshold = 1 if threshold is None else threshold
        self._base_ptoday = ptoday
        self._productivity = productivity

        # パートナー単位の成功統計
        self.partner_success_rate: dict[str, float] = defaultdict(lambda: 0.5)
        self.partner_negotiations: dict[str, int] = defaultdict(int)
        self.partner_successes: dict[str, int] = defaultdict(int)

        # プロフィットトレンドのモード切替用
        self.last_profits: list[float] = []
        self.aggressive_mode = False
        self.conservative_mode = False

        # ===== データ収集足場 (後続レイヤで使用) =====
        # tp_history[product_id] = 直近 30 日の trading price
        self.tp_history: dict[int, deque] = {}
        # 自分の外生契約量 (一次層 / 最終層のみ意味あり)
        self.my_exog_input_history: deque = deque(maxlen=30)
        self.my_exog_output_history: deque = deque(maxlen=30)
        # 市場全体のサマリ (各日の総量・平均価格)
        self.market_volume_history: deque = deque(maxlen=30)
        # パートナー別の交渉履歴
        self.partner_log: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))

    # ======================
    # Time-driven callbacks
    # ======================

    def init(self):
        """シミュレーション開始時の初期化フック。tp_history を商品数で初期化。"""
        self.tp_history = {p: deque(maxlen=30) for p in range(self.awi.n_products)}

    def before_step(self):
        """毎日の最初: 外生契約量と市場サマリを記録。"""
        # 自分の外生契約 (中間層は両方ゼロ)
        self.my_exog_input_history.append(int(self.awi.current_exogenous_input_quantity))
        self.my_exog_output_history.append(int(self.awi.current_exogenous_output_quantity))

        # 市場全体の外生サマリ (publish_exogenous_summary 有効時のみ)
        try:
            summary = self.awi.exogenous_contract_summary
            if summary:
                # summary は [(quantity, total_price), ...] 形式
                self.market_volume_history.append(tuple(summary))
        except Exception:  # noqa: BLE001
            pass

    def step(self):
        """毎日の終わり: 閾値とモードを更新する。"""
        super().step()

        # 全商品の trading price をスナップショット記録
        try:
            for p, tp in enumerate(self.awi.trading_prices):
                if p in self.tp_history:
                    self.tp_history[p].append(float(tp))
        except Exception:  # noqa: BLE001
            pass

        base_threshold = self.awi.n_lines * 0.1

        # 在庫圧力で τ を補正
        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)
        if inventory_ratio < 0.3:
            self._threshold = max(1, int(base_threshold * 1.5))
        elif inventory_ratio > 0.8:
            self._threshold = max(1, int(base_threshold * 0.7))
        else:
            self._threshold = max(1, int(base_threshold))

        # 残時間で τ を補正
        time_left = (self.awi.n_steps - self.awi.current_step) / self.awi.n_steps
        if time_left < 0.3:  # 終盤
            self._threshold = int(self._threshold * 1.8)
            self.aggressive_mode = True
        elif time_left < 0.6:  # 中盤
            self._threshold = int(self._threshold * 1.3)

        # 残高トレンドでモード切替
        current_balance = getattr(self.awi, "current_balance", 0)
        # NOTE: 元 AS0 にあった dead code をベースラインとして残す (後続で除去)
        if len(self.last_profits) > 0:
            current_balance - sum(self.last_profits)

        self.last_profits.append(current_balance)
        if len(self.last_profits) > 5:
            self.last_profits.pop(0)

        if len(self.last_profits) >= 3:
            recent_trend = (
                sum(self.last_profits[-3:]) / 3 - sum(self.last_profits[:2]) / 2
            )
            if recent_trend < -50:
                self.conservative_mode = True
            elif recent_trend > 50:
                self.conservative_mode = False

    # =============================
    # Partner performance tracking
    # =============================

    def update_partner_performance(self, partner: str, success: bool) -> None:
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
        self.partner_success_rate[partner] = (
            self.partner_successes[partner] / self.partner_negotiations[partner]
        )

    def estimate_appetite(self, partner: str, lookback: int = 8) -> float:
        """0..1 の "食欲" スコア。1=大量取引したがる、0=不要 / 拒絶しがち。

        信号:
          A. 直近成立量の正規化値 (n_lines に対する比)
          B. 譲歩スピード (rounds_used 少 → 即決 → 飢えてる)
          C. 直近成功率
        """
        log = list(self.partner_log[partner])
        if len(log) < 1:
            return 0.5  # コールドスタート: 中立な事前分布
        recent = log[-lookback:]

        # A: 平均成立数量
        success_recs = [r for r in recent if r.success]
        n_lines = max(1, self.awi.n_lines)
        if success_recs:
            avg_q = sum(r.agreed_q for r in success_recs) / len(success_recs)
            qty_score = min(1.0, avg_q / n_lines)
        else:
            qty_score = 0.0

        # B: 譲歩スピード (1 - used/total). round_total=0 ガード
        valid = [r for r in recent if r.rounds_total > 0]
        if valid:
            speed_score = sum(
                1.0 - r.rounds_used / r.rounds_total for r in valid
            ) / len(valid)
        else:
            speed_score = 0.5

        # C: 成功率
        success_rate = sum(1 for r in recent if r.success) / len(recent)

        return 0.5 * qty_score + 0.2 * speed_score + 0.3 * success_rate

    def get_effective_ptoday(self) -> float:
        base = self._base_ptoday
        if self.aggressive_mode:
            return min(0.9, base + 0.15)
        if self.conservative_mode:
            return max(0.5, base - 0.1)
        return base

    def select_partners_by_performance(
        self, partners, ratio: float | None = None
    ) -> list[str]:
        """成功率降順で上位を選び、探索枠で最大2人をランダム追加。"""
        if not partners:
            return []
        if ratio is None:
            ratio = self.get_effective_ptoday()
        scored = sorted(
            ((p, self.partner_success_rate[p]) for p in partners),
            key=lambda x: x[1],
            reverse=True,
        )
        select_count = max(1, int(len(partners) * ratio))
        selected = [p for p, _ in scored[:select_count]]
        remaining = [p for p, _ in scored[select_count:]]
        if remaining and len(selected) < len(partners):
            extra = min(2, len(remaining), len(partners) - len(selected))
            selected.extend(random.sample(remaining, extra))
        return selected

    # ======================
    # Negotiation callbacks
    # ======================

    def first_proposals(self) -> dict[str, Outcome | None]:
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()

        first_dict: dict[str, Outcome | None] = {}
        future_suppliers: list[str] = []
        future_consumers: list[str] = []

        for k, q in distribution.items():
            if q > 0:
                price = self.smart_price(k, is_first_proposal=True)
                first_dict[k] = (q, s, price)
            elif self.is_supplier(k):
                future_suppliers.append(k)
            elif self.is_consumer(k):
                future_consumers.append(k)

        response: dict[str, Outcome | None] = {}
        response |= first_dict

        # ニーズが満たされた相手には未来日提案
        top_suppliers = self.select_partners_by_performance(future_suppliers)
        top_consumers = self.select_partners_by_performance(future_consumers)
        response |= self.future_supplie_offer(top_suppliers)
        response |= self.future_consume_offer(top_consumers)

        return response

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response: dict[str, SAOResponse] = {}
        awi = self.awi

        for edge_needs, all_partners, _issues in [
            (awi.needed_supplies, awi.my_suppliers, awi.current_input_issues),
            (awi.needed_sales, awi.my_consumers, awi.current_output_issues),
        ]:
            # Layer 1: 動的生産目標を使用
            day_production = self.daily_throughput_target()

            # 今日の needs を side ごとに計算
            needs = 0
            if all_partners and self.is_supplier(all_partners[0]):
                needs = int(
                    day_production
                    - awi.current_inventory_input
                    - awi.total_supplies_at(awi.current_step)
                )
            elif all_partners and self.is_consumer(all_partners[0]):
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = int(
                        max(
                            0,
                            min(
                                awi.n_lines,
                                day_production + awi.current_inventory_input,
                            )
                            - awi.total_sales_at(awi.current_step),
                        )
                    )

            # 受信オファーを当日 / 未来日に分割 (有効価格のみ)
            partners = {p for p in all_partners if p in offers.keys()}
            current_step_offers: dict[str, Outcome] = {}
            future_step_offers: dict[str, Outcome] = {}

            for p in partners:
                o = offers[p]
                if o is None:
                    continue
                if not self.is_valid_price(o[UNIT_PRICE], p):
                    continue
                if o[TIME] == awi.current_step:
                    current_step_offers[p] = o
                else:
                    future_step_offers[p] = o

            current_step_partners = set(current_step_offers.keys())

            # 未来納期: 累積数量が needs_at(t) を超えない範囲で受諾
            duplicate_list = [0] * awi.n_steps
            for p, o in future_step_offers.items():
                t = o[TIME]
                if t < awi.n_steps:
                    if (
                        o[QUANTITY] + duplicate_list[t - 1]
                        <= self.needs_at(t, p)
                    ):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, o)
                        duplicate_list[t - 1] += o[QUANTITY]
                        self.update_partner_performance(p, True)

            # 当日納期: 部分集合列挙で needs に最も近い組合せを探索
            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))
            for i, partner_ids in enumerate(plist):
                if not partner_ids:
                    continue
                offered = sum(current_step_offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                quality_bonus = sum(
                    self.partner_success_rate[p] for p in partner_ids
                ) / len(partner_ids)
                adjusted_diff = diff * (2.0 - quality_bonus)
                if offered - needs >= 0:
                    if adjusted_diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = adjusted_diff, i
                else:
                    if adjusted_diff < best_minus_diff and offered > 0:
                        best_minus_diff, best_index_minus = adjusted_diff, i

            has_accept = True
            if (
                best_plus_diff
                <= self._threshold * (2.0 if self.aggressive_mode else 1.0)
                and best_index_plus >= 0
                and len(plist[best_index_plus]) > 0
            ):
                best_idx = best_index_plus
            elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
                best_idx = best_index_minus
            else:
                has_accept = False

            # === 2段階受け入れ戦略 ===
            # Phase 1: 全体最適化で needs マッチするパートナーを ACCEPT
            if has_accept and needs > 0:
                partner_ids = plist[best_idx]
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))

                # 全体最適マッチングパートナーは受諾
                response.update(
                    {
                        k: SAOResponse(
                            ResponseType.ACCEPT_OFFER, current_step_offers[k]
                        )
                        for k in partner_ids
                    }
                )
                for k in partner_ids:
                    self.update_partner_performance(k, True)
                for k in others:
                    self.update_partner_performance(k, False)

                # Phase 2: 受諾されなかったパートナーに対しては
                # Layer 3/4 の個別配分ロジックで反提案を作成
                if others:
                    others_s = [x for x in others if self.is_supplier(x)]
                    others_c = [x for x in others if self.is_consumer(x)]

                    # Layer 3: 個別パートナーへの appetite-based allocation
                    if others_s:
                        allocation = self.distribute_todays_needs(others_s)
                        for k, q in allocation.items():
                            if k in response:
                                continue
                            if q > 0:
                                price = self.smart_price(
                                    k,
                                    is_counter_offer=True,
                                    state=states.get(k),
                                )
                                response[k] = SAOResponse(
                                    ResponseType.REJECT_OFFER,
                                    (q, awi.current_step, price),
                                )
                            else:
                                # 現在ステップで量がない場合は未来日
                                future_offers = self.future_supplie_offer([k])
                                for fk, fv in future_offers.items():
                                    response[fk] = SAOResponse(
                                        ResponseType.REJECT_OFFER, fv
                                    )

                    if others_c:
                        allocation = self.distribute_todays_needs(others_c)
                        for k, q in allocation.items():
                            if k in response:
                                continue
                            if q > 0:
                                price = self.smart_price(
                                    k,
                                    is_counter_offer=True,
                                    state=states.get(k),
                                )
                                response[k] = SAOResponse(
                                    ResponseType.REJECT_OFFER,
                                    (q, awi.current_step, price),
                                )
                            else:
                                # 現在ステップで量がない場合は未来日
                                future_offers = self.future_consume_offer([k])
                                for fk, fv in future_offers.items():
                                    response[fk] = SAOResponse(
                                        ResponseType.REJECT_OFFER, fv
                                    )
            else:
                # 受諾候補が無いとき: 残りの相手に対案を出す
                other_partners = {
                    p
                    for p in all_partners
                    if p not in response.keys() and p in self.negotiators.keys()
                }
                distribution = self.distribute_todays_needs(other_partners)
                fs: list[str] = []
                fc: list[str] = []
                for k, q in distribution.items():
                    if q > 0:
                        price = self.smart_price(
                            k, is_counter_offer=True, state=states.get(k)
                        )
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, awi.current_step, price),
                        )
                    elif self.is_supplier(k):
                        fs.append(k)
                    elif self.is_consumer(k):
                        fc.append(k)

                fs_dict = self.future_supplie_offer(fs)
                fc_dict = self.future_consume_offer(fc)
                for k, x in fs_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in fc_dict.items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    # =========
    # Pricing
    # =========

    def _boulware_price(
        self,
        partner: str,
        state: SAOState | None,
        e: float = 0.3,
        urgency_factor: float = 1.0,
    ) -> float | None:
        """Boulware 時間譲歩で提案価格を返す。

        Args:
            partner: 相手 ID
            state: 現在の SAOState (None なら関連時刻 = 0 として最良価格を出す)
            e: 譲歩関数の指数。e<1 で Boulware (粘る), e>1 で Conceder (早く譲歩)
            urgency_factor: 1.0 を中心とした倍率。Aggressive モード時など
                            交渉を早く決めたい場合に > 1.0 を使う
        """
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value

        # 関連時刻 t ∈ [0, 1]
        t = getattr(state, "relative_time", 0.0) if state else 0.0
        try:
            t = float(t)
        except Exception:  # noqa: BLE001
            t = 0.0
        t = max(0.0, min(1.0, t * urgency_factor))

        # Boulware: concession = t^(1/e)。e=0.3 だと終盤までほぼ 0
        concession = t ** (1.0 / max(e, 0.01))

        if self.is_consumer(partner):
            # 売る側: maxp スタート → minp 方向へ譲歩
            price = maxp - concession * (maxp - minp)
        else:
            # 買う側: minp スタート → maxp 方向へ譲歩
            price = minp + concession * (maxp - minp)
        # 範囲内の整数に丸める (scml の price issue は int)
        return int(round(max(minp, min(maxp, price))))

    def smart_price(
        self,
        partner: str,
        is_first_proposal: bool = False,
        is_counter_offer: bool = False,
        state: SAOState | None = None,
    ):
        """AS0 オリジナルの信頼度ベース価格決定 (Boulware は撤回)。

        TODO: round が 1 で終わる現状では時間譲歩は無意味。
              将来的に「相手が REJECT 続けるとき」に発動する別レイヤを検討。
        """
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        success_rate = self.partner_success_rate[partner]

        if self.is_consumer(partner):
            if is_first_proposal:
                return maxp
            base_concession = 0.7
            partner_bonus = success_rate * 0.15
            urgency_bonus = 0.1 if self.aggressive_mode else 0.0
            final_rate = base_concession - partner_bonus - urgency_bonus
            return max(maxp * final_rate, minp)
        else:
            if is_first_proposal:
                return minp
            base_markup = 1.2
            partner_penalty = success_rate * 0.15
            urgency_penalty = 0.1 if self.aggressive_mode else 0.0
            final_rate = base_markup + partner_penalty + urgency_penalty
            return min(minp * final_rate, maxp)

    def is_valid_price(self, price, partner: str) -> bool:
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        if self.is_consumer(partner):
            return price >= minp
        if self.is_supplier(partner):
            return price <= maxp
        return False

    def best_price(self, partner: str):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        pissue = nmi.issues[UNIT_PRICE]
        return pissue.min_value if self.is_supplier(partner) else pissue.max_value

    # =====================================================
    # STEP 1: Adaptive Pricing based on Agreement Rate
    # =====================================================

    def get_agreement_rate(self, partner: str) -> float:
        """相手との合意率を計算 (成功 / 交渉回数)"""
        negotiations = self.partner_negotiations.get(partner, 0)
        if negotiations == 0:
            return 0.5  # デフォルト値（未交渉）
        successes = self.partner_successes.get(partner, 0)
        return successes / negotiations

    def adaptive_price(self, partner: str) -> int | None:
        """Step 1: 合意率に応じて価格を動的に調整"""
        base_price = self.best_price(partner)
        if base_price is None:
            return None

        agreement_rate = self.get_agreement_rate(partner)
        nmi = self.get_nmi(partner)
        if nmi is None:
            return base_price

        pissue = nmi.issues[UNIT_PRICE]
        pmin, pmax = pissue.min_value, pissue.max_value

        # 合意率に応じて価格を調整
        if agreement_rate < 0.3:
            # ほぼ合意しない → 大幅に譲歩（15%）
            adjustment = 0.85
        elif agreement_rate < 0.5:
            # 合意率低い → 少し譲歩（8%）
            adjustment = 0.92
        elif agreement_rate < 0.7:
            # 合意率中程度 → 現在価格維持
            adjustment = 1.0
        else:
            # 合意率高い → さらに有利に（5%）
            adjustment = 1.05

        adjusted_price = int(base_price * adjustment)
        # 価格レンジ内に収める
        adjusted_price = max(pmin, min(pmax, adjusted_price))
        return adjusted_price

    # =====================================================
    # STEP 2: Dynamic Partner Distribution by Needs Size
    # =====================================================

    def select_partners_dynamically(self, needs: int, all_partners: list[str]) -> list[str]:
        """Step 2: needs のサイズに応じてパートナー数を動的に選択"""
        if not all_partners:
            return []

        n_partners = len(all_partners)
        # needs のサイズで分配先を決める
        if needs <= self.awi.n_lines * 0.2:
            # needs が小さい → 最有力相手に絞る（1-2人）
            select_ratio = max(1, int(n_partners * 0.15))
        elif needs <= self.awi.n_lines * 0.5:
            # needs が中程度 → 30%
            select_ratio = max(1, int(n_partners * 0.3))
        else:
            # needs が大きい → 標準的に70%
            select_ratio = max(1, int(n_partners * 0.7))

        # 成功率でソート（高い順）
        sorted_partners = sorted(
            all_partners,
            key=lambda p: self.get_agreement_rate(p),
            reverse=True
        )
        return sorted_partners[:select_ratio]

    # =====================================================
    # STEP 3: Time-Dependent Concession Thresholds
    # =====================================================

    def _allowed_mismatch(self, relative_time: float) -> int:
        """Step 3: 時間依存の許容誤差。CautiousOneShotAgent に基づく。

        Args:
            relative_time: 交渉の相対時刻 [0, 1]

        Returns:
            許容される needs との誤差 (単位数)
        """
        mismatch_exp = 4.0  # CautiousOneShotAgent のデフォルト
        mismatch_max = int(self.awi.n_lines * 0.3)

        # 交渉開始時 (t=0): threshold = 0 (完全一致)
        # 交渉終了時 (t=1): threshold = mismatch_max (大幅譲歩)
        allowed = int(mismatch_max * (relative_time ** mismatch_exp))
        return allowed

    def _overordering_fraction(self, relative_time: float) -> float:
        """Step 3: 時間に応じて over-ordering 率を減少させる。

        初期（t=0）は多めに注文、終盤（t=1）は正確さを重視。
        """
        overordering_exp = 0.4  # CautiousOneShotAgent のデフォルト
        overordering_max = 0.2
        overordering_min = 0.0

        fraction = (
            overordering_max
            - (overordering_max - overordering_min) * (relative_time ** overordering_exp)
        )
        return fraction

    # =====================================================
    # STEP 4: Asymmetric Penalty-Aware Acceptance
    # =====================================================

    def get_disposal_and_shortage_costs(self) -> tuple[float, float]:
        """Step 4: 現在の disposal cost と shortage penalty を取得"""
        disposal_cost = self.awi.current_disposal_cost
        shortage_penalty = self.awi.current_shortfall_penalty
        return disposal_cost, shortage_penalty

    def _allowed_mismatch_asymmetric(self, relative_time: float, is_buying: bool) -> tuple[float, float]:
        """Step 4: Penalty 非対称性を考慮した許容誤差。

        買い手（shortage penalty が高い）と売り手（disposal cost が高い）
        で異なる許容度を返す。

        Args:
            relative_time: 交渉の相対時刻 [0, 1]
            is_buying: 買い手なら True, 売り手なら False

        Returns:
            (min_allowed, max_allowed) - 不足と過剰の許容範囲
        """
        mismatch_exp = 4.0

        if is_buying:
            # 買い手：shortage penalty が高いので、不足は許さない
            # 過剰は許す（一時的に持ってられる）
            overmismatch_max = int(self.awi.n_lines * 0.2)
            undermismatch_min = int(self.awi.n_lines * -0.3)
        else:
            # 売り手：disposal cost が高いので、過剰は許さない
            # 不足は許す（後で調達）
            overmismatch_max = int(self.awi.n_lines * 0.0)
            undermismatch_min = int(self.awi.n_lines * -0.4)

        # 時間に応じて譲歩
        max_allowed = overmismatch_max + (0 - overmismatch_max) * (relative_time ** mismatch_exp)
        min_allowed = undermismatch_min + (0 - undermismatch_min) * (relative_time ** mismatch_exp)

        return min_allowed, max_allowed

    # ===============
    # Demand sizing
    # ===============

    def needs_at(self, step: int, partner: str) -> int:
        awi = self.awi
        # Layer 1: 動的生産目標を使用
        day_production = self.daily_throughput_target()
        if self.is_supplier(partner):
            return int(
                day_production
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
            )
        if self.is_consumer(partner):
            return int(
                max(
                    0,
                    min(awi.n_lines, day_production + awi.current_inventory_input)
                    - awi.total_sales_at(step),
                )
            )
        return 0

    def is_consumer(self, partner: str) -> bool:
        return partner in self.awi.my_consumers

    def is_supplier(self, partner: str) -> bool:
        return partner in self.awi.my_suppliers

    # ===================
    # Layer 1: Market Momentum + Inventory Buffer
    # ===================

    def price_momentum(self, product: int, window: int = 5) -> float:
        """Trading price のモメンタム。

        Returns: -1..1 の範囲。
          - 負: 下落トレンド（安い）
          - 正: 上昇トレンド（高い）
        """
        h = self.tp_history.get(product, deque())
        if len(h) < window:
            return 0.0
        recent_start = max(0, len(h) - window // 2)
        older_start = max(0, len(h) - window)
        recent = sum(list(h)[recent_start:]) / max(1, len(h) - recent_start)
        older = sum(list(h)[older_start : recent_start]) / max(1, recent_start - older_start)
        if older < 1e-6:
            return 0.0
        return (recent - older) / older

    def daily_throughput_target(self) -> int:
        """市況 + 在庫から動的に日次生産目標を調整。

        戦略：
          - Input tp 下落 → 安く買えるから多めに仕入れ（×1.3）
          - Input tp 上昇 → 高いから控えめ（×0.7）
          - 在庫低い → 生産追いつかないからキャッチアップ（×1.3）
          - 在庫多い → Storage cost の負担から売却優先（×0.6）
        """
        awi = self.awi
        base = awi.n_lines * 0.7  # AS0 デフォルト
        inv_ratio = 0.5  # デフォルト値 (try-except で定義されない場合用)

        # Input tp モメンタム
        try:
            input_mom = self.price_momentum(awi.my_input_product)
            if input_mom < -0.02:  # 下落
                base *= 1.3
            elif input_mom > 0.05:  # 上昇
                base *= 0.7
        except Exception:  # noqa: BLE001
            pass

        # 在庫圧力
        try:
            inv_ratio = awi.current_inventory_input / max(1, awi.n_lines)
            if inv_ratio < 0.3:  # 低い
                base *= 1.3
            elif inv_ratio > 0.8:  # 多い
                base *= 0.6
        except Exception:  # noqa: BLE001
            pass

        # Middle layer 特化：外生需要がないので特別処理
        if awi.is_middle_level:
            try:
                input_mom = self.price_momentum(awi.my_input_product)
                # 入力が安い + 在庫が少ない → 積極的に買い増し
                if input_mom < -0.02 and inv_ratio < 0.4:
                    base *= 1.5  # 安い時にバイインする
                # 高在庫は危険なので、より保守的に
                elif inv_ratio > 0.8:
                    base *= 0.5
            except Exception:  # noqa: BLE001
                if inv_ratio > 0.8:
                    base *= 0.5

        return int(max(1, base))

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))

        suppliers = [p for p in partners if self.is_supplier(p)]
        consumers = [p for p in partners if self.is_consumer(p)]

        awi = self.awi
        # Layer 1: 動的生産目標を使用
        day_production = self.daily_throughput_target()
        supply_needs = int(
            day_production
            - awi.current_inventory_input
            - awi.total_supplies_at(awi.current_step)
        )
        consume_needs = int(
            max(
                0,
                min(awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step),
            )
        )

        if suppliers and supply_needs > 0:
            selected = self.select_partners_by_performance(suppliers)
            response |= self._distribute_to_partners(selected, supply_needs)

        if (
            consumers
            and consume_needs > 0
            and awi.total_sales_at(awi.current_step) <= awi.n_lines
        ):
            selected = self.select_partners_by_performance(consumers)
            response |= self._distribute_to_partners(selected, consume_needs)

        return response

    def _distribute_to_partners(self, partners: list[str], needs: int) -> dict[str, int]:
        """AS0 オリジナルの数量配分 (ランダムシャッフル)。Layer 3 は撤回。"""
        response = dict(zip(partners, repeat(0)))
        if not partners or needs <= 0:
            return response

        partners = list(partners)
        random.shuffle(partners)
        ptoday = self.get_effective_ptoday()
        selected = partners[: max(1, int(ptoday * len(partners)))]
        n = len(selected)
        # needs < n の時に下流配分の多様性を加える (ランダム削減)
        # ただし最低でも必要数を満たせるだけの相手は確保
        if needs < n and n > 1:
            # 過剰なパートナーを削減。ただし needs の数だけは確保
            min_partners = max(1, min(needs, n))
            target = random.randint(min_partners, n)
            selected = random.sample(selected, target)
            n = len(selected)
        if n > 0:
            response |= dict(zip(selected, distribute(needs, n)))
        return response

    # ==================
    # Forward contracts
    # ==================

    def future_supplie_offer(self, partner_list: list[str]) -> dict[str, Outcome]:
        return self._forward_contract_offers(partner_list, is_supply=True)

    def future_consume_offer(self, partner_list: list[str]) -> dict[str, Outcome]:
        return self._forward_contract_offers(partner_list, is_supply=False)

    def _forward_contract_offers(
        self, partner_list: list[str], is_supply: bool
    ) -> dict[str, Outcome]:
        """50% / 30% / 20% でパートナーを t+1, t+2, t+3 に振り分ける Forward Contracting。"""
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response: dict[str, Outcome] = {}
        if not partner_list:
            return response

        sorted_partners = self.select_partners_by_performance(partner_list, ratio=1.0)
        step1 = sorted_partners[: int(len(sorted_partners) * 0.5)]
        step2 = sorted_partners[
            int(len(sorted_partners) * 0.5) : int(len(sorted_partners) * 0.8)
        ]
        step3 = sorted_partners[int(len(sorted_partners) * 0.8) :]

        # Layer 1: 動的生産目標を使用
        p_target = self.daily_throughput_target()

        for offset, partners in [(1, step1), (2, step2), (3, step3)]:
            t = s + offset
            if t >= n or not partners:
                continue
            if is_supply:
                step_needs = int(
                    max(
                        0,
                        (p_target - awi.current_inventory_input - awi.total_supplies_at(t))
                        / 3,
                    )
                )
            else:
                if awi.total_sales_at(t) > awi.n_lines:
                    continue
                step_needs = int(
                    max(
                        0,
                        min(awi.n_lines, p_target + awi.current_inventory_input)
                        - awi.total_sales_at(t),
                    )
                    / 3
                )
            if step_needs <= 0:
                continue
            dist = dict(zip(partners, distribute(step_needs, len(partners))))
            for k, q in dist.items():
                if q > 0:
                    response[k] = (q, t, self.best_price(k))
        return response

    # =============================
    # Negotiation result callbacks
    # =============================

    def on_negotiation_failure(  # type: ignore[override]
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """交渉失敗を partner_log に記録。"""
        other = next((p for p in partners if p != self.id), None)
        if other is None:
            return
        rounds_used = getattr(state, "step", 0) or 0
        rounds_total = getattr(mechanism, "n_steps", 0) or 0
        self.partner_log[other].append(
            NegoRecord(
                day=self.awi.current_step,
                is_consumer=other in self.awi.my_consumers,
                success=False,
                rounds_used=int(rounds_used),
                rounds_total=int(rounds_total),
            )
        )

    def on_negotiation_success(  # type: ignore[override]
        self, contract: Contract, mechanism: StdAWI
    ) -> None:
        """交渉成功 (= 契約成立) を partner_log に記録。"""
        partners = list(getattr(contract, "partners", []))
        other = next((p for p in partners if p != self.id), None)
        if other is None:
            return
        agreement = getattr(contract, "agreement", None) or {}
        nmi_state = getattr(mechanism, "state", None)
        rounds_used = getattr(nmi_state, "step", 0) if nmi_state is not None else 0
        rounds_total = getattr(mechanism, "n_steps", 0) or 0
        self.partner_log[other].append(
            NegoRecord(
                day=self.awi.current_step,
                is_consumer=other in self.awi.my_consumers,
                success=True,
                rounds_used=int(rounds_used),
                rounds_total=int(rounds_total),
                agreed_q=int(agreement.get("quantity", 0)),
                agreed_p=float(agreement.get("unit_price", 0.0)),
                agreed_t=int(agreement.get("time", 0)),
            )
        )


# ============================================================
# 提出エージェント SBD (LeanAgentV2 + BO tuned params)
#   - A0:     V5 safe-target/value-accept after 25% of the game
#   - middle: AS0 default に委譲 (_AS0Base 経由)
#   - A_last: Penguin-style delegation
# tuned params 由来: data/best_params_production_20260608_111653.json
# A0 V5 採用メモ:
#   data/a0_v5_combined60_20260614.summary.csv で SBD を上回った
#   v5_ideal25_sw1 を提出用に移植。戻す場合は _A0_V5_ENABLED = False。
# ============================================================
from .lean_agent_v2 import LeanAgentV2 as _LeanAgentV2  # noqa: E402
from .penguinagent import PenguinAgent as _PenguinAgentBase  # noqa: E402

_TUNED_CELL_OVERRIDES = {
    ('A0', 'supply_surplus'): {'needs_mult': 1.49, 'accept_th': 1,  'future_mult': 0.92},
    ('A0', 'balanced'):       {'needs_mult': 1.35, 'accept_th': 9,  'future_mult': 1.96},
    ('A0', 'demand_surplus'): {'needs_mult': 0.80, 'accept_th': 10, 'future_mult': 1.63},
}


class SBD(_LeanAgentV2):
    """ANAC 2026 SCML Std 提出エージェント."""

    _A0_V5_ENABLED = True
    _A0_V5_SWITCH_FRACTION = 0.25
    _A0_V5_ACCEPT_STORAGE_WEIGHT = 1.0
    _A0_V5_ACCEPT_IDEAL_FIRST = True
    _A0_V5_ACCEPT_MIN_GAIN = 0.0
    _A0_V5_UFUN_STORAGE_CANDIDATES = 5
    _A0_V5_UFUN_SHORTFALL_CANDIDATES = 5
    _A0_EARLY_H7_ENABLED = True
    _A0_EARLY_H7_HORIZON = 7
    _A0_EARLY_H7_Q = 2
    _A0_EARLY_H7_UNTIL_FRACTION = 0.18
    _A0_LATE_BROAD_SELL_ENABLED = True
    _A0_LATE_BROAD_SELL_FRACTION = 0.50
    # Final-tournament candidates kept as switches:
    #   A_last: ratchet own observed exogenous demand instead of jumping to 0.9*n_lines.
    #   middle: last_mid09 uses a fixed 0.9*n_lines stock cap with safe selling.
    _A_LAST_HYB_ENABLED = True
    _A_LAST_RATCHET_BUFFER = 1.0
    _A_LAST_RATCHET_FLOOR = 2.0
    _A_LAST_RATCHET_M4_FLOOR_FACTOR = 0.3
    _MIDDLE_LOAD_CAP_ENABLED = True
    _MIDDLE_LOW_CAP_FACTOR = 0.9
    _MIDDLE_HIGH_CAP_FACTOR = 0.9
    _MIDDLE_LOAD_THRESHOLD = 0.8
    # SCML 0.8.x can generate four-process standard worlds where the L1-L2
    # negotiation edge is effectively broken.  In that case L1 can buy from L0
    # but cannot reliably sell to L2, so carrying stock there is mostly storage
    # loss.  Keep this as a switch so the submission can be reverted quickly if
    # the environment is fixed.
    _M4_EDGE_GUARD_ENABLED = True
    _M4_LEVEL1_CAP_FACTOR = 0.0
    _M4_LEVEL2_CAP_FACTOR = 0.9

    def __init__(self, *args, **kwargs):
        self._a0_v5_enabled = bool(
            kwargs.pop("a0_v5_enabled", self._A0_V5_ENABLED)
        )
        self._a0_v5_switch_fraction = float(
            kwargs.pop("a0_v5_switch_fraction", self._A0_V5_SWITCH_FRACTION)
        )
        self._a0_v5_accept_storage_weight = max(
            0.0,
            float(
                kwargs.pop(
                    "a0_v5_accept_storage_weight",
                    self._A0_V5_ACCEPT_STORAGE_WEIGHT,
                )
            ),
        )
        self._a0_v5_accept_ideal_first = bool(
            kwargs.pop(
                "a0_v5_accept_ideal_first",
                self._A0_V5_ACCEPT_IDEAL_FIRST,
            )
        )
        self._a0_v5_accept_min_gain = float(
            kwargs.pop("a0_v5_accept_min_gain", self._A0_V5_ACCEPT_MIN_GAIN)
        )
        self._a0_early_h7_enabled = bool(
            kwargs.pop("a0_early_h7_enabled", self._A0_EARLY_H7_ENABLED)
        )
        self._a0_late_broad_sell_enabled = bool(
            kwargs.pop(
                "a0_late_broad_sell_enabled",
                self._A0_LATE_BROAD_SELL_ENABLED,
            )
        )
        self._a_last_hyb_enabled = bool(
            kwargs.pop("a_last_hyb_enabled", self._A_LAST_HYB_ENABLED)
        )
        self._middle_load_cap_enabled = bool(
            kwargs.pop("middle_load_cap_enabled", self._MIDDLE_LOAD_CAP_ENABLED)
        )
        kwargs.setdefault('productivity', 0.84)
        kwargs.setdefault('partner_alpha', 0.44)
        kwargs.setdefault('forward_horizon', 2)
        kwargs.setdefault('cell_overrides', _TUNED_CELL_OVERRIDES)
        self._a_last_ratchet_max_seen_exo = 0.0
        super().__init__(*args, **kwargs)
        self._ptoday = 0.70

    def before_step(self):
        super().before_step()
        self._a_last_note_current_exogenous()

    def _use_penguin_last(self) -> bool:
        try:
            return bool(self.awi.is_last_level)
        except Exception:
            return False

    def _use_a_last_hyb(self) -> bool:
        return bool(self._a_last_hyb_enabled) and self._use_penguin_last()

    def _a_last_n_agents(self) -> int:
        try:
            return max(1, int(self.awi.n_competitors) + 1)
        except Exception:
            try:
                return max(1, len(self.awi.my_competitors) + 1)
            except Exception:
                return 1

    def _a_last_market_exo_per_agent(self) -> float:
        try:
            summary = self.awi.exogenous_contract_summary
            if summary:
                return float(summary[-1][0]) / float(self._a_last_n_agents())
        except Exception:
            pass
        return float(self.awi.current_exogenous_output_quantity or 0)

    def _a_last_own_exo(self, step: int) -> float:
        if int(step) == int(self.awi.current_step):
            return float(self.awi.current_exogenous_output_quantity or 0)
        return self._a_last_market_exo_per_agent()

    def _a_last_note_current_exogenous(self) -> None:
        try:
            if not self.awi.is_last_level:
                return
            current = float(self.awi.current_exogenous_output_quantity or 0)
        except Exception:
            return
        previous = float(getattr(self, "_a_last_ratchet_max_seen_exo", 0.0))
        self._a_last_ratchet_max_seen_exo = max(previous, current)

    def _a_last_ratchet_floor(self) -> float:
        try:
            if int(self.awi.n_products) == 5:
                return float(self.awi.n_lines) * float(
                    self._A_LAST_RATCHET_M4_FLOOR_FACTOR
                )
        except Exception:
            pass
        return float(self._A_LAST_RATCHET_FLOOR)

    def _a_last_target_at(self, step: int) -> int:
        self._a_last_note_current_exogenous()
        n_lines = int(self.awi.n_lines)
        max_seen = float(getattr(self, "_a_last_ratchet_max_seen_exo", 0.0))
        base = max(
            self._a_last_ratchet_floor(),
            max_seen + float(self._A_LAST_RATCHET_BUFFER),
        )
        return max(0, min(n_lines, int(math.ceil(base))))

    def _a_last_supply_need_at(self, step: int) -> int:
        awi = self.awi
        return int(
            self._a_last_target_at(step)
            - int(awi.current_inventory_input or 0)
            - int(awi.total_supplies_at(step))
        )

    def _call_penguin_last(self, fn, *args, **kwargs):
        old_productivity = getattr(self, "_productivity", 0.7)
        old_threshold = getattr(self, "_threshold", 1)
        old_ptoday = getattr(self, "_ptoday", 0.70)
        try:
            if self._use_a_last_hyb():
                target = self._a_last_target_at(int(self.awi.current_step))
                self._productivity = target / max(1.0, float(self.awi.n_lines))
            else:
                self._productivity = 0.7
            self._ptoday = 0.70
            try:
                self._threshold = self.awi.n_lines * 0.1
            except Exception:
                self._threshold = 1
            return fn(self, *args, **kwargs)
        finally:
            self._productivity = old_productivity
            self._threshold = old_threshold
            self._ptoday = old_ptoday

    def step(self):
        super().step()
        if self._use_penguin_last():
            self._threshold = self.awi.n_lines * 0.1

    def _a0_v5_use_safe_target(self) -> bool:
        if not self._a0_v5_enabled:
            return False
        try:
            if not self.awi.is_first_level:
                return False
            switch_step = int(
                round(int(self.awi.n_steps) * self._a0_v5_switch_fraction)
            )
            return int(self.awi.current_step) >= switch_step
        except Exception:
            return False

    def _use_a0_early_h7(self) -> bool:
        try:
            if not bool(self._a0_early_h7_enabled) or not self.awi.is_first_level:
                return False
            step = int(self.awi.current_step)
            n_steps = max(1, int(self.awi.n_steps))
            return step < int(round(n_steps * float(self._A0_EARLY_H7_UNTIL_FRACTION)))
        except Exception:
            return False

    def _use_a0_late_broad_sell(self) -> bool:
        try:
            return (
                bool(self._a0_late_broad_sell_enabled)
                and bool(self.awi.is_first_level)
                and int(self.awi.current_step)
                > int(round(int(self.awi.n_steps) * float(self._A0_LATE_BROAD_SELL_FRACTION)))
            )
        except Exception:
            return False

    def _lean_needs(self) -> tuple[int, int]:
        if not self._a0_v5_use_safe_target():
            return super()._lean_needs()

        awi = self.awi
        s = int(awi.current_step)
        exogenous = int(awi.current_exogenous_input_quantity or 0)
        inventory = int(awi.current_inventory_input or 0)
        signed_sales = int(awi.total_sales_at(s))
        target = min(int(awi.n_lines), exogenous + inventory)
        return 0, max(0, target - signed_sales)

    def _use_middle_load_cap(self) -> bool:
        try:
            return bool(self._middle_load_cap_enabled) and bool(self.awi.is_middle_level)
        except Exception:
            return False

    def _use_m4_edge_guard(self) -> bool:
        try:
            return (
                bool(self._M4_EDGE_GUARD_ENABLED)
                and bool(self.awi.is_middle_level)
                and int(self.awi.n_products) == 5
            )
        except Exception:
            return False

    def _m4_middle_cap_factor(self) -> float | None:
        if not self._use_m4_edge_guard():
            return None
        level = int(self.awi.my_input_product)
        if level == 1:
            return float(self._M4_LEVEL1_CAP_FACTOR)
        if level == 2:
            return float(self._M4_LEVEL2_CAP_FACTOR)
        return None

    def _middle_n_agents(self) -> int:
        try:
            return max(1, int(self.awi.n_competitors) + 1)
        except Exception:
            try:
                return max(1, len(self.awi.my_competitors) + 1)
            except Exception:
                return 1

    def _middle_current_downstream_load(self) -> float:
        try:
            summary = self.awi.exogenous_contract_summary
            if not summary:
                return 0.0
            total = float(summary[-1][0])
            per_middle = total / float(self._middle_n_agents())
            return per_middle / max(1.0, float(self.awi.n_lines))
        except Exception:
            return 0.0

    def _middle_stock_cap(self) -> int:
        n_lines = int(self.awi.n_lines)
        factor = self._m4_middle_cap_factor()
        if factor is None:
            factor = (
                self._MIDDLE_HIGH_CAP_FACTOR
                if self._middle_current_downstream_load()
                >= float(self._MIDDLE_LOAD_THRESHOLD)
                else self._MIDDLE_LOW_CAP_FACTOR
            )
        return max(0, min(n_lines, int(math.ceil(float(n_lines) * factor))))

    def _middle_supply_need_at(self, step: int) -> int:
        return int(
            self._middle_stock_cap()
            - int(self.awi.current_inventory_input or 0)
            - int(self.awi.total_supplies_at(step))
        )

    def _middle_sell_target_at(self, step: int) -> int:
        available = int(self.awi.current_inventory_input or 0) + int(
            self.awi.total_supplies_at(step)
        )
        return max(0, min(int(self.awi.n_lines), available))

    def _middle_consume_need_at(self, step: int) -> int:
        return int(
            max(
                0,
                self._middle_sell_target_at(step) - int(self.awi.total_sales_at(step)),
            )
        )

    def _a0_late_broad_sell_distribution(self, consumers: list[str], total_q: int) -> dict[str, int]:
        response = dict(zip(consumers, repeat(0)))
        if not consumers or total_q <= 0:
            return response
        ordered = self.select_partners_by_performance(consumers, ratio=1.0)
        safe_q = min(int(total_q), int(self.awi.n_lines))
        response |= dict(zip(ordered, distribute(safe_q, len(ordered))))
        return response

    def first_proposals(self):
        if self._use_penguin_last():
            return self._call_penguin_last(_PenguinAgentBase.first_proposals)
        return super().first_proposals()

    def counter_all(self, offers, states):
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.counter_all, offers, states
            )
        if self._a0_v5_use_safe_target():
            return self._a0_v5_counter_all(offers, states)
        if self._use_middle_load_cap():
            return self._middle_load_counter_all(offers, states)
        return super().counter_all(offers, states)

    def _middle_load_counter_all(self, offers, states):
        response: dict[str, SAOResponse] = {}
        awi = self.awi
        s = int(awi.current_step)

        for needs, all_partners in [
            (max(0, self._middle_supply_need_at(s)), awi.my_suppliers),
            (max(0, self._middle_consume_need_at(s)), awi.my_consumers),
        ]:
            partners = {p for p in all_partners if p in offers}
            current_step_offers: dict[str, Any] = {}
            future_step_offers: dict[str, Any] = {}

            for p in partners:
                offer = offers[p]
                if offer is None:
                    continue
                if not self.is_valid_price(offer[UNIT_PRICE], p):
                    continue
                if int(offer[TIME]) == s:
                    current_step_offers[p] = offer
                else:
                    future_step_offers[p] = offer

            duplicate_list = [0] * int(awi.n_steps)
            for p, offer in future_step_offers.items():
                t = int(offer[TIME])
                idx = t - 1
                if 0 <= idx < len(duplicate_list):
                    future_needs = max(0, int(self.needs_at(t, p)))
                    if int(offer[QUANTITY]) + duplicate_list[idx] <= future_needs:
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                        duplicate_list[idx] += int(offer[QUANTITY])
                        self.update_partner_performance(p, True)

            current_step_partners = set(current_step_offers.keys())
            best_index_plus = -1
            best_plus_diff = float("inf")
            best_index_minus = -1
            best_minus_diff = float("inf")

            plist = list(powerset(current_step_partners))
            for i, partner_ids in enumerate(plist):
                if not partner_ids:
                    continue
                offered = sum(int(current_step_offers[p][QUANTITY]) for p in partner_ids)
                diff = abs(offered - needs)
                quality_bonus = sum(self.partner_success_rate[p] for p in partner_ids) / len(
                    partner_ids
                )
                adjusted_diff = diff * (2.0 - quality_bonus)
                if offered - needs >= 0:
                    if adjusted_diff < best_plus_diff and needs > 0:
                        best_plus_diff, best_index_plus = adjusted_diff, i
                elif offered > 0 and adjusted_diff < best_minus_diff:
                    best_minus_diff, best_index_minus = adjusted_diff, i

            has_accept = True
            if (
                best_plus_diff
                <= self._threshold * (2.0 if self.aggressive_mode else 1.0)
                and best_index_plus >= 0
                and len(plist[best_index_plus]) > 0
            ):
                best_idx = best_index_plus
            elif best_index_minus >= 0 and len(plist[best_index_minus]) > 0:
                best_idx = best_index_minus
            else:
                has_accept = False
                best_idx = -1

            if has_accept and needs > 0:
                partner_ids = plist[best_idx]
                response.update(
                    {
                        p: SAOResponse(ResponseType.ACCEPT_OFFER, current_step_offers[p])
                        for p in partner_ids
                    }
                )
                for p in partner_ids:
                    self.update_partner_performance(p, True)
                others = [
                    p
                    for p in current_step_partners
                    if p not in partner_ids and p not in response
                ]
                for p in others:
                    self.update_partner_performance(p, False)
                others_s = [p for p in others if self.is_supplier(p)]
                others_c = [p for p in others if self.is_consumer(p)]
                for p, offer in self.future_supplie_offer(others_s).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                for p, offer in self.future_consume_offer(others_c).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                continue

            other_partners = {
                p for p in all_partners if p not in response and p in self.negotiators
            }
            distribution = self.distribute_todays_needs(other_partners)
            future_s: list[str] = []
            future_c: list[str] = []
            for p, qty in distribution.items():
                if qty > 0:
                    response[p] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (
                            int(qty),
                            s,
                            self.smart_price(
                                p,
                                is_counter_offer=True,
                                state=states.get(p),
                            ),
                        ),
                    )
                elif self.is_supplier(p):
                    future_s.append(p)
                elif self.is_consumer(p):
                    future_c.append(p)
            for p, offer in self.future_supplie_offer(future_s).items():
                response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            for p, offer in self.future_consume_offer(future_c).items():
                response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        return response

    def _a0_v5_counter_all(self, offers, states):
        response: dict[str, SAOResponse] = {}
        awi = self.awi
        s = int(awi.current_step)
        supply_needs, consume_needs = self._lean_needs()

        for needs, all_partners, is_buying in [
            (supply_needs, awi.my_suppliers, True),
            (consume_needs, awi.my_consumers, False),
        ]:
            partners = {p for p in all_partners if p in offers}
            current_step_offers: dict[str, Any] = {}
            future_step_offers: dict[str, Any] = {}
            for p in partners:
                offer = offers[p]
                if offer is None or not self.is_valid_price(offer[UNIT_PRICE], p):
                    continue
                if int(offer[TIME]) == s:
                    current_step_offers[p] = offer
                else:
                    future_step_offers[p] = offer

            duplicate_list = [0] * int(awi.n_steps)
            for p, offer in future_step_offers.items():
                step = int(offer[TIME])
                if step <= awi.n_steps:
                    if (
                        int(offer[QUANTITY]) + duplicate_list[step - 1]
                        <= self.needs_at(step, p)
                    ):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                        duplicate_list[step - 1] += int(offer[QUANTITY])

            partner_ids: tuple[str, ...] = tuple()
            if needs > 0 and current_step_offers:
                if is_buying:
                    partner_ids = self._a0_v5_sbd_quantity_subset(
                        current_step_offers, needs
                    )
                else:
                    partner_ids = self._a0_v5_value_subset(current_step_offers)

            if partner_ids and needs > 0:
                response.update(
                    {
                        p: SAOResponse(ResponseType.ACCEPT_OFFER, current_step_offers[p])
                        for p in partner_ids
                    }
                )
                others = [
                    p
                    for p in current_step_offers
                    if p not in partner_ids and p not in response
                ]
                others_s = [p for p in others if self.is_supplier(p)]
                others_c = [p for p in others if self.is_consumer(p)]
                for p, offer in self.future_supplie_offer(others_s).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                for p, offer in self.future_consume_offer(others_c).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                continue

            other_partners = {
                p for p in all_partners if p not in response and p in self.negotiators
            }
            if needs > 0:
                distribution = self.distribute_todays_needs(other_partners)
                future_s, future_c = [], []
                for p, qty in distribution.items():
                    if qty > 0:
                        response[p] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (int(qty), s, self.price(p)),
                        )
                    elif self.is_supplier(p):
                        future_s.append(p)
                    elif self.is_consumer(p):
                        future_c.append(p)
                for p, offer in self.future_supplie_offer(future_s).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                for p, offer in self.future_consume_offer(future_c).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            else:
                future_s = [p for p in other_partners if self.is_supplier(p)]
                future_c = [p for p in other_partners if self.is_consumer(p)]
                for p, offer in self.future_supplie_offer(future_s).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                for p, offer in self.future_consume_offer(future_c).items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        return response

    def _a0_v5_sbd_quantity_subset(
        self, current_step_offers: dict[str, Any], needs: int
    ) -> tuple[str, ...]:
        best_plus: tuple[str, ...] = tuple()
        best_plus_diff = float("inf")
        best_minus: tuple[str, ...] = tuple()
        best_minus_diff = float("inf")
        for partner_ids in powerset(current_step_offers.keys()):
            offered = sum(int(current_step_offers[p][QUANTITY]) for p in partner_ids)
            diff = abs(offered - needs)
            if offered - needs >= 0:
                if diff < best_plus_diff and needs > 0:
                    best_plus_diff = diff
                    best_plus = tuple(partner_ids)
            elif diff < best_minus_diff and offered > 0:
                best_minus_diff = diff
                best_minus = tuple(partner_ids)
        if best_plus and best_plus_diff <= self._threshold:
            return best_plus
        return best_minus

    def _a0_v5_value_subset(
        self, current_step_offers: dict[str, Any]
    ) -> tuple[str, ...]:
        target_total = self._a0_v5_safe_target_today()
        exogenous = int(self.awi.current_exogenous_input_quantity or 0)
        signed_sales = int(self.awi.total_sales_at(self.awi.current_step))
        executable_new_sales = max(0, int(target_total) - signed_sales)

        best_ideal: tuple[str, ...] = tuple()
        best_ideal_key: tuple[float, int, int] | None = None
        found_ideal = False
        storage_rows: list[tuple[int, float, int, tuple[str, ...]]] = []
        shortfall_rows: list[tuple[int, float, float, int, tuple[str, ...]]] = []

        def subset_qty(partner_ids: tuple[str, ...]) -> int:
            return sum(int(current_step_offers[p][QUANTITY]) for p in partner_ids)

        def subset_revenue(partner_ids: tuple[str, ...]) -> float:
            return sum(
                int(current_step_offers[p][QUANTITY])
                * float(current_step_offers[p][UNIT_PRICE])
                for p in partner_ids
            )

        def executable_revenue(partner_ids: tuple[str, ...]) -> float:
            remaining = executable_new_sales
            total = 0.0
            for p in sorted(
                partner_ids,
                key=lambda x: float(current_step_offers[x][UNIT_PRICE]),
                reverse=True,
            ):
                if remaining <= 0:
                    break
                q = int(current_step_offers[p][QUANTITY])
                take = min(q, remaining)
                total += float(take) * float(current_step_offers[p][UNIT_PRICE])
                remaining -= take
            return total

        for partner_ids in powerset(current_step_offers.keys()):
            partner_ids = tuple(partner_ids)
            offered = subset_qty(partner_ids)
            total_offered = signed_sales + offered
            revenue = subset_revenue(partner_ids)
            if exogenous <= total_offered <= int(target_total):
                ideal_key = (
                    -float(revenue),
                    abs(int(target_total) - total_offered),
                    len(partner_ids),
                )
                if best_ideal_key is None or ideal_key < best_ideal_key:
                    best_ideal_key = ideal_key
                    best_ideal = partner_ids
                    found_ideal = True
                continue
            storage_gap = exogenous - total_offered
            shortfall_gap = total_offered - int(target_total)
            if storage_gap > 0:
                storage_rows.append(
                    (storage_gap, -float(revenue), len(partner_ids), partner_ids)
                )
            elif shortfall_gap > 0:
                shortfall_rows.append(
                    (
                        shortfall_gap,
                        -float(executable_revenue(partner_ids)),
                        -float(revenue),
                        len(partner_ids),
                        partner_ids,
                    )
                )

        if self._a0_v5_accept_ideal_first and found_ideal:
            return best_ideal

        storage_rows.sort()
        shortfall_rows.sort()
        candidate_subsets = [
            row[3] for row in storage_rows[: int(self._A0_V5_UFUN_STORAGE_CANDIDATES)]
        ]
        candidate_subsets.extend(
            row[4]
            for row in shortfall_rows[: int(self._A0_V5_UFUN_SHORTFALL_CANDIDATES)]
        )

        empty_value, _ = self._a0_v5_adjusted_ufun_value({})
        best_any: tuple[str, ...] = tuple()
        best_any_key: tuple[float, int, int, int, float] | None = None
        best_any_value = float("-inf")

        for partner_ids in candidate_subsets:
            offer_map = {p: current_step_offers[p] for p in partner_ids}
            value, info = self._a0_v5_adjusted_ufun_value(offer_map)
            offered = subset_qty(partner_ids)
            total_offered = signed_sales + offered
            revenue = subset_revenue(partner_ids)
            shortfall = int(getattr(info, "shortfall_quantity", 0) or 0)
            added_inventory = self._a0_v5_added_inventory_from_info(info)
            any_key = (
                -float(value),
                shortfall,
                added_inventory,
                abs(int(target_total) - total_offered),
                -float(revenue),
            )
            if best_any_key is None or any_key < best_any_key:
                best_any_key = any_key
                best_any = tuple(partner_ids)
                best_any_value = float(value)

        min_gain = self._a0_v5_accept_min_gain
        if best_any and best_any_value >= empty_value + min_gain:
            return best_any
        return tuple()

    def _a0_v5_safe_target_today(self) -> int:
        awi = self.awi
        return min(
            int(awi.n_lines),
            int(awi.current_exogenous_input_quantity or 0)
            + int(awi.current_inventory_input or 0),
        )

    def _a0_v5_adjusted_ufun_value(self, offers: dict[str, Any]) -> tuple[float, Any]:
        info = self.ufun.from_offers(
            offers,
            return_info=True,
            ignore_signed_contracts=False,
        )
        remaining_steps = max(0, int(self.awi.n_steps) - int(self.awi.current_step) - 1)
        added_inventory = self._a0_v5_added_inventory_from_info(info)
        future_storage_cost = (
            added_inventory
            * self._a0_v5_input_storage_unit_cost()
            * remaining_steps
            * self._a0_v5_accept_storage_weight
        )
        return float(info.utility) - future_storage_cost, info

    def _a0_v5_added_inventory_from_info(self, info: Any) -> int:
        producible = int(getattr(info, "producible", 0) or 0)
        exogenous = int(self.awi.current_exogenous_input_quantity or 0)
        return max(0, exogenous - producible)

    def _a0_v5_input_storage_unit_cost(self) -> float:
        try:
            scale = float(self.awi.penalty_multiplier(True, None))
        except Exception:
            scale = 1.0
        return float(self.awi.current_storage_cost or 0.0) * scale

    def distribute_todays_needs(self, partners=None):
        if self._use_a_last_hyb():
            if partners is None:
                partners = self.negotiators.keys()
            partners = list(partners)
            response = dict(zip(partners, repeat(0)))
            suppliers = [p for p in partners if self.is_supplier(p)]
            consumers = [p for p in partners if self.is_consumer(p)]

            supply_needs = self._a_last_supply_need_at(int(self.awi.current_step))
            if suppliers and supply_needs > 0:
                response |= self.distribute_todays_supplie_consume_needs(
                    suppliers, supply_needs
                )

            consume_needs = (
                self.needs_at(int(self.awi.current_step), consumers[0])
                if consumers
                else 0
            )
            if (
                consumers
                and consume_needs > 0
                and self.awi.total_sales_at(self.awi.current_step) <= self.awi.n_lines
            ):
                response |= self.distribute_todays_supplie_consume_needs(
                    consumers, consume_needs
                )
            return response
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.distribute_todays_needs, partners
            )
        if self._use_middle_load_cap():
            if partners is None:
                partners = self.negotiators.keys()
            partners = list(partners)
            response = dict(zip(partners, repeat(0)))
            suppliers = [p for p in partners if self.is_supplier(p)]
            consumers = [p for p in partners if self.is_consumer(p)]
            s = int(self.awi.current_step)
            supply_needs = max(0, self._middle_supply_need_at(s))
            consume_needs = max(0, self._middle_consume_need_at(s))
            if suppliers and supply_needs > 0:
                selected = self.select_partners_by_performance(suppliers)
                response |= self._distribute_to_partners(selected, supply_needs)
            if consumers and consume_needs > 0:
                selected = self.select_partners_by_performance(consumers)
                response |= self._distribute_to_partners(selected, consume_needs)
            return response
        if self._use_a0_late_broad_sell():
            if partners is None:
                partners = self.negotiators.keys()
            partners = list(partners)
            response = dict(zip(partners, repeat(0)))
            suppliers = [p for p in partners if self.is_supplier(p)]
            consumers = [p for p in partners if self.is_consumer(p)]
            if suppliers:
                response |= super().distribute_todays_needs(suppliers)
            if consumers:
                _supply_needs, consume_needs = self._lean_needs()
                response |= self._a0_late_broad_sell_distribution(
                    consumers, max(0, int(consume_needs))
                )
            return response
        return super().distribute_todays_needs(partners)

    def distribute_todays_supplie_consume_needs(self, partners, needs):
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.distribute_todays_supplie_consume_needs,
                partners,
                needs,
            )
        return self._distribute_to_partners(partners, needs)

    def future_supplie_offer(self, partner_list):
        if self._use_a_last_hyb():
            awi = self.awi
            s = int(awi.current_step)
            n = int(awi.n_steps)
            partners = list(partner_list)
            response: dict[str, Outcome] = {}
            chunks = [
                partners[: int(len(partners) * 0.5)],
                partners[int(len(partners) * 0.5) : int(len(partners) * 0.8)],
                partners[int(len(partners) * 0.8) :],
            ]
            for offset, chunk in enumerate(chunks, start=1):
                future_step = s + offset
                if future_step >= n or not chunk:
                    continue
                needs = int(self._a_last_supply_need_at(future_step) / 3)
                if needs <= 0:
                    continue
                allocation = dict(zip(chunk, distribute(needs, len(chunk))))
                for partner, qty in allocation.items():
                    if qty > 0:
                        response[partner] = (
                            int(qty),
                            future_step,
                            self.best_price(partner),
                        )
            return response
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.future_supplie_offer, partner_list
            )
        if self._use_middle_load_cap():
            return self._middle_future_offers(partner_list, is_supply=True)
        return super().future_supplie_offer(partner_list)

    def future_consume_offer(self, partner_list):
        if self._use_a0_early_h7():
            try:
                awi = self.awi
                response = dict(super().future_consume_offer(partner_list))
                s = int(awi.current_step)
                t = s + int(self._A0_EARLY_H7_HORIZON)
                if t >= int(awi.n_steps):
                    return response
                q = max(1, int(self._A0_EARLY_H7_Q))
                for p in partner_list:
                    if p in response or not self.is_consumer(p):
                        continue
                    response[p] = (q, t, self.best_price(p))
                return response
            except Exception:
                return super().future_consume_offer(partner_list)
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.future_consume_offer, partner_list
            )
        if self._use_middle_load_cap():
            return self._middle_future_offers(partner_list, is_supply=False)
        return super().future_consume_offer(partner_list)

    def _middle_future_offers(self, partner_list: list[str], is_supply: bool) -> dict[str, Outcome]:
        partners = list(partner_list)
        if not partners:
            return {}
        awi = self.awi
        s = int(awi.current_step)
        n = int(awi.n_steps)
        horizon = min(2, max(0, n - s - 1))
        if horizon <= 0:
            return {}

        if horizon == 1:
            chunks = [partners]
        else:
            split = max(1, len(partners) // 2)
            chunks = [partners[:split], partners[split:]]

        response: dict[str, Outcome] = {}
        for offset, chunk in enumerate(chunks, start=1):
            t = s + offset
            if t >= n or not chunk:
                continue
            needs = (
                max(0, self._middle_supply_need_at(t))
                if is_supply
                else max(0, self._middle_consume_need_at(t))
            )
            needs = int(needs / horizon)
            if needs <= 0:
                continue
            allocation = self._weighted_distribute_quantities(needs, chunk, is_supply)
            for partner, qty in zip(chunk, allocation):
                if qty > 0:
                    response[partner] = (int(qty), t, self.best_price(partner))
        return response

    def needs_at(self, step: int, partner: str) -> int:
        if self._use_a_last_hyb():
            awi = self.awi
            if self.is_supplier(partner):
                return self._a_last_supply_need_at(step)
            if self.is_consumer(partner):
                target = self._a_last_target_at(step)
                return int(
                    max(
                        0,
                        min(
                            int(awi.n_lines),
                            target + int(awi.current_inventory_input or 0),
                        )
                        - int(awi.total_sales_at(step)),
                    )
                )
            return 0
        if self._use_penguin_last():
            return self._call_penguin_last(_PenguinAgentBase.needs_at, step, partner)
        if self._use_middle_load_cap():
            if self.is_supplier(partner):
                return self._middle_supply_need_at(step)
            if self.is_consumer(partner):
                return self._middle_consume_need_at(step)
            return 0
        return super().needs_at(step, partner)

    def is_valid_price(self, price, partner: str) -> bool:
        if self._use_penguin_last():
            return self._call_penguin_last(
                _PenguinAgentBase.is_valid_price, price, partner
            )
        return super().is_valid_price(price, partner)

    def best_price(self, partner: str):
        if self._use_penguin_last():
            return self._call_penguin_last(_PenguinAgentBase.best_price, partner)
        return super().best_price(partner)

    def price(self, partner: str):
        if self._use_penguin_last():
            return self._call_penguin_last(_PenguinAgentBase.price, partner)
        return super().price(partner)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    competition = sys.argv[1] if len(sys.argv) > 1 else "std"
    profile = sys.argv[2] if len(sys.argv) > 2 else "lite"
    run([SBD], competition=competition, profile=profile)
