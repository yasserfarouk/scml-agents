#!/usr/bin/env python
"""
Rohn: ANAC2026 Standard Track 提出用エージェント
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from itertools import chain, combinations, repeat

from negmas import SAOResponse, ResponseType
from numpy.random import choice

from scml.std import StdSyncAgent
from scml.std.common import QUANTITY, TIME, UNIT_PRICE

__all__ = ["Rohn"]


# ======================================================================
# フィーチャーフラグ
# ======================================================================

# ── L0 専用 ──────────────────────────────────────────────────────────
# 売りフロアを storage_cost × スケールで引き下げる（L0 向け在庫圧力対策）
FEATURE_L0_FLOOR_SCALE         = True
L0_FLOOR_SCALE                 = 3.0

# 超過在庫（inv > n_lines × L0_EXCESS_THRESHOLD）で floor=0 かつ安値提案
FEATURE_L0_EXCESS_LIQUIDATION  = True
L0_EXCESS_THRESHOLD            = 2
L0_EXCESS_INITIAL_SELL_RATE    = 0.30
L0_EXCESS_BASE_SELL_RATE       = 0.40

# 終盤（残ステップ ≤ L0_ENDGAME_STEPS）で floor=0 の清算モード
FEATURE_L0_ENDGAME_LIQUIDATION = True
L0_ENDGAME_STEPS               = 10

# 全消費者へのリーチ & フル生産目標（n_lines - sales_at）
FEATURE_L0_FULL_PRODUCTION     = True
FEATURE_L0_FULL_CONSUMER_REACH = True

# 先物提案ホライゾン（s+1 〜 s+L0_FUTURE_HORIZON を全消費者に提案）
L0_FUTURE_HORIZON              = 20
FEATURE_L0_MAX_FUTURE          = True   # 先物ニーズ = n_lines - sales_at

# 外生供給価格を buy_book に記録（原価基準の更新）
FEATURE_L0_EXO_PRICE_TRACK     = True

# 在庫連動売り価格引き下げ
L0_SELL_RATE_BASE              = 0.60
L0_SELL_RATE_EMERGENCY         = 0.40
L0_EMERGENCY_THRESHOLD         = 3
L0_SELL_PUSH_FACTOR            = 2
L0_SELL_PUSH_REDUCTION         = 0.25

# 初回売り提案を下げる（ON: L1 が即受け入れやすくなる）
FEATURE_L0_AGGRESSIVE_FIRST    = True
L0_INITIAL_SELL_RATE           = 0.60

# 先物売り提案を下げる（ON: L1 が先物提案を即受け入れやすくなる）
FEATURE_L0_FUTURE_LOWER_PRICE  = True
L0_FUTURE_SELL_RATE            = 0.40

# 旧互換フラグ（通常 False）
FEATURE_L0_SELL_GUARANTEE      = False
FEATURE_L0_FLOOR_RELEASE       = False
L0_FLOOR_RELEASE_THRESHOLD     = 0

# L0 在庫清算モード（price >= floor なら数量不問で即受け入れ）
# OFF にすると通常の L0 ロジックに戻る
FEATURE_L0_CLEARANCE_MODE      = True
L0_CLEARANCE_SELL_RATE         = 0.10   # 提案価格 = minp + (maxp-minp) × このレート

# ── Last 層専用 ───────────────────────────────────────────────────────
# 全サプライヤーに先物 LAST_FUTURE_HORIZON ステップ分を提案
FEATURE_LAST_FULL_SUPPLIER_REACH = True
LAST_FUTURE_HORIZON              = 20

# 買いニーズ = n_lines - supplies_at（フル生産目標）
FEATURE_LAST_FULL_PRODUCTION     = True

# 在庫ガードなし（供給不足ペナルティを避けるため在庫は多めでよい）
FEATURE_LAST_INVENTORY_GUARD_OFF = True

# 外生需要価格を sell_book に記録 → _max_buy_price が正確になる
FEATURE_LAST_EXO_PRICE_TRACK     = True

# 買い価格レート（Middle=1.2 より高め）
LAST_BUY_RATE_BASE               = 1.4

# 外生需要量を学習し、需要追従型の仕入れ量を計算する
FEATURE_LAST_DEMAND_TRACK = True
LAST_DEMAND_WINDOW        = 10
LAST_DEMAND_BUFFER        = 1.2

# 外生需要連動 needs（needs target = max(当ステップ exo_out, 過去移動平均)）
FEATURE_LAST_NEEDS_EXO_DRIVEN = True
LAST_NEEDS_SCALE              = 1.3   # バッファ係数

# Last 買いを maxp まで許容（shortfall penalty > 利益帯超過コスト）
FEATURE_LAST_ACCEPT_AT_MAXP   = True

# exo_out ベースの買い上限（利益確定できる価格帯だけで仕入れ交渉）
FEATURE_LAST_EXO_CEILING      = True
LAST_BUY_CEILING_MARGIN       = 0.5   # breakeven からのマージン

# needs 計算で inventory_output（製品在庫）を差し引く
FEATURE_LAST_NEEDS_INV_OUT    = True

# needs バッファを時間逓減型に変更（序盤大きく・終盤ゼロへ）
FEATURE_LAST_NEEDS_TIME_DECAY = True

# Last 買い提案価格を minp から ceiling まで t^exp で滑らか譲歩
FEATURE_LAST_BUY_CONCEDE      = True
LAST_BUY_CONCESSION_EXP       = 0.5   # 0.5 = 凹型（序盤に早く・終盤にゆっくり譲歩）

# n_lines を超えた生産目標を排除（過剰仕入れ防止）
FEATURE_LAST_TIGHT_NEEDS      = True

# 終盤バッファ（残ステップが閾値以下のときのみバッファを追加）
# ON : buffer = n_lines × FACTOR × (1 - remaining/THRESHOLD)  (remaining < THRESHOLD)
#            = 0                                               (remaining >= THRESHOLD)
# OFF: 旧バッファ（FEATURE_LAST_NEEDS_TIME_DECAY と同じ挙動）
FEATURE_LAST_LATE_BUFFER      = True
LAST_BUFFER_STEP_THRESHOLD    = 72   # 損益分岐ステップ数 = shortfall_penalty / storage_cost ≈ 27/0.0135
LAST_BUFFER_FACTOR            = 0.5  # 最大でも半ライン分の余剰仕入れ

# 最小安全バッファ（agent068）: buffer = max(LAST_MIN_SAFETY_BUFFER, late_buf)
# 序盤〜中盤のバッファ完全ゼロを回避して shortfall を減らす
FEATURE_LAST_MIN_SAFETY_BUFFER = True
LAST_MIN_SAFETY_BUFFER         = 2

# 固定床の係数（agent069/070/071 共有）: n_lines × これ を生産目標/買い目標の下限にする
LAST_EARLY_FIXED_PRODUCTIVITY  = 0.7

# 需要追従の固定床（agent070）: production_target = max(固定床, 需要追従)
# → AS0 の固定値（n_lines × 0.7）を絶対に下回らない保証
FEATURE_LAST_DEMAND_FLOOR      = True

# buy target 固定床（agent071）: _last_buy_target に max(base, 固定床) を適用し
#   needs_at / counter_all / future_supplie_offer の3経路の買い目標を一致させる
FEATURE_LAST_BUY_TARGET_FLOOR  = True

# ── 共通 ─────────────────────────────────────────────────────────────
FEATURE_HOLDING_COST_FLOOR         = True
FEATURE_PROFIT_BAND                = True
PROFIT_BAND_MARGIN                 = 0
FEATURE_INVENTORY_PRESSURE         = True
INVENTORY_PRESSURE_SCALE           = 1.0
FEATURE_INVENTORY_GUARD            = True
FEATURE_FIXED_THRESHOLD            = True
FEATURE_LATE_GAME_THRESHOLD        = True
FEATURE_SUCCESS_RATE_PRICE         = True
FEATURE_POSITION_AWARE_PRICE       = True
FEATURE_COMPETITIVE_PRICE          = True
COMPETITIVE_PRICE_DISCOUNT         = 0.97
FEATURE_BUYER_PRICE_TRACK          = True
BUYER_TRACK_DISCOUNT               = 0.97
BUYER_TRACK_MIN_SAMPLES            = 5
BUYER_TRACK_WINDOW                 = 30
FEATURE_RAW_DIFF_SEARCH            = True
FEATURE_FALLBACK_THRESHOLD         = True
FEATURE_NO_AGGRESSIVE_MODE         = True
FEATURE_AGGRESSIVE_BUY_SUPPRESSION = True
FEATURE_INVENTORY_SELL_PUSH        = True
SELL_PUSH_FACTOR                   = 5
SELL_PUSH_REDUCTION                = 0.10
FEATURE_CHEAP_SUPPLIER_FIRST       = True
CHEAP_SUPPLIER_PRICE_WEIGHT        = 2.0
FEATURE_STEP_LOG                   = False  # 診断時のみ True


# ======================================================================
# ユーティリティ
# ======================================================================

def distribute(q: int, n: int) -> list[int]:
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# ======================================================================
# エージェント本体
# ======================================================================

class Rohn(StdSyncAgent):

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold    = threshold if threshold is not None else 1
        self._base_ptoday  = ptoday
        self._productivity = productivity
        self._layer: str   = "middle"  # init() で上書き

        self.partner_success_rate  = defaultdict(lambda: 0.5)
        self.partner_negotiations  = defaultdict(int)
        self.partner_successes     = defaultdict(int)
        self.price_concessions     = defaultdict(float)
        self.last_profits          = []
        self.aggressive_mode       = False
        self.conservative_mode     = False

        self._buy_book:  defaultdict = defaultdict(list)
        self._sell_book: defaultdict = defaultdict(list)

        self._buyer_offer_history: deque[float] = deque(maxlen=BUYER_TRACK_WINDOW)
        self._estimated_market_price: float | None = None

        # Last 層: 外生需要量の履歴（需要追従型仕入れに使用）
        self._exo_demand_history: deque[int]   = deque(maxlen=LAST_DEMAND_WINDOW)
        self._avg_exo_demand_qty: float | None = None

    # ------------------------------------------------------------------
    # init: 層を一度だけ確定する
    # ------------------------------------------------------------------

    def init(self):
        super().init()
        if self.awi.is_first_level:
            self._layer = "L0"
        elif self.awi.is_last_level:
            self._layer = "last"
        else:
            self._layer = "middle"

    # ------------------------------------------------------------------
    # 利益帯ヘルパー
    # ------------------------------------------------------------------

    def _production_cost(self) -> float:
        c = self.awi.catalog_prices
        return max(1.0, float(c[self.awi.my_output_product]) - float(c[self.awi.my_input_product]))

    def _avg_book_price(self, book: dict, step: int):
        entries = book.get(step, [])
        if not entries:
            return None
        total_qty = sum(q for _, q in entries)
        return (sum(p * q for p, q in entries) / total_qty) if total_qty > 0 else None

    def _l0_floor_released(self) -> bool:
        return (FEATURE_L0_FLOOR_RELEASE
                and self._layer == "L0"
                and self.awi.current_inventory_input >= self.awi.n_lines * L0_FLOOR_RELEASE_THRESHOLD)

    def _is_excess_inventory(self) -> bool:
        return (FEATURE_L0_EXCESS_LIQUIDATION
                and self._layer == "L0"
                and self.awi.current_inventory_input > self.awi.n_lines * L0_EXCESS_THRESHOLD)

    def _min_sell_price(self, step: int) -> float:
        """売値下限。L0 は保管コストスケール付きフロア、clearance モードでは自然なフロア。"""
        # L0 clearance: スケール係数なしの自然な経済的フロア
        if self._layer == "L0" and FEATURE_L0_CLEARANCE_MODE:
            if not FEATURE_PROFIT_BAND:
                return 0.0
            if self._is_excess_inventory():
                return 0.0
            remaining = max(0, self.awi.n_steps - step)
            if FEATURE_L0_ENDGAME_LIQUIDATION and remaining <= L0_ENDGAME_STEPS:
                return 0.0
            prod_cost = self._production_cost()
            avg_buy   = self._avg_book_price(self._buy_book, step)
            if avg_buy is None:
                avg_buy = float(self.awi.catalog_prices[self.awi.my_input_product])
            base_cost = avg_buy + prod_cost + PROFIT_BAND_MARGIN
            sc = self.awi.current_storage_cost
            return max(0.0, base_cost - sc * remaining)

        if not FEATURE_PROFIT_BAND:
            return 0.0
        if self._l0_floor_released():
            return 0.0
        if self._is_excess_inventory():
            return 0.0

        prod_cost = self._production_cost()
        avg_buy   = self._avg_book_price(self._buy_book, step)
        if avg_buy is None:
            avg_buy = float(self.awi.catalog_prices[self.awi.my_input_product])
        base_cost = avg_buy + prod_cost + PROFIT_BAND_MARGIN

        if FEATURE_HOLDING_COST_FLOOR and self._layer == "L0":
            remaining = max(0, self.awi.n_steps - step)
            if FEATURE_L0_ENDGAME_LIQUIDATION and remaining <= L0_ENDGAME_STEPS:
                return 0.0
            sc = self.awi.current_storage_cost
            effective_sc = sc * L0_FLOOR_SCALE if FEATURE_L0_FLOOR_SCALE else sc
            return max(0.0, base_cost - effective_sc * remaining)

        if FEATURE_INVENTORY_PRESSURE:
            sc        = self.awi.current_storage_cost
            remaining = max(0, self.awi.n_steps - self.awi.current_step)
            if self._layer == "L0":
                inv = self.awi.current_inventory_input
                emergency = inv > self.awi.n_lines * L0_EMERGENCY_THRESHOLD
                scale = L0_FLOOR_SCALE * (2.0 if emergency else 1.0)
            else:
                scale = INVENTORY_PRESSURE_SCALE
            return max(0.0, base_cost - sc * remaining * scale)

        return max(0.0, base_cost)

    def _max_buy_price(self, step: int) -> float:
        """買値上限。Last 層は外生需要価格を sell_book に記録済みなので自動反映される。"""
        if not FEATURE_PROFIT_BAND:
            return float("inf")
        prod_cost = self._production_cost()
        avg_sell  = self._avg_book_price(self._sell_book, step)
        if avg_sell is None:
            avg_sell = float(self.awi.catalog_prices[self.awi.my_output_product])
        return avg_sell - prod_cost - PROFIT_BAND_MARGIN

    def _record_trade(self, partner: str, offer: tuple):
        if not FEATURE_PROFIT_BAND:
            return
        step  = offer[TIME]
        price = float(offer[UNIT_PRICE])
        qty   = int(offer[QUANTITY])
        if self.is_supplier(partner):
            self._buy_book[step].append((price, qty))
        elif self.is_consumer(partner):
            self._sell_book[step].append((price, qty))

    # ------------------------------------------------------------------
    # 在庫ガード
    # ------------------------------------------------------------------

    def _inventory_guard(self, buy_need: int) -> int:
        if not FEATURE_INVENTORY_GUARD:
            return buy_need
        if self._layer == "last" and FEATURE_LAST_INVENTORY_GUARD_OFF:
            return buy_need
        progress = self.awi.current_step / max(1, self.awi.n_steps)
        if progress < 0.5:
            max_inv = self.awi.n_lines * 5
        elif progress < 0.8:
            max_inv = self.awi.n_lines * 3
        else:
            max_inv = self.awi.n_lines * 1
        inv = self.awi.current_inventory_input
        if inv > 1.5 * max_inv:
            return 0
        elif inv > max_inv:
            return int(buy_need * 0.5)
        return buy_need

    def _last_buy_target(self) -> float:
        """Last 層の1ステップあたり仕入れ目標量（agent071: 固定床つき）。

        base:
          FEATURE_LAST_NEEDS_EXO_DRIVEN が ON のとき
            max(当ステップ外生需要, 過去移動平均)（step0 から実需要を反映）。
          OFF のとき 学習5件以上で移動平均 / 未満で n_lines。
        agent071:
          Last 層では max(base, n_lines × LAST_EARLY_FIXED_PRODUCTIVITY) を返し、
          counter_all / future_supplie_offer / needs_at の買い目標床を一致させる。
        """
        if self._layer == "last" and FEATURE_LAST_NEEDS_EXO_DRIVEN:
            exo_now  = float(self.awi.current_exogenous_output_quantity)
            hist_avg = self._avg_exo_demand_qty
            if hist_avg is not None:
                base = max(exo_now, hist_avg, 1.0)
            else:
                base = max(exo_now, 1.0)
        elif FEATURE_LAST_DEMAND_TRACK and self._avg_exo_demand_qty is not None and len(self._exo_demand_history) >= 5:
            base = self._avg_exo_demand_qty
        else:
            base = float(self.awi.n_lines)

        # agent071: Last 層のみ固定床（n_lines × 0.7）を適用
        if self._layer == "last" and FEATURE_LAST_BUY_TARGET_FLOOR:
            floor = self.awi.n_lines * LAST_EARLY_FIXED_PRODUCTIVITY
            return max(base, floor)
        return base

    def _threshold_multiplier(self, is_consumer_side: bool) -> float:
        if not self.aggressive_mode:
            return 1.0
        return 2.0 if (not FEATURE_AGGRESSIVE_BUY_SUPPRESSION or is_consumer_side) else 1.0

    def _is_layer0_emergency(self) -> bool:
        return (self._layer == "L0"
                and self.awi.current_inventory_input > self.awi.n_lines * L0_EMERGENCY_THRESHOLD)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self):
        super().step()
        current = self.awi.current_step

        for s in [k for k in self._buy_book if k < current]:
            del self._buy_book[s]
        for s in [k for k in self._sell_book if k < current]:
            del self._sell_book[s]

        # L0: 外生供給価格を buy_book に記録（原価基準の更新）
        if self._layer == "L0" and FEATURE_L0_EXO_PRICE_TRACK and FEATURE_PROFIT_BAND:
            exo_qty   = self.awi.current_exogenous_input_quantity
            exo_total = self.awi.current_exogenous_input_price
            if exo_qty > 0 and exo_total > 0:
                self._buy_book[current].append((float(exo_total) / float(exo_qty), int(exo_qty)))

        # Last: 外生需要価格を sell_book に記録（買い上限の基準に使う）
        if self._layer == "last" and FEATURE_LAST_EXO_PRICE_TRACK and FEATURE_PROFIT_BAND:
            exo_qty   = self.awi.current_exogenous_output_quantity
            exo_total = self.awi.current_exogenous_output_price
            if exo_qty > 0 and exo_total > 0:
                self._sell_book[current].append((float(exo_total) / float(exo_qty), int(exo_qty)))

        # Last: 外生需要量を履歴に追加し、移動平均を更新（仕入れ量の目標算出に使用）
        if self._layer == "last" and FEATURE_LAST_DEMAND_TRACK:
            exo_qty = self.awi.current_exogenous_output_quantity
            if exo_qty > 0:
                self._exo_demand_history.append(exo_qty)
            if self._exo_demand_history:
                self._avg_exo_demand_qty = (
                    sum(self._exo_demand_history) / len(self._exo_demand_history)
                )

        # デバッグログ（L0 のみ）
        if FEATURE_STEP_LOG and self._layer == "L0":
            s     = current
            n     = self.awi.n_lines
            sales = self.awi.total_sales_at(s)
            inv   = self.awi.current_inventory_input
            exo   = self.awi.current_exogenous_input_quantity
            fill  = sales / max(1, n)
            excess = self._is_excess_inventory()
            marker = "★" if sales >= n else ("△" if sales >= n * 0.5 else "✗")
            # print(
            #     f"[{self.name}] step={s:3d} sales={sales:2d}/{n} ({fill*100:5.1f}%) {marker} "
            #     f"inv={inv:4d} exo={exo} {'[EXCESS]' if excess else ''}"
            # )

        # スレッショルド更新
        if FEATURE_FIXED_THRESHOLD:
            base = self.awi.n_lines * 0.1
            if FEATURE_LATE_GAME_THRESHOLD:
                time_left = (self.awi.n_steps - current) / self.awi.n_steps
                if time_left <= 0.3:
                    self._threshold = max(1, int(base * 1.5))
                elif time_left <= 0.6:
                    self._threshold = max(1, int(base * 1.3))
                else:
                    self._threshold = max(1, int(base))
            else:
                self._threshold = max(1, int(base))

        current_balance = getattr(self.awi, "current_balance", 0)
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

    # ------------------------------------------------------------------
    # パートナー選択・管理
    # ------------------------------------------------------------------

    def update_partner_performance(self, partner, success):
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
        self.partner_success_rate[partner] = (
            self.partner_successes[partner] / self.partner_negotiations[partner]
        )

    def get_effective_ptoday(self):
        base = self._base_ptoday
        if self.aggressive_mode:
            return min(0.9, base + 0.15)
        elif self.conservative_mode:
            return max(0.5, base - 0.1)
        return base

    def select_partners_by_performance(self, partners, ratio=None):
        if not partners:
            return []
        if ratio is None:
            ratio = self.get_effective_ptoday()
        scored = sorted(partners, key=lambda p: self.partner_success_rate[p], reverse=True)
        n = max(1, int(len(scored) * ratio))
        selected = scored[:n]
        remaining = scored[n:]
        if remaining and len(selected) < len(partners):
            selected.extend(random.sample(remaining, min(2, len(remaining))))
        return selected

    # ------------------------------------------------------------------
    # first_proposals
    # ------------------------------------------------------------------

    def first_proposals(self):
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()
        first_dict   = {}
        future_sup   = []
        future_con   = []
        for k, q in distribution.items():
            if q > 0:
                first_dict[k] = (q, s, self.smart_price(k, is_first_proposal=True))
            elif self.is_supplier(k):
                future_sup.append(k)
            elif self.is_consumer(k):
                future_con.append(k)
        return (first_dict
                | self.future_supplie_offer(self.select_partners_by_performance(future_sup))
                | self.future_consume_offer(self.select_partners_by_performance(future_con)))

    # ------------------------------------------------------------------
    # counter_all
    # ------------------------------------------------------------------

    def counter_all(self, offers, states):
        # L0 は在庫清算型ロジックにディスパッチ
        if self._layer == "L0" and FEATURE_L0_CLEARANCE_MODE:
            return self._l0_clearance_counter(offers, states)

        response = {}
        awi = self.awi

        if FEATURE_BUYER_PRICE_TRACK and self._layer != "middle":
            for p, offer in offers.items():
                if offer is not None and self.is_consumer(p):
                    self._buyer_offer_history.append(float(offer[UNIT_PRICE]))
            if len(self._buyer_offer_history) >= BUYER_TRACK_MIN_SAMPLES:
                self._estimated_market_price = (
                    sum(self._buyer_offer_history) / len(self._buyer_offer_history)
                )
            else:
                self._estimated_market_price = None

        for all_partners, issues in [
            (awi.my_suppliers, awi.current_input_issues),
            (awi.my_consumers, awi.current_output_issues),
        ]:
            is_consumer_side = self.is_consumer(all_partners[0])
            day_prod = awi.n_lines * self._productivity
            needs = 0

            if self.is_supplier(all_partners[0]):
                if self._layer == "last" and FEATURE_LAST_DEMAND_TRACK:
                    target = self._last_buy_target()
                    needs  = max(0, int(target * LAST_DEMAND_BUFFER)
                                 - awi.current_inventory_input
                                 - awi.total_supplies_at(awi.current_step))
                elif self._layer == "last" and FEATURE_LAST_FULL_PRODUCTION:
                    needs = max(0, awi.n_lines - awi.total_supplies_at(awi.current_step))
                else:
                    needs = int(day_prod
                                - awi.current_inventory_input
                                - awi.total_supplies_at(awi.current_step))
                    needs = self._inventory_guard(needs)

            elif self.is_consumer(all_partners[0]):
                if self._layer == "L0" and FEATURE_L0_FULL_PRODUCTION:
                    needs = max(0, awi.n_lines - awi.total_sales_at(awi.current_step))
                else:
                    needs = int(max(0,
                        min(awi.n_lines, day_prod + awi.current_inventory_input)
                        - awi.total_sales_at(awi.current_step)))
                    if self._layer == "L0" and FEATURE_L0_SELL_GUARANTEE:
                        inv = awi.current_inventory_input
                        if inv > 0:
                            needs = max(needs, min(awi.n_lines, inv))

            partners = {p for p in all_partners if p in offers}
            cur_offers = {}
            fut_offers = {}
            for p in partners:
                if offers[p] is None:
                    continue
                t = offers[p][TIME]
                if self.is_valid_price(offers[p][UNIT_PRICE], p, step=t):
                    (cur_offers if t == awi.current_step else fut_offers)[p] = offers[p]

            # 先物を先に受け入れ
            dup = [0] * awi.n_steps
            for p, x in fut_offers.items():
                step = x[TIME]
                if step < awi.n_steps:
                    if x[QUANTITY] + dup[step - 1] <= self.needs_at(step, p):
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, x)
                        dup[step - 1] += x[QUANTITY]
                        self.update_partner_performance(p, True)
                        self._record_trade(p, x)

            # 当日オファーから最良組み合わせを選択
            cur_partners = {p for p in partners if p in cur_offers}
            plist = list(powerset(cur_partners))
            best_plus_idx  = -1
            best_plus_diff = float("inf")
            best_minus_idx = -1
            best_minus_diff = float("inf")

            for i, pid in enumerate(plist):
                if not pid:
                    continue
                offered = sum(cur_offers[p][QUANTITY] for p in pid)
                diff = abs(offered - needs)
                if FEATURE_RAW_DIFF_SEARCH:
                    score = diff
                    if FEATURE_CHEAP_SUPPLIER_FIRST and not is_consumer_side and self._layer != "middle":
                        avg_p = sum(cur_offers[p][UNIT_PRICE] for p in pid) / len(pid)
                        max_p = max(1.0, issues[UNIT_PRICE].max_value)
                        score = diff + CHEAP_SUPPLIER_PRICE_WEIGHT * (avg_p / max_p)
                else:
                    quality = sum(self.partner_success_rate[p] for p in pid) / len(pid)
                    score = diff * (2.0 - quality)
                if offered - needs >= 0:
                    if score < best_plus_diff and needs > 0:
                        best_plus_diff, best_plus_idx = score, i
                else:
                    if score < best_minus_diff and offered > 0:
                        best_minus_diff, best_minus_idx = score, i

            has_accept = True
            mul = self._threshold_multiplier(is_consumer_side)
            if best_plus_diff <= self._threshold * mul and best_plus_idx >= 0 and plist[best_plus_idx]:
                best_idx = best_plus_idx
            elif best_minus_idx >= 0 and plist[best_minus_idx]:
                best_idx = best_minus_idx
            elif cur_partners and needs > 0:
                best_single = min(
                    (i for i, ps in enumerate(plist) if len(ps) == 1),
                    key=lambda i: abs(sum(cur_offers[p][QUANTITY] for p in plist[i]) - needs),
                    default=-1,
                )
                if best_single >= 0:
                    if FEATURE_FALLBACK_THRESHOLD:
                        sd = abs(sum(cur_offers[p][QUANTITY] for p in plist[best_single]) - needs)
                        best_idx = best_single if sd <= self._threshold * mul else -1
                        has_accept = best_idx >= 0
                    else:
                        best_idx = best_single
                else:
                    has_accept = False
            else:
                has_accept = False

            if has_accept and needs > 0:
                accepted = plist[best_idx]
                others   = list(set(cur_partners) - set(accepted) - set(response))
                response.update({
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, cur_offers[k])
                    for k in accepted
                })
                for k in accepted:
                    self.update_partner_performance(k, True)
                    self._record_trade(k, cur_offers[k])
                for k in others:
                    self.update_partner_performance(k, False)
                for k, x in self.future_supplie_offer([x for x in others if self.is_supplier(x)]).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer([x for x in others if self.is_consumer(x)]).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
            else:
                other = {p for p in all_partners if p not in response and p in self.negotiators}
                dist  = self.distribute_todays_needs(other)
                fsup, fcon = [], []
                for k, q in dist.items():
                    if q > 0:
                        response[k] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, awi.current_step,
                             self.smart_price(k, state=states.get(k), is_counter_offer=True))
                        )
                    elif self.is_supplier(k):
                        fsup.append(k)
                    elif self.is_consumer(k):
                        fcon.append(k)
                for k, x in self.future_supplie_offer(fsup).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(fcon).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    # ------------------------------------------------------------------
    # L0 在庫清算型 counter_all
    # ------------------------------------------------------------------

    def _l0_clearance_counter(self, offers, states):
        """
        L0 在庫清算型 counter_all:
          1. price >= floor なら数量・needs マッチング不問で即受け入れ
          2. floor 未満なら clearance 価格でカウンター
          3. offers に含まれない消費者には「最近接・未予約ステップ」への先物提案
        """
        response = {}
        awi = self.awi
        n_consumers = max(1, len(awi.my_consumers))
        accepted_today = 0

        consumers_with_offers = [
            p for p in awi.my_consumers
            if p in offers and offers[p] is not None
        ]

        for p in consumers_with_offers:
            offer = offers[p]
            price = float(offer[UNIT_PRICE])
            qty   = int(offer[QUANTITY])
            step  = int(offer[TIME])
            floor = self._min_sell_price(step)

            if step == awi.current_step:
                cap = max(0, awi.n_lines - awi.total_sales_at(awi.current_step) - accepted_today)
            else:
                cap = max(0, awi.n_lines - awi.total_sales_at(step))

            nmi = self.get_nmi(p)
            if nmi is None:
                continue
            mn = nmi.issues[UNIT_PRICE].min_value
            mx = nmi.issues[UNIT_PRICE].max_value
            clearance_price = max(floor, mn + (mx - mn) * L0_CLEARANCE_SELL_RATE)

            if price >= floor and cap > 0:
                if qty <= cap:
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    self._record_trade(p, offer)
                    self.update_partner_performance(p, True)
                    if step == awi.current_step:
                        accepted_today += qty
                else:
                    response[p] = SAOResponse(
                        ResponseType.REJECT_OFFER, (cap, step, clearance_price)
                    )
            else:
                sell_qty = max(1, (awi.n_lines - awi.total_sales_at(awi.current_step))
                               // n_consumers)
                if sell_qty > 0 and cap > 0:
                    response[p] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (sell_qty, awi.current_step, clearance_price),
                    )
                self.update_partner_performance(p, False)

        # offers に含まれない消費者: 先物提案
        others_con = [
            p for p in awi.my_consumers
            if p not in response and p in self.negotiators
        ]
        for k, x in self.future_consume_offer(others_con).items():
            response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    # ------------------------------------------------------------------
    # smart_price（層別ディスパッチ）
    # ------------------------------------------------------------------

    def smart_price(self, partner, state=None, is_first_proposal=False,
                    is_counter_offer=False, step=None):
        if step is None:
            step = self.awi.current_step

        # ── Last 買い: minp から ceiling まで t^exp で滑らか譲歩（agent066）──
        if self._layer == "last" and self.is_supplier(partner) and FEATURE_LAST_BUY_CONCEDE:
            nmi = self.get_nmi(partner)
            if nmi is None:
                return None
            awi  = self.awi
            minp = nmi.issues[UNIT_PRICE].min_value
            maxp = nmi.issues[UNIT_PRICE].max_value

            exo_out_qty   = float(awi.current_exogenous_output_quantity or 0)
            exo_out_total = float(awi.current_exogenous_output_price    or 0)
            if exo_out_qty > 0:
                exo_out_unit = exo_out_total / exo_out_qty
            else:
                exo_out_unit = float(awi.catalog_prices[awi.my_output_product])
            prod_cost = self._production_cost()
            ceiling   = exo_out_unit - prod_cost - LAST_BUY_CEILING_MARGIN
            effective_ceiling = max(minp, min(ceiling, maxp))

            if is_first_proposal:
                t = 0.0
            elif state is not None:
                t = float(getattr(state, "relative_time", 0.0))
            else:
                t = 0.0

            price = minp + (effective_ceiling - minp) * (t ** LAST_BUY_CONCESSION_EXP)
            return max(minp, min(int(price + 0.5), maxp))

        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        minp, maxp = nmi.issues[UNIT_PRICE].min_value, nmi.issues[UNIT_PRICE].max_value

        # ── L0 売り: clearance 価格（agent063）──────────────────────────
        if self._layer == "L0" and self.is_consumer(partner) and FEATURE_L0_CLEARANCE_MODE:
            floor = self._min_sell_price(step)
            return max(floor, minp + (maxp - minp) * L0_CLEARANCE_SELL_RATE)

        # ── L0 売り（通常）───────────────────────────────────────────────
        if self._layer == "L0" and self.is_consumer(partner):
            excess = self._is_excess_inventory()
            if is_first_proposal:
                if excess:
                    price = minp + (maxp - minp) * L0_EXCESS_INITIAL_SELL_RATE
                elif FEATURE_L0_AGGRESSIVE_FIRST:
                    price = minp + (maxp - minp) * L0_INITIAL_SELL_RATE
                else:
                    price = maxp
            elif FEATURE_SUCCESS_RATE_PRICE:
                if excess:
                    base_rate = L0_EXCESS_BASE_SELL_RATE
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * L0_SELL_PUSH_FACTOR))
                        base_rate = max(0.20, base_rate - p * L0_SELL_PUSH_REDUCTION)
                else:
                    emergency = self._is_layer0_emergency()
                    base_rate = L0_SELL_RATE_EMERGENCY if emergency else L0_SELL_RATE_BASE
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * L0_SELL_PUSH_FACTOR))
                        base_rate = max(0.30 if emergency else 0.40,
                                        base_rate - p * L0_SELL_PUSH_REDUCTION)
                price = max(maxp * (base_rate - self.partner_success_rate[partner] * 0.15), minp)
            else:
                nego_t = getattr(state, "relative_time", 0.5) if state else 0.5
                game_t = self.awi.current_step / max(1, self.awi.n_steps)
                t = min(1.0, nego_t + game_t * 0.2)
                excess = self._is_excess_inventory()
                if excess:
                    target_rate = L0_EXCESS_BASE_SELL_RATE
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * L0_SELL_PUSH_FACTOR))
                        target_rate = max(0.20, target_rate - p * L0_SELL_PUSH_REDUCTION)
                else:
                    emergency = self._is_layer0_emergency()
                    target_rate = L0_SELL_RATE_EMERGENCY if emergency else L0_SELL_RATE_BASE
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * L0_SELL_PUSH_FACTOR))
                        target_rate = max(0.30 if emergency else 0.40,
                                        target_rate - p * L0_SELL_PUSH_REDUCTION)
                price = max(minp, maxp - (maxp - max(minp, maxp * target_rate)) * t)

        # ── Last 買い（FEATURE_LAST_BUY_CONCEDE=False のフォールバック）──
        elif self._layer == "last" and self.is_supplier(partner):
            if is_first_proposal:
                price = minp
            elif FEATURE_SUCCESS_RATE_PRICE:
                rate  = LAST_BUY_RATE_BASE + self.partner_success_rate[partner] * 0.15
                price = min(minp * rate, maxp)
            else:
                nego_t = getattr(state, "relative_time", 0.5) if state else 0.5
                game_t = self.awi.current_step / max(1, self.awi.n_steps)
                t = min(1.0, nego_t + game_t * 0.2)
                target = min(maxp, minp * LAST_BUY_RATE_BASE)
                price  = min(maxp, minp + (target - minp) * t)

        # ── 共通（Middle 全般・L0 買い・Last 売り）───────────────────────
        else:
            if is_first_proposal:
                price = maxp if self.is_consumer(partner) else minp
            elif FEATURE_SUCCESS_RATE_PRICE:
                if self.is_consumer(partner):
                    base_rate = 0.7
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * SELL_PUSH_FACTOR))
                        base_rate = max(0.5, base_rate - p * SELL_PUSH_REDUCTION)
                    price = max(maxp * (base_rate - self.partner_success_rate[partner] * 0.15), minp)
                else:
                    base_rate = (1.3 if FEATURE_POSITION_AWARE_PRICE and self._layer == "last"
                                 else 1.2)
                    price = min(minp * (base_rate + self.partner_success_rate[partner] * 0.15), maxp)
            else:
                nego_t = getattr(state, "relative_time", 0.5) if state else 0.5
                game_t = self.awi.current_step / max(1, self.awi.n_steps)
                t = min(1.0, nego_t + game_t * 0.2)
                if self.aggressive_mode:
                    t = min(1.0, t * 1.3)
                if self.is_consumer(partner):
                    target = max(minp, maxp * 0.7)
                    if FEATURE_INVENTORY_SELL_PUSH:
                        p = min(1.0, self.awi.current_inventory_input
                                / max(1, self.awi.n_lines * SELL_PUSH_FACTOR))
                        target = max(minp, maxp * max(0.5, 0.7 - p * SELL_PUSH_REDUCTION))
                    price = max(minp, maxp - (maxp - target) * t)
                else:
                    rate   = 1.3 if FEATURE_POSITION_AWARE_PRICE and self._layer == "last" else 1.2
                    target = min(maxp, minp * rate)
                    price  = min(maxp, minp + (target - minp) * t)

        # ── Profit Band 適用 ──────────────────────────────────────────────
        if FEATURE_PROFIT_BAND:
            if self.is_consumer(partner):
                floor = self._min_sell_price(step)
                price = max(price, min(floor, maxp))
            else:
                ceiling = self._max_buy_price(step)
                price   = min(price, max(ceiling, minp))

        # ── 競合価格キャップ（売り側のみ）────────────────────────────────
        if self.is_consumer(partner):
            cap1, cap2 = None, None
            if FEATURE_COMPETITIVE_PRICE and self._layer != "middle":
                m1 = float(self.awi.trading_prices[self.awi.my_output_product])
                if m1 > 0:
                    cap1 = m1 * COMPETITIVE_PRICE_DISCOUNT
            if FEATURE_BUYER_PRICE_TRACK and self._layer != "middle" and self._estimated_market_price:
                cap2 = self._estimated_market_price * BUYER_TRACK_DISCOUNT
            if cap1 is not None and cap2 is not None:
                w2 = min(1.0, len(self._buyer_offer_history) / BUYER_TRACK_WINDOW)
                price = min(price, max((1 - w2) * cap1 + w2 * cap2, minp))
            elif cap1 is not None:
                price = min(price, max(cap1, minp))
            elif cap2 is not None:
                price = min(price, max(cap2, minp))

        return price

    # ------------------------------------------------------------------
    # is_valid_price
    # ------------------------------------------------------------------

    def is_valid_price(self, price, partner, step=None):
        # Last 買い: exo_out ベースの上限（agent065）
        if self._layer == "last" and self.is_supplier(partner) and FEATURE_LAST_EXO_CEILING:
            nmi = self.get_nmi(partner)
            if nmi is None:
                return False
            awi  = self.awi
            minp = nmi.issues[UNIT_PRICE].min_value
            maxp = nmi.issues[UNIT_PRICE].max_value

            exo_out_qty   = float(awi.current_exogenous_output_quantity or 0)
            exo_out_total = float(awi.current_exogenous_output_price    or 0)
            if exo_out_qty > 0:
                exo_out_unit = exo_out_total / exo_out_qty
            else:
                exo_out_unit = float(awi.catalog_prices[awi.my_output_product])

            prod_cost = self._production_cost()
            ceiling   = exo_out_unit - prod_cost - LAST_BUY_CEILING_MARGIN
            effective_ceiling = max(minp, min(ceiling, maxp))
            return price <= effective_ceiling

        # Last 買い: maxp まで許容（FEATURE_LAST_EXO_CEILING=False のフォールバック、agent064）
        if self._layer == "last" and self.is_supplier(partner) and FEATURE_LAST_ACCEPT_AT_MAXP:
            nmi = self.get_nmi(partner)
            if nmi is None:
                return False
            maxp = nmi.issues[UNIT_PRICE].max_value
            return price <= maxp

        # 共通ロジック（agent062）
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        minp, maxp = nmi.issues[UNIT_PRICE].min_value, nmi.issues[UNIT_PRICE].max_value
        if self.is_consumer(partner):
            if price < minp:
                return False
            if self._l0_floor_released():
                return True
            if FEATURE_PROFIT_BAND and step is not None:
                floor = self._min_sell_price(step)
                if floor <= maxp and price < floor:
                    return False
        elif self.is_supplier(partner):
            if price > maxp:
                return False
            if FEATURE_PROFIT_BAND and step is not None:
                ceiling = self._max_buy_price(step)
                if ceiling >= minp and price > ceiling:
                    return False
        else:
            return False
        return True

    # ------------------------------------------------------------------
    # needs_at
    # ------------------------------------------------------------------

    def needs_at(self, step, partner) -> int:
        awi = self.awi

        # Last 層 needs: n_lines 上限付き + 固定床 + 終盤/最小安全バッファ（agent070）
        if (self._layer == "last"
                and self.is_supplier(partner)
                and FEATURE_LAST_NEEDS_EXO_DRIVEN
                and FEATURE_LAST_DEMAND_TRACK):

            # 需要追従ターゲット（n_lines 上限あり）
            if FEATURE_LAST_TIGHT_NEEDS:
                exo_now   = float(awi.current_exogenous_output_quantity)
                hist_avg  = self._avg_exo_demand_qty
                raw_target = (
                    max(exo_now, hist_avg) if hist_avg is not None else exo_now
                )
                demand_target = min(int(raw_target), awi.n_lines)
            else:
                demand_target = int(self._last_buy_target())

            # 固定床と需要追従の大きい方を生産目標とする（agent070）
            if FEATURE_LAST_DEMAND_FLOOR:
                fixed_floor = int(awi.n_lines * LAST_EARLY_FIXED_PRODUCTIVITY)
                production_target = max(fixed_floor, demand_target)
            else:
                production_target = demand_target

            inv_in  = awi.current_inventory_input
            inv_out = awi.current_inventory_output if FEATURE_LAST_NEEDS_INV_OUT else 0

            if FEATURE_LAST_LATE_BUFFER:
                remaining = awi.n_steps - awi.current_step
                if remaining < LAST_BUFFER_STEP_THRESHOLD:
                    late_buf = int(
                        awi.n_lines * LAST_BUFFER_FACTOR
                        * (1 - remaining / LAST_BUFFER_STEP_THRESHOLD)
                    )
                else:
                    late_buf = 0
            elif FEATURE_LAST_NEEDS_TIME_DECAY:
                late_buf = int(awi.n_lines * (1 - (awi.current_step + 1) / awi.n_steps))
            else:
                late_buf = int(production_target * (LAST_NEEDS_SCALE - 1))

            # 最小安全バッファ（agent068）
            if FEATURE_LAST_MIN_SAFETY_BUFFER:
                buffer = max(LAST_MIN_SAFETY_BUFFER, late_buf)
            else:
                buffer = late_buf

            need = max(
                0,
                production_target + buffer
                - inv_in - inv_out
                - awi.total_supplies_at(step),
            )
            return min(need, awi.n_lines)

        # 共通ロジック（agent062）
        day_prod = awi.n_lines * self._productivity
        if self.is_supplier(partner):
            if self._layer == "last" and FEATURE_LAST_DEMAND_TRACK:
                target = self._last_buy_target()
                need   = max(0, int(target * LAST_DEMAND_BUFFER)
                             - awi.current_inventory_input
                             - awi.total_supplies_at(step))
                return need
            elif self._layer == "last" and FEATURE_LAST_FULL_PRODUCTION:
                return max(0, awi.n_lines - awi.total_supplies_at(step))
            need = int(day_prod - awi.current_inventory_input - awi.total_supplies_at(step))
            return self._inventory_guard(need)
        elif self.is_consumer(partner):
            if self._layer == "L0" and FEATURE_L0_FULL_PRODUCTION:
                return max(0, awi.n_lines - awi.total_sales_at(step))
            return int(max(0, min(awi.n_lines, day_prod + awi.current_inventory_input)
                           - awi.total_sales_at(step)))
        return 0

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))
        sup_list = [p for p in partners if self.is_supplier(p)]
        con_list = [p for p in partners if self.is_consumer(p)]
        awi      = self.awi
        day_prod = awi.n_lines * self._productivity

        if self._layer == "last" and FEATURE_LAST_DEMAND_TRACK:
            target    = self._last_buy_target()
            sup_needs = max(0, int(target * LAST_DEMAND_BUFFER)
                            - awi.current_inventory_input
                            - awi.total_supplies_at(awi.current_step))
        elif self._layer == "last" and FEATURE_LAST_FULL_PRODUCTION:
            sup_needs = max(0, awi.n_lines - awi.total_supplies_at(awi.current_step))
        else:
            sup_needs = int(day_prod - awi.current_inventory_input
                            - awi.total_supplies_at(awi.current_step))
            sup_needs = self._inventory_guard(sup_needs)

        if self._layer == "L0" and FEATURE_L0_FULL_PRODUCTION:
            con_needs = max(0, awi.n_lines - awi.total_sales_at(awi.current_step))
        else:
            con_needs = int(max(0,
                min(awi.n_lines, day_prod + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step)))
            if self._layer == "L0" and FEATURE_L0_SELL_GUARANTEE and awi.current_inventory_input > 0:
                con_needs = max(con_needs, min(awi.n_lines, awi.current_inventory_input))

        if sup_list and sup_needs > 0:
            if self._layer == "last" and FEATURE_LAST_FULL_SUPPLIER_REACH:
                response |= self.distribute_todays_supplie_consume_needs(
                    list(sup_list), sup_needs, full_reach=True)
            else:
                response |= self.distribute_todays_supplie_consume_needs(
                    self.select_partners_by_performance(sup_list), sup_needs)

        if con_list and con_needs > 0 and awi.total_sales_at(awi.current_step) <= awi.n_lines:
            if self._layer == "L0" and FEATURE_L0_FULL_CONSUMER_REACH:
                response |= self.distribute_todays_supplie_consume_needs(
                    list(con_list), con_needs, full_reach=True)
            else:
                response |= self.distribute_todays_supplie_consume_needs(
                    self.select_partners_by_performance(con_list), con_needs)

        return response

    def distribute_todays_supplie_consume_needs(
            self, partners, needs, full_reach=False) -> dict[str, int]:
        response = dict(zip(partners, repeat(0)))
        if not partners:
            return response
        random.shuffle(partners)
        if not full_reach:
            ptoday = self.get_effective_ptoday()
            partners = partners[:max(1, int(ptoday * len(partners)))]
        n = len(partners)
        if not full_reach and needs < n > 0:
            partners = random.sample(partners, random.randint(1, min(needs, n)))
            n = len(partners)
        if n > 0:
            response |= dict(zip(partners, distribute(needs, n)))
        return response

    # ------------------------------------------------------------------
    # 先物提案
    # ------------------------------------------------------------------

    def future_supplie_offer(self, partner_list):
        """買い先への先物提案。Last 層は全サプライヤーに 20 ステップ提案。"""
        if self._layer == "last" and FEATURE_LAST_FULL_SUPPLIER_REACH:
            return self._future_supplie_offer_last(partner_list)
        return self._future_supplie_offer_default(partner_list)

    def _future_supplie_offer_last(self, partner_list) -> dict:
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response
        partners = self.select_partners_by_performance(partner_list, ratio=1.0)
        target = self._last_buy_target()
        for offset in range(1, LAST_FUTURE_HORIZON + 1):
            future_step = s + offset
            if future_step >= n:
                break
            if FEATURE_LAST_DEMAND_TRACK:
                step_needs = max(0, int(target * LAST_DEMAND_BUFFER)
                                 - awi.current_inventory_input
                                 - awi.total_supplies_at(future_step))
            else:
                step_needs = max(0, awi.n_lines - awi.total_supplies_at(future_step))
            if step_needs <= 0:
                continue
            for k, q in dict(zip(partners, distribute(step_needs, len(partners)))).items():
                if q > 0:
                    buy_price = self.best_price(k)
                    if FEATURE_PROFIT_BAND:
                        nmi = self.get_nmi(k)
                        if nmi is not None:
                            ceiling = self._max_buy_price(future_step)
                            buy_price = min(buy_price, max(ceiling, nmi.issues[UNIT_PRICE].min_value))
                    response[k] = (q, future_step, buy_price)
        return response

    def _future_supplie_offer_default(self, partner_list) -> dict:
        """Middle / L0 の買い先への先物提案（3ステップ）。"""
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response
        sorted_p = self.select_partners_by_performance(partner_list, ratio=1.0)
        grp1 = sorted_p[:int(len(sorted_p) * 0.5)]
        grp2 = sorted_p[int(len(sorted_p) * 0.5):int(len(sorted_p) * 0.8)]
        grp3 = sorted_p[int(len(sorted_p) * 0.8):]
        p = awi.n_lines * self._productivity
        for offset, grp in [(1, grp1), (2, grp2), (3, grp3)]:
            fs = s + offset
            if fs >= n or not grp:
                continue
            sn = int(max(0, (p - awi.current_inventory_input
                             - awi.total_supplies_at(fs)) / 3))
            sn = self._inventory_guard(sn)
            if sn <= 0:
                continue
            for k, q in dict(zip(grp, distribute(sn, len(grp)))).items():
                if q > 0:
                    buy_price = self.best_price(k)
                    if FEATURE_PROFIT_BAND:
                        nmi = self.get_nmi(k)
                        if nmi is not None:
                            ceiling   = self._max_buy_price(fs)
                            buy_price = min(buy_price, max(ceiling, nmi.issues[UNIT_PRICE].min_value))
                    response[k] = (q, fs, buy_price)
        return response

    def future_consume_offer(self, partner_list) -> dict:
        """売り先への先物提案。L0 clearance モードでは最近接未充足ステップへ全員提案。"""
        if self._layer == "L0" and FEATURE_L0_CLEARANCE_MODE:
            return self._l0_urgent_future_offer(partner_list)

        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response

        is_l0 = (self._layer == "L0")
        sorted_p = self.select_partners_by_performance(partner_list, ratio=1.0)

        if is_l0:
            offsets_partners = [(i + 1, sorted_p) for i in range(L0_FUTURE_HORIZON)]
        else:
            grp1 = sorted_p[:int(len(sorted_p) * 0.5)]
            grp2 = sorted_p[int(len(sorted_p) * 0.5):int(len(sorted_p) * 0.8)]
            grp3 = sorted_p[int(len(sorted_p) * 0.8):]
            offsets_partners = [(1, grp1), (2, grp2), (3, grp3)]

        p = awi.n_lines * self._productivity
        for offset, grp in offsets_partners:
            fs = s + offset
            if fs >= n or not grp:
                continue
            if awi.total_sales_at(fs) > awi.n_lines:
                continue
            if is_l0 and FEATURE_L0_MAX_FUTURE:
                sn = max(0, awi.n_lines - awi.total_sales_at(fs))
            elif is_l0:
                exo = awi.current_exogenous_input_quantity
                sn  = int(max(0, min(awi.n_lines, exo + p + awi.current_inventory_input)
                              - awi.total_sales_at(fs)) / 2)
            else:
                sn = int(max(0, min(awi.n_lines, p + awi.current_inventory_input)
                             - awi.total_sales_at(fs)) / 3)
            if sn <= 0:
                continue
            for k, q in dict(zip(grp, distribute(sn, len(grp)))).items():
                if q > 0:
                    nmi_k = self.get_nmi(k)
                    if nmi_k is not None:
                        mn_k = nmi_k.issues[UNIT_PRICE].min_value
                        mx_k = nmi_k.issues[UNIT_PRICE].max_value
                    else:
                        mn_k = mx_k = None

                    if is_l0 and FEATURE_L0_FUTURE_LOWER_PRICE and mn_k is not None:
                        sell_price = mn_k + (mx_k - mn_k) * L0_FUTURE_SELL_RATE
                    else:
                        sell_price = self.best_price(k)

                    if FEATURE_PROFIT_BAND and nmi_k is not None:
                        floor      = self._min_sell_price(fs)
                        sell_price = max(sell_price, min(floor, mx_k))
                    if self._is_excess_inventory() and nmi_k is not None:
                        sell_price = min(sell_price, mn_k + (mx_k - mn_k) * L0_EXCESS_INITIAL_SELL_RATE)
                    response[k] = (q, fs, sell_price)
        return response

    def _l0_urgent_future_offer(self, partner_list) -> dict:
        """
        最も近い未充足の将来ステップに全消費者から clearance 価格で提案する。
        step+1 → step+2 → ... と順に埋めることで「砂漠ゾーン」を解消する。
        """
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response

        target_step = None
        target_need  = 0
        for fs in range(s + 1, n):
            need = max(0, awi.n_lines - awi.total_sales_at(fs))
            if need > 0:
                target_step = fs
                target_need  = need
                break

        if target_step is None or target_need <= 0:
            return response

        qty_each = max(1, target_need // max(1, len(partner_list)))
        for k in partner_list:
            nmi = self.get_nmi(k)
            if nmi is None:
                continue
            mn = nmi.issues[UNIT_PRICE].min_value
            mx = nmi.issues[UNIT_PRICE].max_value
            floor = self._min_sell_price(target_step)
            sell_price = max(floor, mn + (mx - mn) * L0_CLEARANCE_SELL_RATE)
            response[k] = (qty_each, target_step, sell_price)

        return response

    def best_price(self, partner):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        issue = nmi.issues[UNIT_PRICE]
        return issue.min_value if self.is_supplier(partner) else issue.max_value


if __name__ == "__main__":
    import sys
    from myagent.helpers.runner import run
    run([Rohn], sys.argv[1] if len(sys.argv) > 1 else "std")
