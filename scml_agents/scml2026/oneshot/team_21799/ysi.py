#!/usr/bin/env python
"""
Ysi (ANAC2026 SCML OneShot 提出版・2026-06-20)
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from itertools import chain, combinations, repeat

from negmas import ResponseType, SAOResponse
from numpy.random import choice
from scml.oneshot import OneShotSyncAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 成約しやすいパートナーを優先してカウンターオファーを配る
# OFF: 全パートナーを区別せずランダムに配る
FEATURE_PARTNER_SELECTION = True

# 受諾の数量許容差（在庫0志向のため小さく保つ）。
#   0 にすると「needs を超える組合せ」は diff==0（=ぴったり）のときだけ受諾し、
#   それ以外は「needs を下回る最良組合せ」を受諾して残りをカウンターで埋める。
ACCEPT_QTY_TOLERANCE = 0


# ======================================================================
# 補助関数
# ======================================================================

def distribute(q: int, n: int) -> list[int]:
    """q 個を n 個の bin に分配する（できるだけ各 bin に最低1個）。"""
    if n <= 0:
        return []
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

class Agent001(OneShotSyncAgent):
    """
    OneShot 用ベースライン。当日の needed_supplies / needed_sales に
    成約数量をぴったり合わせて在庫0を目指す。
    """

    def __init__(self, *args, ptoday=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        # ptoday: カウンターオファーを配るパートナーの割合（多いほど成約機会↑）
        self._base_ptoday = ptoday

        # パートナー成功率追跡
        self.partner_success_rate = defaultdict(lambda: 0.5)
        self.partner_negotiations = defaultdict(int)
        self.partner_successes = defaultdict(int)

    # ------------------------------------------------------------------
    # パートナー成功率
    # ------------------------------------------------------------------
    def update_partner_performance(self, partner, success):
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
        self.partner_success_rate[partner] = (
            self.partner_successes[partner] / self.partner_negotiations[partner]
        )

    def select_partners_by_performance(self, partners, ratio=None):
        """成功率の高い相手を優先して上位を選ぶ。FEATURE OFF なら全員返す。"""
        partners = list(partners)
        if not partners:
            return []
        if not FEATURE_PARTNER_SELECTION:
            random.shuffle(partners)
            return partners
        if ratio is None:
            ratio = self._base_ptoday

        scored = sorted(
            partners, key=lambda p: self.partner_success_rate[p], reverse=True
        )
        k = max(1, int(len(partners) * ratio))
        selected = scored[:k]
        # 探索のため下位からも少しだけ混ぜる
        remaining = scored[k:]
        if remaining and len(selected) < len(partners):
            extra = min(2, len(remaining), len(partners) - len(selected))
            selected.extend(random.sample(remaining, extra))
        return selected

    # ------------------------------------------------------------------
    # 役割判定（OneShot は当日納品のみ）
    # ------------------------------------------------------------------
    def is_supplier(self, partner):
        """自分が買う相手（input 側）。"""
        return partner in self.awi.my_suppliers

    def is_consumer(self, partner):
        """自分が売る相手（output 側）。"""
        return partner in self.awi.my_consumers

    # ------------------------------------------------------------------
    # 価格（単純な relative_time 線形譲歩）
    # ------------------------------------------------------------------
    def _price_bounds(self, partner):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        issue = nmi.issues[UNIT_PRICE]
        return issue.min_value, issue.max_value

    def _concession_price(self, partner, t: float):
        """
        提示価格。交渉序盤は自分に最も有利な価格、t（relative_time）が進むほど
        相手寄りに線形譲歩する。
          - 売り（consumer 相手）: 高値 maxp から minp へ下げる
          - 買い（supplier 相手）: 安値 minp から maxp へ上げる
        """
        b = self._price_bounds(partner)
        if b is None:
            return None
        mn, mx = b
        t = max(0.0, min(1.0, t))
        if self.is_consumer(partner):
            return mn + (mx - mn) * (1.0 - t)
        else:
            return mn + (mx - mn) * t

    def _price_acceptable(self, partner, price, t: float) -> bool:
        """相手のオファー価格が、現在の譲歩ラインから見て受け入れ可能か。"""
        b = self._price_bounds(partner)
        if b is None:
            return False
        mn, mx = b
        line = self._concession_price(partner, t)
        eps = 1e-9
        if self.is_consumer(partner):
            # 自分は売る → 相手の提示価格が譲歩ライン以上なら良い
            return price >= line - eps
        else:
            # 自分は買う → 相手の提示価格が譲歩ライン以下なら良い
            return price <= line + eps

    # ------------------------------------------------------------------
    # needs（在庫0にするための残り必要量）
    # ------------------------------------------------------------------
    def _needs(self):
        """(買う必要量, 売る必要量) を返す。0 未満は 0 に丸める。"""
        buy = max(0, int(self.awi.needed_supplies))
        sell = max(0, int(self.awi.needed_sales))
        return buy, sell

    # ------------------------------------------------------------------
    # 初回提案
    # ------------------------------------------------------------------
    def first_proposals(self):
        buy_needs, sell_needs = self._needs()
        response = {}

        suppliers = [p for p in self.negotiators if self.is_supplier(p)]
        consumers = [p for p in self.negotiators if self.is_consumer(p)]

        response |= self._distribute_offers(suppliers, buy_needs, t=0.0)
        response |= self._distribute_offers(consumers, sell_needs, t=0.0)
        return response

    def _distribute_offers(self, partners, needs, t: float) -> dict:
        """needs を選別したパートナーに分配して当日オファーを作る。"""
        out = {}
        if needs <= 0 or not partners:
            return out
        chosen = self.select_partners_by_performance(partners)
        if not chosen:
            return out
        # needs が人数より少なければ、配る相手を絞る
        if needs < len(chosen):
            chosen = random.sample(chosen, needs)
        for p, q in zip(chosen, distribute(needs, len(chosen))):
            if q > 0:
                price = self._concession_price(p, t)
                out[p] = (q, self.awi.current_step, price)
        return out

    # ------------------------------------------------------------------
    # カウンター（在庫0の核：powerset 厳密マッチング）
    # ------------------------------------------------------------------
    def counter_all(self, offers, states):
        awi = self.awi
        response = {}
        buy_needs, sell_needs = self._needs()

        for needs, all_partners in [
            (buy_needs, awi.my_suppliers),
            (sell_needs, awi.my_consumers),
        ]:
            # この側で当日・価格 OK のオファーだけを受諾候補にする
            cand = {}
            for p in all_partners:
                o = offers.get(p)
                if o is None:
                    continue
                if o[TIME] != awi.current_step or o[QUANTITY] <= 0:
                    continue
                t = states[p].relative_time if p in states else 1.0
                if self._price_acceptable(p, o[UNIT_PRICE], t):
                    cand[p] = o

            accepted = self._accept_best_combo(cand, needs, response)

            # まだ満たせていない分はカウンターオファーで埋める
            remaining = needs - accepted
            others = [
                p
                for p in all_partners
                if p in self.negotiators and p not in response
            ]
            if remaining > 0 and others:
                t_map = {
                    p: (states[p].relative_time if p in states else 1.0)
                    for p in others
                }
                # 代表時刻として平均 relative_time を使う
                t_avg = sum(t_map.values()) / len(t_map) if t_map else 1.0
                dist = self._distribute_offers(others, remaining, t=t_avg)
                for p, off in dist.items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, off)
            # needs を満たした、または配れない相手は交渉終了（None）
            for p in others:
                if p not in response:
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, None)

        return response

    def _accept_best_combo(self, cand: dict, needs: int, response: dict) -> int:
        """
        受諾候補 cand の部分集合のうち、合計数量が needs に最も近い組合せを受諾する。
        在庫0志向:
          - needs 以上になる組合せは diff <= ACCEPT_QTY_TOLERANCE のときのみ採用
            （過剰仕入れ/過剰販売を避ける）
          - そうでなければ needs 未満の最良組合せを採用し、残りはカウンターで埋める
        戻り値: 実際に受諾した合計数量。
        """
        if needs <= 0 or not cand:
            return 0

        best_over = (float("inf"), None)   # (adjusted_diff, combo) for offered >= needs
        best_under = (float("inf"), None)  # for 0 < offered < needs

        for combo in powerset(cand.keys()):
            if not combo:
                continue
            offered = sum(cand[p][QUANTITY] for p in combo)
            diff = abs(offered - needs)
            # 成功率の高い相手を含む組合せをわずかに優遇（同 diff の tie-break）
            quality = sum(self.partner_success_rate[p] for p in combo) / len(combo)
            adjusted = diff * (2.0 - quality)
            if offered >= needs:
                if adjusted < best_over[0]:
                    best_over = (adjusted, combo)
            else:
                if adjusted < best_under[0]:
                    best_under = (adjusted, combo)

        chosen = None
        over_raw_diff = None
        if best_over[1] is not None:
            combo = best_over[1]
            over_raw_diff = sum(cand[p][QUANTITY] for p in combo) - needs
        if over_raw_diff is not None and over_raw_diff <= ACCEPT_QTY_TOLERANCE:
            chosen = best_over[1]
        elif best_under[1] is not None:
            chosen = best_under[1]
        elif best_over[1] is not None:
            # under が無い（全オファーが needs 以上）。最小超過の組合せを受ける
            chosen = best_over[1]

        if chosen is None:
            return 0

        accepted_qty = 0
        for p in chosen:
            response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, cand[p])
            accepted_qty += cand[p][QUANTITY]
            self.update_partner_performance(p, True)
        # 受諾しなかった候補は成功率を下げる（負例）
        for p in cand:
            if p not in chosen:
                self.update_partner_performance(p, False)
        return accepted_qty


# ======================================================================
# ==== agent003 由来 ====
# ======================================================================






# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 組合せ評価を ufun.from_offers（実効用最大）で行う
# OFF: Agent001 と同じ数量差マッチング（_accept_best_combo）
FEATURE_UFUN_COMBO = True

# ON : 超過受諾の許容量を交渉ラウンドとともに緩める
# OFF: ACCEPT_QTY_TOLERANCE 固定（Agent001 相当）
FEATURE_TIME_RELAXED_ACCEPT = True
# 終盤(t=1)で許す超過量 = ceil(EXCESS_SLOPE * needs)。t に比例して 0→これ まで増える。
EXCESS_SLOPE = 0.34

# ON : 価格を積極的に粘る（自分に有利な価格を長く維持）
# OFF: Agent001 と同じ線形譲歩
FEATURE_AGGRESSIVE_PRICE = True
# 1.0=線形。<1 で粘り強く（有利価格を長く維持）。0.5 でかなり積極的。
#   売り側（高く売る）は積極的に粘る。
PRICE_HOLD_POWER = 0.5
#   買い側は shortfall(高コスト)回避のため、安値を粘りすぎず早めに譲歩して仕入を確保。
#   >1 で序盤から速く譲歩（=高い仕入値も早めに受け入れ、買い逃しを防ぐ）。
BUY_HOLD_POWER = 1.5

# ON : 受諾候補を価格ラインで足切りする（Agent001 流の価格ゲート）
#      OneShot では「価格を粘って受諾を拒否」すると売れ残り(disposal)が
#      価格マージンを上回って損になりやすい。rival(SyncRandom/RandDist/CostAverse)
#      は受諾を価格で足切りしない。既定 OFF（受諾は ufun に任せる）。
#      ※提案価格は FEATURE_AGGRESSIVE_PRICE で別途「高く」粘る。
FEATURE_PRICE_GATE = False

# ON : 提案数量を時間減衰で水増し（rival 流 over-order）。
#      序盤に needs*(1+OVERORDER_MAX) を提案し、部分受諾でも needs に届きやすくする。
#      売り/買い 両側に対称適用。OneShot 売り切り(disposal 回避)の主因対策。
FEATURE_OVERORDER = True
OVERORDER_MAX = 0.2        # 売り側(consumer)序盤(t=0)の水増し率
OVERORDER_MAX_BUY = 0.35   # 買い側(supplier)序盤の水増し率（shortfall回避で多め）
OVERORDER_MIN = 0.0        # 終盤(t=1)の水増し率
OVERORDER_EXP = 0.4        # max→min への減衰の速さ

# ON : 超過受諾の許容量を「買い側は多め（過剰買い=安いdisposal）」に方向別調整
#      買い側の超過許容 = EXCESS_SLOPE_BUY、売り側 = EXCESS_SLOPE。
FEATURE_DIRECTIONAL_EXCESS = True
EXCESS_SLOPE_BUY = 0.7

# ON : 実 disposal/shortfall コスト比に応じて over-order と超過許容を動的調整。
#      shortfall >> disposal の側ほど「不足回避」のため過剰寄りに振る（CostAverse 流）。
#      A/B 実測で base 比 +0.012（有意, 強敵フィールド400ワールド）→ rank11→4 に改善。
FEATURE_PENALTY_ADAPTIVE = True
PENALTY_RATIO_MIN = 0.25   # 比のクリップ下限
PENALTY_RATIO_MAX = 4.0    # 比のクリップ上限
PENALTY_RATIO_DEFAULT = 3.0  # disposal_cost≈0 のとき（不足が圧倒的に痛い）の既定比


class Agent003(Agent001):
    """ufun ベースの組合せ受諾 ＋ 時間緩和受諾 ＋ 積極価格粘り。

    主要チューニング値は __init__ 引数（＝インスタンス属性）として持つので、
    サブクラスで値を変えれば同一ワールド内 A/B 比較ができる。既定値は
    モジュール定数（実験で選んだ現行ベスト）。
    """

    def __init__(
        self,
        *args,
        sell_hold_power: float = PRICE_HOLD_POWER,
        buy_hold_power: float = BUY_HOLD_POWER,
        overorder_max_sell: float = OVERORDER_MAX,
        overorder_max_buy: float = OVERORDER_MAX_BUY,
        excess_slope_sell: float = EXCESS_SLOPE,
        excess_slope_buy: float = EXCESS_SLOPE_BUY,
        ptoday: float = 1.0,  # 全パートナーに配る（A/B で 0.85→1.0 が有利）
        **kwargs,
    ):
        super().__init__(*args, ptoday=ptoday, **kwargs)
        self.sell_hold_power = sell_hold_power
        self.buy_hold_power = buy_hold_power
        self.overorder_max_sell = overorder_max_sell
        self.overorder_max_buy = overorder_max_buy
        self.excess_slope_sell = excess_slope_sell
        self.excess_slope_buy = excess_slope_buy

    # ------------------------------------------------------------------
    # 価格：積極的な粘り（hold=(1-t)^POWER）
    # ------------------------------------------------------------------
    def _concession_price(self, partner, t: float):
        b = self._price_bounds(partner)
        if b is None:
            return None
        mn, mx = b
        t = max(0.0, min(1.0, t))
        is_sell = self.is_consumer(partner)
        if FEATURE_AGGRESSIVE_PRICE:
            power = self.sell_hold_power if is_sell else self.buy_hold_power
            hold = (1.0 - t) ** power
        else:
            hold = 1.0 - t
        # hold=1 → 自分に最も有利な価格、hold=0 → 相手寄りの限界価格
        if is_sell:
            # 売り：有利 = 高値 mx
            return mn + (mx - mn) * hold
        else:
            # 買い：有利 = 安値 mn
            return mx - (mx - mn) * hold

    # ------------------------------------------------------------------
    # 提案数量の方向別調整（OVERASK）
    # ------------------------------------------------------------------
    def _penalty_ratio(self, is_buy: bool) -> float:
        """その日の (不足ペナルティ / 廃棄コスト)。大きいほど不足が痛い → 過剰寄り。"""
        d = float(getattr(self.awi, "current_disposal_cost", 0.0) or 0.0)
        s = float(getattr(self.awi, "current_shortfall_penalty", 0.0) or 0.0)
        if d <= 1e-9:
            return PENALTY_RATIO_DEFAULT
        return max(PENALTY_RATIO_MIN, min(PENALTY_RATIO_MAX, s / d))

    def _overorder_fraction(self, t: float, is_buy: bool) -> float:
        t = max(0.0, min(1.0, t))
        mx = self.overorder_max_buy if is_buy else self.overorder_max_sell
        frac = mx - (mx - OVERORDER_MIN) * (t ** OVERORDER_EXP)
        if FEATURE_PENALTY_ADAPTIVE:
            # 不足が痛い側ほど水増しを増やす（比1.0で等倍, 比4で約2倍）
            frac *= 0.5 + 0.5 * min(2.0, self._penalty_ratio(is_buy))
        return frac

    def _distribute_offers(self, partners, needs, t: float) -> dict:
        partners = list(partners)
        if FEATURE_OVERORDER and needs > 0 and partners:
            is_buy = self.is_supplier(partners[0])
            needs = max(1, int(needs * (1.0 + self._overorder_fraction(t, is_buy))))
        return super()._distribute_offers(partners, needs, t)

    # ------------------------------------------------------------------
    # カウンター（ufun ベース組合せ ＋ 時間緩和受諾）
    # ------------------------------------------------------------------
    def counter_all(self, offers, states):
        awi = self.awi
        response = {}
        buy_needs, sell_needs = self._needs()

        for needs, all_partners in [
            (buy_needs, awi.my_suppliers),
            (sell_needs, awi.my_consumers),
        ]:
            # 当日・数量正・価格 OK のオファーを受諾候補にする
            cand = {}
            for p in all_partners:
                o = offers.get(p)
                if o is None:
                    continue
                if o[TIME] != awi.current_step or o[QUANTITY] <= 0:
                    continue
                t = states[p].relative_time if p in states else 1.0
                if (not FEATURE_PRICE_GATE) or self._price_acceptable(p, o[UNIT_PRICE], t):
                    cand[p] = o

            is_buy = all_partners is awi.my_suppliers
            chosen = self._select_combo(cand, needs, states, is_buy)
            accepted = 0
            for p in chosen:
                response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, cand[p])
                accepted += cand[p][QUANTITY]
                self.update_partner_performance(p, True)
            for p in cand:
                if p not in chosen:
                    self.update_partner_performance(p, False)

            # 残りはカウンターオファーで埋める
            remaining = needs - accepted
            others = [
                p for p in all_partners
                if p in self.negotiators and p not in response
            ]
            if remaining > 0 and others:
                t_map = {
                    p: (states[p].relative_time if p in states else 1.0)
                    for p in others
                }
                t_avg = sum(t_map.values()) / len(t_map) if t_map else 1.0
                dist = self._distribute_offers(others, remaining, t=t_avg)
                for p, off in dist.items():
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, off)
            for p in others:
                if p not in response:
                    response[p] = SAOResponse(ResponseType.REJECT_OFFER, None)

        return response

    # ------------------------------------------------------------------
    # 組合せ選択：効用最大（超過は時間緩和の上限内）
    # ------------------------------------------------------------------
    def _max_excess(self, cand: dict, needs: int, states, is_buy: bool) -> int:
        """受諾で許す超過量の上限（時間緩和 / 買い側は多め）。サブクラスで差し替え可。"""
        if FEATURE_TIME_RELAXED_ACCEPT:
            t = min(
                (states[p].relative_time for p in cand if p in states),
                default=1.0,
            )
            slope = self.excess_slope_buy if (FEATURE_DIRECTIONAL_EXCESS and is_buy) else self.excess_slope_sell
            if FEATURE_PENALTY_ADAPTIVE and is_buy:
                # 不足が痛いほど買い超過許容を広げる（過剰買い=安いdisposal で不足を回避）
                slope *= min(2.0, max(1.0, self._penalty_ratio(is_buy) / 1.5))
            return math.floor(slope * t * needs + 1e-9)
        return ACCEPT_QTY_TOLERANCE

    def _select_combo(self, cand: dict, needs: int, states, is_buy: bool = False) -> tuple:
        if needs <= 0 or not cand:
            return ()

        max_excess = self._max_excess(cand, needs, states, is_buy)

        best_u = -math.inf
        best_combo = None
        for combo in powerset(cand.keys()):
            if not combo:
                continue
            offered = sum(cand[p][QUANTITY] for p in combo)
            if offered - needs > max_excess:
                continue
            u = self._combo_utility(cand, combo, needs)
            if u > best_u:
                best_u = u
                best_combo = combo

        # 上限内の組合せが無い（全候補が超過）場合は最小超過の単体を許容
        if best_combo is None:
            best_combo = min(
                (c for c in powerset(cand.keys()) if c),
                key=lambda c: sum(cand[p][QUANTITY] for p in c) - needs,
                default=None,
            )
            if best_combo is None:
                return ()
        return best_combo

    def _combo_utility(self, cand: dict, combo: tuple, needs: int) -> float:
        """組合せの効用。ufun があれば実効用、無ければ数量差＋成功率の近似。"""
        if FEATURE_UFUN_COMBO and getattr(self, "ufun", None) is not None:
            try:
                return float(self.ufun.from_offers({p: cand[p] for p in combo}))
            except Exception:
                pass
        offered = sum(cand[p][QUANTITY] for p in combo)
        diff = abs(offered - needs)
        quality = sum(self.partner_success_rate[p] for p in combo) / len(combo)
        return -(diff * (2.0 - quality))


# ======================================================================
# ==== agent004 由来 ====
# ======================================================================





# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 相手アンカー型の譲歩（相手の提示価格水準 A へ収束）
# OFF: Agent003 と同じ（極端な限界価格へ収束する固定カーブ）
FEATURE_OPPONENT_AWARE = True

# 相手水準 A の EWMA 平滑化係数（大きいほど直近重視）
OPP_EWMA_ALPHA = 0.3
# ベイズ収縮の事前への重み（観測 n に対し τ 件ぶん事前へ引き寄せる）
OPP_SHRINK_TAU = 3.0
# 事前の正規化価格（市場中央）
OPP_PRIOR_NORM = 0.5


class Agent004(Agent003):
    """Agent003 ＋ 相手アンカー型譲歩（ベイズ収縮）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 相手ごとの正規化提示価格 EWMA と観測数（全シミュを通じて蓄積）
        self._opp_level = defaultdict(lambda: OPP_PRIOR_NORM)
        self._opp_count = defaultdict(int)

    # ------------------------------------------------------------------
    # 相手の提示価格を観測してモデル更新
    # ------------------------------------------------------------------
    def _record_offers(self, offers):
        for p, o in offers.items():
            if o is None:
                continue
            b = self._price_bounds(p)
            if b is None:
                continue
            mn, mx = b
            if mx <= mn:
                continue
            n = (o[UNIT_PRICE] - mn) / (mx - mn)
            n = max(0.0, min(1.0, n))
            if self._opp_count[p] == 0:
                self._opp_level[p] = n
            else:
                self._opp_level[p] = (
                    (1 - OPP_EWMA_ALPHA) * self._opp_level[p] + OPP_EWMA_ALPHA * n
                )
            self._opp_count[p] += 1

    def _opp_anchor(self, partner) -> float:
        """相手水準 A（ベイズ収縮済み・正規化 [0,1]）。"""
        n = self._opp_count[partner]
        if n <= 0:
            return OPP_PRIOR_NORM
        ewma = self._opp_level[partner]
        return (n * ewma + OPP_SHRINK_TAU * OPP_PRIOR_NORM) / (n + OPP_SHRINK_TAU)

    # ------------------------------------------------------------------
    # 価格：相手アンカー型の譲歩
    # ------------------------------------------------------------------
    def _concession_price(self, partner, t: float):
        if not FEATURE_OPPONENT_AWARE:
            return super()._concession_price(partner, t)
        b = self._price_bounds(partner)
        if b is None:
            return None
        mn, mx = b
        t = max(0.0, min(1.0, t))
        is_sell = self.is_consumer(partner)
        # 自分の有利端（正規化）: 売り=1.0(高値) / 買い=0.0(安値)
        favorable = 1.0 if is_sell else 0.0
        anchor = self._opp_anchor(partner)
        power = self.sell_hold_power if is_sell else self.buy_hold_power
        hold = (1.0 - t) ** power
        target = anchor + (favorable - anchor) * hold
        target = max(0.0, min(1.0, target))
        return mn + (mx - mn) * target

    # ------------------------------------------------------------------
    # カウンター: 先に相手モデルを更新してから Agent003 のロジックへ
    # ------------------------------------------------------------------
    def counter_all(self, offers, states):
        if FEATURE_OPPONENT_AWARE:
            self._record_offers(offers)
        return super().counter_all(offers, states)


# ======================================================================
# ==== agent005 由来 ====
# ======================================================================



# 採用した買い側パラメータ（A/B で Last 最良だった MidBuy 値）
OVERORDER_MAX_BUY = 0.20
EXCESS_SLOPE_BUY = 0.40


class Agent005(Agent004):
    """agent004 ＋ 買い側過剰買いの是正（over0.20 / excess0.40）。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("overorder_max_buy", OVERORDER_MAX_BUY)
        kwargs.setdefault("excess_slope_buy", EXCESS_SLOPE_BUY)
        super().__init__(*args, **kwargs)


# ======================================================================
# ==== agent009 由来 ====
# ======================================================================




# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 受諾を「時間で緩む品質しきい値」で gate（AlmostEqual 流・厳選して待つ）
# OFF: Agent005 と同じ（毎ラウンド最良 ufun 組合せを即受諾）
FEATURE_ACCEPT_QUALITY_GATE = True
QUALITY_TH = 0.60      # t=0 での要求品質（AlmostEqual と同じ 0.6）
QUALITY_DECAY = 0.5    # 終盤(t=1)で閾値が (1−DECAY) 倍に下がる


class Agent009(Agent005):
    """Agent005 ＋ 時間で緩む受諾品質しきい値（AlmostEqual 流）。"""

    def _select_combo(self, cand: dict, needs: int, states, is_buy: bool = False) -> tuple:
        combo = super()._select_combo(cand, needs, states, is_buy)
        if not FEATURE_ACCEPT_QUALITY_GATE or not combo or needs <= 0:
            return combo
        offered = sum(cand[p][QUANTITY] for p in combo)
        quality = 1.0 - abs(offered - needs) / max(needs, 1)
        t = min(
            (states[p].relative_time for p in cand if p in states),
            default=1.0,
        )
        threshold = QUALITY_TH * (1.0 - QUALITY_DECAY * t)
        if quality >= threshold:
            return combo
        # 品質不足 → 今ラウンドは受諾せずカウンターで待つ（厳選）
        return ()


# ======================================================================
# ==== agent017 由来 ====
# ======================================================================






FEATURE_SUBMITTABLE_BUY = True
OVER_ASK_MULT = 1.3
TIME_ROUNDS = 21
FORCE_CLOSE_ROUND = 7


class Agent017(Agent009):
    """売り=Agent009 / 買い=自前再実装の不足回避 ufun 受諾。"""

    # ------------------------------------------------------------------
    # ディスパッチ（L1 は自前の買い、L0 は Agent009）
    # ------------------------------------------------------------------
    def first_proposals(self):
        if FEATURE_SUBMITTABLE_BUY and self.awi.is_last_level:
            return self._buy_first_proposals()
        return Agent009.first_proposals(self)

    def counter_all(self, offers, states):
        if FEATURE_SUBMITTABLE_BUY and self.awi.is_last_level:
            return self._buy_counter_all(offers, states)
        return Agent009.counter_all(self, offers, states)

    # ------------------------------------------------------------------
    # 買い側ヘルパ
    # ------------------------------------------------------------------
    def _sample_buy_price(self) -> int:
        iss = self.awi.current_input_issues[UNIT_PRICE]
        return random.randint(int(iss.min_value), int(iss.max_value))

    def _loss_ratio(self, price: int) -> float:
        """不足1個 vs 過剰1個 の限界損失比（ufun 曲線から、退避は β/α）。"""
        try:
            step = self.awi.current_step
            n = int(self.awi.n_lines)
            u = [float(self.ufun.from_offers(((q, step, price),), (False,)))
                 for q in range(0, n + 1)]
            shortfall_loss = u[1] - u[0]
            disposal_loss = u[-2] - u[-1]
            if disposal_loss <= 1e-9:
                return 3.0
            return max(0.25, min(6.0, shortfall_loss / disposal_loss))
        except Exception:
            a = float(getattr(self.awi, "current_disposal_cost", 0) or 0)
            b = float(getattr(self.awi, "current_shortfall_penalty", 0) or 0)
            return 3.0 if a <= 1e-9 else max(0.25, min(6.0, b / a))

    def _buy_first_proposals(self):
        awi = self.awi
        needs = max(0, int(awi.needed_supplies))
        partners = [p for p in awi.my_suppliers if p in self.negotiators]
        if needs <= 0 or not partners:
            return {p: None for p in partners}
        price = self._sample_buy_price()
        step = awi.current_step
        alloc = distribute(max(1, int(needs * OVER_ASK_MULT)), len(partners))
        return {p: ((q, step, price) if q > 0 else None) for p, q in zip(partners, alloc)}

    def _buy_counter_all(self, offers, states):
        awi = self.awi
        cur = awi.current_step
        needs = max(0, int(awi.needed_supplies))
        off = {p: o for p, o in offers.items()
               if p in awi.my_suppliers and o is not None
               and o[TIME] == cur and o[QUANTITY] > 0}
        partners = set(off.keys())
        price = self._sample_buy_price()

        if needs <= 0 or not partners:
            return {p: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for p in offers if p in awi.my_suppliers}

        t = min((states[p].relative_time for p in partners if p in states), default=1.0)
        rnd = round(t * TIME_ROUNDS)
        ratio = self._loss_ratio(price)

        best_u, best_combo = -math.inf, ()
        min_sf, min_combo = 10 ** 9, ()
        for combo in powerset(partners):
            offered = sum(off[p][QUANTITY] for p in combo)
            sf = needs - offered
            if combo:
                try:
                    u = float(self.ufun.from_offers({p: off[p] for p in combo}))
                except Exception:
                    u = -abs(offered - needs)
            else:
                u = -math.inf
            if 0 <= sf < min_sf:
                min_sf, min_combo = sf, combo
            if u > best_u:
                best_u, best_combo = u, combo

        th = max(0, math.floor((ratio / 2.0) * (rnd - 1)))
        best_excess = sum(off[p][QUANTITY] for p in best_combo) - needs
        resp = {}

        def accept(combo, end_others):
            others = partners - set(combo)
            r = {p: SAOResponse(ResponseType.ACCEPT_OFFER, off[p]) for p in combo}
            if end_others:
                r |= {p: SAOResponse(ResponseType.END_NEGOTIATION, None) for p in others}
            return r, others

        if min_sf == 0 and min_combo:
            resp, _ = accept(min_combo, True)
        elif 0 <= best_excess <= th and best_combo:
            resp, _ = accept(best_combo, True)
        elif rnd > FORCE_CLOSE_ROUND and best_combo:
            resp, _ = accept(best_combo, True)
        elif min_sf <= 1 and min_combo:
            resp, others = accept(min_combo, False)
            others = list(others)
            rem = needs - sum(off[p][QUANTITY] for p in min_combo)
            if rem > 0 and others:
                for p, q in zip(others, distribute(rem, len(others))):
                    resp[p] = (SAOResponse(ResponseType.REJECT_OFFER, (q, cur, price))
                               if q > 0 else SAOResponse(ResponseType.REJECT_OFFER, None))
            else:
                for p in others:
                    resp[p] = SAOResponse(ResponseType.REJECT_OFFER, None)
        else:
            plist = list(partners)
            for p, q in zip(plist, distribute(max(1, int(needs * OVER_ASK_MULT)), len(plist))):
                resp[p] = (SAOResponse(ResponseType.REJECT_OFFER, (q, cur, price))
                           if q > 0 else SAOResponse(ResponseType.REJECT_OFFER, None))

        for p in offers:
            if p in awi.my_suppliers and p not in resp:
                resp[p] = SAOResponse(ResponseType.REJECT_OFFER, None)
        return resp


# ======================================================================
# ==== agent019 由来 ====
# ======================================================================





# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 売り L0 のオファー数量配分を受諾確率ポートフォリオ最適化にする
# OFF: Agent017 と同じ（distribute＋over-order）
FEATURE_PORTFOLIO_ALLOC = True

# 相手モデルの平滑化（Beta(1,1) 事前 → 未観測相手は p_accept=0.5）
OPP_PRIOR_A = 1.0
OPP_PRIOR_B = 1.0

# コスト情報が全く取れない日のフォールバック重み（売り逃しを不足より重く）
FALLBACK_UNDER = 1.0
FALLBACK_OVER = 0.4


# ---- 標準正規（math のみで実装）----
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT2PI


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


class Agent019(Agent017):
    """売り L0 のオファー配分を受諾確率ポートフォリオ最適化に置換（独自機構・第1段）。"""

    # ------------------------------------------------------------------
    # 相手モデル（シミュ内・日をまたいで学習。init で毎回リセット＝リーク防止）
    # ------------------------------------------------------------------
    def init(self):
        super().init()
        self._opp_succ = defaultdict(float)   # 成約した負け数（相手別・1日1交渉）
        self._opp_total = defaultdict(float)  # 交渉総数（相手別）

    def _partner_of(self, annotation):
        if not annotation:
            return None
        me = self.id
        s = annotation.get("seller")
        b = annotation.get("buyer")
        if s == me:
            return b
        if b == me:
            return s
        return None

    def on_negotiation_success(self, contract, mechanism):
        p = self._partner_of(getattr(contract, "annotation", None))
        if p is not None:
            self._opp_succ[p] += 1.0
            self._opp_total[p] += 1.0
        return super().on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        p = self._partner_of(annotation)
        if p is not None:
            self._opp_total[p] += 1.0
        return super().on_negotiation_failure(partners, annotation, mechanism, state)

    def _p_accept(self, p) -> float:
        succ = getattr(self, "_opp_succ", {})
        tot = getattr(self, "_opp_total", {})
        s = succ.get(p, 0.0)
        t = tot.get(p, 0.0)
        return (s + OPP_PRIOR_A) / (t + OPP_PRIOR_A + OPP_PRIOR_B)

    # ------------------------------------------------------------------
    # newsvendor の under/over 単価
    # ------------------------------------------------------------------
    def _alloc_costs(self) -> tuple[float, float]:
        awi = self.awi
        dr = float(getattr(awi, "current_disposal_cost", 0.0) or 0.0)
        sr = float(getattr(awi, "current_shortfall_penalty", 0.0) or 0.0)
        try:
            ci = float(awi.catalog_prices[awi.my_input_product])
            co = float(awi.catalog_prices[awi.my_output_product])
        except Exception:
            ci = co = 1.0
        # 売り逃し: 売上(co)を失い、余った input を disposal(dr·ci) する
        under = co + dr * ci
        # 売り過ぎ: 生産不能で shortfall(sr·co)
        over = sr * co
        if under <= 1e-9 and over <= 1e-9:
            return FALLBACK_UNDER, FALLBACK_OVER
        return max(1e-9, under), max(1e-9, over)

    def _exp_under_over(self, qmap: dict, pa: dict, needs: float) -> tuple[float, float]:
        """S=Σ q·Bern(pa) を正規近似し E[(needs−S)⁺], E[(S−needs)⁺] を返す。"""
        mean = 0.0
        var = 0.0
        for p, q in qmap.items():
            a = pa[p]
            mean += q * a
            var += q * q * a * (1.0 - a)
        sd = math.sqrt(var)
        if sd < 1e-9:
            return max(0.0, needs - mean), max(0.0, mean - needs)
        z = (needs - mean) / sd
        over = sd * (_phi(z) - z * (1.0 - _Phi(z)))   # E[(S−needs)⁺]
        over = max(0.0, over)
        under = over + (needs - mean)                 # E[(needs−S)⁺]
        return max(0.0, under), over

    def _alloc_var(self, qmap: dict, pa: dict) -> float:
        """S=Σ q·Bern(pa) の分散 Var[S]=Σ q²·pa·(1−pa)。"""
        return sum(q * q * pa[p] * (1.0 - pa[p]) for p, q in qmap.items())

    def _alloc_objective(self, qmap: dict, pa: dict, needs: float,
                         under_c: float, over_c: float) -> float:
        """配分の最小化目的（既定＝期待ペナルティ）。サブクラスで分散項等を足せる。"""
        u, o = self._exp_under_over(qmap, pa, needs)
        return under_c * u + over_c * o

    # ------------------------------------------------------------------
    # 受諾確率ポートフォリオ配分（貪欲）
    # ------------------------------------------------------------------
    def _alloc_cap_total(self, needs: int, n_partners: int) -> int:
        """配分総量の上限（暴走防止の安全弁）。サブクラスで締められる。"""
        return needs * 3 + n_partners + 2

    def _portfolio_alloc(self, partners: list, needs: int, t: float) -> dict:
        partners = list(partners)
        if needs <= 0 or not partners:
            return {}
        pa = {p: self._p_accept(p) for p in partners}
        under_c, over_c = self._alloc_costs()

        # 相手ごとの数量上限（交渉 issue から）
        qmax = {}
        for p in partners:
            nmi = self.get_nmi(p)
            try:
                qmax[p] = int(nmi.issues[QUANTITY].max_value)
            except Exception:
                qmax[p] = max(1, int(needs))

        q = {p: 0 for p in partners}
        cap_total = self._alloc_cap_total(int(needs), len(partners))
        cur_f = self._alloc_objective(q, pa, needs, under_c, over_c)
        total = 0
        while total < cap_total:
            best_gain, best_p, best_f = 1e-9, None, None
            for p in partners:
                if q[p] >= qmax[p]:
                    continue
                q[p] += 1
                f = self._alloc_objective(q, pa, needs, under_c, over_c)
                q[p] -= 1
                gain = cur_f - f
                if gain > best_gain:
                    best_gain, best_p, best_f = gain, p, f
            if best_p is None:
                break
            q[best_p] += 1
            cur_f = best_f
            total += 1

        step = self.awi.current_step
        out = {}
        for p in partners:
            if q[p] > 0:
                price = self._concession_price(p, t)
                out[p] = (q[p], step, price)
        return out

    # ------------------------------------------------------------------
    # フック：売り L0 のときだけポートフォリオ配分を使う
    # ------------------------------------------------------------------
    def _distribute_offers(self, partners, needs, t: float) -> dict:
        partners = list(partners)
        if FEATURE_PORTFOLIO_ALLOC and partners and self.is_consumer(partners[0]):
            out = self._portfolio_alloc(partners, needs, t)
            if out:
                return out
            # 配分が空（全相手の期待利得が低い）なら従来法で最低限配る
        return super()._distribute_offers(partners, needs, t)


# ======================================================================
# ==== agent023 由来 ====
# ======================================================================



# ======================================================================
# 機能フラグ
# ======================================================================

# 配分総量の上限倍率（agent019 既定は実質 3.0）。小さいほど暴走を強く抑える。
ALLOC_CAP_MULT = 2.0


class Agent023(Agent019):
    """agent019 ＋ 配分総量上限を締めて裾リスクを抑制。"""

    def _alloc_cap_total(self, needs: int, n_partners: int) -> int:
        return int(needs * ALLOC_CAP_MULT) + n_partners + 1


# ======================================================================
# ==== agent024 由来 ====
# ======================================================================






# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 買いに時間で緩む価格我慢を入れる（安く買う）
# OFF: agent023（=agent017 流の買い）と同じ
FEATURE_BUY_PRICE_PATIENCE = True
# このラウンドで価格上限を全開放（pmax まで受諾可）。FORCE_CLOSE_ROUND(7) より前に開く。
# スイープ(MEMO 28)で open3(早開放)が一貫最悪=粘りが効く。open5 が名目最良で採用。
PRICE_OPEN_ROUND = 5


class Agent024(Agent023):
    """agent023 ＋ 買い側の時間緩和価格我慢（input を安く買う）。"""

    # ------------------------------------------------------------------
    # 初回提案：低価格(pmin)でアンカー
    # ------------------------------------------------------------------
    def _buy_first_proposals(self):
        if not FEATURE_BUY_PRICE_PATIENCE:
            return super()._buy_first_proposals()
        awi = self.awi
        needs = max(0, int(awi.needed_supplies))
        partners = [p for p in awi.my_suppliers if p in self.negotiators]
        if needs <= 0 or not partners:
            return {p: None for p in partners}
        iss = awi.current_input_issues[UNIT_PRICE]
        price = int(iss.min_value)   # 最安でアンカー
        step = awi.current_step
        alloc = distribute(max(1, int(needs * OVER_ASK_MULT)), len(partners))
        return {p: ((q, step, price) if q > 0 else None) for p, q in zip(partners, alloc)}

    # ------------------------------------------------------------------
    # カウンター：時間で緩む価格我慢
    # ------------------------------------------------------------------
    def _buy_counter_all(self, offers, states):
        if not FEATURE_BUY_PRICE_PATIENCE:
            return super()._buy_counter_all(offers, states)

        awi = self.awi
        cur = awi.current_step
        needs = max(0, int(awi.needed_supplies))
        off_all = {p: o for p, o in offers.items()
                   if p in awi.my_suppliers and o is not None
                   and o[TIME] == cur and o[QUANTITY] > 0}

        if needs <= 0 or not off_all:
            return {p: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for p in offers if p in awi.my_suppliers}

        iss = awi.current_input_issues[UNIT_PRICE]
        pmin, pmax = int(iss.min_value), int(iss.max_value)
        t = min((states[p].relative_time for p in off_all if p in states), default=1.0)
        rnd = round(t * TIME_ROUNDS)

        # 価格我慢: 序盤は price_th 以下の安いオファーのみ受諾候補。PRICE_OPEN_ROUND で全開放。
        frac = 1.0 if rnd >= PRICE_OPEN_ROUND else (rnd / float(PRICE_OPEN_ROUND))
        price_th = pmin + (pmax - pmin) * frac
        counter_price = int(round(price_th))

        off = {p: o for p, o in off_all.items() if o[UNIT_PRICE] <= price_th + 1e-9}
        part = set(off.keys())
        ratio = self._loss_ratio(counter_price)

        # 数量決定（Agent017 流）を「安い候補 off」の上で行う
        best_u, best_combo = -math.inf, ()
        min_sf, min_combo = 10 ** 9, ()
        for combo in powerset(part):
            offered = sum(off[p][QUANTITY] for p in combo)
            sf = needs - offered
            if combo:
                try:
                    u = float(self.ufun.from_offers({p: off[p] for p in combo}))
                except Exception:
                    u = -abs(offered - needs)
            else:
                u = -math.inf
            if 0 <= sf < min_sf:
                min_sf, min_combo = sf, combo
            if u > best_u:
                best_u, best_combo = u, combo

        th = max(0, math.floor((ratio / 2.0) * (rnd - 1)))
        best_excess = sum(off[p][QUANTITY] for p in best_combo) - needs

        chosen = ()
        if min_sf == 0 and min_combo:
            chosen = min_combo
        elif 0 <= best_excess <= th and best_combo:
            chosen = best_combo
        elif rnd > FORCE_CLOSE_ROUND and best_combo:
            chosen = best_combo
        elif min_sf <= 1 and min_combo:
            chosen = min_combo

        resp = {}
        accepted_qty = 0
        for p in chosen:
            resp[p] = SAOResponse(ResponseType.ACCEPT_OFFER, off[p])
            accepted_qty += off[p][QUANTITY]

        # 未受諾(高価格で除外した相手を含む)は、残量があれば低価格でカウンターして待つ
        remaining = needs - accepted_qty
        rest = [p for p in off_all if p not in resp]
        if remaining > 0 and rest:
            for p, q in zip(rest, distribute(max(1, int(remaining * OVER_ASK_MULT)), len(rest))):
                resp[p] = (SAOResponse(ResponseType.REJECT_OFFER, (q, cur, counter_price))
                           if q > 0 else SAOResponse(ResponseType.REJECT_OFFER, None))
        else:
            for p in rest:
                resp[p] = SAOResponse(ResponseType.REJECT_OFFER, None)

        for p in offers:
            if p in awi.my_suppliers and p not in resp:
                resp[p] = SAOResponse(ResponseType.REJECT_OFFER, None)
        return resp


# ======================================================================
# ==== agent025 由来 ====
# ======================================================================



# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 市況連動で売りの強気度を変える
# OFF: agent024 と同じ（固定 sell_hold_power）
FEATURE_MARKET_ADAPTIVE = True
# 強気度の振れ幅。大きいほど好況/不況で粘りを強く変える。
AGGR_GAIN = 0.6
# margin 正規化のスケール（出力価格に対するマージン比がこの値で F=1 側へ）
MARGIN_SCALE = 0.30
# disposal 正規化のスケール（disposal_cost 分布のおおよその上限 μ~U(0,0.2)）
DISPOSAL_SCALE = 0.20


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Agent025(Agent024):
    """agent024 ＋ 市況連動の売り強気度（好況=強気/不況=保守）。"""

    def _market_favorability(self) -> float:
        awi = self.awi
        try:
            tp = awi.trading_prices
            out_p = float(tp[awi.my_output_product])
            in_p = float(tp[awi.my_input_product])
        except Exception:
            try:
                out_p = float(awi.catalog_prices[awi.my_output_product])
                in_p = float(awi.catalog_prices[awi.my_input_product])
            except Exception:
                return 0.5
        cost = float(getattr(awi.profile, "cost", 0.0) or 0.0)
        margin = out_p - in_p - cost
        margin_norm = _clip01(margin / max(1e-9, out_p * MARGIN_SCALE))
        disp = float(getattr(awi, "current_disposal_cost", 0.0) or 0.0)
        disp_norm = _clip01(disp / DISPOSAL_SCALE)
        return _clip01(0.5 * margin_norm + 0.5 * (1.0 - disp_norm))

    def _concession_price(self, partner, t: float):
        # 売り(consumer 相手)のときだけ市況連動で粘りを調整。買い側は agent024 のまま。
        if not (FEATURE_MARKET_ADAPTIVE and self.is_consumer(partner)):
            return super()._concession_price(partner, t)
        f = self._market_favorability()
        old = self.sell_hold_power
        self.sell_hold_power = max(0.2, old * (1.0 - AGGR_GAIN * (f - 0.5)))
        try:
            return super()._concession_price(partner, t)
        finally:
            self.sell_hold_power = old


# ======================================================================
# ==== agent030 由来 ====
# ======================================================================





# ======================================================================
# 機能フラグ
# ======================================================================

# ON : 売り(consumer 相手)に相手別価格フロアを課す（価格鈍感な相手から margin を抜く）
FEATURE_ELASTIC_SELL = True
# ON : 買い(supplier 相手)にも課す。買い側変更は過去裏目が多いので既定 OFF。
FEATURE_ELASTIC_BUY = False

# EMA の学習率（1日数交渉と疎なので緩め）
EMA_ALPHA = 0.30
# 信頼度が飽和する成約数（これ未満は floor を弱める）
CONF_K = 4.0
# 相手の受諾有利度から引く安全マージン（取りこぼし防止。大きいほど保守）
PRICE_SLACK = 0.10
# rbar の初期値（データ無し＝中立 0.5。floor は conf=0 で無効なので実害なし）
RBAR_DEFAULT = 0.5


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Agent030(Agent025):
    """agent025 ＋ パートナー別価格弾力性フロア（価格鈍感な相手を価格搾取）。"""

    def init(self):
        super().init()
        # 相手別: 実際に取れた自分有利度 r の EMA と観測回数
        self._rbar = defaultdict(lambda: RBAR_DEFAULT)
        self._rn = defaultdict(float)

    def _bounds_for(self, partner):
        """価格レンジ (mn, mx)。交渉中は nmi、成約後は当日の awi issues から取る。"""
        b = self._price_bounds(partner)
        if b is not None:
            return b
        try:
            awi = self.awi
            issues = (awi.current_output_issues if self.is_consumer(partner)
                      else awi.current_input_issues)
            iss = issues[UNIT_PRICE]
            return iss.min_value, iss.max_value
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 成約観測：相手から取れた有利度 r を EMA 更新
    # ------------------------------------------------------------------
    def on_negotiation_success(self, contract, mechanism):
        try:
            p = self._partner_of(getattr(contract, "annotation", None))
            agr = getattr(contract, "agreement", None) or {}
            price = agr.get("unit_price", None)
            if p is not None and price is not None:
                b = self._bounds_for(p)
                if b is not None:
                    mn, mx = b
                    span = mx - mn
                    if span > 1e-9:
                        if self.is_consumer(p):       # 売り：高値ほど有利
                            r = (float(price) - mn) / span
                        else:                          # 買い：安値ほど有利
                            r = (mx - float(price)) / span
                        r = _clip01(r)
                        old = self._rbar[p]
                        self._rbar[p] = (1.0 - EMA_ALPHA) * old + EMA_ALPHA * r
                        self._rn[p] += 1.0
        except Exception:
            pass
        return super().on_negotiation_success(contract, mechanism)

    # ------------------------------------------------------------------
    # 価格：base(agent025) に相手別の «下げ止め» フロアを課す
    # ------------------------------------------------------------------
    def _concession_price(self, partner, t: float):
        base = super()._concession_price(partner, t)
        if base is None:
            return base
        is_sell = self.is_consumer(partner)
        if not ((is_sell and FEATURE_ELASTIC_SELL) or
                ((not is_sell) and FEATURE_ELASTIC_BUY)):
            return base
        b = self._price_bounds(partner)
        if b is None:
            return base
        mn, mx = b
        span = mx - mn
        if span <= 1e-9:
            return base
        # base 価格を «自分有利度» に変換
        if is_sell:
            base_r = (base - mn) / span
        else:
            base_r = (mx - base) / span
        conf = min(1.0, self._rn[partner] / CONF_K)
        if conf <= 0.0:
            return base
        floor_r = max(0.0, self._rbar[partner] - PRICE_SLACK)
        final_r = max(base_r, conf * floor_r)
        final_r = _clip01(final_r)
        # 有利度を価格に戻す
        if is_sell:
            return mn + span * final_r
        else:
            return mx - span * final_r


# ======================================================================
# ==== agent036 由来（買いL1 = 本家 CostAverse の忠実再実装） ====
# ======================================================================

def distribute_evenly(total: int, n: int) -> list[int]:
    """total を n 個へ均等分配（本家 CostAverse の distribute_evenly 同等）。"""
    if n <= 0:
        return [0] * max(0, n)
    base = total // n
    rem = total % n
    dist = [base + (1 if i < rem else 0) for i in range(n)]
    random.shuffle(dist)
    return dist


class Agent036(Agent009):
    """売り=Agent009 / 買い=本家 CostAverse の忠実再実装（買いメソッドを Agent037 が借用）。"""

    def first_proposals(self):
        if self.awi.is_last_level:
            return self._buy_first_proposals()
        return Agent009.first_proposals(self)

    def counter_all(self, offers, states):
        if self.awi.is_last_level:
            return self._buy_counter_all(offers, states)
        return Agent009.counter_all(self, offers, states)

    def _util(self, combo_offers: tuple) -> float:
        try:
            return float(self.ufun.from_offers(
                combo_offers,
                tuple([self.awi.is_first_level] * len(combo_offers)),
                False, False,
            ))
        except Exception:
            offered = sum(o[QUANTITY] for o in combo_offers)
            return -abs(offered)

    def _buy_price(self) -> int:
        iss = self.awi.current_input_issues[UNIT_PRICE]
        return random.randint(int(iss.min_value), int(iss.max_value))

    def _buy_first_proposals(self):
        awi = self.awi
        needs = int(awi.needed_supplies)
        partners = [p for p in awi.my_suppliers if p in self.negotiators]
        if needs <= 0 or not partners:
            return {p: None for p in partners}
        price = self._buy_price()
        step = awi.current_step
        dist = distribute_evenly(int(needs * 1.3), len(partners))
        return {p: ((q, step, price) if q > 0 else None) for p, q in zip(partners, dist)}

    def _buy_counter_all(self, offers, states):
        awi = self.awi
        step = awi.current_step
        needs = int(awi.needed_supplies)
        sup_offers = {p: o for p, o in offers.items()
                      if p in awi.my_suppliers and o is not None
                      and o[TIME] == step and o[QUANTITY] > 0}
        partners = set(sup_offers.keys())
        if needs <= 0 or not partners:
            return {p: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for p in offers if p in awi.my_suppliers}
        price = self._buy_price()
        u = [self._util(((q, step, price),)) for q in range(0, 13)]
        shortfall_loss = u[1] - u[0]
        disposal_loss = u[-2] - u[-1]
        ratio = (shortfall_loss / disposal_loss) if abs(disposal_loss) > 1e-12 else float("inf")
        current_time = min(s.relative_time for s in states.values()) if states else 1.0
        current_round = round(current_time * 21)
        plist = list(powerset(partners))
        best_diff, best_idx = float("inf"), -1
        min_sf, min_idx = 10, -1
        for i, ids in enumerate(plist):
            offered = sum(sup_offers[p][QUANTITY] for p in ids)
            sf = needs - offered
            diff = -self._util(tuple(sup_offers[k] for k in ids))
            if 0 <= sf < min_sf:
                min_sf, min_idx = sf, i
            if diff < best_diff:
                best_diff, best_idx = diff, i
        if ratio == float("inf"):
            th = 10 ** 9
        else:
            th = max(0, math.floor((ratio / 2.0) * (current_round - 1)))
        partner_ids = plist[min_idx]
        others = list(partners.difference(partner_ids))
        best_ids = plist[best_idx]
        best_excess = sum(sup_offers[p][QUANTITY] for p in best_ids) - needs
        best_others = list(partners.difference(best_ids))

        def acc(k):
            return SAOResponse(ResponseType.ACCEPT_OFFER, sup_offers[k])

        def end(k):
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        def rej(q):
            return (SAOResponse(ResponseType.REJECT_OFFER, (q, step, price))
                    if q > 0 else SAOResponse(ResponseType.REJECT_OFFER, None))

        resp = {}
        if min_sf == 0:
            resp |= {k: acc(k) for k in partner_ids}
            resp |= {k: end(k) for k in others}
        elif 0 <= best_excess <= th:
            resp |= {k: acc(k) for k in best_ids}
            resp |= {k: end(k) for k in best_others}
        elif current_round > 7 and len(partner_ids) > 0:
            resp |= {k: acc(k) for k in best_ids}
            resp |= {k: end(k) for k in others}
        elif min_sf <= 1:
            offered = sum(sup_offers[p][QUANTITY] for p in partner_ids)
            excess_needs = needs - offered
            if excess_needs < 0:
                resp |= {k: end(k) for k in partners}
            else:
                dist = dict(zip(others, distribute_evenly(int(excess_needs), len(others))))
                resp |= {k: acc(k) for k in partner_ids}
                resp |= {k: rej(q) for k, q in dist.items()}
        else:
            dist = dict(zip(partners, distribute_evenly(int(max(0, needs)), len(partners))))
            resp |= {k: rej(q) for k, q in dist.items()}
        for p in offers:
            if p in awi.my_suppliers and p not in resp:
                resp[p] = SAOResponse(ResponseType.REJECT_OFFER, None)
        return resp


# ======================================================================
# ==== agent037 由来（売りL0 = 独自核 / 買いL1 = agent036 忠実 CostAverse） ====
# ======================================================================

class Agent037(Agent030):
    """売り = agent030 独自核 / 買い = agent036 忠実 CostAverse（メソッド借用）。"""

    _util = Agent036._util
    _buy_price = Agent036._buy_price
    _buy_first_proposals = Agent036._buy_first_proposals
    _buy_counter_all = Agent036._buy_counter_all

    def first_proposals(self):
        if self.awi.is_last_level:
            return self._buy_first_proposals()
        return super().first_proposals()

    def counter_all(self, offers, states):
        if self.awi.is_last_level:
            return self._buy_counter_all(offers, states)
        return super().counter_all(offers, states)


# ======================================================================
# ==== agent039 由来（需要逼迫連動の受諾緊急化・売りL0 独自機構） ====
# ======================================================================
# ON : 売りL0で「自分に来た買い手需要の逼迫度」に応じ受諾しきい値を一律に緩める（disposal回収）
FEATURE_DEMAND_URGENCY = True
DEMAND_REF = 2.0   # 需要余剰がこの値以上なら潤沢（緩めない）
URGENCY = 0.5      # 完全逼迫時にしきい値を下げる最大割合


class Agent039(Agent037):
    """agent037 ＋ 需要逼迫連動の受諾緊急化（売りL0・分散を壊さない一律しきい値調整）。"""

    URGENCY_MIN_T = 0.0
    URGENCY_STRICT = False

    def init(self):
        super().init()
        self._sell_offered = defaultdict(float)
        self._sell_day = -1

    def counter_all(self, offers, states):
        awi = self.awi
        if awi.is_first_level:
            if awi.current_step != self._sell_day:
                self._sell_day = awi.current_step
                self._sell_offered = defaultdict(float)
            cur = awi.current_step
            for p, o in offers.items():
                if (p in awi.my_consumers and o is not None
                        and o[TIME] == cur and o[QUANTITY] > 0):
                    if o[QUANTITY] > self._sell_offered[p]:
                        self._sell_offered[p] = float(o[QUANTITY])
        return super().counter_all(offers, states)

    def _select_combo(self, cand: dict, needs: int, states, is_buy: bool = False) -> tuple:
        # agent009 の固定ゲートを一時無効化し「ゲート前の最良 combo」を取得（チェーン非依存）
        g = globals()
        saved = g["FEATURE_ACCEPT_QUALITY_GATE"]
        g["FEATURE_ACCEPT_QUALITY_GATE"] = False
        try:
            combo = super()._select_combo(cand, needs, states, is_buy)
        finally:
            g["FEATURE_ACCEPT_QUALITY_GATE"] = saved
        if not saved or not combo or needs <= 0:
            return combo
        offered = sum(cand[p][QUANTITY] for p in combo)
        quality = 1.0 - abs(offered - needs) / max(needs, 1)
        t = min((states[p].relative_time for p in cand if p in states), default=1.0)
        threshold = QUALITY_TH * (1.0 - QUALITY_DECAY * t)
        if (FEATURE_DEMAND_URGENCY and (not is_buy) and self.awi.is_first_level
                and t >= self.URGENCY_MIN_T):
            offered_now = sum(self._sell_offered.values())
            surplus = offered_now - needs
            if self.URGENCY_STRICT:
                tight = 1.0 if surplus < 0 else 0.0
            else:
                tight = _clip01(1.0 - surplus / DEMAND_REF)
            threshold *= (1.0 - URGENCY * tight)
        if quality >= threshold:
            return combo
        return ()


# ======================================================================
# ==== 提出クラス ====
# ======================================================================
__all__ = ["Ysi"]


class Ysi(Agent037):
    """提出版エージェント（0620 = Agent037 ベース／urgency 無し）。

    構成:
      売りL0 = 独自核（newsvendor ポートフォリオ配分 + 市況連動売り強気 + 相手別価格弾力性）
               ※ 0618版にあった「需要逼迫連動の受諾緊急化(urgency)」は本版では外している。
      買いL1 = 本家 CostAverse の忠実再実装（提出可・両フィールド頑健）

    切り分け実験用: 0618(=Agent039, urgency 有り) からの差分は urgency の有無のみ。
    """
    pass


if __name__ == "__main__":
    # 単体動作確認（自己完結・myagent 非依存）: Ysi を簡易ワールドで1回走らせてスコア表示。
    from scml.oneshot import SCML2024OneShotWorld
    from scml.oneshot.agents import GreedyOneShotAgent

    world = SCML2024OneShotWorld(
        **SCML2024OneShotWorld.generate(
            agent_types=[Ysi, GreedyOneShotAgent], n_steps=50, n_agents_per_process=2),
        construct_graphs=False,
    )
    world.run()
    print("Ysi 単体動作 OK / scores:", world.scores())
