#!/usr/bin/env python
from __future__ import annotations

import os
import json
import ast
import random
import math
from collections import Counter, defaultdict
from itertools import repeat

from negmas import *
from numpy.random import choice
from scml.std import *

__all__ = ["GS3"]

# GS3: GS2 base + market shortage state, yen-based DP score, production control, and linear Q approximation. Reads/saves Q-tables in knapQ10.


def distribute(q: int, n: int) -> list[int]:
    """Distributes q units over n bins with at least one item per non-empty bin when possible."""
    if n <= 0:
        return []
    if q <= 0:
        return [0] * n
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]

FEATURE_DIM = 13
class GS3(StdSyncAgent):
    ACT_PROFIT = 0
    ACT_BALANCE = 1
    ACT_NEED = 2
    ACT_TRUST = 3
    ACT_REJECT = 4

    ACTION_NAMES = {
        ACT_PROFIT: "PROFIT",
        ACT_BALANCE: "BALANCE",
        ACT_NEED: "NEED",
        ACT_TRUST: "TRUST",
        ACT_REJECT: "REJECT",
    }

    # 提出用では、Qテーブル・オフラインスナップショット・訪問回数・線形重みを
    # すべてインスタンス変数として保持する。
    # これにより、knapQ10 から事前学習Q値を読み込んでも、同一Pythonプロセス内の
    # 別world/別インスタンスへQ状態が共有されない。
    FEATURE_DIM = 13

    def __init__(
        self,
        *args,
        threshold=None,
        ptoday=0.70,
        productivity=0.7,
        output_dir="knapQ10",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1
        self._threshold = threshold
        self._base_ptoday = ptoday
        self._productivity = productivity
        # 出力先は環境変数で上書き可能。容量制限対策として /tmp などを指定できる。
        self.output_dir = os.environ.get("GS3_Q_DIR", output_dir)

        self.partner_success_rate = defaultdict(lambda: 0.5)
        self.partner_negotiations = defaultdict(int)
        self.partner_successes = defaultdict(int)
        self.partner_total_qty = defaultdict(int)
        self.partner_total_price_score_sell = defaultdict(float)
        self.partner_total_price_score_buy = defaultdict(float)
        self.partner_rejections = defaultdict(int)

        self.last_balances = []
        self.initial_balance = None
        # sideごとの市場価格優位性の履歴。状態に市場トレンドを入れるために使う。
        self.market_price_history = {"buy": [], "sell": []}
        self.aggressive_mode = False
        self.conservative_mode = False

        # 提出用: Q値・線形近似重みは更新せず、学習済みQを固定参照する。
        self.q_alpha = 0.0
        self.q_gamma = 0.85

        # 提出用: 探索率は低く固定する。
        self.q_epsilon_start = 0.01
        self.q_epsilon_min = 0.01
        self.q_epsilon = 0.01

        


        # オフライン・オンライン融合設定
        # 序盤は学習済みQ値を強く使い、同じ状態でのオンライン更新回数が増えるほど
        # その場の学習結果を強く使う。
        self.offline_q_weight_start = 0.75
        self.offline_q_weight_min = 0.20
        self.offline_confidence_scale = 6.0
        self.online_confidence_scale = 8.0

        # AS23と同様に、保存済みQ値を次回読み込む。ただし完全には消さず、
        # 古いQ値の影響を少し弱めるために読み込み時に減衰させる。
        self.q_value_decay = 1.0
        self.q_persistent_file = os.path.join(self.output_dir, "gs3_qtable_persistent.json")

        # 提出用: Qテーブルは各インスタンス専用に作成し、knapQ10 の保存済みQ値を
        # そのインスタンスへコピーして使う。クラス変数へは保存しないため、
        # world間・インスタンス間でQ状態を共有しない。
        self.q_sell = defaultdict(lambda: list(self._initial_q_values("sell")))
        self.q_buy = defaultdict(lambda: list(self._initial_q_values("buy")))
        self.w_sell = {act: [0.0] * GS3.FEATURE_DIM for act in range(5)}
        self.w_buy = {act: [0.0] * GS3.FEATURE_DIM for act in range(5)}
        self.q_offline_sell = {}
        self.q_offline_buy = {}
        self.q_visits_sell = defaultdict(lambda: [0, 0, 0, 0, 0])
        self.q_visits_buy = defaultdict(lambda: [0, 0, 0, 0, 0])
        self._load_persistent_q_tables()
        self._ensure_offline_q_snapshot()

        self.last_sell_policy_state = None
        self.last_sell_policy_action = self.ACT_BALANCE
        self.last_buy_policy_state = None
        self.last_buy_policy_action = self.ACT_BALANCE

        # ---- 出力用ログ ----
        self.q_update_history = []
        self.balance_history = []
        self.step_summary_history = []
        self._artifacts_saved = False

    def _update_linear_epsilon(self):
        """提出用では探索率を 0.01 に固定する。"""
        self.q_epsilon_start = 0.01
        self.q_epsilon_min = 0.01
        self.q_epsilon = 0.01

    def _load_persistent_q_tables(self):
        """保存済みQテーブルを、このインスタンス専用Qテーブルへ読み込む。

        クラス変数には読み込まないため、同一Pythonプロセス内の別world/別インスタンスへ
        Q状態は共有されない。提出用では読み込み後の更新・保存もしない。
        """
        if not os.path.exists(self.q_persistent_file):
            return

        try:
            with open(self.q_persistent_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for side, table in [("sell", self.q_sell), ("buy", self.q_buy)]:
                saved_table = data.get(side, {})
                for state_repr, values in saved_table.items():
                    try:
                        state = ast.literal_eval(state_repr)
                    except Exception:
                        continue

                    if not isinstance(values, list) or len(values) != 5:
                        continue

                    state = self._normalize_state_key(state)
                    table[state] = [float(v) * self.q_value_decay for v in values]

            for key, values in data.get("w_sell", {}).items():
                try:
                    action = int(key)
                    if 0 <= action < 5 and isinstance(values, list):
                        vals = [float(v) * self.q_value_decay for v in values[:GS3.FEATURE_DIM]]
                        self.w_sell[action] = vals + [0.0] * (GS3.FEATURE_DIM - len(vals))
                except Exception:
                    continue
            for key, values in data.get("w_buy", {}).items():
                try:
                    action = int(key)
                    if 0 <= action < 5 and isinstance(values, list):
                        vals = [float(v) * self.q_value_decay for v in values[:GS3.FEATURE_DIM]]
                        self.w_buy[action] = vals + [0.0] * (GS3.FEATURE_DIM - len(vals))
                except Exception:
                    continue

        except Exception as e:
            pass

    def _save_persistent_q_tables(self):
        """提出用ではQテーブルを保存しない。"""
        return



    def step(self):
        super().step()
        if self.is_upstream_level():
            self.q_gamma = max(self.q_gamma, 0.87)
            self.q_epsilon_min = max(self.q_epsilon_min, 0.01)    

        self.aggressive_mode = False
        self.conservative_mode = False

        base_threshold = self.awi.n_lines * 0.1
        inventory_ratio = self.awi.current_inventory_input / max(1, self.awi.n_lines)

        if inventory_ratio < 0.3:
            self._threshold = max(1, int(base_threshold * 1.5))
        elif inventory_ratio > 0.8:
            self._threshold = max(1, int(base_threshold * 0.7))
        else:
            self._threshold = max(1, int(base_threshold))

        time_left = (self.awi.n_steps - self.awi.current_step) / max(1, self.awi.n_steps)
        if time_left < 0.3:
            self._threshold = max(1, int(self._threshold * 1.8))
            self.aggressive_mode = True
        elif time_left < 0.6:
            self._threshold = max(1, int(self._threshold * 1.3))

        balance = float(getattr(self.awi, "current_balance", 0.0))
        if self.initial_balance is None:
            self.initial_balance = balance
        self.last_balances.append(balance)
        self.balance_history.append({
            "step": int(self.awi.current_step),
            "balance": balance,
            "inventory_input": int(getattr(self.awi, "current_inventory_input", 0)),
            "inventory_output": int(getattr(self.awi, "current_inventory_output", 0)),
            "epsilon": float(self.q_epsilon),
        })
        if len(self.last_balances) > 6:
            self.last_balances.pop(0)

        if len(self.last_balances) >= 4:
            old_avg = sum(self.last_balances[:2]) / 2.0
            new_avg = sum(self.last_balances[-2:]) / 2.0
            trend = new_avg - old_avg
            # 固定値 -50 ではなく、初期資金・ライン数に応じた相対閾値を使う。
            base_balance = abs(float(self.initial_balance)) if self.initial_balance is not None else abs(balance)
            dynamic_threshold = max(50.0, 0.03 * max(1.0, base_balance), 5.0 * max(1, int(getattr(self.awi, "n_lines", 1))))
            if trend < -dynamic_threshold:
                self.conservative_mode = True
            elif trend > dynamic_threshold:
                self.conservative_mode = False

        self._update_linear_epsilon()

        if self.awi.current_step == self.awi.n_steps - 1 and not self._artifacts_saved:
            self._save_learning_artifacts()

    def _time_bin(self):
        ratio = self.awi.current_step / max(1, self.awi.n_steps - 1)
        if ratio < 0.33:
            return 0
        if ratio < 0.66:
            return 1
        return 2

    def _inventory_bin(self):
        inv = self.awi.current_inventory_input
        cap = max(1, self.awi.n_lines)
        ratio = inv / cap
        if ratio < 0.3:
            return 0
        if ratio < 0.8:
            return 1
        return 2

    def _need_bin(self, needs):
        cap = max(1, self.awi.n_lines)
        ratio = max(0, needs) / cap
        if ratio < 0.3:
            return 0
        if ratio < 0.7:
            return 1
        return 2

    def _offer_price_bin(self, offers, side):
        if not offers:
            return 1
        vals = [self.normalized_price_advantage(p, offer, side) for p, offer in offers.items()]
        if not vals:
            return 1
        avg = sum(vals) / len(vals)
        if avg < 0.33:
            return 0
        if avg < 0.66:
            return 1
        return 2

    def _current_market_price_average(self, offers, side):
        """現在オファー群の平均価格優位性を返す。

        値が大きいほど、そのsideにとって有利な価格である。
        buyなら安いほど高く、sellなら高いほど高くなる。
        """
        if not offers:
            return 0.5
        vals = [self.normalized_price_advantage(p, offer, side) for p, offer in offers.items()]
        return sum(vals) / max(1, len(vals)) if vals else 0.5

    def _market_trend_bin(self, side, offers):
        """市場全体の価格優位性トレンドを3段階に離散化する。

        0: 前回より不利
        1: ほぼ横ばい
        2: 前回より有利
        """
        current_avg = self._current_market_price_average(offers, side)
        history = self.market_price_history.get(side, [])
        if not history:
            return 1
        diff = current_avg - history[-1]
        if diff < -0.08:
            return 0
        if diff > 0.08:
            return 2
        return 1

    def _remember_market_price_average(self, side, offers):
        """次ステップ以降の状態化に使うため、市場価格平均を記録する。"""
        if not offers:
            return
        history = self.market_price_history.setdefault(side, [])
        history.append(self._current_market_price_average(offers, side))
        if len(history) > 6:
            del history[0]

    def _partner_quality_score(self, p, side):
        success = self.partner_success_rate[p]
        deals = self.partner_negotiations[p]
        volume = self.partner_total_qty[p]
        volume_score = min(1.0, volume / max(1, self.awi.n_lines * 2))
        if deals > 0:
            avg_price = (
                self.partner_total_price_score_sell[p] / deals
                if side == "sell"
                else self.partner_total_price_score_buy[p] / deals
            )
        else:
            avg_price = 0.5
        return 0.55 * success + 0.30 * avg_price + 0.15 * volume_score

    def _partner_quality_bin(self, partners, side):
        if not partners:
            return 1
        avg = sum(self._partner_quality_score(p, side) for p in partners) / len(partners)
        if avg < 0.40:
            return 0
        if avg < 0.70:
            return 1
        return 2

    def _balance_bin(self):
        """現在の資金状態を3段階に離散化する。

        0: 初期資金から大きく低下、または赤字
        1: 初期資金付近
        2: 初期資金から十分に増加
        """
        balance = float(getattr(self.awi, "current_balance", 0.0))
        if self.initial_balance is None:
            self.initial_balance = balance

        base = max(1.0, abs(float(self.initial_balance)))
        ratio = balance / base

        if balance < 0 or ratio < 0.80:
            return 0
        if ratio < 1.20:
            return 1
        return 2

    def _market_supply_bins(self):
        """外部レポートから市場全体の需給逼迫度を2値で推定する。

        戻り値:
            input_shortage_bin: 自分の入力製品が市場で不足気味なら1
            output_shortage_bin: 自分の出力製品が市場で不足気味なら1

        extrinsic_reports が取得できない環境では中立値0を返す。
        """
        total_input_produced = 0
        total_output_produced = 0
        try:
            reports = getattr(self.awi, "extrinsic_reports", None)
            if reports:
                for r in reports:
                    product = getattr(r, "product", None)
                    quantity = int(getattr(r, "quantity", 0) or 0)
                    if product == getattr(self.awi, "my_input_product", None):
                        total_input_produced += quantity
                    elif product == getattr(self.awi, "my_output_product", None):
                        total_output_produced += quantity
        except Exception:
            return 0, 0

        threshold = max(1, int(getattr(self.awi, "n_lines", 1)) * 2)
        input_shortage_bin = 1 if total_input_produced < threshold else 0
        output_shortage_bin = 1 if total_output_produced < threshold else 0
        return input_shortage_bin, output_shortage_bin

    def _normalize_state_key(self, state):
        """旧FS1の8要素状態との互換用に、balance_bin=1を補完する。

        FS1: (level, inv, need, time, price, quality, aggressive, conservative)
        GS3: (level, inv, need, time, price, quality, balance, aggressive, conservative)
        """
        if isinstance(state, tuple) and len(state) == 8:
            # 旧FS1: balance_bin, market_trend_bin, market shortage bins を補完
            return state[:6] + (1,) + state[6:] + (1, 0, 0)
        if isinstance(state, tuple) and len(state) == 9:
            # 旧GS1/GS3: market_trend_bin, market shortage bins を補完
            return state + (1, 0, 0)
        if isinstance(state, tuple) and len(state) == 10:
            # 旧GS2: market shortage bins を補完
            return state + (0, 0)
        return state

    def make_q_state(self, side, offers, needs):
        # 既存状態に、市場価格トレンドと市場全体の需給逼迫度を追加する。
        input_shortage_bin, output_shortage_bin = self._market_supply_bins()
        return (
            self._level_index(),
            self._inventory_bin(),
            self._need_bin(needs),
            self._time_bin(),
            self._offer_price_bin(offers, side),
            self._partner_quality_bin(list(offers.keys()), side),
            self._balance_bin(),
            int(self.aggressive_mode),
            int(self.conservative_mode),
            self._market_trend_bin(side, offers),
            input_shortage_bin,
            output_shortage_bin,
        )

    def _level_index(self) -> int:
        """SCMLの層番号を安全に推定する。

        まず AWI 側の属性を確認し，取れない場合はエージェントIDの
        ``@0``，``@1`` のような表記から推定する。最後の保険として，
        仕入れ先がない場合を0層，販売先がない場合を最終層とみなす。
        """
        for obj in [getattr(self, "awi", None), getattr(getattr(self, "awi", None), "profile", None)]:
            if obj is None:
                continue
            for name in ["level", "process", "production_level"]:
                value = getattr(obj, name, None)
                if isinstance(value, int):
                    return max(0, min(3, value))
        try:
            sid = str(getattr(self, "id", ""))
            if "@" in sid:
                return max(0, min(3, int(sid.rsplit("@", 1)[1])))
        except Exception:
            pass
        try:
            if len(getattr(self.awi, "my_suppliers", [])) == 0:
                return 0
            if len(getattr(self.awi, "my_consumers", [])) == 0:
                return 3
        except Exception:
            pass
        return 1

    def _initial_q_values(self, side: str) -> list[float]:
        """層・buy/sell別の初期Q値。

        これまでの学習曲線から，0層は調達不足で崩れやすく，1層は
        BALANCE寄りが安定，2層はPROFIT一辺倒だと大敗しやすく，
        3層はsell側のPROFIT/TRUSTが強い傾向があった。そのため，
        初期Q値を層ごとに分ける。

        並び順は [PROFIT, BALANCE, NEED, TRUST, REJECT]。
        """
        level = self._level_index()
        # 行動順: [PROFIT, BALANCE, NEED, TRUST, REJECT]
        # 方針: NEED > BALANCE > TRUST > PROFIT > REJECT
        # buy/sell の両方で、必要量を最優先し、BALANCEで安定化、TRUST/PROFITは微調整にする。
        if side == "buy":
            if level == 0:
                return [0.18, 0.58, 0.76, 0.30, -0.45]
            if level == 1:
                return [0.18, 0.62, 0.72, 0.32, -0.42]
            if level == 2:
                return [0.20, 0.58, 0.68, 0.34, -0.45]
            return [0.16, 0.54, 0.62, 0.30, -0.50]
        else:
            # sell側はbuy側と同じ比率に近づけるため、
            # NEEDを少し下げ、BALANCEを明確に上げる。
            if level == 0:
                return [0.16, 0.64, 0.65, 0.27, -0.45]
            if level == 1:
                return [0.16, 0.68, 0.69, 0.28, -0.42]
            if level == 2:
                return [0.18, 0.64, 0.65, 0.29, -0.45]
            return [0.16, 0.60, 0.59, 0.28, -0.50]

    def _ensure_q_state_initialized(self, side: str, state):
        """未知状態に層別初期Q値を入れる。"""
        table = self._q_table(side)
        if state not in table:
            table[state] = list(self._initial_q_values(side))

    def _layer_action_weights(self, action, side: str):
        """層・buy/sell別のナップサック評価重み。

        戻り値は
            (price_w, trust_w, dev_penalty, overflow_penalty, underfill_penalty)
        の5要素。

        方針は buy/sell ともに
            NEED > BALANCE > TRUST > PROFIT > REJECT
        とし、必要量充足を主軸、BALANCEを次点、TRUST/PROFITは微調整にする。
        """
        level = self._level_index()

        if side == "buy":
            if level == 0:
                table = {
                    self.ACT_PROFIT:  (2.1, 0.8, 1.5, 1.7, 2.3),
                    self.ACT_BALANCE: (2.3, 1.0, 1.8, 1.5, 3.0),
                    self.ACT_NEED:    (1.7, 0.9, 2.5, 1.2, 3.8),
                    self.ACT_TRUST:   (1.6, 2.0, 1.9, 1.4, 2.8),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            elif level == 1:
                table = {
                    self.ACT_PROFIT:  (2.2, 0.9, 1.5, 2.2, 1.8),
                    self.ACT_BALANCE: (2.4, 1.3, 2.0, 2.0, 2.6),
                    self.ACT_NEED:    (1.9, 1.0, 2.3, 1.7, 3.2),
                    self.ACT_TRUST:   (1.7, 2.6, 1.9, 2.1, 2.4),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            elif level == 2:
                table = {
                    self.ACT_PROFIT:  (2.4, 1.0, 1.6, 2.3, 2.0),
                    self.ACT_BALANCE: (2.4, 1.4, 2.0, 2.1, 2.7),
                    self.ACT_NEED:    (1.9, 1.1, 2.3, 1.9, 3.2),
                    self.ACT_TRUST:   (1.8, 2.7, 1.9, 2.2, 2.5),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            else:
                table = {
                    self.ACT_PROFIT:  (2.0, 1.1, 1.7, 2.6, 1.8),
                    self.ACT_BALANCE: (2.1, 1.4, 2.1, 2.6, 2.3),
                    self.ACT_NEED:    (1.7, 1.1, 2.3, 2.2, 2.7),
                    self.ACT_TRUST:   (1.7, 2.8, 1.9, 2.7, 2.0),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
        else:
            if level == 0:
                table = {
                    self.ACT_PROFIT:  (1.8, 0.8, 1.5, 1.8, 1.9),
                    self.ACT_BALANCE: (2.4, 1.2, 2.2, 1.9, 3.0),
                    self.ACT_NEED:    (1.7, 0.9, 2.1, 1.5, 2.7),
                    self.ACT_TRUST:   (1.7, 2.2, 1.8, 1.8, 2.1),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            elif level == 1:
                table = {
                    self.ACT_PROFIT:  (2.0, 0.9, 1.6, 2.1, 1.7),
                    self.ACT_BALANCE: (2.6, 1.4, 2.3, 2.1, 2.9),
                    self.ACT_NEED:    (1.8, 1.0, 2.1, 1.8, 2.7),
                    self.ACT_TRUST:   (1.7, 2.3, 1.9, 2.1, 2.2),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            elif level == 2:
                table = {
                    self.ACT_PROFIT:  (2.2, 1.0, 1.6, 2.3, 1.8),
                    self.ACT_BALANCE: (2.7, 1.5, 2.3, 2.3, 2.9),
                    self.ACT_NEED:    (1.9, 1.0, 2.1, 2.0, 2.7),
                    self.ACT_TRUST:   (1.8, 2.4, 1.9, 2.3, 2.3),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
            else:
                table = {
                    self.ACT_PROFIT:  (2.4, 1.0, 1.5, 2.3, 1.5),
                    self.ACT_BALANCE: (2.7, 1.4, 2.2, 2.2, 2.6),
                    self.ACT_NEED:    (1.9, 0.9, 2.0, 1.9, 2.4),
                    self.ACT_TRUST:   (1.8, 2.4, 1.7, 2.1, 1.9),
                    self.ACT_REJECT:  (0.0, 0.0, 0.0, 0.0, 0.0),
                }
        return table.get(action, table[self.ACT_REJECT])

    def _reward_weights(self, side: str, action=None) -> dict:
        """層・buy/sell別のQ学習報酬重み。

        方針は buy/sell ともに NEED > BALANCE > TRUST > PROFIT > REJECT。
        そのため必要量充足(fill/under/need_bonus/shortage_extra)を強め、
        price/trustは補助的に使う。
        """
        level = self._level_index()
        w = dict(price=3.2, trust=1.1, fill=3.5, dev=2.1, over=1.9,
                 under=2.6, need_bonus=0.8, shortage_extra=0.8,
                 reject_need=-3.2, reject_aggressive=-4.4,
                 future_over=0.65, low_trust=0.8, balance_gain=0.0002)

        if level == 0:
            if side == "buy":
                w.update(price=2.8, trust=0.8, fill=4.6, dev=2.3, over=1.4,
                         under=3.6, need_bonus=1.2, shortage_extra=1.2,
                         reject_need=-3.8, reject_aggressive=-5.2,
                         future_over=0.45, low_trust=0.6, balance_gain=0.00015)
            else:
                w.update(price=2.9, trust=0.8, fill=3.8, dev=2.2, over=1.6,
                         under=2.7, need_bonus=0.65, shortage_extra=0.9,
                         reject_need=-3.4, reject_aggressive=-4.8,
                         future_over=0.50, low_trust=0.7, balance_gain=0.00024)
        elif level == 1:
            if side == "sell":
                w.update(price=3.0, trust=1.0, fill=3.5, dev=2.1, over=1.9,
                         under=2.5, need_bonus=0.60, shortage_extra=0.8,
                         reject_need=-3.2, reject_aggressive=-4.4,
                         future_over=0.65, low_trust=0.8, balance_gain=0.00028)
            else:
                w.update(price=3.2, trust=1.1, fill=3.8, dev=2.1, over=1.9,
                         under=2.8, need_bonus=0.9, shortage_extra=0.9,
                         reject_need=-3.2, reject_aggressive=-4.4,
                         future_over=0.65, low_trust=0.8, balance_gain=0.0002)
        elif level == 2:
            if side == "sell":
                w.update(price=3.2, trust=1.1, fill=3.3, dev=2.0, over=2.1,
                         under=2.4, need_bonus=0.55, shortage_extra=0.8,
                         reject_need=-3.1, reject_aggressive=-4.3,
                         future_over=0.75, low_trust=0.9, balance_gain=0.00032)
            else:
                w.update(price=3.2, trust=1.2, fill=3.7, dev=2.1, over=2.0,
                         under=2.9, need_bonus=0.9, shortage_extra=1.0,
                         reject_need=-3.3, reject_aggressive=-4.5,
                         future_over=0.70, low_trust=0.9, balance_gain=0.00022)
        elif level >= 3:
            if side == "sell":
                w.update(price=3.4, trust=1.1, fill=3.0, dev=1.8, over=1.9,
                         under=2.1, need_bonus=0.50, shortage_extra=0.7,
                         reject_need=-3.0, reject_aggressive=-4.0,
                         future_over=0.60, low_trust=0.9, balance_gain=0.00036)
            else:
                w.update(price=3.0, trust=1.2, fill=3.2, dev=2.2, over=2.6,
                         under=2.2, need_bonus=0.7, shortage_extra=0.8,
                         reject_need=-2.8, reject_aggressive=-3.8,
                         future_over=0.85, low_trust=0.9, balance_gain=0.00025)
        return w

    def _state_features(self, state):
        """状態タプルを線形近似用の特徴量ベクトルへ変換する。

        各binをおおむね0〜1へ正規化し、最後にバイアス1.0を追加する。
        未知状態でもQ値を一般化して推定できるようにする。
        """
        state = self._normalize_state_key(state)
        vals = list(state) if isinstance(state, tuple) else []
        # 12個の状態特徴 + bias = 13次元。足りない場合は0で補完する。
        vals = (vals + [0] * 12)[:12]
        scales = [3, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1]
        features = []
        for v, scale in zip(vals, scales):
            try:
                features.append(float(v) / max(1.0, float(scale)))
            except Exception:
                features.append(0.0)
        features.append(1.0)
        return features

    def _weight_table(self, side):
        return self.w_sell if side == "sell" else self.w_buy

    def get_approx_q_value(self, side, state, action):
        weights = self._weight_table(side)[action]
        features = self._state_features(state)
        return sum(float(f) * float(w) for f, w in zip(features, weights))

    def _approx_q_values(self, side, state):
        return [self.get_approx_q_value(side, state, action) for action in range(5)]

    def _update_approx_weights(self, side, state, action, reward, next_state):
        """提出用では線形近似重みを更新しない。"""
        return



    def _q_table(self, side):
        return self.q_sell if side == "sell" else self.q_buy

    def _offline_q_table(self, side):
        return self.q_offline_sell if side == "sell" else self.q_offline_buy

    def _visit_table(self, side):
        return self.q_visits_sell if side == "sell" else self.q_visits_buy

    def _ensure_offline_q_snapshot(self):
        """読み込み済みQ値をオフライン知識として固定保存する。

        self.q_sell/self.q_buy はオンライン更新で変化するため、
        実行開始時点のコピーを別に持つ。これにより、
        「学習済みQ値を参照しつつ、現在のworldで上書き学習する」形にできる。
        """
        if not self.q_offline_sell:
            self.q_offline_sell = {state: list(values) for state, values in self.q_sell.items()}
        if not self.q_offline_buy:
            self.q_offline_buy = {state: list(values) for state, values in self.q_buy.items()}

    def _offline_weight(self, side, state):
        """オフラインQ値をどれだけ信じるかを返す。

        序盤は保存済みQ値を強めに使う。
        ただし同じ状態・行動でオンライン更新が増えるほど、
        現在の環境に合わせたQ値を優先する。
        """
        total_steps = max(1, int(getattr(self.awi, "n_steps", 1)) - 1)
        current_step = int(getattr(self.awi, "current_step", 0))
        progress = min(1.0, max(0.0, current_step / total_steps))

        base = self.offline_q_weight_start - (self.offline_q_weight_start - self.offline_q_weight_min) * progress
        visits = sum(self._visit_table(side)[state])
        online_conf = visits / (visits + self.online_confidence_scale)
        return max(self.offline_q_weight_min, base * (1.0 - online_conf))

    def _blended_q_values(self, side, state):
        """オフラインQ値とオンラインQ値を混ぜた方針決定用Q値。"""
        online_table = self._q_table(side)
        offline_table = self._offline_q_table(side)

        online_values = online_table[state]
        offline_values = offline_table.get(state)
        if offline_values is None:
            return list(online_values)

        w = self._offline_weight(side, state)
        approx_values = self._approx_q_values(side, state)
        # 既知状態はQテーブルを主に使い、線形近似を補助的に混ぜる。
        approx_weight = 0.20
        return [
            (1.0 - approx_weight) * (w * float(offline_values[i]) + (1.0 - w) * float(online_values[i]))
            + approx_weight * float(approx_values[i])
            for i in range(5)
        ]

    def choose_q_action(self, side, state):
        # 未知状態でもBALANCE固定にせず，層・buy/sell別の初期Q値を入れてから選ぶ。
        self._ensure_q_state_initialized(side, state)

        if random.random() < self.q_epsilon:
            level = self._level_index()
            # 行動順: [PROFIT, BALANCE, NEED, TRUST, REJECT]
            # buy/sellとも NEED を主軸、BALANCE を次点、TRUST/PROFIT は微調整にする。
            if side == "buy" and level == 0:
                weights = [0.08, 0.34, 0.42, 0.11, 0.05]
            elif side == "buy" and level == 1:
                weights = [0.08, 0.36, 0.40, 0.12, 0.04]
            elif side == "buy" and level == 2:
                weights = [0.10, 0.34, 0.38, 0.14, 0.04]
            elif side == "buy":
                weights = [0.08, 0.34, 0.38, 0.14, 0.06]
            elif side == "sell" and level == 0:
                weights = [0.06, 0.40, 0.38, 0.10, 0.06]
            elif side == "sell" and level == 1:
                weights = [0.06, 0.42, 0.36, 0.10, 0.06]
            elif side == "sell" and level == 2:
                weights = [0.08, 0.40, 0.36, 0.10, 0.06]
            else:
                weights = [0.06, 0.40, 0.36, 0.10, 0.08]
            return random.choices(
                [self.ACT_PROFIT, self.ACT_BALANCE, self.ACT_NEED, self.ACT_TRUST, self.ACT_REJECT],
                weights=weights,
                k=1,
            )[0]

        values = self._blended_q_values(side, state)
        return max(range(len(values)), key=lambda i: values[i])

    def update_q_value(self, side, state, action, reward, next_state):
        """提出用ではQ値・訪問回数・線形近似重みを更新しない。"""
        return



    def action_weights(self, action, side=None):
        """ナップサック評価の重み。

        各層・buy/sellで役割が異なるため，層別に重みを切り替える。
        戻り値は必ず5要素:
            (price_w, trust_w, dev_penalty, overflow_penalty, underfill_penalty)
        """
        if side is None:
            side = "buy" if self.is_upstream_level() else "sell"
        return self._layer_action_weights(action, side)

    def q_concession_factor(self, side, action):
        if side == "sell":
            return {
                self.ACT_PROFIT: 1.00,
                self.ACT_BALANCE: 0.95,
                self.ACT_NEED: 0.88,
                self.ACT_TRUST: 0.92,
                self.ACT_REJECT: 0.84,
            }[action]
        return {
            self.ACT_PROFIT: 1.00,
            self.ACT_BALANCE: 1.05,
            self.ACT_NEED: 1.12,
            self.ACT_TRUST: 1.08,
            self.ACT_REJECT: 1.16,
        }[action]

    def update_partner_performance(self, partner, success, offer=None, side=None):
        self.partner_negotiations[partner] += 1
        if success:
            self.partner_successes[partner] += 1
        else:
            self.partner_rejections[partner] += 1

        self.partner_success_rate[partner] = self.partner_successes[partner] / max(1, self.partner_negotiations[partner])

        if success and offer is not None and side is not None:
            q = int(offer[QUANTITY])
            self.partner_total_qty[partner] += q
            price_score = self.normalized_price_advantage(partner, offer, side)
            if side == "sell":
                self.partner_total_price_score_sell[partner] += price_score
            else:
                self.partner_total_price_score_buy[partner] += price_score

    def normalized_price_advantage(self, partner, offer, side):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return 0.5
        issue = nmi.issues[UNIT_PRICE]
        pmin, pmax = issue.min_value, issue.max_value
        span = max(1, pmax - pmin)
        price = offer[UNIT_PRICE]
        if side == "sell":
            return (price - pmin) / span
        return (pmax - price) / span

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def is_upstream_level(self):
        """最上流寄りかどうかを本番環境でも使える形で判定する。

        仕入れ先が存在しない場合を上流とみなし、上流だけNEED偏重を弱める。
        """
        return len(getattr(self.awi, "my_suppliers", [])) == 0


    def _is_late_game(self):
        """終盤判定。探索率や学習率は変えず、契約フィルタだけに使う。"""
        try:
            return int(getattr(self.awi, "current_step", 0)) >= int(0.90 * int(getattr(self.awi, "n_steps", 1)))
        except Exception:
            return False

    def _inventory_quantity(self, product, fallback_attr=None):
        """current_inventory辞書を優先して在庫数を安全に取得する。"""
        try:
            inv = getattr(self.awi, "current_inventory", None)
            if inv is not None:
                if hasattr(inv, "get"):
                    return max(0, int(inv.get(product, 0)))
                return max(0, int(inv[product]))
        except Exception:
            pass
        if fallback_attr:
            try:
                return max(0, int(getattr(self.awi, fallback_attr, 0)))
            except Exception:
                pass
        return 0

    def _current_output_inventory(self):
        return self._inventory_quantity(
            getattr(self.awi, "my_output_product", None),
            "current_inventory_output",
        )

    def _current_input_inventory(self):
        return self._inventory_quantity(
            getattr(self.awi, "my_input_product", None),
            "current_inventory_input",
        )

    def _late_quantity_limit(self, side):
        """終盤に一度に受ける/出す数量の上限を返す。"""
        n_lines = max(1, int(getattr(self.awi, "n_lines", 1)))
        if side == "buy":
            return max(1, int(n_lines * 0.8))
        inv = self._current_output_inventory()
        return inv if inv > 0 else n_lines

    def _late_reject_filter(self, side, partner, offer):
        """終盤の事故を避けるための安全フィルタ。

        Q値や探索率には触らず、終盤だけ危険な契約を強制拒否する。
        - buy : 高値買い、買いすぎ、納期が遅すぎる買いを避ける
        - sell: 在庫超過売り、安すぎる売り、納期が危ない売りを避ける
        """
        if not self._is_late_game() or offer is None:
            return False

        try:
            quantity = int(offer[QUANTITY])
            delivery_time = int(offer[TIME])
            price = float(offer[UNIT_PRICE])
        except Exception:
            return False

        step = int(getattr(self.awi, "current_step", 0))
        n_steps = int(getattr(self.awi, "n_steps", step + 1))
        n_lines = max(1, int(getattr(self.awi, "n_lines", 1)))

        if quantity <= 0:
            return True

        # すでに間に合わない納期は拒否。終盤の遠すぎる買いも、生産・販売に回せない可能性が高い。
        if delivery_time < step:
            return True
        if side == "buy" and delivery_time > step + 1:
            return True
        if side == "sell" and delivery_time <= step and quantity > max(1, int(n_lines * 0.8)):
            return True

        catalog = self._catalog_price_for_side(side)
        nmi = self.get_nmi(partner)
        if nmi is not None:
            try:
                issue = nmi.issues[UNIT_PRICE]
                pmin, pmax = float(issue.min_value), float(issue.max_value)
            except Exception:
                pmin, pmax = 0.0, max(1.0, catalog * 2.0)
        else:
            pmin, pmax = 0.0, max(1.0, catalog * 2.0)

        if side == "sell":
            inventory = self._current_output_inventory()
            already_sold = 0
            try:
                already_sold = int(self.awi.total_sales_at(delivery_time))
            except Exception:
                pass
            safe_stock = max(0, inventory - already_sold)

            # 終盤に在庫以上の売り約束を作らない。
            if quantity > max(0, safe_stock) and quantity > max(1, int(n_lines * 0.8)):
                return True

            # 終盤の安すぎる売りを拒否。ただし最後の1割は在庫処分のため少し緩める。
            last_10_percent = step >= int(0.9 * max(1, n_steps))
            min_sell_price = max(pmin, 0.75 * catalog)
            if last_10_percent:
                min_sell_price = max(pmin, 0.70 * catalog)
            if price < min_sell_price:
                return True

        else:  # buy
            # 終盤の高値買いを拒否。NEED/aggressive時でも極端な高値は避ける。
            max_buy_price = min(pmax, 1.20 * catalog)
            if price > max_buy_price:
                return True

            # 終盤の買いすぎを拒否。
            if quantity > self._late_quantity_limit("buy"):
                return True

        return False

    def _late_adjust_offer_quantity(self, partner, offer):
        """自分から出す終盤オファーの数量を安全側に丸める。"""
        if not self._is_late_game() or offer is None:
            return offer
        try:
            q, t, p = int(offer[QUANTITY]), int(offer[TIME]), offer[UNIT_PRICE]
        except Exception:
            return offer

        side = "buy" if self.is_supplier(partner) else "sell"
        if side == "buy":
            q = min(q, self._late_quantity_limit("buy"))
            # 終盤の遠い買いは、材料を活用しづらいため翌ステップまでに寄せる。
            t = min(t, int(getattr(self.awi, "current_step", 0)) + 1)
        else:
            inv = max(0, self._current_output_inventory())
            q = min(q, inv) if inv > 0 else q
        if q <= 0:
            return None
        return (q, t, p)

    def _expected_input_until(self, delivery_time):
        """現在から納期までに使える見込み原材料数を保守的に推定する。

        current_inventory_input に加え、すでに成立済みの将来供給
        total_supplies_at(t) を納期まで足す。取得できない環境では
        現在在庫だけで判定する。
        """
        try:
            step = int(getattr(self.awi, "current_step", 0))
            delivery_time = int(delivery_time)
            current_input = self._current_input_inventory()
            future_input = 0
            for t in range(max(step + 1, 0), max(step + 1, delivery_time + 1)):
                try:
                    future_input += max(0, int(self.awi.total_supplies_at(t)))
                except Exception:
                    pass
            return max(0, current_input + future_input)
        except Exception:
            return max(0, self._current_input_inventory())

    def _sell_supply_capacity_by_time(self, delivery_time):
        """指定納期までに安全に売れる製品数量の上限を推定する。

        いま持っている完成品 + 納期までに原材料から生産できる量を上限にする。
        これにより、Q値が受諾を選んでも、納期未達ペナルティを起こしやすい
        売り契約をルールベースで止める。
        """
        try:
            step = int(getattr(self.awi, "current_step", 0))
            delivery_time = int(delivery_time)
            n_lines = max(1, int(getattr(self.awi, "n_lines", 1)))
            current_output = max(0, self._current_output_inventory())

            # 当日納期は、基本的に現在の完成品だけを安全在庫として扱う。
            if delivery_time <= step:
                return current_output

            producible_steps = max(0, delivery_time - step)
            line_capacity = producible_steps * n_lines
            available_input = self._expected_input_until(delivery_time)
            future_output = min(line_capacity, available_input)
            return max(0, current_output + future_output)
        except Exception:
            return max(0, self._current_output_inventory())

    def _supply_reject_filter(self, side, partner, offer, reserved_by_time=None):
        """売り契約の供給能力チェック。

        終盤限定ではなく、全ステップで適用する安全装置。
        個別契約だけでなく、同じ納期ですでに受けた売り契約量
        reserved_by_time も足して判定する。
        """
        if side != "sell" or offer is None:
            return False
        try:
            q = int(offer[QUANTITY])
            t = int(offer[TIME])
        except Exception:
            return False
        if q <= 0:
            return True

        already_reserved = 0
        if reserved_by_time is not None:
            try:
                already_reserved = int(reserved_by_time.get(t, 0))
            except Exception:
                already_reserved = 0

        capacity = self._sell_supply_capacity_by_time(t)
        # 以前は capacity - safety_margin でかなり保守的に拒否していた。
        # gs3_4 では供給能力の10%程度まで攻めを許し、機会損失を減らす。
        capacity_with_buffer = int(capacity * 1.10)
        return q + already_reserved > max(0, capacity_with_buffer)


    def is_valid_price(self, price, partner):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return False
        issue = nmi.issues[UNIT_PRICE]
        return issue.min_value <= price <= issue.max_value

    def best_price(self, partner):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        issue = nmi.issues[UNIT_PRICE]
        return issue.min_value if self.is_supplier(partner) else issue.max_value

    def get_effective_ptoday(self):
        if self.aggressive_mode:
            return min(0.9, self._base_ptoday + 0.15)
        if self.conservative_mode:
            return max(0.5, self._base_ptoday - 0.10)
        return self._base_ptoday

    def select_partners_by_performance(self, partners, side, ratio=None):
        if not partners:
            return []
        if ratio is None:
            ratio = self.get_effective_ptoday()
        scored = [(p, self._partner_quality_score(p, side)) for p in partners]
        scored.sort(key=lambda x: x[1], reverse=True)
        select_count = max(1, int(len(partners) * ratio))
        selected = [p for p, _ in scored[:select_count]]
        remaining = [p for p, _ in scored[select_count:]]
        if remaining and len(selected) < len(partners):
            extra_count = min(2, len(remaining), len(partners) - len(selected))
            selected.extend(random.sample(remaining, extra_count))
        return selected

    def smart_price(self, partner, is_first_proposal=False, is_counter_offer=False):
        nmi = self.get_nmi(partner)
        if nmi is None:
            return None
        pissue = nmi.issues[UNIT_PRICE]
        pmin, pmax = pissue.min_value, pissue.max_value
        success_rate = self.partner_success_rate[partner]

        if self.is_consumer(partner):
            action = self.last_sell_policy_action
            if is_first_proposal:
                if action == self.ACT_PROFIT:
                    return pmax
                if action == self.ACT_NEED or self.aggressive_mode:
                    return max(pmin, int(0.92 * pmax))
                return max(pmin, int(0.97 * pmax))
            base_concession = 0.74
            partner_bonus = success_rate * 0.15
            urgency_bonus = 0.10 if self.aggressive_mode else 0.0
            final_rate = (base_concession - partner_bonus - urgency_bonus) * self.q_concession_factor("sell", action)
            return max(int(round(pmax * final_rate)), pmin)

        action = self.last_buy_policy_action
        if is_first_proposal:
            if action == self.ACT_PROFIT:
                return pmin
            if action == self.ACT_NEED or self.aggressive_mode:
                return min(pmax, int(1.08 * pmin))
            return min(pmax, int(1.03 * pmin))
        base_markup = 1.18
        partner_penalty = success_rate * 0.15
        urgency_penalty = 0.10 if self.aggressive_mode else 0.0
        final_rate = (base_markup + partner_penalty + urgency_penalty) * self.q_concession_factor("buy", action)
        return min(int(round(pmin * final_rate)), pmax)

    def price(self, partner):
        return self.smart_price(partner, is_counter_offer=True)

    def needs_at(self, step, partner):
        awi = self.awi
        p = awi.n_lines * self._productivity
        if self.is_supplier(partner):
            return int(p - awi.current_inventory_input - awi.total_supplies_at(step))
        return int(max(0, min(awi.n_lines, p + awi.current_inventory_input) - awi.total_sales_at(step)))

    def distribute_todays_supplie_consume_needs(self, partners, needs, side):
        response = dict(zip(partners, repeat(0)))
        if not partners or needs <= 0:
            return response
        partners = self.select_partners_by_performance(partners, side)
        partners = partners[: max(1, int(self.get_effective_ptoday() * len(partners)))]
        n_partners = len(partners)
        if needs < n_partners:
            partners = random.sample(partners, random.randint(1, min(needs, n_partners)))
            n_partners = len(partners)
        if n_partners > 0:
            response |= dict(zip(partners, distribute(needs, n_partners)))
        return response

    def distribute_todays_needs(self, partners=None):
        if partners is None:
            partners = self.negotiators.keys()
        response = dict(zip(partners, repeat(0)))
        suppliers = [p for p in partners if self.is_supplier(p)]
        consumers = [p for p in partners if self.is_consumer(p)]
        awi = self.awi
        p = awi.n_lines * self._productivity
        supply_needs = int(p - awi.current_inventory_input - awi.total_supplies_at(awi.current_step))
        sale_needs = int(max(0, min(awi.n_lines, p + awi.current_inventory_input) - awi.total_sales_at(awi.current_step)))
        if suppliers and supply_needs > 0:
            response |= self.distribute_todays_supplie_consume_needs(suppliers, supply_needs, "buy")
        if consumers and sale_needs > 0 and awi.total_sales_at(awi.current_step) <= awi.n_lines:
            response |= self.distribute_todays_supplie_consume_needs(consumers, sale_needs, "sell")
        return response

    def _future_day_weight(self, step_offset, side):
        if self.aggressive_mode:
            return {1: 0.55, 2: 0.30, 3: 0.15}.get(step_offset, 0.0)
        if self.conservative_mode:
            return {1: 0.35, 2: 0.35, 3: 0.30}.get(step_offset, 0.0)
        return {1: 0.45, 2: 0.33, 3: 0.22}.get(step_offset, 0.0)

    def future_supplie_offer(self, partner_list):
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response
        sorted_partners = self.select_partners_by_performance(partner_list, "buy", ratio=1.0)
        p = awi.n_lines * self._productivity
        for step_offset in [1, 2, 3]:
            if s + step_offset >= n:
                continue
            weight = self._future_day_weight(step_offset, "buy")
            step_needs = int(max(0, p - awi.current_inventory_input - awi.total_supplies_at(s + step_offset)) * weight)
            if step_needs <= 0:
                continue
            k = max(1, int(len(sorted_partners) * weight + 0.999))
            partners = sorted_partners[:k]
            dist = dict(zip(partners, distribute(step_needs, len(partners))))
            for partner, q in dist.items():
                if q > 0:
                    price = self.best_price(partner)
                    if price is None:
                        continue
                    offer = self._late_adjust_offer_quantity(partner, (q, s + step_offset, price))
                    if offer is not None:
                        response[partner] = offer
        return response

    def future_consume_offer(self, partner_list):
        awi = self.awi
        s, n = awi.current_step, awi.n_steps
        response = {}
        if not partner_list:
            return response
        sorted_partners = self.select_partners_by_performance(partner_list, "sell", ratio=1.0)
        p = awi.n_lines * self._productivity
        for step_offset in [1, 2, 3]:
            if s + step_offset >= n:
                continue
            if awi.total_sales_at(s + step_offset) > awi.n_lines:
                continue
            weight = self._future_day_weight(step_offset, "sell")
            step_needs = int(max(0, min(awi.n_lines, p + awi.current_inventory_input) - awi.total_sales_at(s + step_offset)) * weight)
            if step_needs <= 0:
                continue
            k = max(1, int(len(sorted_partners) * weight + 0.999))
            partners = sorted_partners[:k]
            dist = dict(zip(partners, distribute(step_needs, len(partners))))
            for partner, q in dist.items():
                if q > 0:
                    price = self.best_price(partner)
                    if price is None:
                        continue
                    offer = self._late_adjust_offer_quantity(partner, (q, s + step_offset, price))
                    if offer is not None:
                        response[partner] = offer
        return response

    def first_proposals(self):
        s = self.awi.current_step
        distribution = self.distribute_todays_needs()
        response = {}
        future_suppliers, future_consumers = [], []
        for partner, q in distribution.items():
            if q > 0:
                price = self.smart_price(partner, is_first_proposal=True)
                if price is None:
                    continue
                offer = self._late_adjust_offer_quantity(partner, (q, s, price))
                if offer is None:
                    continue
                response[partner] = offer
            elif self.is_supplier(partner):
                future_suppliers.append(partner)
            elif self.is_consumer(partner):
                future_consumers.append(partner)
        response |= self.future_supplie_offer(future_suppliers)
        response |= self.future_consume_offer(future_consumers)
        if self._is_late_game():
            response = {p: x for p, x in response.items() if x is not None}
        return response

    def _catalog_price_for_side(self, side):
        """buy/sellに対応する製品のカタログ価格を安全に返す。"""
        try:
            product = self.awi.my_output_product if side == "sell" else self.awi.my_input_product
            prices = getattr(self.awi, "catalog_prices", None)
            if prices is not None:
                return float(prices[product])
        except Exception:
            pass
        return 1.0

    def select_subset_by_dp(self, offers_dict, needs, side, action):
        if not offers_dict:
            return [], 0, float("-inf")
        if action == self.ACT_REJECT:
            return [], 0, 0.0

        items = []
        total_q = 0
        for p, offer in offers_dict.items():
            q = int(offer[QUANTITY])
            if q <= 0:
                continue
            price_adv = self.normalized_price_advantage(p, offer, side)
            trust = self._partner_quality_score(p, side)
            items.append((p, q, price_adv, trust))
            total_q += q
        if not items:
            return [], 0, float("-inf")

        if action == self.ACT_NEED or self.aggressive_mode:
            # Q方針がNEEDを選んだときは、DP側でも不足回避を優先する。
            cap_limit = max(needs + self._threshold * 3, int(needs * 2.2) + 1, self.awi.n_lines * 3)
        else:
            cap_limit = max(needs + self._threshold * 2, int(needs * 1.6) + 1, self.awi.n_lines * 2)
        cap = max(1, min(total_q, cap_limit))
        price_w, trust_w, dev_penalty, overflow_penalty, underfill_penalty = self.action_weights(action, side)
        if action == self.ACT_NEED or self.aggressive_mode:
            overflow_penalty *= 0.55
            underfill_penalty *= 1.20
        dp = [None] * (cap + 1)
        dp[0] = (0.0, [])

        for p, q, price_adv, trust in items:
            offer = offers_dict[p]
            price = float(offer[UNIT_PRICE])
            catalog = self._catalog_price_for_side(side)
            success_rate = self.partner_success_rate[p]
            if side == "sell":
                direct_profit = (price - catalog) * q
                breach_risk_penalty = (1.0 - success_rate) * (catalog * 0.5) * q
                yen_value = direct_profit - breach_risk_penalty
            else:
                direct_savings = (catalog - price) * q
                inv_input = float(getattr(self.awi, "current_inventory_input", 0))
                penalty_avoidance_value = (catalog * 0.5) * q if inv_input < getattr(self.awi, "n_lines", 1) else 0.0
                yen_value = direct_savings + penalty_avoidance_value
            # 円換算値をカタログ価格で正規化し、既存の信頼度重みと合わせる。
            monetary_score = max(-5.0 * q, min(5.0 * q, yen_value / max(1.0, catalog)))
            item_value = price_w * monetary_score + trust_w * trust * q
            for c in range(cap, q - 1, -1):
                if dp[c - q] is None:
                    continue
                cand_score = dp[c - q][0] + item_value
                cand_list = dp[c - q][1] + [p]
                if dp[c] is None or cand_score > dp[c][0]:
                    dp[c] = (cand_score, cand_list)

        best_obj, best_set, best_qty = float("-inf"), [], 0
        for c, val in enumerate(dp):
            if val is None:
                continue
            item_score, partner_list = val
            deviation = abs(c - needs)
            overflow = max(0, c - needs)
            underfill = max(0, needs - c)
            obj = (
                item_score
                - dev_penalty * deviation
                - overflow_penalty * overflow
                - underfill_penalty * underfill
            )
            if self.aggressive_mode:
                obj += 0.35 * min(c, needs)
            if self.conservative_mode and overflow > 0:
                obj -= 0.6 * overflow
            if obj > best_obj:
                best_obj, best_set, best_qty = obj, partner_list, c
        return best_set, best_qty, best_obj

    def _normalize_reward(self, reward: float) -> float:
        """Q値更新を安定させるため、報酬を -1〜1 付近に正規化する。

        tanh を使うことで、大きすぎる正報酬・負報酬の影響を抑える。
        状態には新しい要素を追加しないため、状態数は増えない。
        """
        return math.tanh(float(reward) / 3.0)

    def compute_q_reward(self, selected_partners, offers_dict, needs, side, action=None):
        """Q学習用報酬。

        以前の式は price_score に数量 q を掛けた値をそのまま足していたため、
        取引量が大きいだけで報酬が大きくなり、利益・過不足・信頼度の比較が
        不安定になりやすかった。ここでは 0〜1 付近に正規化した特徴量で報酬を作る。

        DS1改良点:
        1. REJECT の価値を状況で場合分けする。
           ただし状態には新しいbinを追加しないので、状態数は増えない。
        2. 最後に tanh 正規化を入れ、極端な報酬でQ値が壊れにくくする。
        """
        # そもそも必要量がない場合:
        # 何も受けないのは自然なので少しだけ良い。
        # 逆に不要なのに受ける場合は在庫・過剰契約リスクとして悪くする。
        if needs <= 0:
            raw_reward = 0.3 if not selected_partners else -1.5
            return self._normalize_reward(raw_reward)

        # 何も選ばなかった場合。
        # これは REJECT 行動、またはDPで受ける相手が選ばれなかった場合に対応する。
        # 状態数を増やさないため、既存の aggressive/conservative mode だけで調整する。
        if not selected_partners:
            rw = self._reward_weights(side, action)
            if action == self.ACT_REJECT:
                if self.aggressive_mode:
                    raw_reward = rw["reject_aggressive"]
                elif self.conservative_mode:
                    raw_reward = max(-1.6, rw["reject_need"] * 0.60)
                else:
                    raw_reward = rw["reject_need"]
            else:
                raw_reward = rw["reject_need"] * 0.85
            return self._normalize_reward(raw_reward)

        total_q = 0
        weighted_price_score = 0.0
        trust_score = 0.0
        for p in selected_partners:
            offer = offers_dict[p]
            q = int(offer[QUANTITY])
            total_q += q
            weighted_price_score += q * self.normalized_price_advantage(p, offer, side)
            trust_score += self._partner_quality_score(p, side)

        avg_price_score = weighted_price_score / max(1, total_q)
        avg_trust = trust_score / max(1, len(selected_partners))
        fill_rate = min(total_q, needs) / max(1, needs)
        deviation_ratio = abs(total_q - needs) / max(1, needs)
        overflow_ratio = max(0, total_q - needs) / max(1, needs)
        underfill_ratio = max(0, needs - total_q) / max(1, needs)

        rw = self._reward_weights(side, action)
        raw_reward = (
            rw["price"] * avg_price_score
            + rw["trust"] * avg_trust
            + rw["fill"] * fill_rate
            - rw["dev"] * deviation_ratio
            - rw["over"] * overflow_ratio
            - rw["under"] * underfill_ratio
        )

        # NEED は常時強化せず、不足を小さく抑えたときだけ報酬を足す。
        # 逆に大きく不足した場合は、BALANCE一強にならないように不足を明示的に罰する。
        shortage_ratio = underfill_ratio
        if action == self.ACT_NEED and shortage_ratio < 0.2:
            raw_reward += rw["need_bonus"]
        elif shortage_ratio > 0.45:
            raw_reward -= rw["shortage_extra"]

        # 将来の在庫過多・過剰契約リスクを軽く罰する。
        # これにより、目先の fill_rate だけで大量契約する事故を抑える。
        future_over_ratio = max(0.0, total_q - needs - self._threshold) / max(1, self.awi.n_lines)
        if future_over_ratio > 0:
            raw_reward -= rw["future_over"] * min(1.5, future_over_ratio)

        # 危険な契約を受けた場合の追加調整。
        # ここも報酬側の調整だけなので、状態数は増えない。
        if overflow_ratio > 0.7:
            raw_reward -= 1.8
        if avg_trust < 0.35 and total_q >= needs:
            raw_reward -= rw["low_trust"]

        if self.aggressive_mode and total_q > 0:
            raw_reward += 0.2
        if self.conservative_mode and overflow_ratio > 0:
            raw_reward -= 0.8

        return self._normalize_reward(raw_reward)

    def counter_all(self, offers, states):
        response = {}
        awi = self.awi

        for _, all_partners, _issues in [
            (awi.needed_supplies, awi.my_suppliers, awi.current_input_issues),
            (awi.needed_sales, awi.my_consumers, awi.current_output_issues),
        ]:
            if not all_partners:
                continue

            side = "buy" if self.is_supplier(all_partners[0]) else "sell"
            day_production = awi.n_lines * self._productivity

            if side == "buy":
                needs = int(day_production - awi.current_inventory_input - awi.total_supplies_at(awi.current_step))
            else:
                if awi.total_sales_at(awi.current_step) <= awi.n_lines:
                    needs = int(max(0, min(awi.n_lines, day_production + awi.current_inventory_input) - awi.total_sales_at(awi.current_step)))
                else:
                    needs = 0

            partners = {p for p in all_partners if p in offers.keys()}
            current_step_offers = {}
            future_step_offers = {}

            for p in partners:
                if offers[p] is None:
                    continue
                if not self.is_valid_price(offers[p][UNIT_PRICE], p):
                    continue
                if offers[p][TIME] == awi.current_step:
                    current_step_offers[p] = offers[p]
                else:
                    future_step_offers[p] = offers[p]

            duplicate_list = [0 for _ in range(awi.n_steps)]
            sell_reserved_by_time = defaultdict(int)
            for p, offer in future_step_offers.items():
                step = offer[TIME]
                if self._supply_reject_filter(side, p, offer, sell_reserved_by_time):
                    self.update_partner_performance(p, False)
                    continue
                if self._late_reject_filter(side, p, offer):
                    self.update_partner_performance(p, False)
                    continue
                if 1 <= step <= awi.n_steps and offer[QUANTITY] + duplicate_list[step - 1] <= max(0, self.needs_at(step, p) + self._threshold):
                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    duplicate_list[step - 1] += offer[QUANTITY]
                    if side == "sell":
                        sell_reserved_by_time[int(step)] += int(offer[QUANTITY])
                    self.update_partner_performance(p, True, offer, side)

            if side == "sell" and current_step_offers:
                safe_current_step_offers = {}
                for p, offer in current_step_offers.items():
                    if self._supply_reject_filter(side, p, offer, sell_reserved_by_time):
                        self.update_partner_performance(p, False)
                        continue
                    safe_current_step_offers[p] = offer
                current_step_offers = safe_current_step_offers

            current_step_partners = set(current_step_offers.keys())
            has_accept_offer = False
            selected_partners = []
            selected_qty = 0
            action = None
            reward = None

            if current_step_offers and needs > 0:
                state = self.make_q_state(side, current_step_offers, needs)
                action = self.choose_q_action(side, state)
                if side == "sell":
                    self.last_sell_policy_state = state
                    self.last_sell_policy_action = action
                else:
                    self.last_buy_policy_state = state
                    self.last_buy_policy_action = action

                if action != self.ACT_REJECT:
                    selected_partners, selected_qty, _ = self.select_subset_by_dp(current_step_offers, needs, side, action)
                    has_accept_offer = len(selected_partners) > 0

                reward = self.compute_q_reward(selected_partners, current_step_offers, needs, side, action)
                next_state = self.make_q_state(side, current_step_offers, max(0, needs - selected_qty))
                self.update_q_value(side, state, action, reward, next_state)

                self.step_summary_history.append(
                    {
                        "step": int(awi.current_step),
                        "side": side,
                        "needs": int(needs),
                        "selected_qty": int(selected_qty),
                        "selected_partner_count": int(len(selected_partners)),
                        "action": int(action),
                        "action_name": self.ACTION_NAMES.get(action, str(action)),
                        "reward": float(reward),
                    }
                )
                self._remember_market_price_average(side, current_step_offers)

            flag = 0
            if has_accept_offer and needs > 0:
                partner_ids = selected_partners
                others = list(current_step_partners.difference(partner_ids))
                others = list(set(others) - set(response.keys()))

                safe_partner_ids = []
                for k in partner_ids:
                    offer = current_step_offers[k]
                    if self._supply_reject_filter(side, k, offer, sell_reserved_by_time):
                        self.update_partner_performance(k, False)
                        others.append(k)
                        continue
                    if self._late_reject_filter(side, k, offer):
                        self.update_partner_performance(k, False)
                        others.append(k)
                        continue
                    safe_partner_ids.append(k)
                    response[k] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    if side == "sell":
                        sell_reserved_by_time[int(offer[TIME])] += int(offer[QUANTITY])
                    self.update_partner_performance(k, True, offer, side)

                for k in others:
                    if k not in safe_partner_ids:
                        self.update_partner_performance(k, False)

                others_s = [x for x in others if self.is_supplier(x)]
                others_c = [x for x in others if self.is_consumer(x)]

                for k, x in self.future_supplie_offer(others_s).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(others_c).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                flag = 1

            if flag != 1:
                other_partners = {_ for _ in all_partners if _ not in response.keys() and _ in self.negotiators.keys()}
                distribution = self.distribute_todays_needs(other_partners)
                future_supplie_partner = []
                future_consume_partner = []

                for k, q in distribution.items():
                    if q > 0:
                        price = self.smart_price(k, is_counter_offer=True)
                        if price is None:
                            continue
                        offer = self._late_adjust_offer_quantity(k, (q, awi.current_step, price))
                        if offer is None:
                            continue
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                    elif self.is_supplier(k):
                        future_supplie_partner.append(k)
                    elif self.is_consumer(k):
                        future_consume_partner.append(k)

                for k, x in self.future_supplie_offer(future_supplie_partner).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)
                for k, x in self.future_consume_offer(future_consume_partner).items():
                    response[k] = SAOResponse(ResponseType.REJECT_OFFER, x)

        return response

    def production_inputs(self) -> dict[int, int] | None:
        """市場が悪いときに生産を抑制する。

        SELL側がREJECTを選んだ場合や、売り市場トレンドが悪い場合は、
        原材料を全量投入せず、在庫として残すことで赤字生産を避ける。
        scml側がこのメソッドを使わない環境では無視される。
        """
        try:
            input_product = self.awi.my_input_product
            available_input = int(getattr(self.awi, "current_inventory_input", 0))
            bad_sell_market = self._market_trend_bin("sell", {}) == 0
            if self.last_sell_policy_action == self.ACT_REJECT or (bad_sell_market and self.conservative_mode):
                return {input_product: 0}
            return {input_product: max(0, available_input)}
        except Exception:
            return None

    # ============================
    # 出力補助
    # ============================
    def _q_table_to_dataframe(self, side: str, level: int | None = None) -> pd.DataFrame:
        table = self._q_table(side)
        rows = []
        for state, values in table.items():
            state_level = state[0] if isinstance(state, tuple) and len(state) >= 1 and isinstance(state[0], int) else -1
            if level is not None and state_level != level:
                continue
            balance_bin = state[6] if isinstance(state, tuple) and len(state) >= 9 else None
            market_trend_bin = state[9] if isinstance(state, tuple) and len(state) >= 10 else None
            row = {"level": int(state_level), "balance_bin": balance_bin, "market_trend_bin": market_trend_bin, "state": repr(state)}
            for i, v in enumerate(values):
                row[f"Q_{self.ACTION_NAMES[i]}"] = float(v)
            row["best_action"] = self.ACTION_NAMES[max(range(len(values)), key=lambda i: values[i])]
            rows.append(row)
        columns = ["level", "balance_bin", "market_trend_bin", "state", "Q_PROFIT", "Q_BALANCE", "Q_NEED", "Q_TRUST", "Q_REJECT", "best_action"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows).sort_values(["level", "balance_bin", "state"]).reset_index(drop=True)

    def _save_learning_artifacts(self):
        """提出用では学習成果物を出力しない。"""
        self._artifacts_saved = True
        return
