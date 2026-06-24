# market_analyst.py
from __future__ import annotations


class MarketAnalyst:
    """市場動向を解析し意思決定モードと交渉枠を提供するユーティリティクラス。

    役割:
    - 収益差分の履歴を保持してトレンドを検出する
    - `mode`（Aggressive/Conservative/Neutral）を決定する
    - 本日の需要配分比率 `ptoday` と交渉上限 `tau` を動的に算出する
    """

    def __init__(self):
        """内部状態を初期化する。

        - `profit_history`: 直近の収益差分のリスト
        - `prev_balance`: 直前ステップの残高（トレンド計算用）
        - `mode`, `tau`, `ptoday` はデフォルト値で初期化
        """
        self.profit_history: list[float] = []
        self.prev_balance: float | None = None
        self.mode: str = "Neutral"
        self.tau: int = 1
        self.tau_buy: int = 1
        self.tau_sell: int = 1
        self.ptoday: float = 0.70

    def reset(self, initial_balance: float):
        """シミュレーション開始時に呼ばれるリセットメソッド。

        引数:
        - `initial_balance`: 初期残高（分析の基準値）
        """
        self.prev_balance = initial_balance
        self.profit_history = []
        self.mode = "Neutral"
        self.tau = 1
        self.tau_buy = 1
        self.tau_sell = 1
        self.ptoday = 0.70

    def update_mode_and_cap(self, current_balance: float, current_step: int, n_steps: int, n_lines: int, current_inventory: list[int]):
        """利益トレンドと在庫水準から戦略モードと `tau` を更新する。

        処理概要:
        1. 収益差分を `profit_history` に追加して短期/中期のトレンドを評価
        2. トレンドに基づいて `mode` を Aggressive / Conservative / Neutral に切替
        3. `mode` によって `ptoday`（本日配分率）を調整
        4. 在庫比率（It / n_lines）と残りステップに応じて `tau`（交渉上限）を算出
        """
        # 1. 利益トレンド検出 & モード切り替え
        if self.prev_balance is not None:
            current_profit = current_balance - self.prev_balance
            self.profit_history.append(current_profit)
        self.prev_balance = current_balance

        if len(self.profit_history) >= 5:
            b_recent = sum(self.profit_history[-3:]) / 3
            b_old = sum(self.profit_history[-5:-3]) / 2
            delta_b = b_recent - b_old
            if delta_b > 50:
                self.mode = "Aggressive"
            elif delta_b < -50:
                self.mode = "Conservative"
            else:
                self.mode = "Neutral"
        else:
            self.mode = "Neutral"

        # 2. Daily Adoption Ratio (ptoday) の調整
        p0 = 0.70
        if self.mode == "Aggressive":
            self.ptoday = min(0.90, p0 + 0.15)
        elif self.mode == "Conservative":
            self.ptoday = max(0.50, p0 - 0.10)
        else:
            self.ptoday = p0

        # 3. 動的交渉キャップ (tau_t) の算出
        It = sum(current_inventory)
        rt = It / max(1, n_lines)
        f_inv = 1.5 if rt < 0.3 else (0.7 if rt > 0.8 else 1.0)

        phi_t = (n_steps - current_step) / max(1, n_steps)
        f_time = 0.8 if phi_t < 0.2 else (1.0 if phi_t < 0.6 else 1.3)

        tau_hat = 0.1 * n_lines
        base_tau = max(1, int(tau_hat * f_inv * f_time))
        self.tau = base_tau

        # 非対称キャップ: 在庫状況に応じて買い枠／売り枠を分離
        It = sum(current_inventory)
        # ライン数の2.5倍を越えると過剰在庫と見なす（売却を広げ買いを絞る）
        if It > (n_lines * 3):
            self.tau_buy = 1
            self.tau_sell = max(1, n_lines * 3)
        else:
            rt = It / max(1, n_lines)
            if rt < 0.3:
                self.tau_buy = max(1, int(base_tau * 1.4))
                self.tau_sell = max(1, int(base_tau * 0.8))
            elif rt > 1.0:
                self.tau_buy = max(1, int(base_tau * 0.8))
                self.tau_sell = max(1, int(base_tau * 1.4))
            else:
                self.tau_buy = base_tau
                self.tau_sell = base_tau

        if self.mode == "Aggressive":
            self.tau_buy = int(self.tau_buy * 1.5)
            self.tau_sell = int(self.tau_sell * 1.2)
        elif self.mode == "Conservative":
            self.tau_buy = max(1, int(self.tau_buy * 0.7))
            self.tau_sell = max(1, int(self.tau_sell * 0.8))