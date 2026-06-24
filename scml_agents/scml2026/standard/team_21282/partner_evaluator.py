# partner_evaluator.py
from __future__ import annotations
import random
from collections import defaultdict


class PartnerEvaluator:
    """パートナーの交渉履歴を保持し、選択やスコアを提供するユーティリティクラス。

    役割:
    - 交渉の成功/失敗や合計約定量を記録する
    - パートナーの信頼度（sigma）を算出する
    - `select_partners` で当日のアクティブ候補を絞り込む
    """

    def __init__(self):
        """内部履歴を初期化する。

        - `negotiation_history` は partner -> {success, fail, total_quantity} を保持
        """
        # パートナーごとの履歴統計
        self.negotiation_history = defaultdict(lambda: {"success": 0, "fail": 0, "total_quantity": 0})

    def record_success(self, partner: str, quantity: int):
        """交渉成功を記録する。

        - `quantity` は成立した数量を累積するために使う
        """
        self.negotiation_history[partner]["success"] += 1
        self.negotiation_history[partner]["total_quantity"] += quantity

    def record_failure(self, partner: str):
        """交渉失敗を記録する。"""
        self.negotiation_history[partner]["fail"] += 1

    def get_sigma(self, partner: str) -> float:
        """パートナーの信頼度（成功率）を返す。

        初回は中立値 0.5 を返す。
        """
        record = self.negotiation_history[partner]
        s = record["success"]
        f = record["fail"]
        if s + f == 0:
            return 0.5  # 初対面時の初期信頼度
        return s / (s + f)

    def select_partners(self, all_partners: list[str], ptoday: float, mode: str) -> list[str]:
        """与えられたパートナー群から当日の交渉対象を選択する。

        - 上位 `ptoday` 割合を主力として選ぶ
        - 余り枠からモードに応じた探索サンプルを含める
        """
        if not all_partners:
            return []
        
        sorted_partners = sorted(all_partners, key=lambda p: self.get_sigma(p), reverse=True)
        k = max(1, int(len(all_partners) * ptoday))
        primary_partners = sorted_partners[:k]
        remaining_partners = sorted_partners[k:]
        
        # モードに応じた市場探索枠
        exp_count = 2 if mode == "Aggressive" else (1 if mode == "Neutral" else 0)
        exploration_partners = random.sample(remaining_partners, min(len(remaining_partners), exp_count)) if remaining_partners else []
        
        return list(set(primary_partners + exploration_partners))