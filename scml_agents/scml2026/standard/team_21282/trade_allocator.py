# trade_allocator.py
from __future__ import annotations
from typing import Callable


class TradeAllocator:
    """取引量配分の静的ユーティリティ。

    役割: 総需要を与えられたアクティブパートナーに均等（または重み付け）で配分する。
    """

    @staticmethod
    def distribute_even(total_needed: int, partners: list[str], sigma_func: Callable[[str], float]) -> dict[str, int]:
        """総需要をアクティブパートナーへ均等に配分する (Even Allocation).

        - `sigma_func` を用いてパートナーをソート（信頼度の高い順）して配分を安定化する。
        - 剰余は上位パートナーに 1 ずつ割り当てる方式を採用する。

        Returns:
            dict: partner -> 配分数量
        """
        if not partners or total_needed <= 0:
            return {p: 0 for p in partners}
        
        sorted_partners = sorted(partners, key=lambda p: sigma_func(p), reverse=True)
        n = len(sorted_partners)
        base = total_needed // n
        remainder = total_needed % n
        
        allocation = {}
        for i, p in enumerate(sorted_partners):
            allocation[p] = base + (1 if i < remainder else 0)
        return allocation