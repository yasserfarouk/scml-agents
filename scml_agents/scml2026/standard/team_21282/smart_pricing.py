# smart_pricing.py
from __future__ import annotations
from typing import Any
from scml.oneshot.common import UNIT_PRICE


class SmartPricing:
    """価格決定のヘルパー。

    NMI（交渉メタ情報）と AWI、パートナー信頼度 `sigma`、および戦略 `mode` を元に
    提案すべき `UNIT_PRICE` を算出するユーティリティクラス。
    """

    @staticmethod
    # 役割: NMI/AWI/戦略モードを考慮して合理的な価格を返す
    def get_price(nmi: Any, awi: Any, is_seller: bool, sigma: float, mode: str) -> int:
        """信頼度と戦略モードに基づいたスマート価格を算出する。

        - nmi が None の場合は AWI の issue 範囲から端点を返すフォールバックを行う
        - `sigma` が高いほど相手に有利な価格に近づける
        - `mode` によって譲歩幅の係数 (`alpha`, `beta`) を調整する
        """
        if not nmi:
            if is_seller:
                return awi.current_output_issues[UNIT_PRICE].max_value if awi.current_output_issues else 100
            else:
                return awi.current_input_issues[UNIT_PRICE].min_value if awi.current_input_issues else 0
                
        pissue = nmi.issues[UNIT_PRICE]
        minp, maxp = pissue.min_value, pissue.max_value
        delta_p = maxp - minp

        # モードによるマージン幅調整のパラメータ
        if mode == "Aggressive":
            alpha = beta = 0.2
        elif mode == "Conservative":
            alpha = beta = 0.1
        else:
            alpha = beta = 0.15

        if is_seller:
            # ベース価格: モードと sigma による目標価格
            base_price = maxp - alpha * (1.0 - sigma) * delta_p

            # 在庫比率に応じたマイルド割引（ライン数の1倍を超えた分に応じて最大30%）
            out_inv = getattr(awi, 'current_inventory_output', 0)
            n_lines = max(1, getattr(awi, 'n_lines', 1))
            overstock_ratio = max(0.0, (out_inv / n_lines) - 1.0)
            discount_pct = min(0.30, overstock_ratio * 0.1)  # ライン数 x4で最大30%割引になる想定

            price = base_price - (delta_p * discount_pct)
            return int(max(minp, min(maxp, price)))
        else:
            price = minp + beta * (1.0 - sigma) * delta_p
            return int(max(minp, min(maxp, price)))