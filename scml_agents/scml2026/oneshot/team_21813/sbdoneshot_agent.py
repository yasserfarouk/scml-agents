#!/usr/bin/env python
"""
**Submitted to ANAC 2026 SCML (OneShot track)**
*Authors* Hajime Endo <endo@katfuji.lab.tuat.ac.jp> Tokyo University of Agriculture and Technology

Rchan (ANAC 2025 OneShot 2位) を継承し、count-based leverage layer を追加。

戦略の柱:
  Stage 1: Rchan の partner modeling と Cautious 系交渉戦略を継承
    - 売買非対称 _allowed_mismatch
    - 終盤 (step > 0.5*n_steps) の top-1 partner 集中
    - Over-ordering with time decay
    - Partner ごとの first_accept_prob / agent_ave_quantity 追跡

  Stage 2: count-based leverage layer
    init() で leverage_ratio = n_them / n_us を計算 (n_them = 相手 level 人数,
    n_us = 自分 level 人数 = my_competitors + 1):
      ratio < 0.7  : outnumbered ─ 自分側人数多い．partner 希少 → 譲歩多めに取りに行く
      0.7-1.3      : balanced    ─ Rchan 通常動作
      ratio > 1.3  : advantaged  ─ 自分側人数少ない．partner 豊富 → cherry-pick

    _allowed_mismatch の出力 (th_min, th_max) に上記 class ごとの倍率を乗算:
      outnumbered: th_max *= 1.3, th_min *= 1.3  (tolerance UP)
      advantaged : th_max *= 0.7, th_min *= 0.7  (tolerance DOWN)
      balanced   : 倍率なし

設計判断:
  - SC 形状の人数比 (volume と独立な count-based 指標) は世界生成時に確定的 →
    luck classification (5-step 観測ベース) のような不安定性なし
  - benchmark で V8 (Cautious + leverage layer) が overall +0.004 over Cautious 確認.
    Rchan ベースの V9 は overall 1.067 で Rchan 1.069 とほぼ同等 (noise floor 内),
    Cautious 1.054 を +0.013 で上回る.

挙動の保証:
  leverage_class == 'balanced' の世界では Rchan 完全等価.
  outnumbered/advantaged 環境でのみ tolerance を補正する保守的拡張.
"""
from __future__ import annotations

from scml_agents.scml2025.oneshot.takafam.rchan import Rchan

__all__ = ["SBDOneShot"]


class SBDOneShot(Rchan):
    """Rchan + count-based leverage 補正."""

    def __init__(
        self,
        *args,
        outnumbered_th: float = 0.7,
        advantaged_th: float = 1.3,
        outnumbered_max_mult: float = 1.3,
        outnumbered_min_mult: float = 1.3,
        advantaged_max_mult: float = 0.7,
        advantaged_min_mult: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outnumbered_th = outnumbered_th
        self.advantaged_th = advantaged_th
        self.outnumbered_max_mult = outnumbered_max_mult
        self.outnumbered_min_mult = outnumbered_min_mult
        self.advantaged_max_mult = advantaged_max_mult
        self.advantaged_min_mult = advantaged_min_mult
        self.leverage_ratio: float = 1.0
        self.leverage_class: str = "balanced"

    def init(self):
        super().init()
        try:
            n_us = len(self.awi.my_competitors) + 1
        except Exception:
            n_us = 1
        try:
            them = (
                self.awi.my_consumers
                if self.awi.is_first_level
                else self.awi.my_suppliers
            )
            n_them = len(them) if them else 1
        except Exception:
            n_them = 1
        self.leverage_ratio = float(n_them) / max(1.0, float(n_us))
        if self.leverage_ratio < self.outnumbered_th:
            self.leverage_class = "outnumbered"
        elif self.leverage_ratio > self.advantaged_th:
            self.leverage_class = "advantaged"
        else:
            self.leverage_class = "balanced"

    def _allowed_mismatch(self, r, n_others, is_selling):
        th_min, th_max = super()._allowed_mismatch(r, n_others, is_selling)
        if self.leverage_class == "outnumbered":
            th_max = th_max * self.outnumbered_max_mult
            th_min = th_min * self.outnumbered_min_mult
        elif self.leverage_class == "advantaged":
            th_max = th_max * self.advantaged_max_mult
            th_min = th_min * self.advantaged_min_mult
        return th_min, th_max
