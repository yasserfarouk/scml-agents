"""
LeanAgentV2: LeanAgent に state-conditional 戦略を加えた版.

拡張内容:
  1) Market state 分類: step 5 で supply_surplus / balanced / demand_surplus に確定
  2) (位置 × 市況) 7 セルで needs/accept/future multiplier を変える (middle は state 無関係)
  3) future contract horizon を tunable (1-10 日先まで)
  4) partner 別 luckiness 重み付け配分 (cash from FinancialReport で luckiness 推定)
"""
from __future__ import annotations

from itertools import repeat
from typing import Any

from negmas import ResponseType, SAOResponse

from scml.std import QUANTITY, TIME, UNIT_PRICE
from .lean_agent import LeanAgent
from .penguinagent import distribute


# (位置, 市況) → param dict のデフォルト
# 「中庸」セルは V1 LeanAgent と一致するよう accept_th=1 / future_mult=1.0
# 「アグレッシブ」セルだけ params を変える
DEFAULT_CELL_PARAMS = {
    ('A0', 'supply_surplus'):    {'needs_mult': 1.2, 'accept_th': 10, 'future_mult': 1.5},
    ('A0', 'balanced'):          {'needs_mult': 1.0, 'accept_th': 1,  'future_mult': 1.0},  # 中庸=V1
    ('A0', 'demand_surplus'):    {'needs_mult': 1.0, 'accept_th': 1,  'future_mult': 1.0},  # 中庸=V1
    ('A_last', 'supply_surplus'): {'needs_mult': 0.8, 'accept_th': 1,  'future_mult': 0.8},
    ('A_last', 'balanced'):       {'needs_mult': 1.0, 'accept_th': 1,  'future_mult': 1.0},  # 中庸=V1
    ('A_last', 'demand_surplus'): {'needs_mult': 1.0, 'accept_th': 10, 'future_mult': 1.5},
    ('middle', 'any'):           {'needs_mult': 1.0, 'accept_th': 1,  'future_mult': 1.0},  # 中庸=V1
}

# 市況分類閾値 (BO 対象にしてもよい)
SUPPLY_SURPLUS_TH = 0.15
DEMAND_SURPLUS_TH = -0.15

OBSERVATION_STEPS = 5  # state ロックする step


class LeanAgentV2(LeanAgent):
    """state-conditional LeanAgent."""

    def __init__(
        self,
        *args,
        productivity: float = 0.7,
        # global tunable
        forward_horizon: int = 3,
        partner_alpha: float = 0.0,
        # state-conditional override (cell key → param dict)
        cell_overrides: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, productivity=productivity, **kwargs)
        self._forward_horizon = max(1, int(forward_horizon))
        self._partner_alpha = float(partner_alpha)
        # 蓄積バッファ (state 判定用)
        self._exo_summary_history: list = []
        self._state_locked = False
        self._market_state: str = 'balanced'
        # cell params (override 適用)
        self._cell_params = {**DEFAULT_CELL_PARAMS}
        if cell_overrides:
            for k, v in cell_overrides.items():
                if k in self._cell_params:
                    self._cell_params[k].update(v)
        # active params (state ロック後にセットされる)
        self._needs_mult = 1.0
        self._accept_th = 3
        self._future_mult = 1.0

    # ----------------------------
    # 観測 & state classify
    # ----------------------------
    def step(self):
        # state ロック前は exogenous_contract_summary を蓄積
        if not self._state_locked and self.awi.current_step < OBSERVATION_STEPS:
            try:
                summ = self.awi.exogenous_contract_summary
                if summ:
                    self._exo_summary_history.append([float(q) for (q, p) in summ])
            except Exception:
                pass

        # state ロック
        if not self._state_locked and self.awi.current_step >= OBSERVATION_STEPS:
            self._market_state = self._classify_market_state()
            self._state_locked = True
            self._load_cell_params()

        super().step()

    def _classify_market_state(self) -> str:
        if not self._exo_summary_history:
            return 'balanced'
        q0_sum = sum(s[0] for s in self._exo_summary_history if s)
        qn_sum = sum(s[-1] for s in self._exo_summary_history if s)
        if q0_sum + qn_sum == 0:
            return 'balanced'
        pressure = (q0_sum - qn_sum) / max(q0_sum, qn_sum)
        if pressure > SUPPLY_SURPLUS_TH:
            return 'supply_surplus'
        if pressure < DEMAND_SURPLUS_TH:
            return 'demand_surplus'
        return 'balanced'

    def _position_key(self) -> str:
        if self.awi.is_first_level:
            return 'A0'
        if self.awi.is_last_level:
            return 'A_last'
        return 'middle'

    def _load_cell_params(self):
        pos = self._position_key()
        if pos == 'middle':
            key = ('middle', 'any')
        else:
            key = (pos, self._market_state)
        params = self._cell_params.get(key, self._cell_params[('middle', 'any')])
        self._needs_mult = params['needs_mult']
        self._accept_th = params['accept_th']
        self._future_mult = params['future_mult']

    # ----------------------------
    # _lean_needs: 倍率適用
    # ----------------------------
    def _lean_needs(self):
        supply_needs, consume_needs = super()._lean_needs()
        return int(supply_needs * self._needs_mult), int(consume_needs * self._needs_mult)

    # ----------------------------
    # counter_all: accept_threshold 上書き
    # ----------------------------
    def counter_all(self, offers, states):
        if self._state_locked:
            original = self._threshold
            self._threshold = max(1, int(self._accept_th))
            try:
                return super().counter_all(offers, states)
            finally:
                self._threshold = original
        return super().counter_all(offers, states)

    # ----------------------------
    # 未来契約: horizon と multiplier 適用
    # default param (horizon=3, future_mult=1.0) のときは parent (PENGUIN 50/30/20) を呼ぶ
    # ----------------------------
    def future_supplie_offer(self, partner_list):
        if self._forward_horizon == 3 and self._future_mult == 1.0:
            return super().future_supplie_offer(partner_list)
        return self._future_offer_generic(partner_list, is_supply=True)

    def future_consume_offer(self, partner_list):
        if self._forward_horizon == 3 and self._future_mult == 1.0:
            return super().future_consume_offer(partner_list)
        return self._future_offer_generic(partner_list, is_supply=False)

    def _future_offer_generic(self, partner_list, is_supply: bool) -> dict:
        awi = self.awi
        s = awi.current_step
        n = awi.n_steps
        response = dict()
        if not partner_list:
            return response

        h = max(1, min(self._forward_horizon, n - s - 1))
        if h <= 0:
            return response

        # partners を h 等分 (front-loaded: 多めに前段へ)
        partners = list(partner_list)
        n_partners = len(partners)
        # 各 day i (1..h) に振る partner 数
        per_day = max(1, n_partners // h)
        partner_chunks = []
        idx = 0
        for i in range(h):
            chunk = partners[idx: idx + per_day] if i < h - 1 else partners[idx:]
            partner_chunks.append(chunk)
            idx += per_day

        p_target = awi.n_lines * self._productivity

        for i, chunk in enumerate(partner_chunks, start=1):
            future_step = s + i
            if future_step >= n:
                continue
            if not chunk:
                continue
            # その day における needs
            if is_supply:
                day_need = int(
                    (p_target - awi.current_inventory_input - awi.total_supplies_at(future_step))
                    / h * self._future_mult
                )
            else:
                day_need = int(
                    max(0,
                        min(awi.n_lines, p_target + awi.current_inventory_input)
                        - awi.total_sales_at(future_step))
                    / h * self._future_mult
                )
            if day_need <= 0:
                continue
            # partner_alpha で重み付け配分 (cash 推定)
            allocation = self._weighted_distribute_quantities(day_need, chunk, is_supply)
            for partner, qty in zip(chunk, allocation):
                if qty > 0:
                    response[partner] = (qty, future_step, self.best_price(partner))

        return response

    # ----------------------------
    # 数量配分: partner luckiness 重み付け
    # MyAgent の _distribute_to_partners を override (LeanAgent が MyAgent ベースになったので)
    # ----------------------------
    def _distribute_to_partners(self, partners, needs):
        """MyAgent の _distribute_to_partners を partner_alpha で重み付け化."""
        response = dict(zip(partners, repeat(0)))
        if not partners or needs <= 0:
            return response
        import random
        partners = list(partners)
        random.shuffle(partners)
        ptoday = self.get_effective_ptoday()
        selected = partners[: max(1, int(ptoday * len(partners)))]
        n = len(selected)
        if n == 0:
            return response

        if self._partner_alpha == 0.0:
            allocation = distribute(needs, n)
        else:
            is_supply = bool(self.is_supplier(selected[0]))
            allocation = self._weighted_distribute_quantities(needs, selected, is_supply)
        response |= dict(zip(selected, allocation))
        return response

    def _weighted_distribute_quantities(self, needs: int, partners: list[str], is_supply: bool) -> list[int]:
        """needs を partners に partner luckiness で重み付け配分."""
        if needs <= 0 or not partners:
            return [0] * len(partners)
        if self._partner_alpha == 0.0:
            return distribute(needs, len(partners))

        # luckiness 取得 (cash)。downstream (consumer) のとき有効、upstream (supplier) は推定弱
        luck = []
        for p in partners:
            try:
                reports = self.awi.reports_of_agent(p)
                if reports:
                    latest = max(reports.keys())
                    luck.append(float(reports[latest].cash))
                else:
                    luck.append(None)
            except Exception:
                luck.append(None)

        valid = [v for v in luck if v is not None]
        if not valid or all(v == valid[0] for v in valid):
            return distribute(needs, len(partners))

        mean_luck = sum(valid) / len(valid)
        scale = max(1.0, max(abs(v - mean_luck) for v in valid))
        weights = []
        for v in luck:
            if v is None:
                weights.append(1.0)
            else:
                w = 1.0 + self._partner_alpha * ((v - mean_luck) / scale)
                weights.append(max(0.1, w))

        total_w = sum(weights)
        fractional = [needs * w / total_w for w in weights]
        integer_alloc = [int(f) for f in fractional]
        remainder = needs - sum(integer_alloc)
        order = sorted(range(len(weights)), key=lambda i: fractional[i] - integer_alloc[i], reverse=True)
        for i in range(remainder):
            integer_alloc[order[i % len(order)]] += 1
        return integer_alloc
