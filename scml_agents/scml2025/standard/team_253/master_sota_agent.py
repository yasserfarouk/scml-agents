#!/usr/bin/env python
"""
Author:
    1. Sota Sakaguchi  [email]sakaguchi.sota@otsukalab.nitech.ac.jp
    2. Takanobu Otsuka [email]otsuka.takanobu@nitech.ac.jp
"""

from __future__ import annotations
from typing import Any
from scml.std import StdAWI
from negmas import Contract, SAOState
from scml.std.agent import StdSyncAgent

from .record_manager import RecordManager
from .proposal_strategy import ProposalStrategy

__all__ = ["XenoSotaAgent"]


class XenoSotaAgent(StdSyncAgent):
    def __init__(self, *args, threshold=0.9, ptoday=0.6, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.ptoday = ptoday
        self.productivity = productivity

        self.record = None  # 交渉履歴の管理
        self.ue = None  # 効用関数
        self.scorer = None  # 交渉相手のスコアリング
        self.ps = None  # 交渉戦略
        self.sda = None  # 需要供給管理

    def init(self):
        self.record = RecordManager(self.awi)
        self.ps = ProposalStrategy(
            self.awi,
            self.negotiators.keys(),
            self.record,
            self.productivity,
            self.ptoday,
        )

    def before_step(self):
        negotiators = self.active_negotiators
        if self.awi.current_step == 0:
            negotiators = self.negotiators
        self.ps.before_step(negotiators)

    def step(self):
        self.ps.step()

    def first_proposals(self):
        # nmiの取得 + set
        nmis = {}
        for partner_id in self.active_negotiators:
            nmis[partner_id] = self.get_nmi(partner_id)
        self.ps.set_nmis(nmis)

        return self.ps.generate_initial_offer(self.active_negotiators)

    def counter_all(self, offers, states):
        return self.ps.generate_counter_offer(offers, self.threshold)

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        self.record.set_failure(self.id, partners)

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        self.record.set_success(self.id, contract)
