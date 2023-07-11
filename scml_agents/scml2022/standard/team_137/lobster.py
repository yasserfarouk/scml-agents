#!/usr/bin/env python
from __future__ import annotations

# required for typing
from typing import Any

import numpy as np
from negmas import Breach, Contract, Issue
from negmas.sao import SAONMI, SAOState, SAONegotiator
from negmas.helpers import humanize_time
from scml.scml2020 import Failure

# required for development
# required for running the test tournament
import time
from setuptools import PEP420PackageFinder
from tabulate import tabulate
from scml.utils import anac2022_collusion, anac2022_std, anac2022_oneshot
from scml.scml2020 import SCML2020Agent
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    DoNothingAgent,
)

from scml.scml2020.common import QUANTITY
from scml.scml2020.common import TIME
from scml.scml2020.common import UNIT_PRICE
from typing import Dict
from .myinfo import myinfo
from .controllerA import SyncControllerA
from .controllerB import SyncControllerB
from negmas.sao import SAOSyncController


from scml.scml2020.common import is_system_agent

__all__ = ["Lobster"]


class Lobster(SCML2020Agent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    def set_price(self):
        prices = (
            self.awi.catalog_prices
            if not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        cost = self.awi.profile.costs[self.awi.my_input_product].min()

        # 買い側#0 min 1 typ 2 max
        p10 = "INF"
        p11 = prices[self.awi.my_input_product]  # 言い値
        p12 = prices[self.awi.my_input_product]  # 妥協ライン

        # 売り側#0 min 1 typ 2 max
        p20 = prices[self.awi.my_output_product]  # 妥協ライン
        p21 = prices[self.awi.my_output_product]  # 言い値
        p22 = "INF"

        # min_margine
        while p20 - p12 - cost <= self.Imyinfo.min_margine + 2:
            p20 += 1
            p12 -= 1
        while p20 - p12 - cost > self.Imyinfo.min_margine:
            p20 -= 1
            p12 += 1

        # max_margine
        while p21 - p11 - cost <= self.Imyinfo.max_margine + 2:
            p21 += 1
            p11 -= 1
        while p21 - p11 - cost > self.Imyinfo.max_margine:
            p21 -= 1
            p11 += 1

        while p11 <= 0 or p21 <= 0:  # プラスになるまで移動
            p11 += 1
            p21 += 1
        while p12 <= 0 or p20 <= 0:  # プラスになるまで移動
            p12 += 1
            p20 += 1

        p11 = int(p11)
        p12 = int(p12)
        p20 = int(p20)
        p21 = int(p21)
        self.Imyinfo.p1 = (p10, p11, p12)
        self.Imyinfo.p2 = (p20, p21, p22)

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""
        # print("debug:init")
        self.Imyinfo = myinfo(self)
        # self.controllers: Dict[str, SAOSyncController] = None
        self.controllers: Dict[str, SAOSyncController] = {
            "A": SyncControllerA(
                is_seller=False,
                parent=self,
                Imyinfo=self.Imyinfo,
            ),
            "B": SyncControllerB(
                is_seller=True,
                parent=self,
                Imyinfo=self.Imyinfo,
            ),
        }

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        # print("debug:before_step")

        self.set_price()
        self.Imyinfo.set_need()

    def step(self):
        """Called at at the END of every production step (day)"""
        # print("debug:step")
        # breachした時の在庫管理
        prices = (
            self.awi.catalog_prices
            if not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )

        price = prices[self.awi.my_output_product]
        price_range = (
            int(min(self.Imyinfo.p2[0], 0.5 * price)),
            int(max(self.Imyinfo.p2[1], 1.5 * price)),
        )
        # 売り契約のネゴシエーション予約
        self._negotiation_request(True, (1, self.awi.n_lines), price_range)

        # 買い契約のネゴシエーション予約
        price = prices[self.awi.my_input_product]
        price_range = (
            min(self.Imyinfo.p1[1], 0.5 * price),
            max(self.Imyinfo.p1[2], 1.5 * price),
        )
        self._negotiation_request(False, (1, self.awi.n_lines), price_range)

    # ================================
    # Negotiation Control and Feedback
    # ================================
    def _negotiation_request(self, seller, quantity_range, price_range):
        self.awi.request_negotiations(
            not seller,
            self.awi.my_output_product
            if (seller == True)
            else self.awi.my_input_product,
            quantity_range,
            price_range,
            time=(self.Imyinfo.first_day, self.Imyinfo.last_day),
            controller=self.controllers["B" if (seller == True) else "A"],
        )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: SAONMI,
    ) -> SAONegotiator | None:
        """Called whenever an agent requests a negotiation with you.
        Return either a negotiator to accept or None (default) to reject it"""
        # print("debug:respond_to_negotiation_request")

        # 日時が範囲外（かぶっていない）。
        if (
            issues[TIME].min_value > self.Imyinfo.last_day + 1
            or issues[TIME].max_value
            < self.Imyinfo.first_day + 1  # 次の日のはずだが調整できないので大体で。
        ):
            return None

        if (
            issues[QUANTITY].max_value < 1
            or self.awi.n_lines < issues[QUANTITY].min_value
        ):
            return None
        if annotation["seller"] == self.id:
            if not (
                issues[UNIT_PRICE].min_value
                <= self.Imyinfo.p2[0]
                <= issues[UNIT_PRICE].max_value
            ):
                return None
            if not (
                issues[UNIT_PRICE].min_value
                <= self.Imyinfo.p2[1]
                <= issues[UNIT_PRICE].max_value
            ):
                return None
        else:
            if not (
                issues[UNIT_PRICE].min_value
                <= self.Imyinfo.p1[1]
                <= issues[UNIT_PRICE].max_value
            ):
                return None
            if not (
                issues[UNIT_PRICE].min_value
                <= self.Imyinfo.p1[2]
                <= issues[UNIT_PRICE].max_value
            ):
                return None

        controller = None
        # 売り手買い手と製品の組み合わせが正しい。
        if (
            annotation["buyer"] == self.id
            and annotation["product"] == self.awi.my_input_product
        ):
            controller = self.controllers["A"]
        elif (
            annotation["seller"] == self.id
            and annotation["product"] == self.awi.my_output_product
        ):
            controller = self.controllers["B"]

        if controller is None:
            return None
        return controller.create_negotiator()

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: SAONMI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""
        # print("debug:on_negotiation_failure")
        return None

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""
        # print("debug:on_negotiation_success")
        return None

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_all_contracts(self, contracts: list[Contract]) -> list[str | None]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        # print("debug:sign_all_contracts")
        self.Imyinfo.set_need()  # 一旦初期化

        signatures = [None] * len(contracts)
        # sort contracts by goodness of price, time and then put system contracts first within each time-step
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["time"]
                if (self.awi.is_last_level == False)
                else -1 * x[0].agreement["time"],  # いつ買えるか分からないのでできるだけ最後の方からとっていきたい
                -1 * x[0].agreement["unit_price"]
                if x[0].annotation["seller"] == self.id
                else x[0].agreement["unit_price"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
            ),
        )

        s = self.awi.current_step
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle.
            if t < s:
                continue

            if is_system_agent(contract.annotation["buyer"]):
                if not (
                    self.Imyinfo.first_day + 10 <= t < self.awi.n_steps - 10
                ):  # 早すぎると間に合わない可能性あり。 TODO
                    continue

            if is_seller:
                flag = self.controllers["B"]._check_timequantity((q, t, u))
                if flag == True:
                    signatures[indx] = self.id
                    self.controllers["B"].reg_secure((q, t, u))
                # print(self.awi.current_step,":",contract.annotation["buyer"],":",signatures[indx],":",contract.agreement)
            else:
                flag = self.controllers["A"]._check_timequantity((q, t, u))
                if flag == True:
                    signatures[indx] = self.id
                    self.controllers["A"].reg_secure((q, t, u))
                # print(self.awi.current_step,":",contract.annotation["seller"],":",signatures[indx],":",contract.agreement)
        return signatures

    def on_contracts_finalized(
        self,
        signed: list[Contract],
        cancelled: list[Contract],
        rejectors: list[list[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in
        a step (day)"""
        # print("debug:on_contracts_finalized")
        for sign in signed:
            if sign.annotation["seller"] == self.id:
                self.Imyinfo.set_contractB(sign.agreement)
            else:
                self.Imyinfo.set_contractA(sign.agreement)

        return None

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""
        # print("debug:on_contract_executed")
        if contract.annotation["seller"] == self.id:
            # 自分が売り手(インベントリBから移動完了)
            pass
        else:
            # 自分が買い手（インベントリAに移動完了）→すぐに製品に加工する手配をとる。
            steps, _ = self.awi.schedule_production(
                process=contract.annotation["product"],
                repeats=contract.agreement["quantity"],
                step=(self.awi.current_step, self.awi.n_steps),  # TODO n_steps
                line=-1,
                partial_ok=True,
                override=False,
                method="earliest",
            )

        return None

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution
        (i.e. resoluion is None)"""
        # print("debug:on_contract_breached")
        return None

    # ====================
    # Production Callbacks
    # ====================

    def on_failures(self, failures: list[Failure]) -> None:
        """Called when production fails. If you are careful in
        what you order in `confirm_production`, you should never see that."""
        # print("debug:on_failures")
        return None

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: list[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)

        self.Imyinfo.set_bankrapt(agent, contracts, quantities, compensation_money)
        return None
