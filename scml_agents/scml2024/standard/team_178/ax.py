#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.std import *

# required for typing
from negmas import *

__all__ = ["AX"]


class AX(StdSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `StdUFun` in the docs for more details).
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    # 使っていない。代わりにproposeが呼び出されている
    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        return dict(
            zip(
                self.negotiators.keys(),
                (self.good_offer(_) for _ in self.negotiators.keys()),
            )
        )

    # 使っていない。代わりにrespondが呼び出されている
    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """

        return self.respond(offers, states)

    # =====================
    # Time-Driven Callbacks
    # =====================
    # SimpleAgentを基にしている。production_levelとfuture_concessionは使わなかった
    def __init__(self, *args, production_level=1, future_concession=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_level = production_level
        self.future_concession = future_concession

    def propose(self, negotiator_id: str, state):
        return self.good_offer(negotiator_id, state)

    # 交渉への返事。物が必要でいい値ならACCEPT　そうでないならREJECT offerが無いならEND
    def respond(self, negotiator_id, state, source=""):
        # accept any quantity needed at a good price
        offer = state.current_offer
        if offer is None:
            return ResponseType.END_NEGOTIATION
        elif self.is_needed(negotiator_id, offer) and self.is_good_price(
            negotiator_id, offer, state
        ):
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    # 望まないオファーは受けない！
    def is_needed(self, partner, offer):
        if offer is None:
            return False
        # last_levelは製品を早く届けてもらわないと困るため、3日以上は待たない
        elif self.awi.is_last_level and offer[TIME] > self.awi.current_step + 3:
            return False
        # middle_levelは全体のstepの半分を超えた段階では何も買わない（おそらく買ったところで売れない）
        elif (
            self.is_supplier(partner)
            and self.awi.is_middle_level
            and offer[TIME] / self.awi.n_steps >= 0.5
        ):
            return False
        # オファーがあってから出荷予定日までに作れる製品の量（今までに注文が入ったぶんも加味）と今回売る量を比較
        elif (
            self.is_consumer(partner)
            and (
                (offer[TIME] - self.awi.current_step)
                * min(self.awi.n_lines, self.awi.current_inventory_input)
            )
            - self.awi.total_sales_between(self.awi.current_step + 1, offer[TIME])
            < offer[QUANTITY]
        ):
            return False
        # n_lines*0.7(last_levelは0.9)持ってたら追加注文はしない
        elif self.is_supplier(partner) and (
            (
                self.awi.total_future_supplies + self.awi.current_inventory_input
                > self.awi.n_lines * 0.7
                and not self.awi.is_last_level
            )
            or self.awi.current_inventory_input > self.awi.n_lines * 0.9
        ):
            return False
        if self.is_consumer(partner):
            return offer[QUANTITY] <= min(
                int(self._needs(partner, offer[TIME])),
                int(
                    max(
                        self.step_progress()
                        * (
                            (offer[TIME] - self.awi.current_step)
                            * min(self.awi.n_lines, self.awi.current_inventory_input)
                            - self.awi.total_sales_between(
                                self.awi.current_step + 1, offer[TIME]
                            )
                        )
                        / (len(self.awi.my_consumers)),
                        1,
                    )
                ),
            )
        # 買い過ぎも厄介だが、買わなさすぎはもっとよくない
        elif self.awi.is_last_level:
            return offer[QUANTITY] <= max(
                2 * self.awi.n_lines / len(self.awi.my_suppliers), 1
            )
        else:
            return offer[QUANTITY] <= max(
                self.awi.n_lines / len(self.awi.my_suppliers), 1
            )

    # 相手の提案がいい値かを判定　最安値と最高値の間だったら基本的にTrueを返す
    def is_good_price(self, partner, offer, state):
        # ending the negotiation is bad
        if offer is None:
            return False
        nmi = self.get_nmi(partner)
        if not nmi:
            return False
        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value
        # 交渉の進行度 提案しては断ってを最大で20（21?）回繰り返す（rの値は0,1/21,2/21……というように増加）
        r = state.relative_time
        if self.is_consumer(partner):
            # first_level以外は最初に最高値で物を売るよう提案→交渉の進行度に合わせて妥協する
            if not self.awi.is_first_level:
                return offer[UNIT_PRICE] >= int(minp + (1 - r * r) * (maxp - minp))
            # first_levelは最安値で物を売るが、もしもエージェント数に対してn_lines数が足りていない場合は少しだけ条件を絞る
            if self.awi.n_lines < len(self.awi.my_suppliers):
                return offer[UNIT_PRICE] >= minp + (0.1) * (maxp - minp)
            return offer[UNIT_PRICE] >= minp
        # last_levelは最高値で物を買おうとする
        elif self.awi.is_last_level:
            return offer[UNIT_PRICE] <= maxp
        # last_level以外は最初に最安値→進行度に合わせて妥協
        else:
            offer[UNIT_PRICE] <= minp + (r * r) * (maxp - minp)

    # こちらからの提案
    def good_offer(self, partner, state):
        nmi = self.get_nmi(partner)
        if not nmi:
            return None
        # input量が十分ならもう買い物はしない
        elif self.is_supplier(partner) and (
            (
                self.awi.total_future_supplies + self.awi.current_inventory_input
                > self.awi.n_lines * 0.7
                and not self.awi.is_last_level
            )
            or self.awi.current_inventory_input > self.awi.n_lines * 0.9
        ):
            return None

        issues = nmi.issues
        qissue = issues[QUANTITY]
        pissue = issues[UNIT_PRICE]
        for basetime in sorted(list(issues[TIME].all)):
            needed = self._needs(partner, basetime)
            # 最終日は何もしない
            if self.awi.n_steps - self.awi.current_step == 1:
                return None

            # t = basetime + random.randint(1, min(3,self.awi.n_steps - self.awi.current_step-1))
            if needed <= 0:
                continue
            offer = [-1] * 3
            # ask for as much as I need for this day
            # おそらく最短でも契約を決めた2日後に実行される
            if self.awi.is_last_level:
                offer[TIME] = self.awi.current_step + 2
            else:
                offer[TIME] = basetime + 2
                # middle_levelは全体のstepの半分を超えた段階では何も買わない（おそらく買ったところで売れない）
            if (
                self.is_supplier(partner)
                and self.awi.is_middle_level
                and offer[TIME] / self.awi.n_steps >= 0.5
            ):
                return None
            # 売る量はmin(売りたい在庫の数,売ることのできる最大数)
            if self.is_consumer(partner):
                if self.awi.current_inventory_input == 0:
                    return None
                # 売る量を決定　なお一度に売る量はn_lines/人数までにしないと在庫が足りない
                offer[QUANTITY] = min(
                    int(self._needs(partner, offer[TIME])),
                    int(
                        max(
                            (
                                (offer[TIME] - self.awi.current_step)
                                * min(
                                    self.awi.n_lines, self.awi.current_inventory_input
                                )
                                - self.awi.total_sales_between(
                                    self.awi.current_step + 1, offer[TIME]
                                )
                            )
                            / (len(self.awi.my_consumers)),
                            1,
                        )
                    ),
                )
                # 与えられた期間内に物を用意できなさそうならより先の日までの提案をする
                if offer[QUANTITY] > (
                    (offer[TIME] - self.awi.current_step) * self.awi.n_steps
                ) - self.awi.total_sales_between(
                    self.awi.current_step + 1, offer[TIME]
                ):
                    continue
            # 買う量を決定　買い過ぎないよう気を付ける
            elif self.awi.is_last_level:
                offer[QUANTITY] = int(
                    max(2 * self.awi.n_lines / len(self.awi.my_suppliers), 1)
                )

            else:
                offer[QUANTITY] = int(
                    max(self.awi.n_lines / len(self.awi.my_suppliers), 1)
                )
                # min(needed, int(max((self.awi.n_lines - self.awi.current_inventory_input - self.awi.total_future_supplies)/(len(self.awi.my_suppliers)),1)))

            # concede linearly on price
            minp, maxp = pissue.min_value, pissue.max_value
            r = state.relative_time
            # 在庫が無いなら売らない
            if self.is_consumer(partner):
                if self.awi.current_inventory_input == 0:
                    return None
                # first_level以外は強気な値段設定で売ろうとするが、交渉が進むごとに値下げする
                elif not self.awi.is_first_level:
                    offer[UNIT_PRICE] = int(minp + (1 - r * r) * (maxp - minp))
                # first_levelは安値で売ろうとするが、相手が多い場合は少しだけ条件を絞る
                elif self.awi.n_lines < len(self.awi.my_suppliers):
                    offer[UNIT_PRICE] = int(minp + (0.1) * (maxp - minp))
                else:
                    offer[UNIT_PRICE] = int(minp)
            # last_levelは高値で物を買う
            elif self.awi.is_last_level:
                offer[UNIT_PRICE] = int(maxp)
            # それ以外は安値で買おうとする(時間経過で妥協する)
            else:
                offer[UNIT_PRICE] = int(minp + (r * r) * (maxp - minp))
            return tuple(offer)
        # just end the negotiation if I need nothing
        return None

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def _needs(self, partner, t):
        # find my needs today
        # 与えられた期間内で売ることのできる量を返す
        if self.is_consumer(partner):
            total_needs = min(self.awi.current_inventory_input, self.awi.n_lines) * (
                t - self.awi.current_step - 1
            ) - self.awi.total_sales_between(self.awi.current_step + 1, t - 1)
        # n_lines - (持っているinput + 買う予定のinput) を教える n_lines以上なら0
        else:
            total_needs = max(
                0,
                self.awi.n_lines
                - self.awi.current_inventory_input
                - self.awi.total_future_supplies,
            )
        return int(total_needs)

    # first_levelがconsumer相手に物を売る場合、後半になるにつれて取引相手が少なくなると思われるため、一度に売る量を増やしていく（生産が間に合う保証はない）
    def step_progress(self):
        if self.awi.is_first_level:
            if self.awi.current_step / self.awi.n_steps >= 0.8:
                return min(4, len(self.awi.my_consumers))
            if self.awi.current_step / self.awi.n_steps >= 0.5:
                return min(2, len(self.awi.my_consumers))
            else:
                return 1
        else:
            return 1

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([AX], sys.argv[1] if len(sys.argv) > 1 else "std")
