import random
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from negmas import AgentWorldInterface
from negmas import ControlledNegotiator
from negmas import LinearUtilityFunction
from negmas import MechanismState
from negmas import Outcome
from negmas import PolyAspiration
from negmas import ResponseType
from negmas import UtilityFunction
from negmas import make_issue
from negmas import make_os
from negmas import outcome_is_valid
from negmas.common import NegotiatorMechanismInterface
from negmas.events import Notification
from negmas.events import Notifier
from negmas.helpers import instantiate
from negmas.sao import SAOController
from negmas.sao import SAONegotiator
from negmas.sao import SAOResponse
from negmas.sao import SAOState
from negmas.sao import SAOSyncController
from negmas.sao.negotiators.controlled import ControlledSAONegotiator

from scml.scml2020.common import QUANTITY
from scml.scml2020.common import TIME
from scml.scml2020.common import UNIT_PRICE

from .myinfo import myinfo

# our controller
class SyncControllerA(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent needs and with lowest price
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        Imyinfo: "myinfo",
        parent,  #:"ABDPStd"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.Imyinfo = Imyinfo
        self.parent = parent

        # find out my needs and the amount secured lists

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        issues = self.negotiators[negotiator_id][0].nmi.issues
        return outcome_is_valid(offer, issues)

    def _check_timequantity(self, offer: "Outcome") -> bool:  # time and quantity
        info = self.Imyinfo

        if not (info.first_day <= offer[TIME] < info.last_day):  # 範囲外の日程
            return False

        if info.invB_checkday < offer[TIME]:
            return False  # 入荷最終日以降の話

        # 上限を超えない
        if offer[QUANTITY] > info.max_input:
            return False

        # ラストのレベルの時。先にBUYを決めているので。マイナス（出荷）に間に合う日までに仕入れたい
        temp = 0
        for i in reversed(range(info.first_day, info.last_day)):
            if (
                self.Imyinfo.secure_inventoryB[i]
                - self.Imyinfo.secure_inventoryB[i + 1]
                > 0
            ):
                temp = i
                break

        temp_last = (
            info.last_day if (self.parent.awi.is_last_level == False) else temp - 1
        )

        # ラインが空いている
        t_q = offer[QUANTITY]
        for i in range(offer[TIME], temp_last):  # CHECK
            m = min(t_q, info.input_needs[i])
            t_q -= m
            if t_q == 0:
                break

        if t_q != 0:
            return False
        return True

    def check_makeA(self, offer: "Outcome") -> bool:
        info = self.Imyinfo
        # price
        if info.p1[1] < offer[UNIT_PRICE]:
            return False  # 高いのでダメ

        if self._check_timequantity(offer) == False:
            return False  # 在庫ダメ

        return True

    def check_makeB(self, offer: "Outcome") -> bool:
        info = self.Imyinfo
        # price
        if info.p1[2] < offer[UNIT_PRICE]:
            return False  # 高いのでダメ

        if self._check_timequantity(offer) == False:
            return False  # 在庫ダメ

        return True

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.

        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        responses = {
            nid: SAOResponse(ResponseType.NO_RESPONSE, None) for nid in offers.keys()
        }
        for nid, offer in sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE]):
            # ここでひとつづつ見ていく。(安い順)
            negotiator = self.negotiators[nid][0]
            if negotiator.nmi is None:
                continue
            nmi = negotiator.nmi
            state = states[nid]

            flag = False
            if nmi.n_steps - state.step > self.Imyinfo.negotiator_laststep:
                flag = self.check_makeA(offer)  # 妥協しない
            else:
                flag = self.check_makeB(offer)  # 妥協

            if flag == True:
                ######ACC######
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            else:
                ######COUNTER######
                responses[nid] = SAOResponse(
                    ResponseType.REJECT_OFFER, self.best_proposal(nid)
                )

        return responses

    def reg_secure(self, offer: "Outcome") -> bool:
        info = self.Imyinfo
        # AGREE返信でSECURE登録
        t_q = offer[QUANTITY]
        for i in range(offer[TIME], info.last_day):  # REGISTER
            m = min(t_q, info.input_needs[i])
            t_q -= m
            info.input_needs[i] -= m
            if t_q == 0:
                break
        info.max_input -= offer[QUANTITY]

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        """Update the secured quantities whenever a negotiation ends"""
        # AGREE返信でSECURE登録
        #
        if state.agreement is not None:  # 成立
            self.reg_secure(state.agreement)
        return None

    def best_proposal(self, nid: str) -> Outcome:
        """
        Finds the best proposal for the given negotiation

        Args:
            nid: Negotiator ID

        Returns:
            The outcome with highest utility and the corresponding utility
        """
        info = self.Imyinfo
        negotiator = self.negotiators[nid][0]
        if negotiator.nmi is None:
            return None
        nmi = negotiator.nmi
        t_min = nmi.issues[TIME].min_value
        t_max = nmi.issues[TIME].max_value
        p_min = nmi.issues[UNIT_PRICE].min_value
        p_max = nmi.issues[UNIT_PRICE].max_value
        q_min = nmi.issues[QUANTITY].min_value
        q_max = nmi.issues[QUANTITY].max_value
        q_max = min(q_max, self.parent.awi.n_lines)  # 一度の取引で大量に入荷しない
        # Price
        p = self.Imyinfo.p1[2]

        t = max(info.first_day, t_min)

        if not (info.first_day <= t < info.last_day):  # 最終日は含めない
            return None

        q = min(q_max, info.max_input)
        count = 0
        for i in range(t, info.last_day):
            count += info.input_needs[i]
        q = min(q, count)
        if q <= 0:
            return None

        if not (p_min <= p <= p_max):
            return None
        if not (q_min <= q <= q_max):
            return None
        if not (t_min <= t <= t_max):
            return None

        return (q, t, p)

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return {nid: self.best_proposal(nid) for nid in self.negotiators.keys()}
