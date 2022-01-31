import warnings
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Collection, List, Optional, Tuple, Union

from negmas import Contract
from negmas.outcomes import Issue, Outcome
from negmas.preferences import UtilityFunction, Value
from scml.oneshot.ufun import OneShotUFun
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE, is_system_agent


class UFunCalc:
    @staticmethod
    def find_insert_index(l: List, e, key: Callable):
        """Finds the index at which to insert e into l, assuming l is sorted per key"""
        i = 0
        for item in l:
            if key(e) <= key(item):
                return i
            i += 1
        return i

    @staticmethod
    def is_sorted(l: List, key: Callable):
        return all(key(l[i]) <= key(l[i + 1]) for i in range(len(l) - 1))

    @staticmethod
    def order(x):
        offer, is_output, is_exogenous = x
        return -offer[UNIT_PRICE] if is_output else offer[UNIT_PRICE]

    @staticmethod
    def order_output(offer):
        return -offer[UNIT_PRICE]

    @staticmethod
    def order_input(offer):
        return offer[UNIT_PRICE]

    def __init__(self, ufun: OneShotUFun, is_selling: bool):
        if not ufun.ex_qin and not ufun.ex_qout:
            warnings.warn("qin = qout = 0")
        exog_offer = (0, 0, 0)
        if ufun.ex_qin:
            exog_offer = (ufun.ex_qin, 0, ufun.ex_pin / ufun.ex_qin)
        elif ufun.ex_qout:
            exog_offer = (ufun.ex_qout, 0, ufun.ex_pout / ufun.ex_qout)

        self.exog = exog_offer
        self.exog_q = self.exog[QUANTITY]
        self.exog_p = self.exog[UNIT_PRICE] * self.exog[QUANTITY]
        self.f_exog_out = bool(ufun.ex_qout)

        self.ufun = ufun
        self.is_selling = is_selling

        self.persistent_pin = 0
        self.persistent_qin = 0

    def set_persistent_offers(self, persistent_offers: List[Tuple[int, ...]]):
        """Set the set of non-exogenous offers that will be included in all calculations"""
        offer_key = self.order_output if self.is_selling else self.order_input
        offers = sorted(persistent_offers, key=offer_key)
        self.persistent_offers = offers
        if not self.is_selling:  # input offers
            self.persistent_p = sum(
                o[UNIT_PRICE] * o[QUANTITY] for o in self.persistent_offers
            )
            self.persistent_q = sum(o[QUANTITY] for o in self.persistent_offers)

    def ufun_from_offer(
        self, new_offer: Optional[Tuple], return_producible=False
    ) -> Union[float, Tuple[float, int]]:
        """
        NOTES: speedups based on removing deecopy (so, will modify inputs)
        and requiring sorted offers on input
        and assuming we aren't going bankrupt
        """
        non_exog_offers = self.persistent_offers.copy()

        if new_offer is not None:
            offer_key = self.order_output if self.is_selling else self.order_input
            idx = self.find_insert_index(l=non_exog_offers, e=new_offer, key=offer_key)
            non_exog_offers[idx:idx] = [new_offer]

        # disable this for performance
        # assert self.is_sorted(zip_l, key=self.order)

        # initialize some variables
        qin, qout, pin, pout = 0, 0, 0, 0
        # qin_bar = 0
        pout_bar = 0

        # input_offers: List[Tuple[int, ...]] = []
        output_offers: List[Tuple[int, ...]] = []
        if self.is_selling:
            # input_offers = self.exog
            output_offers = non_exog_offers
            pin, qin = self.exog_p, self.exog_q
        else:
            # input_offers = non_exog_offers
            output_offers = [self.exog]
            pin, qin = self.persistent_p, self.persistent_q
            if new_offer is not None:
                pin += new_offer[UNIT_PRICE] * new_offer[QUANTITY]
                qin += new_offer[QUANTITY]

        # we calculate the total quantity we are are required to pay for `qin` and
        # the associated amount of money we are going to pay `pin`. Moreover,
        # we calculate the total quantity we can actually buy given our limited
        # money balance (`qin_bar`).
        # for offer in input_offers:
        #     topay_this_time = offer[UNIT_PRICE] * offer[QUANTITY]
        #     pin += topay_this_time
        #     qin += offer[QUANTITY]

        # if not going_bankrupt:
        qin_bar = qin

        # calculate the maximum amount we can produce given our limited production
        # capacity and the input we CAN BUY
        n_lines = self.ufun.n_lines
        producible = min(qin_bar, n_lines)

        # find the total sale quantity (qout) and money (pout). Moreover find
        # the actual amount of money we will receive
        done_selling = False
        for offer in output_offers:
            if not done_selling:
                if qout + offer[QUANTITY] >= producible:
                    # assert producible >= qout, f"producible {producible}, qout {qout}"
                    can_sell = producible - qout
                    done_selling = True
                else:
                    can_sell = offer[QUANTITY]
                pout_bar += can_sell * offer[UNIT_PRICE]
            pout += offer[UNIT_PRICE] * offer[QUANTITY]
            qout += offer[QUANTITY]

        # should never produce more than we signed to sell
        producible = min(producible, qout)

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        producible = min(qin, self.ufun.n_lines, producible)

        # the scale with which to multiply disposal_cost and shortfall_penalty
        # if no scale is given then the unit price will be used.
        output_penalty = self.ufun.output_penalty_scale
        if output_penalty is None:
            output_penalty = pout / qout if qout else 0
        output_penalty *= self.ufun.shortfall_penalty * max(0, qout - producible)
        input_penalty = self.ufun.input_penalty_scale
        if input_penalty is None:
            input_penalty = pin / qin if qin else 0
        input_penalty *= self.ufun.disposal_cost * max(0, qin - producible)

        # call a helper method giving it the total quantity and money in and out.
        return self.from_aggregates(
            qin, qout, producible, pin, pout_bar, input_penalty, output_penalty
        )

    def from_aggregates(
        self,
        qin: int,
        qout_signed: int,
        qout_sold: int,
        pin: int,
        pout: int,
        input_penalty,
        output_penalty,
    ) -> float:
        produced = min(qin, self.ufun.n_lines, qout_sold)
        # a = pout - pin - input_penalty - output_penalty
        # b = self.ufun.production_cost * produced
        # return a - b
        return (
            pout
            - pin
            - self.ufun.production_cost * produced
            - input_penalty
            - output_penalty
        )
