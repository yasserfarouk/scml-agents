import math
import random
from typing import Callable, List

import numpy as np
from negmas import SAOResponse
from numpy.lib.arraysetops import _unpack_tuple

from .bilat_ufun import BilatUFun, BilatUFunAvg
from .nash_calc_godfather import NashCalcGodfather
from .negotiation_history import BilateralHistory
from .offer import Offer
from .spaces import *


class Strategy:
    """Bilateral negotiation strategy. Must return a move, but NOT
    Moves.ACCEPT or Moves.END on the first round"""

    def __call__(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> Move:
        raise NotImplementedError

    @staticmethod
    def _is_first_proposal(histories: List[BilateralHistory]):
        """Is this a first proposal (meaning we must return an Offer type)?"""
        return not histories[-1].offers()


class StrategyMin(Strategy):
    """Offers min price, quantity. Never accepts."""

    def __call__(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> Move:
        offer_space = ufun.offer_space
        p = offer_space.min_p
        q = offer_space.max_q
        # p = min(offer_space.min_p + round_num, offer_space.max_p)
        # q = min(offer_space.min_q + round_num, offer_space.max_q)
        return Offer(p, q)


class StrategyRandom(Strategy):
    """Accepts offers half the time, and half the time proposes a random offer"""

    def __call__(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> Move:
        if not histories:
            raise ValueError("No histories provided")

        if not self._is_first_proposal(histories) and np.random.random() < 0.25:
            # after first round, 25% chance of accepting
            return Moves.ACCEPT
        else:
            all_offers = ufun.offer_space.offer_set()
            if not len(all_offers):
                raise ValueError("No possible moves")
            return np.random.choice(list(all_offers))


class StrategyAspiration(Strategy):
    ASPIRATION_EXPONENT = 1.0  # linear

    def _set_state(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> None:
        """Store ufun, histories, and a list of (util, offer) pairs in ascending order of utility"""
        self.ufun = ufun
        self.histories = histories

        self.utils_offers = []
        self.os = ufun.offer_space
        for p in range(self.os.min_p, self.os.max_p + 1):
            for q in range(self.os.min_q, self.os.max_q + 1):
                o = Offer(p, q)
                self.utils_offers.append((ufun(o), o))

        self.utils_offers.sort(
            key=lambda x: (x[0], -x[1].quantity)
        )  # highest quantities come first
        self.best_util, self.best_offer = self.utils_offers[-1]

    def init(self) -> None:
        """Called at the beginning of __call__. Any necessary precomputations
        for start_end_util and ufun_inverse"""
        pass

    def get_start_end_util(self):
        """Get the start and end utils. (In this case, start from our best, move toward reserve.)
        Overload for different behavior."""
        start_util = self.best_util
        end_util = self.ufun(self.os.reserve)
        return start_util, end_util

    def ufun_inverse(self, threshold: float) -> Offer:
        """In general, gets an offer with a utility around threshold.
        This specific implementation gets the outcome with lowest utility
        above the threshold (n.b.: price/quantity not very stable),
        or maximum-utility outcome if none clears the threshold.
        Overload for different behavior."""
        # TODO: binary search
        for idx in range(len(self.utils_offers)):
            util, offer = self.utils_offers[idx]
            if util >= threshold:
                return offer

        # fallback
        return self.best_offer

    def _get_target_utility(
        self, t: float, start_util: float, end_util: float
    ) -> float:
        asp_val = 1.0 - math.pow(t, self.ASPIRATION_EXPONENT)
        return asp_val * start_util + (1 - asp_val) * end_util

    def __call__(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> Move:
        self._set_state(ufun, histories)

        self.init()
        start_util, end_util = self.get_start_end_util()

        if self.best_util < ufun(self.os.reserve):
            if self._is_first_proposal(histories):
                return self.best_offer  # can't use Moves.END
            else:
                return Moves.END

        current_history = histories[-1]
        standing_offer = current_history.standing_offer()

        t = current_history.est_frac_complete()
        target_util = self._get_target_utility(t, start_util, end_util)
        standing_util = (
            ufun(standing_offer) if standing_offer is not None else -math.inf
        )

        if standing_util >= target_util and not self._is_first_proposal(histories):
            return Moves.ACCEPT
        else:
            return self.ufun_inverse(target_util)


class StrategyParetoAspiration(StrategyAspiration):
    """Abstract pareto aspiration agent. Does not implement an opponent model."""

    PCT = 0.5

    def get_opp_ufun(self) -> BilatUFun:
        raise NotImplementedError

    def init(self) -> None:
        """Called at the beginning of __call__. Any necessary precomputations
        for start_end_util and ufun_inverse_above_threshold"""
        self.opp_ufun = self.get_opp_ufun()
        # assert self.ufun.offer_space == self.opp_ufun.offer_space, "mine {}, {}; theirs {}, {}".format(
        #     self.ufun.offer_space,
        #     self.opp_ufun.offer_space)

        self.nc = NashCalcGodfather(self.ufun, self.opp_ufun)
        # print(self.nc.frontier_offers())

    def get_start_end_util(self):
        """Move toward a utility threshold between nash point util and reserve util"""
        # start: same as stock StrategyAspiration
        start_util = self.best_util

        # end: partway between nbp and reserve
        nbp_util = self.nc.nash_point_my_util()
        reserve_util = self.ufun(self.os.reserve)
        end_util = nbp_util * self.PCT + reserve_util * (1 - self.PCT)

        return start_util, end_util

    def ufun_inverse(self, threshold: float) -> Offer:
        """Closest below threshold (TODO: what about nearest to threshold?)"""
        dist = float("inf")
        chosen = None
        for o in self.nc.frontier_offers():
            u = self.ufun(o)
            if threshold - u >= 0 and threshold - u < dist:
                dist = threshold - u
                chosen = o
        return chosen if chosen is not None else self.best_offer


class StrategySimpleParetoAspiration(StrategyParetoAspiration):
    def get_opp_ufun(self) -> BilatUFun:
        return BilatUFunAvg(self.os, self.histories[-1])


class StrategyGoldfishParetoAspiration(StrategyParetoAspiration):
    PCT = 0.75
    ASPIRATION_EXPONENT = 0.5

    def get_opp_ufun(self) -> BilatUFun:
        standing_offer = self.histories[-1].standing_offer()
        ex_q = standing_offer.quantity if standing_offer else None
        return BilatUFunAvg(self.os, self.histories[-1], expected_exog_quant=ex_q)


class StrategyHardnosedGoldfish(StrategyGoldfishParetoAspiration):
    ASPIRATION_EXPONENT = 4.0


class StrategyHardnosedBigger(StrategyGoldfishParetoAspiration):
    ASPIRATION_EXPONENT = 4.0
    PCT = 0.75


class StrategyHardnosedSmaller(StrategyGoldfishParetoAspiration):
    ASPIRATION_EXPONENT = 4.0
    PCT = 0.25


class StrategySoftnosedGoldfish(StrategyGoldfishParetoAspiration):
    ASPIRATION_EXPONENT = 0.25


class StrategyParameterizedGoldfish(StrategyGoldfishParetoAspiration):
    def __init__(self, asp_ex, pct):
        self.ASPIRATION_EXPONENT = asp_ex
        self.PCT = pct


class StrategyParameterizedRegAsp(StrategyGoldfishParetoAspiration):
    def __init__(self, asp_ex):
        self.ASPIRATION_EXPONENT = asp_ex


class StrategyCheatingParetoAspiration(StrategyParetoAspiration):
    def __init__(self, ufun_getter: Callable[[], BilatUFun]) -> None:
        self._ufun_getter = ufun_getter
        super().__init__()

    def get_opp_ufun(self) -> BilatUFun:
        opp_ufun = self._ufun_getter()
        if opp_ufun is not None:
            return opp_ufun
        else:
            return BilatUFunAvg(self.os, self.histories[-1])


class StrategyProphetAspiration(StrategyParetoAspiration):
    """Requires the bilateral ufuns of every step in the negotiation history to be recorded
    (or at least the normalized utility values)"""

    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory], prev_ufuns
    ) -> Move:
        self._set_state(ufun, histories)

        self.init()
        start_util, end_util = self.get_start_end_util()

        if self.best_util < ufun(self.os.reserve):
            if self._is_first_proposal(histories):
                return self.best_offer  # can't use Moves.END
            else:
                return Moves.END

        current_history = histories[-1]
        standing_offer = current_history.standing_offer()

        t = current_history.est_frac_complete()
        target_util = self._get_target_utility(t, start_util, end_util)
        standing_util = (
            ufun(standing_offer) if standing_offer is not None else -math.inf
        )

        if self.acceptance(
            histories, prev_ufuns, standing_util
        ) and not self._is_first_proposal(histories):
            return Moves.ACCEPT
        else:
            return self.ufun_inverse(target_util)

    def acceptance(self, histories, ufuns, standing_util):
        t = histories[-1].est_frac_complete()
        curr_step = len(histories[-1].opp_offers())
        selected_indices = [-1]

        sum = 0
        for i in range(min(10, len(histories) - 1)):
            index = -1
            while index in selected_indices:
                index = random.randrange(0, len(histories - 1))
            selected_indices.append(index)
            hist = histories[index].opp_offers()
            max_offer = 0
            for j in range(curr_step, len(hist)):
                # uf = ufuns[index][j]
                ## if we had stored normalized utilities could use
                offer_util_norm = histories[index].opp_offers_utils[j]

                # offer_util_norm = self.normalize_util(uf, uf(hist[j]))
                if offer_util_norm > max_offer:
                    max_offer = offer_util_norm
            sum += max_offer

        expected_util = sum / 10

        adjusted_standing_util = standing_util / self.util_range(self.ufun)

        return adjusted_standing_util > 0.5 * expected_util

    def price_range(self, a_ufun):
        return a_ufun.offer_space.min_p, a_ufun.offer_space.max_p

    def mq(self, a_ufun):
        return a_ufun.offer_space.max_q

    def normalize_util(self, a_ufun, utility):
        max_val = float("-inf")
        min_val = float("inf")
        for q in range(1, self.mq(a_ufun) + 1):
            for p in range(
                self.price_range(a_ufun)[0], self.price_range(a_ufun)[1] + 1
            ):
                offer = Offer(p, q)
                value = a_ufun(offer)
                if value > max_val:
                    max_val = value
                if value < min_val:
                    min_val = value

        u_range = max_val - min_val
        return (utility - min_val) / u_range
