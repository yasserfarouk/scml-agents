import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from negmas import AgentMechanismInterface
from scml.oneshot import OneShotAgent

from .bilat_ufun import BilatUFun, BilatUFunDummy
from .negotiation_history import BilateralHistory, SCMLHistory
from .offer import Offer
from .outcome_distr import (
    OutcomeDistr,
    OutcomeDistrMarginal,
    OutcomeDistrPoint,
    OutcomeDistrTable,
    OutcomeDistrUniform,
)
from .simulator import BilatSimulator
from .spaces import *

# import .godfather as godfather
from .strategy import Strategy, StrategyAspiration


def realistic(model_call):
    def call(
        model,
        ufun: BilatUFun,
        histories: List[BilateralHistory],
        enable_realistic_checks=True,
    ) -> OutcomeDistr:
        if enable_realistic_checks:
            if not model._strategy_self:
                # raise RuntimeError("realistic model needs a defined strategy")
                return model_call(model, ufun, histories)

            if histories[-1].is_ended():
                return OutcomeDistrPoint(ufun.outcome_space, histories[-1].outcome())
            next_move = model._strategy_self(ufun, histories)
            if next_move == Moves.ACCEPT:
                return OutcomeDistrPoint(
                    ufun.outcome_space, histories[-1].standing_offer()
                )
            elif next_move == Moves.END:
                return OutcomeDistrPoint(ufun.outcome_space, ufun.outcome_space.reserve)

        return model_call(model, ufun, histories)

    return call


class Model:
    """Bilateral negotiation model"""

    def __init__(self, neg_id: str, strategy_self):
        self._neg_id = neg_id
        self._strategy_self = strategy_self

    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        # note: ignoring prior info for now
        raise NotImplementedError


class ModelDisagreement(Model):
    """Always predicts disagreement"""

    @realistic
    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        return OutcomeDistrPoint(ufun.outcome_space, ufun.outcome_space.reserve)


class ModelRandomPoint(Model):
    """Predicts a randomly drawn point distr"""

    @realistic
    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        pt = np.random.choice(list(ufun.outcome_space.outcome_set()))
        return OutcomeDistrPoint(ufun.outcome_space, pt)


class ModelUniform(Model):
    """Uniform distr"""

    @realistic
    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        return OutcomeDistrUniform(ufun.outcome_space)


# class ModelRollingAvg(Model):
#     """Predicts that negotiations will end at halfway between most recent two offers"""
#     def __call__(self, ufun: BilatUFun, histories: List[BilateralHistory]) -> OutcomeDistr:
#         current_history = histories[-1]
#         last_offers = current_history.offers()[-2:]
#         if len(last_offers) >= 2:
#             a, b = last_offers[0], last_offers[1]
#             if isinstance(a, Offer) and isinstance(b, Offer):
#                 p = round((a.price + b.price) / 2)
#                 q = round((a.quantity + b.quantity) / 2)
#                 o = Offer(p, q)
#                 if o in ufun.outcome_space.outcome_set():
#                     return OutcomeDistrPoint(ufun.outcome_space, o)

#         m = ModelUniform(self._neg_id)
#         return m(ufun, histories)


class ModelCheating(Model):
    """Cheats, using opponent's calculated marginal utilities. Only works against GodfatherAgent."""

    def __init__(
        self,
        ufun_getter: Callable[[], Optional[BilatUFun]],
        strategy_self=StrategyAspiration(),
        strategy_opp=StrategyAspiration(),
    ):
        self._get_opp_ufun = ufun_getter
        self._strategy_self = strategy_self
        self._strategy_opp = strategy_opp

    @realistic
    def __call__(
        self, self_ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        opp_ufun = self._get_opp_ufun()
        if opp_ufun is None:
            m = ModelEmpirical(neg_id="", strategy_self=self._strategy_self)
            return m(self_ufun, histories)

        current_history = copy.deepcopy(
            histories[-1]
        )  # very important! can't mess with this history
        outcome = BilatSimulator.simulate_negotiation(
            self_ufun,
            opp_ufun,
            self._strategy_self,
            self._strategy_opp,
            current_history,
        )

        return OutcomeDistrPoint(self_ufun.outcome_space, outcome)
