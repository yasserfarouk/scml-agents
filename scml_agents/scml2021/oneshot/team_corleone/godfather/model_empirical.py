import copy
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
from negmas import AgentMechanismInterface
from scml.oneshot import OneShotAgent

from .bilat_ufun import BilatUFun, BilatUFunDummy
from .model import Model, realistic
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


class ModelEmpirical(Model):
    """Uses empirical distribution of outcomes in prior (finished) negotiations, after
    a uniform prior with the weight of 10 observation"""

    prior_bias = 10  # prior is weighted equal to 10 observed outcomes

    def __init__(self, neg_id: str, strategy_self, prior_bias=10):
        super().__init__(neg_id, strategy_self)
        self.prior_bias = prior_bias

    @realistic
    def __call__(
        self, ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        complete_histories = [h for h in histories if h.is_ended()]

        outcome_set = ufun.outcome_space.outcome_set()

        total_weight = len(complete_histories) + self.prior_bias
        distr: Dict[Outcome, float] = {}

        if total_weight > 0:
            # prior
            distr = {
                o: (
                    self.prior_bias / total_weight
                    if o == ufun.outcome_space.reserve
                    else 0
                )
                for o in outcome_set
            }

            # empirical distr
            for h in complete_histories:
                o = h.outcome()
                if o not in outcome_set:
                    o.price = max(o.price, ufun.offer_space.min_p)
                    o.price = min(o.price, ufun.offer_space.max_p)
                distr[o] += 1 / total_weight
        else:
            # default to uniform when no data available
            distr = {o: 1 / len(outcome_set) for o in outcome_set}

        # assert abs(sum(distr.values()) - 1.0) < 0.001
        return OutcomeDistrTable(ufun.outcome_space, distr)
