import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

from .offer import Offer
from .spaces import *


class OutcomeDistr:
    def __init__(self, o: OutcomeSpace) -> None:
        self.outcome_space = o

    def __call__(self, o: Outcome) -> float:
        raise NotImplementedError

    def sample(self) -> Outcome:
        raise NotImplementedError

    def marginalize(self) -> Tuple[List[float], float]:
        est_p = 0.0
        q_probs = [0.0] * 11

        for o in self.outcome_space.outcome_set():
            prob = self(o)
            q_probs[o.quantity] += prob
            est_p += prob * o.price

        return q_probs, est_p

    def distance(self, other: "OutcomeDistr") -> float:
        """Total variation distance
        https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures"""
        if not self.outcome_space == other.outcome_space:
            raise ValueError(
                "Can only find distance between distributions over same outcome space"
            )
        return max(abs(self(o) - other(o)) for o in self.outcome_space.outcome_set())

    def vis(self, fig, ax):
        x, y, z = [], [], []
        for o in self.outcome_space.outcome_set():
            x.append(o.price)
            y.append(o.quantity)
            z.append(self(o))
        plot = ax.scatter(x, y, c=z)
        ax.set_title("probability by p/q")
        ax.set_xlabel("price")
        ax.set_ylabel("quant")
        fig.colorbar(plot, ax=ax)

    def __repr__(self) -> str:
        rstr = "<"
        for o in self.outcome_space.outcome_set():
            if self(o) > 0:
                rstr += "({0.price}, {0.quantity}) -> {1},".format(o, self(o))
        return rstr[:-1] + ">"

    def __eq__(self, other) -> bool:
        return all(
            abs(self(o) - other(o)) < 0.01 for o in self.outcome_space.outcome_set()
        )


class OutcomeDistrFunc(OutcomeDistr):
    def __init__(
        self,
        outcome_space: OutcomeSpace,
        eval_fun: Callable[[Outcome], float],
        sample_fun: Callable[[], Outcome],
    ):
        self.eval_fun = eval_fun
        self.sample_fun = sample_fun
        super().__init__(outcome_space)

    def __call__(self, o: Outcome) -> float:
        return self.eval_fun(o)

    def sample(self) -> Outcome:
        return self.sample_fun()


class OutcomeDistrPoint(OutcomeDistr):
    def __init__(self, outcome_space: OutcomeSpace, point: Outcome):
        self.point = point
        super().__init__(outcome_space)

    def __call__(self, o: Outcome):
        return 1 if o == self.point else 0

    def sample(self) -> Outcome:
        return self.point


class OutcomeDistrUniform(OutcomeDistr):
    def __init__(self, outcome_space: OutcomeSpace):
        self.size = len(outcome_space.outcome_set())
        super().__init__(outcome_space)

    def __call__(self, o: Outcome):
        return 1 / self.size

    def sample(self) -> Outcome:
        x = np.random.random()
        p = 1 / self.size
        for o in self.outcome_space.outcome_set():
            if x < p:
                return o
            x -= p
        warnings.warn("Probabilities do not sum to 1")
        return self.outcome_space.outcome_set().pop()


class OutcomeDistrTable(OutcomeDistr):
    def __init__(self, outcome_space: OutcomeSpace, distr: Dict[Outcome, float]):
        self.distr = distr
        super().__init__(outcome_space)

    def __call__(self, o: Outcome) -> float:
        if o in self.distr:
            return self.distr[o]
        else:
            if o in self.outcome_space.outcome_set():
                raise KeyError(f"Outcome {o} not in distribution domain")
            else:
                raise RuntimeError(
                    "OutcomeDistrTable error: distribution dict does not cover outcome space (missing {})".format(
                        o
                    )
                )

    def sample(self) -> Outcome:
        x = np.random.random()
        for o, p in self.distr.items():
            if x < p:
                return o
            x -= p
        warnings.warn("Probabilities do not sum to 1")
        return self.outcome_space.outcome_set().pop()


class OutcomeDistrRandom(OutcomeDistr):
    """Assigns probabilities randomly to all outcomes in the space"""

    def __init__(self, outcome_space: OutcomeSpace):
        self.distr: Dict[Outcome, float] = {}
        all_outcomes = outcome_space.outcome_set()
        probs = np.random.random(len(all_outcomes))
        probs /= sum(probs)
        for idx, o in enumerate(all_outcomes):
            self.distr[o] = probs[idx]
        super().__init__(outcome_space)

    def __call__(self, o: Outcome) -> float:
        return self.distr[o]

    def sample(self) -> Outcome:
        x = np.random.random()
        for o, p in self.distr.items():
            if x < p:
                return o
            x -= p
        warnings.warn("Probabilities do not sum to 1")
        return self.outcome_space.outcome_set().pop()


class OutcomeDistrMarginal(OutcomeDistr):
    def __init__(self, outcome_space: OutcomeSpace, q_probs: List[float], p_est: float):
        super().__init__(outcome_space)
        if not abs(sum(q_probs) - 1) < 0.001:
            raise RuntimeError("Invalid q_probs for OutcomeDistrMarginal")
        self.q_probs = q_probs
        p_est_int = int(p_est)
        offer_space = outcome_space.offer_space()
        self.p_est = max(min(p_est_int, offer_space.max_p), offer_space.min_p)

    def __call__(self, o: Outcome) -> float:
        if self.p_est == o.price:
            return self.q_probs[o.quantity]
        else:
            return 0

    def sample(self) -> Outcome:
        x = np.random.random()
        for q, prob in enumerate(self.q_probs):
            if x < prob:
                return q
            x -= prob
        warnings.warn("Probabilities do not sum to 1")
        return self.outcome_space.outcome_set().pop()
