from typing import TypeVar, Iterable
from itertools import combinations, chain
import random

T = TypeVar("T")


def powerset(iterable: Iterable[T]) -> Iterable[tuple[T, ...]]:
    s = list(iterable)
    c = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return c


def weighted_sample(items: list[tuple[T, float]]) -> T:
    weight_sum = sum([item[1] for item in items])
    r = random.uniform(0, weight_sum)
    s = 0.0
    for value, weight in items:
        if s + weight > r:
            return value
        else:
            s += weight

    return items[-1][0]
