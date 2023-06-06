from typing import TypeVar

from negmas import Outcome
from scml import UNIT_PRICE, QUANTITY, TIME

T = TypeVar("T")


def get_outcome(unit_price: int, quantity: int, time: int) -> Outcome:
    offer = [0, 0, 0]
    offer[UNIT_PRICE] = unit_price
    offer[QUANTITY] = quantity
    offer[TIME] = time
    return tuple(offer)
