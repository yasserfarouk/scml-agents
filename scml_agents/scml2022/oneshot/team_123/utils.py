from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from typing import Tuple, Union
from negmas import Outcome


def get_proposal(unit_price: int, quantity: int, time: int) -> Outcome:
    offer = [0, 0, 0]
    offer[UNIT_PRICE] = unit_price
    offer[QUANTITY] = quantity
    offer[TIME] = time
    return tuple(offer)


def td_concession_rate(step: int, n_steps: int, e: float) -> float:
    return 1 - ((n_steps - step - 1) / (n_steps - 1)) ** e


def get_price(cr: float, is_selling: bool, limit: Tuple[float, float]) -> float:
    mx = max(limit[0], limit[1])
    mn = min(limit[0], limit[1])
    if is_selling:
        return mx - cr * (mx - mn)
    else:
        return mn + cr * (mx - mn)


def clamp(x: float, limit: Tuple[float, float]) -> float:
    mx = max(limit[0], limit[1])
    mn = min(limit[0], limit[1])
    return min(mx, max(mn, x))


def simple_round(x: Union[float, int]) -> int:
    if type(x) == int:
        return x
    return int((x * 2 + 1) // 2)
