from typing import Tuple

from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE


class Offer:
    """Unit price, quantity pair"""

    def __init__(self, unit_price: int, quantity: int):
        self.price = int(unit_price)
        self.quantity = int(quantity)

    def __repr__(self):
        return "Offer(unit_price={0.price}, quantity={0.quantity})".format(self)

    def getprice(self):
        return self.price

    def getquant(self):
        return self.quantity

    def to_negmas(self, cur_step: int) -> Tuple[int, ...]:
        offer = [-1] * 3
        offer[TIME] = cur_step
        offer[QUANTITY] = self.quantity
        offer[UNIT_PRICE] = self.price
        return tuple(offer)

    def __eq__(self, other):
        if isinstance(other, Offer):
            return self.price == other.price and self.quantity == other.quantity
        return None

    def __hash__(self):
        return hash(repr(self))
