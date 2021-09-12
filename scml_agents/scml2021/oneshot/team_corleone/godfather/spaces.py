from enum import Enum
from typing import Set, Union

from negmas import ResponseType, SAOResponse

from .offer import Offer


class OfferSpace:
    def __init__(self, min_p: int, max_p: int, min_q: int, max_q: int, reserve: Offer):
        self._min_p = min_p
        self._max_p = max_p
        self._min_q = min_q
        self._max_q = max_q
        self._reserve = reserve

    @property
    def min_p(self):
        return self._min_p

    @property
    def max_p(self):
        return self._max_p

    @property
    def min_q(self):
        return self._min_q

    @property
    def max_q(self):
        return self._max_q

    @property
    def reserve(self):
        return self._reserve

    def offer_set(self) -> Set[Offer]:
        return {
            Offer(p, q)
            for p in range(self._min_p, self._max_p + 1)
            for q in range(self._min_q, self._max_q + 1)
        }

    def __eq__(self, other):
        return (
            self._min_p == other._min_p
            and self._max_p == other._max_p
            and self._min_q == other._min_q
            and self._max_q == other._max_q
            and self._reserve == other._reserve
        )

    def __repr__(self):
        return "OfferSpace(min_p={}, max_p={}, min_q={}, max_q={}, reserve={}".format(
            self._min_p, self._max_p, self._min_q, self._max_q, self._reserve
        )


class Moves(Enum):
    ACCEPT = 1
    END = 2


Move = Union[Moves, Offer]


class MoveSpace:
    def __init__(self, offer_space: OfferSpace):
        self._offer_space = offer_space

    def move_set(self) -> Set[Move]:
        moves: Set[Move] = set()
        return moves.union(set(Moves)).union(self._offer_space.offer_set())

    @staticmethod
    def move_to_sao_response(move: Move, cur_step: int) -> SAOResponse:
        if move is Moves.ACCEPT:
            return SAOResponse(ResponseType.ACCEPT_OFFER, None)
        elif move is Moves.END:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        elif isinstance(move, Offer):
            return SAOResponse(ResponseType.REJECT_OFFER, move.to_negmas(cur_step))
        else:
            raise RuntimeError("Unknown move type")


Outcome = Offer


class OutcomeSpace:
    def __init__(self, offer_space: OfferSpace):
        self._offer_space = offer_space

    @property
    def reserve(self):
        return self._offer_space.reserve

    def outcome_set(self) -> Set[Outcome]:
        return self._offer_space.offer_set().union({self._offer_space.reserve})

    def offer_space(self) -> OfferSpace:
        return self._offer_space

    def __eq__(self, other):
        return self._offer_space == other._offer_space
