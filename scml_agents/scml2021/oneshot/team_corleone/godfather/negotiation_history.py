import sys
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tabulate import tabulate

from .offer import Offer
from .spaces import Move, Moves, Outcome, OutcomeSpace


class WorldInfo:
    def __init__(
        self,
        n_negotiation_rounds: int,
        my_level: int,
        n_competitors: int,
        n_partners: int
        # competition_idx: float
        # etc.
    ) -> None:
        self.n_negotiation_rounds = n_negotiation_rounds
        self.my_level = my_level
        self.n_competitors = n_competitors
        self.n_partners = n_partners


class BilateralHistory:
    """
    History of one bilateral SAOP negotiation. Note that the first move (first proposal)
    is a pseudo-move, in that it has only a 50% chance of being seen by the opponent.

    History consists of
    - moves: a series of moves (offer, accept, end)
    - is_failed: did it fail due to timeout or "end negotiation" move? (yes/no)
    - a bunch of other stuff that's constant across time

    N.B.: If the opponent makes an "end negotiation" move, it will not be reflected in the series
    of moves! The negotiation will just say "failed". This is mostly a negmas limitation (no easy
    way to distinguish between timeout and opponent ending negotiation).
    """

    def __init__(
        self, opp_id: str, outcome_space: OutcomeSpace, world_info: WorldInfo
    ) -> None:
        self.opp_id = opp_id
        self.outcome_space = outcome_space
        self.world_info = world_info

        self.prediction_points: List[Dict[str, Any]] = []

        self.moves: List[Tuple[int, Move]] = []
        self.is_failed = False

        # by default, we overcount moves by 0.5 bc of two first proposals
        self.n_moves_adjustment = -0.5

    def register_prediction_point(self, ufun: "BilatUFun") -> None:
        self.prediction_points.append(
            {"trace_idx": len(self.moves), "ufun_params": ufun.poly_fit_3rd_correct()}
        )

    def add_move(self, m: Move, is_me: bool) -> None:
        mover_idx = 1 if is_me else 0
        if self.moves and self.moves[-1][0] == mover_idx:
            if self.moves[-1] == m:
                warnings.warn(
                    "Negotiation history: ignoring second identical move from same mover"
                )
            else:
                warnings.warn(
                    "Negotiation history {} invalid: two moves in a row from mover {}: {}, adding {}, is_me={}".format(
                        self.opp_id, mover_idx, self, m, is_me
                    )
                )
            return
        elif not self.moves and mover_idx == 0:
            warnings.warn(
                "Negotiation history {} invalid: first move from opponent".format(
                    self.opp_id
                )
            )
            self.moves.append((1, self.outcome_space.offer_space().offer_set().pop()))
        if not self.moves and m is Moves.ACCEPT:
            warnings.warn(
                "Negotiation history {} invalid: acceptance on first move".format(
                    self.opp_id
                )
            )
        if self.is_ended():
            warnings.warn(
                "Negotiation history {} invalid: additional move {} after negotiation ended due to {}".format(
                    self.opp_id,
                    m,
                    "failure" if self.is_failed else f"terminal move {self.moves[-1]}",
                )
            )
            return
        self.moves.append((mover_idx, m))

    def est_frac_complete(self):
        f = (len(self.moves) + self.n_moves_adjustment) / (
            2 * self.world_info.n_negotiation_rounds
        )
        return max(0, f)

    def invert(self) -> "BilateralHistory":
        """Returns a new BilateralHistory with self and opponent swapped.
        Note: NOT a deep copy."""
        bh = BilateralHistory(
            opp_id=self.opp_id,
            outcome_space=self.outcome_space,
            world_info=self.world_info,
        )
        for i, m in self.moves:
            bh.add_move(m, not i)
        return bh

    def offers(self) -> List[Move]:
        return [m for i, m in self.moves]

    def my_offers(self) -> List[Move]:
        return [m for i, m in self.moves if i]

    def opp_offers(self) -> List[Move]:
        return [m for i, m in self.moves if not i]

    def standing_offer(self) -> Optional[Offer]:
        """Offer to which we are responding, or None"""
        i, m = (
            self.moves[-1] if self.moves else (None, None)
        )  # beware: do not remove parens
        return m if i != 1 and isinstance(m, Offer) else None

    def whose_turn(self) -> Optional[int]:
        """Whose turn is it to move? Assumes we should always move first"""
        if not self.moves:
            return 1
        else:
            i, m = self.moves[-1]
            return 0 if i else 1

    def is_ended(self) -> bool:
        if self.is_failed:
            return True
        elif not self.moves:
            return False
        else:
            i, m = self.moves[-1]
            return m in {Moves.ACCEPT, Moves.END}

    def fail(self) -> None:
        """Indicate negotiation has failed"""
        if not self.moves:
            warnings.warn(
                "Negotiation failed with no moves... uh-oh. Probably means we timed out on first_proposals()"
            )
        self.is_failed = True

    def outcome(self) -> Outcome:
        """Returns the outcome of the negotiation -- either
        agreed-upon offer or reserve value. If negotiation
        has not ended, raises ValueError."""
        _, last_move = (
            self.moves[-1] if self.moves else (None, None)
        )  # beware: do not remove parens
        if last_move is Moves.END or self.is_failed:
            return self.outcome_space.reserve
        elif last_move is Moves.ACCEPT:
            _, agreement = self.moves[-2]
            if not isinstance(agreement, Outcome):
                warnings.warn("BilateralHistory failed to assure validity")
                return self.outcome_space.reserve
            return agreement
        else:
            raise RuntimeError("In-progress negotation has no outcome")

    def __repr__(self) -> str:
        return "<bilat_hist: {}, failed: {}>".format(
            [str(m) for m in self.moves], self.is_failed
        )


class SCMLHistory:
    """Keep track of received offers across many days and many negotiators"""

    def __init__(self, world_info: WorldInfo) -> None:
        self.round = 0
        self.negotiations: Dict[str, List[BilateralHistory]] = {}
        self.outcome_spaces: Dict[str, OutcomeSpace] = {}
        self.world_info = world_info

    def register_agent(self, opp_id: str, outcome_space: OutcomeSpace) -> None:
        if not opp_id in self.negotiations:
            if self.round != 0:
                warnings.warn(f"Cannot add negotiator {opp_id} at round {self.round}")
            self.outcome_spaces[opp_id] = outcome_space
            self.negotiations[opp_id] = [self._make_bilat_hist(opp_id)]

    def next_round(self) -> None:
        self.round += 1
        for opp_id, neg_list in self.negotiations.items():
            neg_list.append(self._make_bilat_hist(opp_id))

    def _make_bilat_hist(self, opp_id: str) -> BilateralHistory:
        return BilateralHistory(
            opp_id=opp_id,
            outcome_space=self.outcome_spaces[opp_id],
            world_info=self.world_info,
        )

    def no_first_proposals(self, opp_id: str) -> None:
        current_negotiation = self.negotiations[opp_id][-1]
        current_negotiation.n_moves_adjustment = -1

    def fail(self, opp_id: str) -> None:
        if opp_id not in self.negotiations:
            warnings.warn(f"failing unknown negotiation {opp_id}")
            return
        current_negotiation = self.negotiations[opp_id][-1]
        current_negotiation.fail()

    def move(self, opp_id: str, move: Move, is_me: bool) -> None:
        if opp_id not in self.negotiations:
            warnings.warn(
                "moving in unknown negotiation {}, {}, is_me={}".format(
                    opp_id, move, is_me
                )
            )
            return
        current_negotiation = self.negotiations[opp_id][-1]
        current_negotiation.add_move(move, is_me=is_me)

    def my_move(self, opp_id: str, move: Move) -> None:
        self.move(opp_id, move, is_me=True)

    def opponent_move(self, opp_id: str, move: Move) -> None:
        self.move(opp_id, move, is_me=False)

    def negotiator_histories(self, opp_id: str) -> List[BilateralHistory]:
        if not opp_id in self.negotiations:
            raise ValueError(
                "Invalid negotiator id {} is not a registered agent ({})".format(
                    opp_id, self.negotiations.keys()
                )
            )
        return self.negotiations[opp_id]

    def current_negotiator_history(self, opp_id: str) -> BilateralHistory:
        return self.negotiator_histories(opp_id)[-1]

    def __repr__(self) -> str:
        return f"<SCMLHistory {self.negotiations}>"
