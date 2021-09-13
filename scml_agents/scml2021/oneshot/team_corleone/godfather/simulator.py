import numpy as np
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

from .bilat_ufun import BilatUFun
from .negotiation_history import BilateralHistory
from .offer import Offer
from .spaces import *
from .strategy import Strategy


class BilatSimulator:
    """Simulates a bilateral negotiation in progress. Not entirely deterministic
    because it does not know exact # of rounds (would be possible to cheat and look
    that up, but it shouldn't matter much)."""

    @staticmethod
    def simulate_negotiation(
        u_a: BilatUFun,
        u_b: BilatUFun,
        s_a: Strategy,
        s_b: Strategy,
        history_a: BilateralHistory,
    ) -> Outcome:
        """history_a: negotiation history from POV of player A."""

        # print("simulating negotiation", file=sys.stderr)

        num_rounds: int = np.random.choice(
            [
                history_a.world_info.n_negotiation_rounds,  # if our first proposal was chosen
                history_a.world_info.n_negotiation_rounds
                + 1,  # if our first proposal was ignored
            ]
        )

        def move_next():
            if history_a.whose_turn() == 1:
                move = s_a(u_a, [history_a])
                history_a.add_move(move, is_me=True)
            else:
                move = s_b(u_b, [history_a.invert()])
                history_a.add_move(move, is_me=False)

        while not history_a.is_ended() and len(history_a.moves) < num_rounds:
            move_next()
            move_next()

        if len(history_a.moves) >= num_rounds:
            history_a.fail()  # hit round limit

        # print("simulated negotiation ends with outcome", history_a.outcome(), file=sys.stderr)
        return history_a.outcome()
