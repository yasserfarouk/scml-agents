"""
shared infoset encoder for the SCML EFR v1 pipeline.

single source of truth for the abstract state key used by both
build_game.py (offline EFG generator) and efr_oneshot_agent.py
(runtime policy lookup). if these two sides ever disagree on the
key format, the agent silently looks up garbage at runtime — that
is the most dangerous failure mode in the whole stack, so this
module is intentionally tiny, pure, and unit-tested.

abstraction follows v1 assumption 4:
  - role         : "S" (seller) or "B" (buyer)
  - my_type      : private cost bucket in {0,1,2}
  - my_exog      : own exogenous quantity bucket in {0,1,2}
  - round        : negotiation round k in {0..K-1}
  - last_offer   : opponent's most recent offer, abstracted to
                   (q_bucket, p_bucket) or None on the first move
  - my_history   : tuple of all offers I have made so far in this
                   negotiation, each as (q_bucket, p_bucket).
                   required for perfect recall — CFR / run_corr_dist
                   produce wrong policies on imperfect-recall games.

partner type and partner exog quantity are NOT in the key — they
are chance moves the player cannot observe (per the EFG semantics).
"""

from dataclasses import dataclass
from typing import Optional

# v1 discretization constants. these are *the* numbers; build_game.py
# and any tests must import them from here, never hardcode.
N_TYPE_BUCKETS = 3
N_EXOG_BUCKETS = 3
N_QTY = 4              # quantity in {1, 2, 3, 4}
N_PRICE = 2            # price in {low, high}
K_ROUNDS = 4

# number of OTHER concurrent partners, beyond this bilateral. drawn at the
# chance node so the CCE can condition on it — matches the runtime signal
# `len(self.negotiators) - 1`. this lets one trained policy generalize over
# daily partner counts instead of collapsing to a fixed prior.
N_OTHER_VALUES = (1, 2, 3, 4)
N_OTHER_BUCKETS = len(N_OTHER_VALUES)

QTY_VALUES = tuple(range(1, N_QTY + 1))      # (1,2,3,4)
PRICE_VALUES = tuple(range(N_PRICE))          # (0=low, 1=high)


@dataclass(frozen=True)
class InfosetKey:
    """abstract decision-node key. hashable, serializable."""

    role: str                              # "S" or "B"
    my_type: int                           # 0..N_TYPE_BUCKETS-1
    my_exog: int                           # 0..N_EXOG_BUCKETS-1
    n_other_idx: int                       # index into N_OTHER_VALUES
    round: int                             # 0..K_ROUNDS-1
    last_offer: Optional[tuple[int, int]]  # opp's last offer, (q,p) or None
    my_history: tuple[tuple[int, int], ...] = ()  # my prior offers, in order

    def __post_init__(self) -> None:
        # cheap invariants — catch drift between caller and encoder
        if self.role not in ("S", "B"):
            raise ValueError(f"role must be S or B, got {self.role!r}")
        if not 0 <= self.my_type < N_TYPE_BUCKETS:
            raise ValueError(f"my_type out of range: {self.my_type}")
        if not 0 <= self.my_exog < N_EXOG_BUCKETS:
            raise ValueError(f"my_exog out of range: {self.my_exog}")
        if not 0 <= self.n_other_idx < N_OTHER_BUCKETS:
            raise ValueError(f"n_other_idx out of range: {self.n_other_idx}")
        if not 0 <= self.round < K_ROUNDS:
            raise ValueError(f"round out of range: {self.round}")
        if self.last_offer is not None:
            q, p = self.last_offer
            if not 0 <= q < N_QTY:
                raise ValueError(f"last_offer q bucket out of range: {q}")
            if not 0 <= p < N_PRICE:
                raise ValueError(f"last_offer p bucket out of range: {p}")
        for q, p in self.my_history:
            if not 0 <= q < N_QTY or not 0 <= p < N_PRICE:
                raise ValueError(f"my_history entry out of range: ({q},{p})")

    @property
    def n_other(self) -> int:
        """number of OTHER concurrent partners implied by n_other_idx."""
        return N_OTHER_VALUES[self.n_other_idx]

    def serialize(self) -> str:
        """canonical string form. used as the .efg infoset label and
        as the lookup key in the dumped policy file."""
        if self.last_offer is None:
            tail = "x"
        else:
            tail = f"{self.last_offer[0]}{self.last_offer[1]}"
        if self.my_history:
            hist = ",".join(f"{q}{p}" for q, p in self.my_history)
        else:
            hist = "x"
        return (
            f"{self.role}|{self.my_type}|{self.my_exog}|{self.n_other_idx}|"
            f"{self.round}|{tail}|{hist}"
        )

    @classmethod
    def parse(cls, s: str) -> "InfosetKey":
        role, mt, me, noi, rnd, tail, hist = s.split("|")
        last = None if tail == "x" else (int(tail[0]), int(tail[1]))
        if hist == "x":
            my_history: tuple[tuple[int, int], ...] = ()
        else:
            my_history = tuple((int(tok[0]), int(tok[1])) for tok in hist.split(","))
        return cls(
            role=role,
            my_type=int(mt),
            my_exog=int(me),
            n_other_idx=int(noi),
            round=int(rnd),
            last_offer=last,
            my_history=my_history,
        )


# ----------------------------------------------------------------------
# bucketing helpers — pure functions, no SCML imports.
# the runtime agent feeds in primitives extracted from `self.awi`;
# build_game.py feeds in the same primitives drawn from the prior.
# ----------------------------------------------------------------------

def bucket_type(cost: float, cost_min: float, cost_max: float) -> int:
    """bucket a continuous private cost into N_TYPE_BUCKETS equal-width bins."""
    return _equal_width_bucket(cost, cost_min, cost_max, N_TYPE_BUCKETS)


def bucket_exog(qty: int, qty_min: int, qty_max: int) -> int:
    """bucket an exogenous quantity into N_EXOG_BUCKETS equal-width bins."""
    return _equal_width_bucket(qty, qty_min, qty_max, N_EXOG_BUCKETS)


def bucket_qty(qty: int) -> int:
    """quantize a real SCML offer quantity to {0..N_QTY-1}.
    clips to the abstract grid; ifq>N_QTY it folds to the top bucket."""
    if qty < 1:
        return 0
    if qty > N_QTY:
        return N_QTY - 1
    return qty - 1


def bucket_price(price: float, price_low: float, price_high: float) -> int:
    """quantize an offer price to {0=low, 1=high} via midpoint split."""
    if price_high <= price_low:
        return 0
    mid = 0.5 * (price_low + price_high)
    return 0 if price <= mid else 1


def _equal_width_bucket(x: float, lo: float, hi: float, n: int) -> int:
    if hi <= lo:
        return 0
    # clip then map to [0, n-1]
    if x <= lo:
        return 0
    if x >= hi:
        return n - 1
    frac = (x - lo) / (hi - lo)
    return min(n - 1, int(frac * n))


# ----------------------------------------------------------------------
# action encoding. abstract action ids are dense integers, used as
# both EFG move labels and policy-table column indices.
# ----------------------------------------------------------------------

# offer actions: enumerate (q_bucket, p_bucket) lexicographically
# id 0..N_QTY*N_PRICE-1 are offers; the next two are accept and end.
N_OFFER_ACTIONS = N_QTY * N_PRICE
ACTION_ACCEPT = N_OFFER_ACTIONS
ACTION_END = N_OFFER_ACTIONS + 1
N_ACTIONS = N_OFFER_ACTIONS + 2


def offer_action_id(q_bucket: int, p_bucket: int) -> int:
    return q_bucket * N_PRICE + p_bucket


def decode_action(action_id: int) -> tuple[str, Optional[tuple[int, int]]]:
    """returns (kind, payload). kind in {'offer','accept','end'}."""
    if action_id == ACTION_ACCEPT:
        return ("accept", None)
    if action_id == ACTION_END:
        return ("end", None)
    if not 0 <= action_id < N_OFFER_ACTIONS:
        raise ValueError(f"unknown action id: {action_id}")
    q = action_id // N_PRICE
    p = action_id % N_PRICE
    return ("offer", (q, p))
