"""
EFRDistOneShotAgent: a competition entry for SCML OneShot that fuses the
winning need-distribution heuristic with an offline-solved extensive-form
regret minimization (EFR) bilateral price policy.

the OneShot day is, mechanically, three joint-across-partners decisions:
  1. allocation  -- split today's need over live partners
  2. acceptance  -- accept the combination of offers nearest the need
  3. price       -- what unit price to ask / when to concede

the bilateral EFR abstraction provably cannot represent (1) or (2): that
joint state is in no infoset. but (3) is exactly the 2-player alternating
-offer price game EFR was solved for. so we decompose by responsibility:

  backbone (SyncRandomOneShotAgent): owns allocation + powerset acceptance.
  EFR policy: owns the per-partner ask price + a price gate on acceptance.

every EFR consult is guarded: on a policy miss or any error we fall back to
the backbone's own behaviour, so the agent is never worse than its parent on
the decisions EFR does not touch and cannot crash a tournament world.
"""

from __future__ import annotations

import base64
import gzip
import random
from typing import Optional

from negmas import ResponseType, SAOResponse
from negmas.outcomes import Outcome
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents import SyncRandomOneShotAgent

from .infoset import (
    InfosetKey,
    K_ROUNDS,
    N_OTHER_VALUES,
    bucket_exog,
    bucket_price,
    bucket_qty,
    bucket_type,
)
from .policy_data import POLICY_B64


def _load_embedded_policy() -> dict[str, list[tuple[str, float]]]:
    """decode the embedded gzip+base64 policy table into a lookup dict.
    one row per infoset: 'label\\taction:prob,action:prob,...'."""
    text = gzip.decompress(base64.b64decode(POLICY_B64)).decode("utf-8")
    table: dict[str, list[tuple[str, float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        label, body = line.split("\t", 1)
        entries: list[tuple[str, float]] = []
        for tok in body.split(","):
            name, prob = tok.split(":")
            entries.append((name, float(prob)))
        table[label] = entries
    return table


class EFRDistOneShotAgent(SyncRandomOneShotAgent):
    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        # parent.init() scales mismatch_max by n_lines -- keep it
        super().init()
        try:
            self._policy = _load_embedded_policy()
        except Exception:
            self._policy = {}
        self._rng = random.Random()
        self._price_gate = True

    def before_step(self) -> None:
        super().before_step()
        # prime ufun bounds for the price gate
        try:
            self.ufun.find_limit(True)
            self.ufun.find_limit(False)
        except Exception:
            pass
        # per-day per-partner trackers
        self._dcount: dict[str, int] = {}
        self._hist: dict[str, list[tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    # EFR policy core
    # ------------------------------------------------------------------

    def _abstract_round(self, nid: str, is_seller: bool) -> int:
        """seller acts at even rounds, buyer at odd; saturate at the last
        in-range round for the role so late SAO rounds keep using the policy."""
        k = self._dcount.get(nid, 0)
        if is_seller:
            max_round = K_ROUNDS - 1 if (K_ROUNDS - 1) % 2 == 0 else K_ROUNDS - 2
            offset = 0
        else:
            max_round = K_ROUNDS - 1 if (K_ROUNDS - 1) % 2 == 1 else K_ROUNDS - 2
            offset = 1
        return min(offset + 2 * k, max_round)

    def _build_key(
        self, nid: str, is_seller: bool, last_offer: Optional[Outcome]
    ) -> Optional[InfosetKey]:
        nmi = self.get_nmi(nid)
        if nmi is None:
            return None
        rnd = self._abstract_round(nid, is_seller)
        if rnd >= K_ROUNDS:
            return None

        if is_seller:
            exog_q = int(self.awi.current_exogenous_input_quantity or 0)
        else:
            exog_q = int(self.awi.current_exogenous_output_quantity or 0)
        max_q = int(getattr(self.awi, "n_lines", 10) or 10)
        my_exog = bucket_exog(exog_q, 0, max_q)

        cost = float(getattr(self.awi.profile, "cost", 0) or 0)
        my_type = bucket_type(cost, 0.0, 5.0)

        last_bucket: Optional[tuple[int, int]] = None
        if last_offer is not None:
            p_issue = nmi.issues[UNIT_PRICE]
            last_bucket = (
                bucket_qty(int(last_offer[QUANTITY])),
                bucket_price(
                    float(last_offer[UNIT_PRICE]),
                    float(p_issue.min_value),
                    float(p_issue.max_value),
                ),
            )

        offset = 0 if is_seller else 1
        expected_hist_len = max(0, (rnd - offset) // 2)
        full_history = self._hist.get(nid, ())
        history = tuple(full_history[-expected_hist_len:]) if expected_hist_len else ()

        n_other_raw = max(0, len(self.negotiators) - 1)
        n_other = max(N_OTHER_VALUES[0], min(N_OTHER_VALUES[-1], n_other_raw))
        n_other_idx = N_OTHER_VALUES.index(n_other)

        try:
            return InfosetKey(
                role="S" if is_seller else "B",
                my_type=my_type,
                my_exog=my_exog,
                n_other_idx=n_other_idx,
                round=rnd,
                last_offer=last_bucket,
                my_history=history,
            )
        except ValueError:
            return None

    def _sample(self, key: InfosetKey) -> Optional[tuple[str, str]]:
        """returns (action_str, kind) or None on miss. kind in {accept,end,offer}."""
        entries = self._policy.get(key.serialize())
        if not entries:
            return None
        names = [e[0] for e in entries]
        weights = [e[1] for e in entries]
        choice = self._rng.choices(names, weights=weights, k=1)[0]
        if choice == "acc":
            return (choice, "accept")
        if choice == "end":
            return (choice, "end")
        return (choice, "offer")

    def _efr_decision(
        self, nid: str, is_seller: bool, last_offer: Optional[Outcome]
    ) -> tuple[Optional[int], Optional[str]]:
        """consult EFR for this bilateral. returns (price_or_None, kind_or_None).

        price is a concrete unit price mapped from the sampled offer action's
        price bucket onto the partner's real price issue; None when EFR sampled
        accept/end or missed. kind is the raw decision so the caller can use the
        accept signal as a price gate."""
        try:
            key = self._build_key(nid, is_seller, last_offer)
            if key is None:
                return (None, None)
            sampled = self._sample(key)
            if sampled is None:
                return (None, None)
            action_str, kind = sampled
            if kind != "offer":
                return (None, kind)
            nmi = self.get_nmi(nid)
            if nmi is None or not (len(action_str) == 3 and action_str[0] == "o"):
                return (None, kind)
            p_bucket = int(action_str[2])
            p_issue = nmi.issues[UNIT_PRICE]
            price = int(p_issue.min_value) if p_bucket == 0 else int(p_issue.max_value)
            return (price, kind)
        except Exception:
            return (None, None)

    def _record(self, nid: str, offer: Outcome) -> None:
        """record my emitted offer to history and advance the decision count
        so the next consult lands at the correct abstract round."""
        try:
            nmi = self.get_nmi(nid)
            if nmi is not None:
                p_issue = nmi.issues[UNIT_PRICE]
                self._hist.setdefault(nid, []).append(
                    (
                        bucket_qty(int(offer[QUANTITY])),
                        bucket_price(
                            float(offer[UNIT_PRICE]),
                            float(p_issue.min_value),
                            float(p_issue.max_value),
                        ),
                    )
                )
            self._dcount[nid] = self._dcount.get(nid, 0) + 1
        except Exception:
            pass

    def _price_ok(self, is_seller: bool, price: float) -> bool:
        try:
            if is_seller:
                return self.ufun.ok_to_sell_at(price)
            return self.ufun.ok_to_buy_at(price)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # proposals -- quantity from the backbone, price from EFR
    # ------------------------------------------------------------------

    def first_proposals(self) -> dict[str, Outcome | None]:
        base = super().first_proposals()
        out: dict[str, Outcome | None] = {}
        for nid, offer in base.items():
            if offer is None:
                out[nid] = None
                continue
            is_seller = nid in self.awi.my_consumers
            price, _ = self._efr_decision(nid, is_seller, last_offer=None)
            if price is None:
                out[nid] = offer
            else:
                new = (offer[QUANTITY], self.awi.current_step, price)
                out[nid] = new
                self._record(nid, new)
        return out

    def counter_all(self, offers, states) -> dict[str, SAOResponse]:
        # let the backbone decide allocation + which combination to accept
        base = super().counter_all(offers, states)
        out: dict[str, SAOResponse] = {}
        for nid, resp in base.items():
            is_seller = nid in self.awi.my_consumers
            opp_offer = offers.get(nid)

            if resp.response == ResponseType.ACCEPT_OFFER:
                # price gate: EFR-aware veto of a quantity-matched but badly
                # priced acceptance. only veto when EFR did not itself say
                # accept and the price fails the ufun threshold -- then counter
                # at the EFR price instead of locking in a bad deal.
                accepted = resp.outcome if resp.outcome is not None else opp_offer
                price, kind = self._efr_decision(nid, is_seller, opp_offer)
                if (
                    self._price_gate
                    and kind != "accept"
                    and accepted is not None
                    and not self._price_ok(is_seller, accepted[UNIT_PRICE])
                ):
                    ask = price if price is not None else self._counter_price(
                        nid, is_seller, accepted
                    )
                    counter = (accepted[QUANTITY], self.awi.current_step, ask)
                    out[nid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                    self._record(nid, counter)
                else:
                    out[nid] = resp
                continue

            if (
                resp.response == ResponseType.REJECT_OFFER
                and resp.outcome is not None
            ):
                # backbone gives quantity + a random price; overwrite the price
                # with EFR's ask. an EFR accept here means the price is good
                # enough -- defer to the backbone's reject (it knows quantity).
                q, _, base_p = resp.outcome
                price, _ = self._efr_decision(nid, is_seller, opp_offer)
                ask = price if price is not None else base_p
                counter = (q, self.awi.current_step, ask)
                out[nid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                self._record(nid, counter)
                continue

            out[nid] = resp
        return out

    def _counter_price(self, nid: str, is_seller: bool, ref: Outcome) -> int:
        """best-for-me price on the partner's issue, used when EFR is silent."""
        nmi = self.get_nmi(nid)
        if nmi is None:
            return int(ref[UNIT_PRICE])
        p_issue = nmi.issues[UNIT_PRICE]
        return int(p_issue.max_value) if is_seller else int(p_issue.min_value)
