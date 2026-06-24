#!/usr/bin/env python
"""
**Submitted to ANAC 2026 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here
"""
from __future__ import annotations

from typing import Any

from scml.std import StdAWI, StdSyncAgent
from negmas import Contract, Outcome, ResponseType, SAOResponse, SAOState

# SCML offers are POSITIONAL tuples: (quantity, time, unit_price)
QUANTITY = 0
TIME = 1
UNIT_PRICE = 2


class EmSel(StdSyncAgent):
    """
    Adaptive Std-track sync agent.

    Key strategy points:
        - Offered QUANTITY is tied to real needs (needed_supplies / needed_sales)
          so the agent stops over-committing -> far fewer negative-score worlds.
        - Price via EMA + linear-trend prediction with a softened concession curve.
        - Fully crash-proof: every negotiator handled in its own try/except.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_history: dict[str, list[float]] = {}
        self.critical_stock_limit = 2
        # softened from 0.88 -> closes more deals, fewer dead-end negotiations
        self.base_max_threshold = 0.80

        self.received_prices_history: dict[str, list[float]] = {}
        self.opponent_concession_rates: dict[str, list[float]] = {}
        self.opponent_personas: dict[str, str] = {}

    # ---------- small safe helpers ----------

    def _safe_nmi(self, negotiator_id):
        try:
            return self.get_nmi(negotiator_id)
        except Exception:
            return None

    def _scalar(self, val) -> float:
        try:
            if val is None:
                return 0.0
            if hasattr(val, "__len__"):
                return float(sum(val))
            return float(val)
        except Exception:
            return 0.0

    def _is_selling(self, nmi) -> bool:
        if nmi is None:
            return True
        try:
            ann = getattr(nmi, "annotation", {}) or {}
            if "seller" in ann:
                return ann["seller"] == self.id
            if "product" in ann:
                return ann["product"] == self.awi.my_output_product
        except Exception:
            pass
        return True

    def _issue_bounds(self, nmi, idx):
        try:
            issues = getattr(nmi, "issues", None)
            if not issues or len(issues) <= idx:
                return None
            issue = issues[idx]
            mn = getattr(issue, "min_value", None)
            mx = getattr(issue, "max_value", None)
            if mn is None or mx is None:
                vals = getattr(issue, "values", None)
                if vals:
                    mn, mx = min(vals), max(vals)
            if mn is None or mx is None:
                return None
            return float(mn), float(mx)
        except Exception:
            return None

    def _target_quantity(self, nmi, am_seller) -> int:
        """Quantity to ask for, based on what we still need, clamped to issue range."""
        qb = self._issue_bounds(nmi, QUANTITY)
        if not qb:
            return 1
        qmin, qmax = int(qb[0]), int(qb[1])
        try:
            need = self._scalar(self.awi.needed_sales if am_seller else self.awi.needed_supplies)
        except Exception:
            need = 0.0
        target = int(round(need)) if need > 0 else qmin
        return max(qmin if qmin > 0 else 1, min(target, qmax))

    def _best_outcome(self, nmi):
        """Aggressive-but-sane opening offer: best price, need-based quantity."""
        try:
            issues = getattr(nmi, "issues", None)
            if not issues:
                return None
            am_seller = self._is_selling(nmi)
            out = list(issues[0:0])  # placeholder; we build by index below
            out = [None] * len(issues)

            # quantity
            if len(issues) > QUANTITY:
                out[QUANTITY] = self._target_quantity(nmi, am_seller)
            # time -> deliver at current step by default
            if len(issues) > TIME:
                tb = self._issue_bounds(nmi, TIME)
                cur = float(getattr(self.awi, "current_step", 0))
                if tb:
                    cur = max(tb[0], min(cur, tb[1]))
                out[TIME] = int(cur)
            # price -> best for us
            if len(issues) > UNIT_PRICE:
                pb = self._issue_bounds(nmi, UNIT_PRICE)
                if pb:
                    out[UNIT_PRICE] = pb[1] if am_seller else pb[0]

            # fill any leftover Nones with the issue min, just in case
            for i, issue in enumerate(issues):
                if out[i] is None:
                    b = self._issue_bounds(nmi, i)
                    out[i] = b[0] if b else 0
            return tuple(out)
        except Exception:
            return None

    # ---------- time-driven ----------

    def init(self):
        pass

    def before_step(self):
        pass

    def step(self):
        pass

    # ---------- negotiation ----------

    def first_proposals(self) -> dict[str, Outcome | None]:
        proposals: dict[str, Outcome | None] = {}
        try:
            ids = list(self.negotiators.keys())
        except Exception:
            return {}
        for nid in ids:
            try:
                proposals[nid] = self._best_outcome(self._safe_nmi(nid))
            except Exception:
                proposals[nid] = None
        return proposals

    def _respond_one(self, nid, offer, state, input_stock, output_stock) -> SAOResponse:
        nmi = self._safe_nmi(nid)
        best = self._best_outcome(nmi)

        if nmi is None or not getattr(nmi, "issues", None) or offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        am_seller = self._is_selling(nmi)
        counter = best if best is not None else offer
        adaptive_modifier = 0.0
        quantity_risk_modifier = 0.0

        try:
            cur_price = float(offer[UNIT_PRICE]) if len(offer) > UNIT_PRICE else None
            cur_qty = float(offer[QUANTITY]) if len(offer) > QUANTITY else 1.0
            predicted = cur_price if cur_price is not None else 0.0
            target_qty = self._target_quantity(nmi, am_seller)

            if cur_price is not None:
                self.received_prices_history.setdefault(nid, []).append(cur_price)
                hist = self.received_prices_history[nid]
                if len(hist) >= 2:
                    ema = hist[0]
                    a = 0.6
                    for p in hist[1:]:
                        ema = a * p + (1 - a) * ema
                    linear = 0.7 * hist[-1] + 0.3 * (hist[-1] - hist[-2])
                    predicted = (linear + ema) / 2

                    last = abs(hist[-1] - hist[-2])
                    self.opponent_concession_rates.setdefault(nid, []).append(last)
                    rates = self.opponent_concession_rates[nid]
                    avg = sum(rates) / len(rates)
                    if avg < 0.2:
                        self.opponent_personas[nid] = "STUBBORN"
                        adaptive_modifier += 0.04 if state.time < 0.7 else -0.03
                    elif avg > 1.2:
                        self.opponent_personas[nid] = "CONCEDER"
                        adaptive_modifier += 0.03
                    else:
                        self.opponent_personas[nid] = "BALANCED"

            # discourage buying way more than we need
            if not am_seller and cur_qty > target_qty + 1:
                quantity_risk_modifier += 0.05

            # build counter
            if cur_price is not None:
                # softened multipliers (1.02 / 0.98) -> converge faster
                dyn = max(cur_price, predicted * 1.02) if am_seller else min(cur_price, predicted * 0.98)
                pb = self._issue_bounds(nmi, UNIT_PRICE)
                if pb:
                    dyn = max(pb[0], min(dyn, pb[1]))

                modified = list(offer)
                modified[UNIT_PRICE] = type(offer[UNIT_PRICE])(dyn)
                # cap quantity to what we actually need
                if len(offer) > QUANTITY and target_qty > 0:
                    modified[QUANTITY] = type(offer[QUANTITY])(min(int(cur_qty), target_qty))
                if len(offer) > TIME and state.time > 0.8:
                    new_t = min(float(offer[TIME]), float(self.awi.current_step + 1))
                    modified[TIME] = type(offer[TIME])(new_t)
                counter = tuple(modified)

        except Exception:
            counter = best if best is not None else offer
            adaptive_modifier = 0.0
            quantity_risk_modifier = 0.0

        try:
            u = self.ufun(offer)
            u = float(u) if u is not None else 0.0
        except Exception:
            u = 0.0

        tp = getattr(state, "time", 0.0)
        threshold = self.base_max_threshold - 0.22 * tp
        if not am_seller and input_stock <= self.critical_stock_limit:
            threshold -= 0.16 if output_stock <= 2 else 0.10
        elif am_seller and output_stock >= 10:
            threshold -= 0.14 if tp > 0.7 else 0.07
        threshold += adaptive_modifier + quantity_risk_modifier
        threshold = max(0.50, threshold)

        if (tp > 0.95 and u >= threshold - 0.04) or u >= threshold:
            return SAOResponse(ResponseType.ACCEPT_OFFER, None)
        if counter is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, counter)

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses: dict[str, SAOResponse] = {}
        try:
            inputs = self.awi.profile.inputs
            outputs = self.awi.profile.outputs
            input_stock = sum(self.awi.state.inventory.get(i, 0) for i in inputs) if inputs else 5
            output_stock = sum(self.awi.state.inventory.get(o, 0) for o in outputs) if outputs else 5
        except Exception:
            input_stock, output_stock = 5, 5

        for nid, offer in offers.items():
            try:
                responses[nid] = self._respond_one(
                    nid, offer, states[nid], input_stock, output_stock
                )
            except Exception:
                responses[nid] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return responses

    # ---------- contract signing ----------

    def sign_all_contracts(self, contracts):  # type: ignore
        """Sign up to our remaining needs, best-priced first.

        Low-risk design:
          - If a side's need is unknown / <= 0 we DO NOT restrict it (sign all on
            that side) so we never starve ourselves.
          - Only the marginal excess beyond a positive need is refused.
          - Any error -> fall back to signing everything (the framework default).
        """
        try:
            n = len(contracts)
        except Exception:
            return []
        try:
            my_id = self.id
            buy_cap = self._scalar(getattr(self.awi, "needed_supplies", 0))
            sell_cap = self._scalar(getattr(self.awi, "needed_sales", 0))
            if buy_cap <= 0:
                buy_cap = float("inf")
            if sell_cap <= 0:
                sell_cap = float("inf")

            parsed = []
            for i, c in enumerate(contracts):
                ann = getattr(c, "annotation", {}) or {}
                agr = getattr(c, "agreement", {}) or {}
                if isinstance(agr, dict):
                    qty = float(agr.get("quantity", 0) or 0)
                    price = float(agr.get("unit_price", agr.get("price", 0)) or 0)
                else:
                    qty = float(agr[QUANTITY]) if (agr and len(agr) > QUANTITY) else 0.0
                    price = float(agr[UNIT_PRICE]) if (agr and len(agr) > UNIT_PRICE) else 0.0
                is_buy = ann.get("buyer", None) == my_id
                is_sell = ann.get("seller", None) == my_id
                parsed.append((i, is_buy, is_sell, qty, price))

            signatures = [None] * n
            buys = sorted((p for p in parsed if p[1]), key=lambda x: x[4])       # cheap first
            sells = sorted((p for p in parsed if p[2]), key=lambda x: -x[4])     # pricey first

            used = 0.0
            for p in buys:
                if used + p[3] <= buy_cap:
                    signatures[p[0]] = my_id
                    used += p[3]
            used = 0.0
            for p in sells:
                if used + p[3] <= sell_cap:
                    signatures[p[0]] = my_id
                    used += p[3]

            # contracts we can't classify -> sign (safe default)
            for p in parsed:
                if not p[1] and not p[2] and signatures[p[0]] is None:
                    signatures[p[0]] = my_id

            return signatures
        except Exception:
            return [self.id] * n

    # ---------- feedback ----------

    def on_negotiation_failure(  # type: ignore
        self, partners, annotation, mechanism, state
    ) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        try:
            ann = getattr(contract, "annotation", {}) or {}
            partner = ann.get("partner_name", "unknown")
            agreement = getattr(contract, "agreement", {}) or {}
            price = agreement.get("unit_price", agreement.get("price", None))
            if price is not None:
                self.opponent_history.setdefault(partner, []).append(price)
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    try:
        from .helpers.runner import run
    except ImportError:
        from helpers.runner import run

    run([EmSel], sys.argv[1] if len(sys.argv) > 1 else "std")