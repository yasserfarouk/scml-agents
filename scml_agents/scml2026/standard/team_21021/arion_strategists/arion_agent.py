# ArionStrategists - SCML Standard track (CS451/551)
# Muhammad Raees Azam (S050683), Mehak Arshid (S050293)
# Extends SyncRandomStdAgent with bundle selection + partner memory.

from __future__ import annotations

import os
from itertools import combinations
from typing import Literal

from negmas import Outcome, ResponseType, SAOResponse, SAOState
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.std.agents.rand import SyncRandomStdAgent

StrategyName = Literal["baseline", "optimize", "search", "game", "hybrid"]

# submission default: game (top-5 tune — ensemble bundles + hybrid pipeline)
DEFAULT_STRATEGY: StrategyName = "game"


class ArionAgent(SyncRandomStdAgent):
    PRODUCTION_NORMAL = 0.50
    PRODUCTION_URGENT = 0.68
    PRODUCTION_BOOST = 0.52
    INV_BOOST_FRAC = 0.28
    INV_LOW_FRAC = 0.28
    MIDDLE_FLOOR_SOFT = 0.30
    SUPPLY_SECURE_FRAC = 0.80
    UTIL_FLOOR_NORMAL = 0.16
    UTIL_FLOOR_URGENT = 0.07
    UTIL_FLOOR_LATE = 0.05
    MAX_SUBSET = 10
    LATE_REL = 0.70
    BEAM_WIDTH = 8
    OVERORDER_FRAC = 0.13
    RELAX_PRICE_SLACK = 0.12
    SALVAGE_PRESSURE = 0.25

    def __init__(self, *args, strategy: StrategyName | None = None, **kwargs):
        kwargs.setdefault("today_target_productivity", 0.50)
        kwargs.setdefault("future_target_productivity", 0.36)
        kwargs.setdefault("today_concession_exp", 1.45)
        kwargs.setdefault("future_concession_exp", 4.0)
        kwargs.setdefault("pfuture", 0.08)
        kwargs.setdefault("today_concentration", 0.38)
        super().__init__(*args, **kwargs)

        picked = strategy
        if picked is None:
            env = os.environ.get("ARION_STRATEGY", "").strip().lower()
            if env in ("baseline", "optimize", "search", "game", "hybrid"):
                picked = env  # type: ignore[assignment]
            else:
                picked = DEFAULT_STRATEGY
        self._strategy: StrategyName = picked  # type: ignore[assignment]

        self._partner_best_buy: dict[str, float] = {}
        self._partner_best_sell: dict[str, float] = {}
        self._partner_offers_seen: dict[str, int] = {}
        self._step_supplies = 0
        self._step_sales = 0
        self._urgent = False
        self._prod_anchor = self.PRODUCTION_NORMAL

    def init(self):
        self._partner_best_buy.clear()
        self._partner_best_sell.clear()
        self._partner_offers_seen.clear()

    def before_step(self):
        ufun = getattr(self, "ufun", None)
        if ufun is not None:
            try:
                ufun.find_limit(True)
                ufun.find_limit(False)
            except MemoryError:
                pass
        self._step_supplies = 0
        self._step_sales = 0
        awi = self.awi
        self._urgent = int(awi.needed_supplies) > 0 or int(awi.needed_sales) > 0
        self._prod_anchor = self._pick_production_anchor()
        self.today_productivity = self._prod_anchor

    def counter_all(self, offers: dict[str, Outcome], states: dict[str, SAOState]):
        """Sync base + ensemble bundles + future accepts + aggressive salvage."""
        min_buy, max_buy, min_sell, max_sell = self._price_bounds()
        c = int(self.awi.current_step)
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)

        needed_supplies, needed_sales = self.estimate_future_needs()
        supply_need = max(
            int(self.awi.needed_supplies),
            self._today_supply_need(),
            int(needed_supplies.get(c, 0)),
        )
        sales_need = max(
            int(self.awi.needed_sales),
            self._today_sales_need(),
            int(needed_sales.get(c, 0)),
        )
        if self.awi.is_middle_level:
            if self._chase_full_production():
                floor_lines = int(self.awi.n_lines * self._prod_anchor)
            else:
                floor_lines = int(self.awi.n_lines * self.MIDDLE_FLOOR_SOFT)
            supply_need = max(supply_need, floor_lines)
            sales_need = max(sales_need, floor_lines)
        needed_supplies[c] = supply_need
        needed_sales[c] = sales_need

        responses = self._counter_sync_style(offers, states)

        buy_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_supplier(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        sell_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_consumer(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        self._note_side_offers(buy_today, selling=False)
        self._note_side_offers(sell_today, selling=True)

        try:
            buy_bundle = self._select_today_bundle(
                buy_today, supply_need, selling=False
            )
            sell_bundle = self._select_today_bundle(
                sell_today, sales_need, selling=True
            )
        except MemoryError:
            buy_bundle, sell_bundle = {}, {}

        n = max(int(self.awi.n_steps) - c, 1)
        for p, off in buy_bundle.items():
            if p in responses:
                continue
            q, t, price = int(off[QUANTITY]), int(off[TIME]), float(off[UNIT_PRICE])
            if q <= 0:
                continue
            today = t == c
            r = float(states[p].relative_time) if today else (t - c) / n
            if self._should_accept_bundle_offer(
                price, r, min_buy, max_buy, today, buying=True
            ):
                responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, off)

        for p, off in sell_bundle.items():
            if p in responses:
                continue
            q, t, price = int(off[QUANTITY]), int(off[TIME]), float(off[UNIT_PRICE])
            if q <= 0:
                continue
            today = t == c
            r = float(states[p].relative_time) if today else (t - c) / n
            if self._should_accept_bundle_offer(
                price, r, min_sell, max_sell, today, buying=False
            ):
                responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, off)

        self._accept_good_priced_offers(
            responses,
            offers,
            states,
            needed_supplies,
            needed_sales,
            min_buy=min_buy,
            max_buy=max_buy,
            min_sell=min_sell,
            max_sell=max_sell,
        )

        got_in = sum(int(buy_today[p][QUANTITY]) for p in responses if p in buy_today)
        got_out = sum(int(sell_today[p][QUANTITY]) for p in responses if p in sell_today)
        if (
            got_in < supply_need
            or got_out < sales_need
            or self._should_salvage(rel)
            or self._supply_pressure() > self.SALVAGE_PRESSURE
            or self._sales_pressure() > self.SALVAGE_PRESSURE
        ):
            self._salvage_today(
                responses, buy_today, sell_today, supply_need, sales_need
            )
        return responses

    def first_proposals(self):
        """Open with Nash-anchored prices instead of extreme best_price."""
        distribution = self.distribute_todays_needs()
        future_suppliers = {k for k, v in distribution.items() if v <= 0}
        unneeded = None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        c = int(self.awi.current_step)
        min_buy, max_buy, min_sell, max_sell = self._price_bounds()
        offers: dict[str, Outcome | None] = {}
        for k, q in distribution.items():
            if q > 0:
                if self.is_supplier(k):
                    price = self._anchor_price(k, buying=True, mn=min_buy, mx=max_buy)
                else:
                    price = self._anchor_price(k, buying=False, mn=min_sell, mx=max_sell)
                offers[k] = (int(q), c, price)
            else:
                offers[k] = unneeded
        return offers | self.distribute_future_offers(list(future_suppliers))

    def _supplies_secured_today(self) -> bool:
        awi = self.awi
        base_target = int(awi.n_lines * self.PRODUCTION_NORMAL)
        have = int(awi.current_inventory_input) + int(awi.total_supplies_at(awi.current_step))
        return have >= int(self.SUPPLY_SECURE_FRAC * base_target)

    def _pick_production_anchor(self) -> float:
        if self._urgent:
            return self.PRODUCTION_URGENT
        awi = self.awi
        lines = max(1, int(awi.n_lines))
        inv = int(awi.current_inventory_input)
        if (
            self._supplies_secured_today()
            and inv >= int(self.INV_BOOST_FRAC * lines)
            and self._pressure_at_prod(self.PRODUCTION_NORMAL, buying=True) < 0.20
        ):
            return self.PRODUCTION_BOOST
        return self.PRODUCTION_NORMAL

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        agr = contract.agreement
        t = agr.get("time", agr.get(TIME))
        price = agr.get("unit_price", agr.get(UNIT_PRICE))
        if t is None or price is None:
            return
        if int(t) != int(self.awi.current_step):
            return
        price = float(price)
        qty = int(agr.get("quantity", agr.get(QUANTITY, 0)))

        if contract.annotation["product"] == self.awi.my_input_product:
            seller = contract.annotation["seller"]
            old = self._partner_best_buy.get(seller, price)
            self._partner_best_buy[seller] = min(old, price)
            self._step_supplies += qty
        else:
            buyer = contract.annotation["buyer"]
            old = self._partner_best_sell.get(buyer, price)
            self._partner_best_sell[buyer] = max(old, price)
            self._step_sales += qty

    def _chase_full_production(self) -> bool:
        if self._urgent:
            return True
        awi = self.awi
        lines = max(1, int(awi.n_lines))
        inv = int(awi.current_inventory_input)
        if inv < int(self.INV_LOW_FRAC * lines):
            return True
        rel = float(getattr(awi, "relative_time", 0.0) or 0.0)
        if rel > 0.45 and not self._supplies_secured_today():
            return True
        return False

    def _today_supply_need(self) -> int:
        awi = self.awi
        explicit = max(0, int(awi.needed_supplies))
        if not self._chase_full_production():
            return explicit
        target = int(awi.n_lines * self._prod_anchor)
        have = int(awi.current_inventory_input) + int(awi.total_supplies_at(awi.current_step))
        return max(0, target - have, explicit)

    def _today_sales_need(self) -> int:
        awi = self.awi
        explicit = max(0, int(awi.needed_sales))
        if not self._chase_full_production():
            return explicit
        if int(awi.total_sales_at(awi.current_step)) > int(awi.n_lines):
            return explicit
        target = int(awi.n_lines * self._prod_anchor)
        cap = min(int(awi.n_lines), target + int(awi.current_inventory_input))
        sold = int(awi.total_sales_at(awi.current_step))
        return max(0, cap - sold, explicit)

    def _supply_pressure(self) -> float:
        return self._pressure_at_prod(self._prod_anchor, buying=True)

    def _sales_pressure(self) -> float:
        return self._pressure_at_prod(self._prod_anchor, buying=False)

    def _pressure_at_prod(self, prod: float, *, buying: bool) -> float:
        awi = self.awi
        lines = max(1, int(awi.n_lines))
        if buying:
            target = int(awi.n_lines * prod)
            have = int(awi.current_inventory_input) + int(awi.total_supplies_at(awi.current_step))
            need = max(0, target - have, int(awi.needed_supplies))
        else:
            if int(awi.total_sales_at(awi.current_step)) > int(awi.n_lines):
                return 0.0
            target = int(awi.n_lines * prod)
            cap = min(int(awi.n_lines), target + int(awi.current_inventory_input))
            sold = int(awi.total_sales_at(awi.current_step))
            need = max(0, cap - sold, int(awi.needed_sales))
        return min(1.0, need / lines)

    def _util_floor(self) -> float:
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            return 0.0
        max_u = getattr(ufun, "max_utility", None)
        if max_u is None or max_u <= 0:
            return 0.0
        frac = self.UTIL_FLOOR_URGENT if self._urgent else self.UTIL_FLOOR_NORMAL
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)
        pressure = max(self._supply_pressure(), self._sales_pressure())
        if pressure > 0.50:
            frac = min(frac, self.UTIL_FLOOR_URGENT)
        if rel > self.LATE_REL:
            frac = min(frac, self.UTIL_FLOOR_LATE)
        if self._urgent and rel > 0.50:
            frac = min(frac, self.UTIL_FLOOR_LATE)
        return frac * float(ufun.max_utility)

    def _should_accept_bundle_offer(
        self, price: float, r: float, mn: int, mx: int, today: bool, *, buying: bool
    ) -> bool:
        if self._price_acceptable(price, r, mn, mx, today, buying=buying):
            return True
        pressure = self._supply_pressure() if buying else self._sales_pressure()
        if not (self._urgent or pressure > 0.40):
            return False
        slack = 1.0 + self.RELAX_PRICE_SLACK * min(1.0, pressure * 1.15)
        if buying:
            return price <= self.buy_price(r, mn, mx, today) * slack
        sell_slack = 1.0 - self.RELAX_PRICE_SLACK * min(1.0, pressure * 1.15)
        return price >= self.sell_price(r, mn, mx, today) * sell_slack

    def _price_at(self, prices, product: int) -> int | None:
        """Read price from dict, list, or numpy array (ANAC server uses arrays)."""
        if prices is None:
            return None
        try:
            if hasattr(prices, "get"):
                val = prices.get(product)
            else:
                val = prices[int(product)]
            if val is None:
                return None
            return int(val)
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _catalog_price(self, product: int, *, buying: bool) -> int | None:
        return self._price_at(getattr(self.awi, "catalog_prices", None), product)

    def _price_acceptable(
        self, price: float, r: float, mn: int, mx: int, today: bool, *, buying: bool
    ) -> bool:
        if buying:
            if self.good2buy(price, r, mn, mx, today):
                return True
            if self._urgent or self._supply_pressure() > 0.5:
                slack = 1.0 + self.RELAX_PRICE_SLACK * self._supply_pressure()
                return price <= self.buy_price(r, mn, mx, today) * slack
        else:
            if self.good2sell(price, r, mn, mx, today):
                return True
            if self._urgent or self._sales_pressure() > 0.5:
                slack = 1.0 - self.RELAX_PRICE_SLACK * self._sales_pressure()
                return price >= self.sell_price(r, mn, mx, today) * slack
        return False

    def _note_side_offers(self, side_offers: dict[str, Outcome], *, selling: bool) -> None:
        for partner, offer in side_offers.items():
            price = float(offer[UNIT_PRICE])
            if selling:
                self._partner_best_sell[partner] = max(
                    self._partner_best_sell.get(partner, price), price
                )
            else:
                self._partner_best_buy[partner] = min(
                    self._partner_best_buy.get(partner, price), price
                )

    def _bundle_util(self, chosen: dict[str, Outcome], *, selling: bool) -> float:
        ufun = getattr(self, "ufun", None)
        if ufun is None or not chosen:
            return -1.0
        try:
            outs = tuple(chosen.values())
            flags = tuple(selling for _ in chosen)
            return float(ufun.from_offers(outs, flags))
        except MemoryError:
            return -1.0
        except Exception:
            return -1.0

    def _qty_cap(self, need: int) -> int:
        extra = int(self.awi.n_lines * self.OVERORDER_FRAC)
        return need + extra

    def _rank_offers(self, side_offers: dict[str, Outcome], *, selling: bool):
        cat = self._catalog_price(
            self.awi.my_output_product if selling else self.awi.my_input_product,
            buying=not selling,
        )

        def key_fn(kv: tuple[str, Outcome]):
            price = int(kv[1][UNIT_PRICE])
            if selling:
                cat_bias = 0 if cat is None else (1 if price >= cat else 0)
                return (-price, -cat_bias)
            cat_bias = 0 if cat is None else (1 if price <= cat else 0)
            return (price, -cat_bias)

        return sorted(side_offers.items(), key=key_fn)

    def _subset_qty(self, chosen: dict[str, Outcome]) -> int:
        return sum(int(o[QUANTITY]) for o in chosen.values())

    def _subset_score(self, chosen, need, util, floor, *, selling: bool):
        if util < floor:
            return (-1.0, -1, -1, -1.0)
        qty = self._subset_qty(chosen)
        if qty > self._qty_cap(need):
            return (-1.0, -1, -1, -1.0)
        gap = abs(qty - need)
        pressure = self._supply_pressure() if not selling else self._sales_pressure()
        if pressure > 0.40:
            gap = int(gap * 0.5)
        covers = 1 if qty >= need else 0
        return (util, covers, -gap, -qty)

    def _best_subset_baseline(self, side_offers, need, *, selling: bool):
        if need <= 0 or not side_offers:
            return {}
        floor = self._util_floor()
        ranked = self._rank_offers(side_offers, selling=selling)
        top = ranked[: self.MAX_SUBSET]
        partners = [p for p, _ in top]
        offers = {p: o for p, o in top}

        if partners:
            best = {}
            best_u = -1.0
            for r in range(1, len(partners) + 1):
                for combo in combinations(partners, r):
                    qty = sum(int(offers[p][QUANTITY]) for p in combo)
                    if qty > self._qty_cap(need):
                        continue
                    trial = {p: offers[p] for p in combo}
                    u = self._bundle_util(trial, selling=selling)
                    if u >= floor and u > best_u:
                        best_u = u
                        best = trial
            if best:
                return best

        chosen = {}
        got = 0
        for partner, offer in ranked:
            trial = dict(chosen)
            trial[partner] = offer
            if self._bundle_util(trial, selling=selling) >= floor:
                chosen = trial
                got += int(offer[QUANTITY])
                if got >= need:
                    break
        pressure = self._supply_pressure() if not selling else self._sales_pressure()
        if (self._urgent or pressure > 0.5) and self._subset_qty(chosen) < need:
            return self._greedy_cover_need(
                side_offers, need, selling=selling, existing=chosen, require_floor=False
            )
        return chosen

    def _greedy_cover_need(
        self,
        side_offers: dict[str, Outcome],
        need: int,
        *,
        selling: bool,
        existing: dict[str, Outcome],
        require_floor: bool,
    ) -> dict[str, Outcome]:
        ranked = self._rank_offers(side_offers, selling=selling)
        chosen = dict(existing)
        got = self._subset_qty(chosen)
        floor = self._util_floor() if require_floor else -1.0
        for partner, offer in ranked:
            if partner in chosen:
                continue
            q = int(offer[QUANTITY])
            if got + q > self._qty_cap(need):
                continue
            trial = dict(chosen)
            trial[partner] = offer
            if require_floor and self._bundle_util(trial, selling=selling) < floor:
                continue
            chosen = trial
            got += q
            if got >= need:
                break
        return chosen

    def _best_subset_optimize(self, side_offers, need, *, selling: bool):
        if need <= 0 or not side_offers:
            return {}
        floor = self._util_floor()
        ranked = self._rank_offers(side_offers, selling=selling)
        top = ranked[: self.MAX_SUBSET]
        partners = [p for p, _ in top]
        offers = {p: o for p, o in top}

        best = {}
        best_score = (-1.0, -1, -1, -1.0)
        for r in range(1, len(partners) + 1):
            for combo in combinations(partners, r):
                trial = {p: offers[p] for p in combo}
                u = self._bundle_util(trial, selling=selling)
                sc = self._subset_score(trial, need, u, floor, selling=selling)
                if sc > best_score:
                    best_score = sc
                    best = trial
        if best:
            return best
        return self._best_subset_baseline(side_offers, need, selling=selling)

    def _best_subset_search(self, side_offers, need, *, selling: bool):
        if need <= 0 or not side_offers:
            return {}
        floor = self._util_floor()
        ranked = self._rank_offers(side_offers, selling=selling)[: self.MAX_SUBSET]

        beam = [{}]
        for partner, offer in ranked:
            nxt = []
            for state in beam:
                nxt.append(state)
                trial = dict(state)
                trial[partner] = offer
                if self._subset_qty(trial) <= self._qty_cap(need):
                    nxt.append(trial)
            scored = []
            for state in nxt:
                if not state:
                    scored.append(((0.0, 0, 0, 0.0), state))
                    continue
                u = self._bundle_util(state, selling=selling)
                scored.append((self._subset_score(state, need, u, floor, selling=selling), state))
            scored.sort(key=lambda x: x[0], reverse=True)
            beam = [s for _, s in scored[: self.BEAM_WIDTH]]

        best = {}
        best_score = (-1.0, -1, -1, -1.0)
        for state in beam:
            if not state:
                continue
            u = self._bundle_util(state, selling=selling)
            sc = self._subset_score(state, need, u, floor, selling=selling)
            if sc > best_score:
                best_score = sc
                best = state
        if best:
            return best
        return self._best_subset_baseline(side_offers, need, selling=selling)

    def _pick_best_bundle(
        self,
        *candidates: dict[str, Outcome],
        need: int,
        selling: bool,
    ) -> dict[str, Outcome]:
        floor = self._util_floor()
        best: dict[str, Outcome] = {}
        best_key = (-1.0, -1, -999, -1.0)
        for cand in candidates:
            if not cand:
                continue
            u = self._bundle_util(cand, selling=selling)
            key = self._subset_score(cand, need, u, floor, selling=selling)
            if key > best_key:
                best_key = key
                best = cand
        return best

    def _select_today_bundle(self, side_offers, need, *, selling: bool):
        mode = self._strategy
        if mode == "optimize":
            return self._best_subset_optimize(side_offers, need, selling=selling)
        if mode == "search":
            return self._best_subset_search(side_offers, need, selling=selling)
        if mode == "hybrid":
            return self._best_subset_search(side_offers, need, selling=selling)
        if mode == "game":
            opt = self._best_subset_optimize(side_offers, need, selling=selling)
            srch = self._best_subset_search(side_offers, need, selling=selling)
            return self._pick_best_bundle(opt, srch, need=need, selling=selling)
        return self._best_subset_baseline(side_offers, need, selling=selling)

    def _nash_reservation(self, mn: int, mx: int, *, buying: bool) -> int:
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)
        t = min(1.0, max(0.0, rel)) ** 1.4
        mid = (mn + mx) / 2.0
        if buying:
            target = mn + (mid - mn) * (0.25 + 0.65 * t)
            return int(min(mx, max(mn, target)))
        target = mx - (mx - mid) * (0.25 + 0.65 * t)
        return int(max(mn, min(mx, target)))

    def _anchor_price_baseline(self, partner: str, *, buying: bool, mn: int, mx: int) -> int:
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)
        t = rel ** 1.6
        if buying:
            anchor = self._partner_best_buy.get(partner, float(mx))
            cap = int(mn + (mx - mn) * (0.35 + 0.55 * t))
            return int(min(cap, max(mn, anchor + 1)))
        anchor = self._partner_best_sell.get(partner, float(mn))
        floor = int(mx - (mx - mn) * (0.35 + 0.55 * t))
        return int(max(floor, min(mx, anchor - 1)))

    def _anchor_price_game(self, partner: str, *, buying: bool, mn: int, mx: int) -> int:
        nash = self._nash_reservation(mn, mx, buying=buying)
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)
        if self._urgent or rel > self.LATE_REL:
            push = 0.30 if self._urgent else 0.18
            if buying:
                nash = int(min(mx, nash + (mx - nash) * push))
            else:
                nash = int(max(mn, nash - (nash - mn) * push))
        if buying:
            cat = self._catalog_price(self.awi.my_input_product, buying=True)
            if cat is not None:
                nash = min(nash, cat + 1)
            hist = self._partner_best_buy.get(partner)
            if hist is not None:
                return int(min(mx, max(mn, min(nash, int(hist) + 1))))
            return int(min(mx, max(mn, nash)))
        cat = self._catalog_price(self.awi.my_output_product, buying=False)
        if cat is not None:
            nash = max(nash, cat - 1)
        hist = self._partner_best_sell.get(partner)
        if hist is not None:
            return int(max(mn, min(mx, max(nash, int(hist) - 1))))
        return int(max(mn, min(mx, nash)))

    def _anchor_price(self, partner: str, *, buying: bool, mn: int, mx: int) -> int:
        if self._strategy in ("game", "hybrid"):
            return self._anchor_price_game(partner, buying=buying, mn=mn, mx=mx)
        return self._anchor_price_baseline(partner, buying=buying, mn=mn, mx=mx)

    def _should_salvage(self, rel: float) -> bool:
        if self._urgent:
            return True
        if rel > self.LATE_REL:
            return True
        if rel > 0.50 and (
            self._supply_pressure() > 0.30 or self._sales_pressure() > 0.30
        ):
            return True
        return False

    def _salvage_today(self, responses, buy_today, sell_today, supply_need, sales_need):
        got_in = sum(int(buy_today[p][QUANTITY]) for p in responses if p in buy_today)
        got_out = sum(int(sell_today[p][QUANTITY]) for p in responses if p in sell_today)
        rem_in = max(0, supply_need - got_in)
        rem_out = max(0, sales_need - got_out)

        for p in sorted(buy_today, key=lambda x: int(buy_today[x][UNIT_PRICE])):
            if rem_in <= 0 or p in responses:
                continue
            q = int(buy_today[p][QUANTITY])
            if q > 0:
                responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, buy_today[p])
                rem_in -= q

        for p in sorted(sell_today, key=lambda x: int(sell_today[x][UNIT_PRICE]), reverse=True):
            if rem_out <= 0 or p in responses:
                continue
            q = int(sell_today[p][QUANTITY])
            if q > 0:
                responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, sell_today[p])
                rem_out -= q

    def _accept_good_priced_offers(
        self,
        responses: dict[str, SAOResponse],
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
        needed_supplies: dict[int, int],
        needed_sales: dict[int, int],
        *,
        min_buy: int,
        max_buy: int,
        min_sell: int,
        max_sell: int,
    ) -> None:
        """SyncRandom-style pass: accept any good-priced partial offer we still need."""
        c = int(self.awi.current_step)
        n = max(int(self.awi.n_steps) - c, 1)
        for is_partner, needs, good_fn, mn, mx in (
            (self.is_supplier, needed_supplies, self.good2buy, min_buy, max_buy),
            (self.is_consumer, needed_sales, self.good2sell, min_sell, max_sell),
        ):
            if mn > mx:
                continue
            for partner, offer in offers.items():
                if partner in responses or not is_partner(partner) or offer is None:
                    continue
                q, t = int(offer[QUANTITY]), int(offer[TIME])
                if q <= 0:
                    continue
                today = t == c
                r = float(states[partner].relative_time) if today else (t - c) / n
                if not good_fn(float(offer[UNIT_PRICE]), r, mn, mx, today):
                    continue
                need_t = int(needs.get(t, 0))
                if 0 < q <= need_t:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    needs[t] = max(0, need_t - q)

    def _price_bounds(self):
        max_sell = self.awi.current_output_issues[UNIT_PRICE].max_value
        min_sell = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )
        min_buy = self.awi.current_input_issues[UNIT_PRICE].min_value
        max_buy = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        return min_buy, max_buy, min_sell, max_sell

    def _counter_sync_style(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """Match SyncRandomStdAgent acceptance; use Nash anchors on counter-offers."""
        min_buy, max_buy, min_sell, max_sell = self._price_bounds()
        needed_supplies, needed_sales = self.estimate_future_needs()
        c = int(self.awi.current_step)
        needed_sales[c] = int(self.awi.needed_sales)
        needed_supplies[c] = int(self.awi.needed_supplies)
        if self.awi.is_middle_level:
            floor = int(self.awi.n_lines * self.today_productivity)
            needed_sales[c] = max(int(needed_sales[c]), floor)
            needed_supplies[c] = max(int(needed_supplies[c]), floor)

        responses: dict[str, SAOResponse] = {}
        n = max(int(self.awi.n_steps) - c, 1)
        for is_partner, needs, good_fn, mn, mx in (
            (self.is_supplier, needed_supplies, self.good2buy, min_buy, max_buy),
            (self.is_consumer, needed_sales, self.good2sell, min_sell, max_sell),
        ):
            if mn > mx:
                continue
            for partner, offer in offers.items():
                if not is_partner(partner) or offer is None:
                    continue
                q, t = int(offer[QUANTITY]), int(offer[TIME])
                if q <= 0:
                    continue
                today = t == c
                r = float(states[partner].relative_time) if today else (t - c) / n
                if not good_fn(float(offer[UNIT_PRICE]), r, mn, mx, today):
                    continue
                need_t = int(needs.get(t, 0))
                if 0 < q < need_t:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    needs[t] = max(0, need_t - q)

        remaining = {k for k in offers if k not in responses}
        distribution = self.distribute_todays_needs(partners=remaining)
        future_partners = {k for k, v in distribution.items() if v <= 0}
        unneeded = None if not self.awi.allow_zero_quantity else (0, c, 0)
        myoffers: dict[str, Outcome | None] = {}
        for partner, q in distribution.items():
            if q > 0:
                if self.is_supplier(partner):
                    mn_p, mx_p = min_buy, max_buy
                else:
                    mn_p, mx_p = min_sell, max_sell
                price = self._anchor_price(
                    partner,
                    buying=self.is_supplier(partner),
                    mn=int(mn_p),
                    mx=int(mx_p),
                )
                myoffers[partner] = (int(q), c, price)
            else:
                myoffers[partner] = unneeded
        myoffers |= self.distribute_future_offers(list(future_partners))
        for k, offer in myoffers.items():
            if k not in responses:
                responses[k] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        return responses

    def _upgrade_urgent_today(
        self,
        responses: dict[str, SAOResponse],
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
        rel: float,
    ) -> None:
        """On urgent/late steps, salvage any remaining today supply/sales gaps."""
        c = int(self.awi.current_step)
        supply_need = int(self.awi.needed_supplies)
        sales_need = int(self.awi.needed_sales)
        if self.awi.is_middle_level:
            floor_lines = int(self.awi.n_lines * self._prod_anchor)
            supply_need = max(supply_need, floor_lines)
            sales_need = max(sales_need, floor_lines)

        buy_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_supplier(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        sell_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_consumer(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        self._note_side_offers(buy_today, selling=False)
        self._note_side_offers(sell_today, selling=True)

        got_in = sum(int(buy_today[p][QUANTITY]) for p in responses if p in buy_today)
        got_out = sum(int(sell_today[p][QUANTITY]) for p in responses if p in sell_today)
        if self._should_salvage(rel) or got_in < supply_need or got_out < sales_need:
            self._salvage_today(responses, buy_today, sell_today, supply_need, sales_need)

    def _counter_smart(self, offers: dict[str, Outcome], states: dict[str, SAOState]):
        min_buy, max_buy, min_sell, max_sell = self._price_bounds()
        needed_supplies, needed_sales = self.estimate_future_needs()
        c = int(self.awi.current_step)
        supply_need = max(
            int(self.awi.needed_supplies),
            self._today_supply_need(),
            int(needed_supplies.get(c, 0)),
        )
        sales_need = max(
            int(self.awi.needed_sales),
            self._today_sales_need(),
            int(needed_sales.get(c, 0)),
        )
        if self.awi.is_middle_level:
            if self._chase_full_production():
                floor_lines = int(self.awi.n_lines * self._prod_anchor)
            else:
                floor_lines = int(self.awi.n_lines * self.MIDDLE_FLOOR_SOFT)
            supply_need = max(supply_need, floor_lines)
            sales_need = max(sales_need, floor_lines)
        needed_supplies[c] = supply_need
        needed_sales[c] = sales_need

        responses: dict[str, SAOResponse] = {}
        rel = float(getattr(self.awi, "relative_time", 0.0) or 0.0)

        buy_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_supplier(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        sell_today = {
            p: offers[p]
            for p in offers
            if offers[p] is not None
            and self.is_consumer(p)
            and int(offers[p][TIME]) == c
            and int(offers[p][QUANTITY]) > 0
        }
        self._note_side_offers(buy_today, selling=False)
        self._note_side_offers(sell_today, selling=True)

        for p, off in self._select_today_bundle(buy_today, supply_need, selling=False).items():
            responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, off)
        for p, off in self._select_today_bundle(sell_today, sales_need, selling=True).items():
            responses[p] = SAOResponse(ResponseType.ACCEPT_OFFER, off)

        if self._should_salvage(rel):
            self._salvage_today(responses, buy_today, sell_today, supply_need, sales_need)
        else:
            got_in = sum(int(buy_today[p][QUANTITY]) for p in responses if p in buy_today)
            got_out = sum(int(sell_today[p][QUANTITY]) for p in responses if p in sell_today)
            if got_in < supply_need or got_out < sales_need:
                self._salvage_today(
                    responses, buy_today, sell_today, supply_need, sales_need
                )

        n = max(int(self.awi.n_steps) - c, 1)
        for is_partner, needs, is_good_price, mn, mx, buy_side in (
            (self.is_supplier, needed_supplies, self.good2buy, min_buy, max_buy, True),
            (self.is_consumer, needed_sales, self.good2sell, min_sell, max_sell, False),
        ):
            if mn > mx:
                continue
            partners = [
                p for p in offers if p not in responses and is_partner(p) and offers[p] is not None
            ]
            partners.sort(key=lambda p: int(offers[p][UNIT_PRICE]), reverse=not buy_side)
            for partner in partners:
                offer = offers[partner]
                q, t = int(offer[QUANTITY]), int(offer[TIME])
                if q <= 0:
                    continue
                today = t == c
                state = states[partner]
                r = float(state.relative_time) if today else (t - c) / n
                price = float(offer[UNIT_PRICE])
                if not self._price_acceptable(price, r, mn, mx, today, buying=buy_side):
                    continue
                need_t = int(needs.get(t, 0))
                if today:
                    if 0 < q <= need_t or (self._urgent and 0 < q < need_t):
                        responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                        needs[t] = max(0, need_t - q)
                elif 0 < q <= need_t:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    needs[t] = max(0, need_t - q)

        self._accept_good_priced_offers(
            responses,
            offers,
            states,
            needed_supplies,
            needed_sales,
            min_buy=min_buy,
            max_buy=max_buy,
            min_sell=min_sell,
            max_sell=max_sell,
        )

        remaining = {k for k in offers if k not in responses}
        distribution = self.distribute_todays_needs(partners=remaining)
        future_partners = {k for k, v in distribution.items() if v <= 0}
        unneeded = None if not self.awi.allow_zero_quantity else (0, c, 0)
        myoffers: dict[str, Outcome | None] = {}
        for partner, q in distribution.items():
            if q > 0:
                if self.is_supplier(partner):
                    mn_p, mx_p = min_buy, max_buy
                else:
                    mn_p, mx_p = min_sell, max_sell
                price = self._anchor_price(
                    partner,
                    buying=self.is_supplier(partner),
                    mn=int(mn_p),
                    mx=int(mx_p),
                )
                myoffers[partner] = (int(q), c, price)
            else:
                myoffers[partner] = unneeded
        myoffers |= self.distribute_future_offers(list(future_partners))
        for k, offer in myoffers.items():
            if k not in responses:
                responses[k] = SAOResponse(ResponseType.REJECT_OFFER, offer)
        return responses


# extra classes for local comparison only
class ArionAgentBaseline(ArionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, strategy="baseline", **kwargs)


class ArionAgentOptimize(ArionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, strategy="optimize", **kwargs)


class ArionAgentSearch(ArionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, strategy="search", **kwargs)


class ArionAgentGame(ArionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, strategy="game", **kwargs)


class ArionAgentHybrid(ArionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, strategy="hybrid", **kwargs)


STRATEGY_VARIANTS: dict[str, type[ArionAgent]] = {
    "baseline": ArionAgentBaseline,
    "optimize": ArionAgentOptimize,
    "search": ArionAgentSearch,
    "game": ArionAgentGame,
    "hybrid": ArionAgentHybrid,
}


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([ArionAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
