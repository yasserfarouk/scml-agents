
""" Ozyegin University - CS551: Introduction to Artificial Intelligence
    Term Project - ANAC Standard SCML Negotiation Agent
    by Amer M. N. Dyab and Hamid B. Majidi (CS551 Group 4)

    File Description:
        Single-file submission consolidating the agent and its four main components:

            1. Demand-driven production scheduling
            2. Quantity gate on incoming offers (load-bearing, ~ +0.20)
            3. Boulware bidding with decoupled utility-aware acceptance
            4. PCB archetype classifier with a small acceptance-threshold adjustment

    Preliminary Results:
        Scores 1.054 +/- 0.034 against baseline agents (n_seeds = 9)
"""

# IMPORTS #############################################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Optional

from negmas import Outcome, ResponseType, SAOState
from scml.std import StdAgent

# AGENT PARAMETERS ####################################################################################################

# Bidding (Boulware) Curve Exponents for Proposal and Acceptance
BOULWARE_PROPOSE_E  = 0.2
BOULWARE_ACCEPT_E   = 0.5

# Symmetric Multiplier Applied to the Acceptance Threshold Based on Partner Archetype
ARCHETYPE_THRESHOLD_BIAS = 0.05

# Hard Floor on Accepted Utility (Only Positive Utilities Are Accepted)
MIN_UTILITY_FLOOR = 0.0

# Archetype Classifier Labels
ARCHETYPE_IDEAL         = "IdealSupplier"
ARCHETYPE_BUDGET        = "BudgetSpecialist"
ARCHETYPE_SUSPICIOUS    = "SuspiciousCheap"
ARCHETYPE_PREMIUM       = "Premium"
ARCHETYPE_UNKNOWN       = "Unknown"

# Cold-Start Gate: Minimum Observations Required Before Classification
MIN_OFFERS      = 5
MIN_OUTCOMES    = 5

# Trust Bands: trust = 1 - min(1, stdev/|mean|)
TRUST_HIGH = 0.70
TRUST_MID = 0.40

# Price-Score Bands: Signed Relative-to-Market Price Deviation
PRICE_PREMIUM = 0.10
PRICE_BUDGET = -0.10

# Reliability Bands (Success Rate)
REL_HIGH = 0.30
REL_MID = 0.15
REL_LOW = 0.05

# PARTNER ARCHETYPE CLASSIFIER ########################################################################################

""" Maps observed negotiation behaviour to one of four PCB-procurement 
    archetypes identified from real-world experience:

        A. IdealSupplier        high trust, mid price, high reliability
        B. BudgetSpecialist     mid-high trust, low price, mid reliability
        C. SuspiciousCheap      low trust, very-low price, low reliability
        D. Premium              mid trust, high price, variable reliability
        E. Unknown              cold-start label (insufficient data)

    The three metrics of evaluation, collected from observation:
    
        1. trust       in [0, 1]        1 minus normalized stdev of partner offers
        2. price       in R             signed deviation from market
        3. reliability in [0, 1]        succ / (succ + fail) from _partner_stats 
"""

@dataclass
class PartnerObservation:

    """ Per-partner observation buffer storing prices and role metadata """

    partner_is_seller: Optional[bool] = None
    offer_prices: list[float] = field(default_factory=list)
    market_at_obs: list[float] = field(default_factory=list)

    def record(
        self,
        price: float,
        market: Optional[float],
        partner_is_seller: Optional[bool] = None,
    ) -> None:

        """ Append one observation and record partner role on first encounter """

        if self.partner_is_seller is None and partner_is_seller is not None:
            self.partner_is_seller = partner_is_seller
        self.offer_prices.append(float(price))
        self.market_at_obs.append(float(market) if market is not None else float("nan"))
        if len(self.offer_prices) > 30:
            self.offer_prices = self.offer_prices[-30:]
            self.market_at_obs = self.market_at_obs[-30:]

def _trust(offers: list[float]) -> float:

    """ Consistency score computed as 1 minus the clamped coefficient of variation """

    m = mean(offers)
    if abs(m) < 1e-9:
        return 0.0
    cv = pstdev(offers) / abs(m)
    return max(0.0, 1.0 - min(1.0, cv))


def _price_score(offers: list[float], markets: list[float]) -> Optional[float]:

    """ Mean signed deviation from market price in relative units """

    pairs = [(p, m) for p, m in zip(offers, markets) if m == m and m > 0]  # NaN-safe
    if not pairs:
        return None
    return mean((p - m) / m for p, m in pairs)

def classify(obs: PartnerObservation, succ: int, fail: int) -> tuple[str, dict]:

    """ Classify partner behavior and return (label, feature_dict) """

    n_offers = len(obs.offer_prices)
    n_outcomes = succ + fail
    if n_offers < MIN_OFFERS or n_outcomes < MIN_OUTCOMES:
        return ARCHETYPE_UNKNOWN, {
            "trust": None, "price": None, "reliability": None,
            "n_offers": n_offers, "n_outcomes": n_outcomes,
            "partner_is_seller": obs.partner_is_seller,
        }

    trust = _trust(obs.offer_prices)
    price = _price_score(obs.offer_prices, obs.market_at_obs)
    reliability = succ / n_outcomes
    features = {
        "trust": trust,
        "price": price,
        "reliability": reliability,
        "n_offers": n_offers,
        "n_outcomes": n_outcomes,
        "partner_is_seller": obs.partner_is_seller,
    }
    # No usable market anchor -> treat as cold-start case
    if price is None:
        return ARCHETYPE_UNKNOWN, features

    # Decision tree: first matching rule determines the label
    if reliability < REL_LOW and price < PRICE_BUDGET:                                   # Cheap and Unreliable Partner
        return ARCHETYPE_SUSPICIOUS, features
    if price > PRICE_PREMIUM and trust >= TRUST_MID:                                     # Consistently Expensive Partner
        return ARCHETYPE_PREMIUM, features
    if price < PRICE_BUDGET and trust >= TRUST_MID and reliability >= REL_MID:           # Honest Cheap
        return ARCHETYPE_BUDGET, features
    if trust >= TRUST_HIGH and abs(price) <= PRICE_PREMIUM and reliability >= REL_HIGH:  # Default Good
        return ARCHETYPE_IDEAL, features
    if reliability < REL_LOW:                                                            # Bottom-tier Fallback
        return ARCHETYPE_SUSPICIOUS, features
    return ARCHETYPE_UNKNOWN, features

# CONTRACT UTILITY MODEL ##############################################################################################

""" Single source of truth for contract valuation,
    consumed by Group4.respond():

        Selling:
            revenue - production_cost - shortfall_penalty * over_qty

        Buying:
            resale_value - price - production_cost
            - disposal_penalty * over_qty

    Penalty rates are obtained from AWI
    (current_shortfall_penalty, current_disposal_cost).

    Capacity is estimated using needed_sales / needed_supplies;
    excess quantity incurs the corresponding penalty.
"""

def contract_utility(agent, *, quantity: int, price: float, selling: bool) -> float:

    """ Estimate signed utility of a single contract (negative = value destroying) """

    awi = getattr(agent, "awi", None)
    if awi is None:
        return 0.0

    # Cached Per-Unit Production Cost

    prod_cost = getattr(agent, "_cached_prod_cost", None)
    if prod_cost is None:
        prod_cost = _compute_prod_cost(awi)
        try:
            agent._cached_prod_cost = prod_cost
        except Exception:
            pass

    short_rate = float(getattr(awi, "current_shortfall_penalty", 0.0) or 0.0)
    disposal_rate = float(getattr(awi, "current_disposal_cost", 0.0) or 0.0)

    # Selling Case
    if selling:
        revenue = price * quantity
        base_cost = prod_cost * quantity
        capacity = max(0, int(getattr(awi, "needed_sales", quantity) or 0))
        over = max(0, quantity - capacity)
        shortfall_penalty = short_rate * over * price
        return revenue - base_cost - shortfall_penalty

    # Buying Case
    output_price = _expected_output_price(awi)
    expected_revenue = output_price * quantity
    base_cost = price * quantity + prod_cost * quantity
    capacity = max(0, int(getattr(awi, "needed_supplies", quantity) or 0))
    over = max(0, quantity - capacity)
    disposal_penalty = disposal_rate * over * price
    return expected_revenue - base_cost - disposal_penalty


def _compute_prod_cost(awi) -> float:

    """ Per-unit production cost from FactoryProfile; 1.0 if unavailable """

    prof = getattr(awi, "profile", None)
    costs = getattr(prof, "costs", None)
    if costs is None:
        return 1.0
    try:
        flat = list(map(float, costs.flatten()))
    except Exception:
        return 1.0
    return sum(flat) / len(flat) if flat else 1.0


def _expected_output_price(awi) -> float:

    """ Resale price for our output: trading_prices > catalog_prices > 0.0 """

    trading = getattr(awi, "trading_prices", None)
    if trading is not None:
        try:
            return float(trading[awi.my_output_product])
        except Exception:
            pass
    catalog = getattr(awi, "catalog_prices", None)
    if catalog is not None:
        try:
            return float(catalog[awi.my_output_product])
        except Exception:
            pass
    return 0.0

# DEMAND-DRIVEN PRODUCTION ############################################################################################

""" Manufacture only what is needed to cover today's signed sells that
    output inventory does not already cover. Cap by available input
    inventory and the factory's daily line count. Never produce
    speculatively: unsold output incurs disposal and storage penalties. """

def schedule_production(agent) -> int:

    """ Units to produce today: min(today_sells - output, input, n_lines) """

    awi = getattr(agent, "awi", None)
    if awi is None:
        return 0

    profile = getattr(awi, "profile", None)
    n_lines = int(getattr(profile, "n_lines", 0)) if profile else 0
    if n_lines <= 0:
        return 0

    today = int(getattr(awi, "current_step", 0))
    sells_by_day = getattr(agent, "_sells_by_day", {}) or {}
    today_sell_qty = int(sells_by_day.get(today, 0))
    if today_sell_qty <= 0:
        return 0

    on_hand_output = int(getattr(awi, "current_inventory_output", 0))
    on_hand_input = int(getattr(awi, "current_inventory_input", 0))
    needed = max(0, today_sell_qty - on_hand_output)

    # Production Limited by Need, Input Inventory, and Daily Capacity

    return min(needed, on_hand_input, n_lines)

# AGENT CLASS - Group4 ################################################################################################

""" Wires the four components into the negotiation lifecycle hooks:

        Component 1 (production)    
            -> step()

        Component 2 (quantity gate) 
            -> respond(), first check

        Component 3a (bidding)      
            -> propose()

        Component 3b (acceptance)   
            -> respond(), utility threshold

        Component 4 (classifier)    
            -> observation in propose/respond
               and threshold adjustment in respond()                       
                                       
"""

class Group4(StdAgent):

    """ SCML standard-track agent """

    # Lifecycle -------------------------------------------------------

    def init(self):

        """ Forward to base; set up state dicts and per-step caches """

        super().init()
        self._sells_by_day: dict[int, int] = {}
        self._partner_stats: dict[str, dict[str, int]] = {}
        self._partner_obs: dict[str, PartnerObservation] = {}
        self._market_cache: dict[bool, Optional[float]] = {}
        self._max_utility_cache: Optional[float] = None

    def step(self):

        """ Per-day hook: schedule production for today's commitments """

        super().step()

        # Invalidate Per-Step Caches

        self._market_cache.clear()
        self._max_utility_cache = None

        qty = schedule_production(self)
        if qty <= 0:
            return

        # Compatibility Guard for SCML Versions

        if not hasattr(self.awi, "schedule_production"):
            return
        try:
            self.awi.schedule_production(
                process=self.awi.my_input_product, repeats=qty
            )
        except Exception:
            pass

    def on_negotiation_success(self, contract, mechanism):

        """ Record successful sales and update partner success statistics """

        ann = contract.annotation or {}
        we_are_seller = ann.get("seller") == self.id
        partner = ann.get("buyer") if we_are_seller else ann.get("seller")
        quantity = int(contract.agreement["quantity"])
        time = int(contract.agreement["time"])
        if we_are_seller:
            self._sells_by_day[time] = self._sells_by_day.get(time, 0) + quantity
        self._record_partner_outcomes([partner] if partner else [], "succ")

    def on_negotiation_failure(self, partners, annotation, mechanism, state):

        """ Update failure statistics for all partners in the callback """

        self._record_partner_outcomes(partners or [], "fail")

    def _record_partner_outcomes(self, partners, outcome: str) -> None:

        """ Update per-partner success/failure counts, excluding self """

        for p in partners:
            if not p or p == self.id:
                continue
            stats = self._partner_stats.setdefault(p, {"succ": 0, "fail": 0})
            stats[outcome] += 1

    # Component 3a - Bidding (Propose) --------------------------------

    def propose(
        self,
        negotiator_id: str,
        state: SAOState,
        dest: Optional[str] = None,
    ) -> Optional[Outcome]:

        """ Build a Boulware-style offer, or skip if we have nothing to add """

        nmi, selling, partner_id = self._thread_role(negotiator_id)
        if nmi is None:
            return None

        need = self._remaining_need(selling)
        if need <= 0:
            return None

        # Issues: (quantity, time, price)

        q_issue, t_issue, p_issue = nmi.issues[0], nmi.issues[1], nmi.issues[2]
        quantity = max(q_issue.min_value, min(need, q_issue.max_value))
        time = t_issue.min_value
        price = self._boulware_target(selling, p_issue, state, BOULWARE_PROPOSE_E)

        # Record Partner Counteroffer for Classifier

        current = getattr(state, "current_offer", None)
        if current is not None:
            self._record_partner_offer(partner_id, float(current[2]), selling)
        return (quantity, time, price)

    def _boulware_target(
        self, selling: bool, p_issue, state: SAOState, e: float
    ) -> float:

        """ Boulware price target: 1 - t**(1/e) interpolation, extreme to market """

        t = float(getattr(state, "relative_time", 0.0) or 0.0)
        t = max(0.0, min(1.0, t))
        aspiration = 1.0 - (t ** (1.0 / e))

        pmin, pmax = float(p_issue.min_value), float(p_issue.max_value)

        # Anchor at Market, else Issue Midpoint

        market = self._market_price(selling)
        if market is None:
            market = (pmin + pmax) / 2.0
        market = max(pmin, min(pmax, market))

        if selling:
            target = market + aspiration * (pmax - market)
        else:
            target = market - aspiration * (market - pmin)
        return max(pmin, min(pmax, target))

    # Components 2 + 3b + 4 - Acceptance (Respond) --------------------

    def respond(
        self,
        negotiator_id: str,
        state: SAOState,
        source: Optional[str] = None,
    ) -> ResponseType:

        """ Quantity gate, then utility threshold with archetype bias """

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        quantity, _time, price = offer
        _, selling, partner_id = self._thread_role(negotiator_id)

        # Record Partner Observation Before Decision

        self._record_partner_offer(partner_id, float(price), selling)

        # Component 2: Quantity Gate

        if quantity > self._remaining_need(selling):
            return ResponseType.REJECT_OFFER

        # Components 3b + 4: Utility Threshold with Archetype Bias

        u = contract_utility(
            self, quantity=int(quantity), price=float(price), selling=selling
        )
        threshold = self.utility_threshold(state)
        threshold *= self._archetype_threshold_multiplier(partner_id)
        if u >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def utility_threshold(self, state: SAOState) -> float:

        """ Boulware-shaped acceptance floor on the utility axis """

        t = float(getattr(state, "relative_time", 0.0) or 0.0)
        t = max(0.0, min(1.0, t))
        aspiration = 1.0 - (t ** (1.0 / BOULWARE_ACCEPT_E))

        # getattr Fallback for Test Stubs that Bypass init()

        max_u = getattr(self, "_max_utility_cache", None)
        if max_u is None:
            max_u = self._max_achievable_utility()
            try:
                self._max_utility_cache = max_u
            except Exception:
                pass
        return MIN_UTILITY_FLOOR + aspiration * (max_u - MIN_UTILITY_FLOOR)

    def _max_achievable_utility(self) -> float:

        """ t=0 anchor: utility at reservation price on the busier side """

        awi = getattr(self, "awi", None)
        if awi is None:
            return 1.0
        sell_need = max(0, int(getattr(awi, "needed_sales", 0)))
        buy_need = max(0, int(getattr(awi, "needed_supplies", 0)))
        selling = sell_need >= buy_need
        qty = max(1, sell_need if selling else buy_need)
        market = self._market_price(selling) or 1.0
        reservation = market * 2.0 if selling else market * 0.5
        u = contract_utility(
            self, quantity=int(qty), price=float(reservation), selling=selling
        )
        return max(MIN_UTILITY_FLOOR, u)

    def _archetype_threshold_multiplier(self, partner_id: Optional[str]) -> float:

        """ Threshold adjustment: Suspicious +bias, Ideal -bias, otherwise 1.0 """

        if not partner_id:
            return 1.0
        label, _ = self.classify_partner(partner_id)
        if label == ARCHETYPE_SUSPICIOUS:
            return 1.0 + ARCHETYPE_THRESHOLD_BIAS
        if label == ARCHETYPE_IDEAL:
            return 1.0 - ARCHETYPE_THRESHOLD_BIAS
        return 1.0

    # Component 4 - Archetype Classifier Wiring -----------------------

    def _record_partner_offer(
        self, partner_id: Optional[str], price: float, selling: bool
    ) -> None:

        """ Buffer one observed partner price against the market anchor """

        if not partner_id:
            return
        obs = self._partner_obs.setdefault(partner_id, PartnerObservation())
        obs.record(price, self._market_price(selling), partner_is_seller=(not selling))

    def classify_partner(self, partner_id: str) -> tuple[str, dict]:

        """ Run the classifier for one partner; returns (label, features) """

        obs = self._partner_obs.get(partner_id)
        if obs is None:
            return ARCHETYPE_UNKNOWN, {}
        stats = self._partner_stats.get(partner_id, {"succ": 0, "fail": 0})
        return classify(obs, stats["succ"], stats["fail"])

    # Helpers Methods ---------------------------------------------------------

    def _thread_role(self, negotiator_id: str):

        """ Return (nmi, we_are_selling, partner_id) for one thread """

        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return None, False, None
        ann = getattr(nmi, "annotation", {}) or {}
        seller, buyer = ann.get("seller"), ann.get("buyer")
        selling = seller == self.id
        return nmi, selling, (buyer if selling else seller)

    def _remaining_need(self, selling: bool) -> int:

        """ Outstanding units to trade on this side today, clamped >= 0 """

        awi = self.awi
        if selling:
            need = getattr(awi, "needed_sales", None)
            if need is None:
                need = int(getattr(awi, "current_exogenous_input_quantity", 0)) - int(
                    getattr(awi, "total_sales", 0)
                )
        else:
            need = getattr(awi, "needed_supplies", None)
            if need is None:
                need = int(getattr(awi, "current_exogenous_output_quantity", 0)) - int(
                    getattr(awi, "total_supplies", 0)
                )
        return max(0, int(need))

    def _market_price(self, selling: bool) -> Optional[float]:

        """ Market price for the relevant product: trading > catalog > None """

        cache = getattr(self, "_market_cache", None)
        if cache is not None and selling in cache:
            return cache[selling]
        awi = self.awi
        product = awi.my_output_product if selling else awi.my_input_product
        result: Optional[float] = None
        for source in ("trading_prices", "catalog_prices"):
            prices = getattr(awi, source, None)
            if prices is None:
                continue
            try:
                result = float(prices[product])
                break
            except Exception:
                continue
        if cache is not None:
            cache[selling] = result
        return result
