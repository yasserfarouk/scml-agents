# Slide 1 — LitaAgentY: Strategy Overview

**Goal**  Build a robust, inventory‑aware negotiation agent for SCML Standard 2025. &#x20;

* Three‑tier supply‑chain awareness *(first / middle / last layer)* &#x20;
* Tight coupling with **InventoryManager** for real‑time stock & capacity control &#x20;
* Modular pipelines for **bidding**, **accepting**, **opponent modeling**, and **profit tuning** &#x20;
* Pluggable RL/heuristic hooks (`decide_with_model`, `update_profit_strategy`) &#x20;

---

# Slide 2 — High‑Level Architecture (Maybe)

```mermaid
    World
        Suppliers((Suppliers))
        Consumers((Consumers))
        Exogenous[Exogenous Contracts]
    IM[InventoryManager\n(raw & product)]
    LA[LitaAgentY]
    Exogenous -.-> IM
    IM --> LA
    LA <--> Suppliers
    LA <--> Consumers
```

* **InventoryManager**   ↔ tracks batches, contracts, production plan, shortages &#x20;
* **Negotiation Loop**   `before_step` → `first_proposals` → `counter_all` → `on_negotiation_success` → `step` &#x20;
* **Exogenous contracts** automatically inserted on boundary layers &#x20;

---

# Slide 3 — Inventory‑Driven Demand Calculation

```python
# before_step()
self.today_insufficient = im.get_today_insufficient(t)
self.total_insufficient = im.get_total_insufficient(t)

raw_need_today = self.today_insufficient           # emergency
raw_need_future = self.total_insufficient          # planned
raw_need_optional = int(raw_need_future * 0.2)     # 20 % buffer
```

| Layer            | Buy Need Source          | Sell Need Source                                      |
| ---------------- | ------------------------ | ----------------------------------------------------- |
| \*\*First  \*\*  | none (exogenous)         | min(raw inventory available, max production capacity) |
| \*\*Middle  \*\* | shortages (table above)  | today’s product inventory                             |
| \*\*Last  \*\*   | shortages                | none (exogenous)                                      |

Demand figures feed `_distribute_todays_needs()` which splits quantity across a *pₜ* fraction of partners. &#x20;

---

# Slide 4 — InventoryManager Implementation

```python
@dataclass
class Batch:
    batch_id: str
    remaining: float
    unit_cost: float
    production_time: int  # day index

class InventoryManager:
    def add_transaction(self, contract: IMContract):
        # 1) register supply/demand contract
        # 2) recompute production plan & shortages
        self.plan_production(self.max_day)

    def plan_production(self, horizon: int):
        """Greedy backward planning (latest‑first) ensuring delivery & capacity"""
        # ① simulate demand & supply per day
        # ② fill backward from due date respecting daily capacity
        # ③ populate self.production_plan & self.insufficient_raw
```

\*\*Key responsibilities  \*\*

1. **Batch FIFO** management (`raw_batches`, `product_batches`) &#x20;
2. **Shortage detection** → `self.insufficient_raw[day]` (daily / total) &#x20;
3. **Production plan** generation per day with *capacity constraints* &#x20;
4. **What‑if simulation** via `simulate_future_inventory()` for agent look‑ahead &#x20;
5. **Convenience** `process_day_operations()` executes receive → produce → deliver in one call. &#x20;

> The negotiation layer queries IM *every step* to keep bids feasible and avoid penalties. &#x20;

---

# Slide 5 — First‑Round Bidding: Mechanics

```python
# first_proposals()
partners = list(self.negotiators.keys())

# 1️⃣ filter partners by layer role
if self.awi.is_first_level:
    partners = [p for p in partners if p in self.awi.my_consumers]
elif self.awi.is_last_level:
    partners = [p for p in partners if p in self.awi.my_suppliers]

# 2️⃣ quantity allocation
allocation = self._distribute_todays_needs(partners)

# 3️⃣ extreme‑anchor pricing
out = {}
for pid, qty in allocation.items():
    if qty <= 0:
        continue
    price = self._best_price(pid)  # max when selling, min when buying
    out[pid] = (qty, self.awi.current_step, price)
return out
```

### `_distribute_todays_needs()`

```python
def _distribute_todays_needs(self, partners):
    k = max(1, int(len(partners) * self._ptoday))  # fraction selected
    chosen = random.sample(partners, k)
    quantities = _distribute(total_need, k)        # random integer split
    return dict(zip(chosen, quantities))
```

* **Layer‑aware filtering** ensures first layer only sells, last layer only buys. &#x20;
* **Random partner subset** (*pₜ*≈0.5 default) → exploration & load balancing. &#x20;
* **Extreme anchors** exploit issue range to leave margin for concession. &#x20;
* Opening offers are always **time = today** on first layer to match capacity. &#x20;

---

# Slide 6 — Emergency Supply Acceptance Logic

```python
penalty = self.awi.current_shortfall_penalty
for pid, offer in sort_by_price(offers):
    if offer[UNIT_PRICE] <= penalty:
        ACCEPT_OFFER(offer)
    else:
        new_price = _apply_concession(pid, best_price(pid), state, offer[UNIT_PRICE])
        REJECT_OFFER((offer[QUANTITY], today, new_price))
```

* Accept if cheaper than **penalty for stock‑out**  .
* Otherwise counter with opponent‑aware concession. &#x20;
* Stop when `today_insufficient` is covered. &#x20;

---

# Slide 7 — Planned & Optional Supply Logic

```python
unit_cost = im.get_inventory_summary(t, PRODUCT)["estimated_average_cost"]
max_price = est_sell_price / (1 + min_profit_margin)
...
```

Optional stock‑up triggers when `price ≤ market_avg * cheap_price_discount` **and** projected inventory < 120 % of demand. &#x20;

---

# Slide 8 — Sales Offer Handling

```python
max_prod = im.get_max_possible_production(t)
if signed_qty + qty > max_prod:
    counter_qty = max_prod - signed_qty
    REJECT → counter(counter_qty, t, price)
...
```

Guarantees **capacity safety** and **profit margin**  .

---

# Slide 9 — Opponent Modeling & Concession Curve

```python
def concession_model(time, opp_rate, beta=0.5):
    return min(1.0, time * (1 + beta * opp_rate))
```

* **Local concession speed** ΔP/P per negotiator. &#x20;
* Mixed with **relative time** to scale own concession. &#x20;
* β controls aggressiveness against fast‑conceding opponents. &#x20;

---

# Slide 10 — Dynamic Profit & Risk Tuning

```python
agent.update_profit_strategy(min_profit_margin=0.12,
                             cheap_price_discount=0.6)
```

* Raise margin when demand high / capacity tight. &#x20;
* Lower discount threshold in bullish markets to stock ahead. &#x20;

---

# Slide 11 — Key Code Path Recap

| Phase                  | Core Method                | Essential Logic                                        |
| ---------------------- | -------------------------- | ------------------------------------------------------ |
| \*\*Init  \*\*         | `init()`                   | instantiate `InventoryManager`                         |
| \*\*Before day  \*\*   | `before_step()`            | pull shortages, exogenous contracts                    |
| \*\*First offers  \*\* | `first_proposals()`        | layer‑aware extreme anchors                            |
| \*\*Respond  \*\*      | `counter_all()`            | route supply & sales offers                            |
|                        | `_process_*`               | emergency / planned / optional / sales handlers        |
| \*\*Success  \*\*      | `on_negotiation_success()` | IM transaction logging                                 |
| \*\*End‑day  \*\*      | `step()`                   | IM `process_day_operations()` & update price averages  |

> **Take‑away:** Inventory‑first reasoning + adaptive concession yields resilient, profitable negotiation across volatile supply chains. &#x20;

---

# Slide 12 — Current Issues & Improvement Directions

**Observed limitations**

1. **Optional stock‑up may over‑buy** — rule uses 120 % cap but doesn’t consider future price trend ⇒ risk of excess inventory.
2. **Linear concession curve** — lacks non‑linear time pressure or Stacked‑Nash heuristic.
3. **Opponent model is shallow** — only price concession speed; no pattern mining or Bayes utility inference.
4. **Exogenous price assumption in Pareto analysis** — real downstream price uncertain, affects profit estimation.
5. **InventoryManager performance** — greedy backward planning is $O(days·contracts)$; for >1000 contracts could lag.
6. **No penalty anticipation** — relies on current `shortfall_penalty` but doesn’t forecast aggregated penalties across steps.


> Continuous profiling & simulation on SCML2025 benchmark worlds will guide iterative tuning.
