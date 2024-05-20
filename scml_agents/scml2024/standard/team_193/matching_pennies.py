#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.oneshot import OneShotAWI, OneShotSyncAgent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState

from scml.scml2020.common import QUANTITY, UNIT_PRICE
from numpy import random
from numpy.random import choice
from collections import Counter
from itertools import chain, combinations
from negmas.gb.common import ResponseType

__all__ = ["MatchingPennies"]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at
    least one item per bin assuming q > n"""

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def distribute_goods(goods, capacities):
    # Initialize bins with 0 goods
    bins = [0] * len(capacities)

    # Continue until all goods are distributed
    while goods > 0:
        # Track if we successfully distributed at least one good in this iteration
        distributed = False

        # Attempt to distribute one good to each bin, starting with the bin
        # with the lowest current value, without exceeding its capacity
        for i, capacity in sorted(enumerate(capacities), key=lambda x: bins[x[0]]):
            if bins[i] < capacity:
                bins[i] += 1
                goods -= 1
                distributed = True
                if goods == 0:
                    break

        # If we couldn't distribute any goods in this iteration, break the loop
        # to prevent an infinite loop (this means all bins are at capacity)
        if not distributed:
            break

    # Return the final distribution of goods across bins
    return bins


class MatchingPennies(OneShotSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `OneShotUFun` in the docs for more details).
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    def distribute_needs(self, match_history) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(dict(zip(partner_ids, distribute(needs, partners))))
        return dist

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs(match_history=False)
        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}

        return d

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return mn + (mx - mn) * (r**4.0)

    def best_subset(self, needs, offers, plist, quantity_cost_tradeoff=0.95):
        best_total_los = float("inf")
        best_quantity_diff, best_indx = float("inf"), -1
        # quantity_cost_tradeoff = 0.90  ### toggle this to adjust the tradeoff!
        for i, partner_ids in enumerate(plist):
            # Calculate quantity
            offered_quantity = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = abs(offered_quantity - needs)

            # Calculate COST, the bigger, the worse!
            total_contracts_cost = sum(
                offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in partner_ids
            )
            penalty = 0

            if self.awi.level == 0:
                if offered_quantity < needs:
                    # surplus!
                    penalty += diff * self.awi.current_disposal_cost
                else:
                    penalty += diff * self.awi.current_shortfall_penalty
            # consumers
            else:
                if offered_quantity < needs:
                    # shortfall
                    penalty += diff * self.awi.current_shortfall_penalty
                if offered_quantity > needs:
                    penalty += diff * self.awi.current_disposal_cost
            # Surplus Penalty
            if offered_quantity > needs + 1:
                continue
            # elif offered_quantity == needs:
            #     best_quantity_diff, best_indx = diff, i
            #     break

            max_utility, min_utility = self.ufun.max_utility, self.ufun.min_utility
            max_diff, min_diff = needs, 0
            total_profit = (
                (self.p * self.q - total_contracts_cost - penalty)
                if self.awi.level == 1
                else (total_contracts_cost - self.p * self.q - penalty)
            )

            # Normalize total_profit and diff
            total_profit_normalized = (
                (total_profit - min_utility) / (max_utility - min_utility)
                if (max_utility - min_utility) != 0
                else 0
            )
            diff_normalized = (
                (diff - min_diff) / (max_diff - min_diff)
                if (max_diff - min_diff) != 0
                else 0
            )

            # Calculating loss with normalized values
            loss = (
                quantity_cost_tradeoff * diff_normalized
                - (1 - quantity_cost_tradeoff) * total_profit_normalized
            )

            if loss < best_total_los:
                best_quantity_diff, best_indx = diff, i
                best_total_los = loss

        return best_quantity_diff, best_indx

    def counter_all(self, offers, states):
        response = dict()
        self.negotiation_steps += 1
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()

            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}
            total_offer_quant = sum(offers[p][QUANTITY] for p in list(partners))

            self.expected_contract_quantity[self.negotiation_steps] *= (
                self.expected_contract_step[self.negotiation_steps]
            )
            self.expected_contract_step[self.negotiation_steps] += 1
            self.expected_contract_quantity[self.negotiation_steps] += total_offer_quant
            self.expected_contract_quantity[self.negotiation_steps] /= (
                self.expected_contract_step[self.negotiation_steps]
            )

            # Update partner's wanted price history
            for partner in list(partners):
                if partner not in self.partner_price_history:
                    self.partner_price_history = [offers[partner][UNIT_PRICE]]
                else:
                    self.partner_price_history.append([offers[partner][UNIT_PRICE]])

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))
            best_diff, best_indx = self.best_subset(
                # QUESTION: how to get max exogenous quantity and min exogenous quantity?
                needs,
                offers,
                plist,
                quantity_cost_tradeoff=(1.0 - self.q / 100),
            )

            partner_ids = plist[best_indx]
            others = list(partners.difference(partner_ids))

            if (sum(offers[p][QUANTITY] for p in partner_ids) > (needs + 1)) or (
                best_diff < needs
                and best_diff > needs / 2
                and needs > 3
                and self.q >= 6
                and total_offer_quant >= needs * 1.5
                and (
                    self.expected_contract_quantity[self.negotiation_steps + 1] >= needs
                    or self.days <= 4
                )
            ):
                # Renegotiate
                capacity = []
                for partner in list(partners):
                    capacity.append(offers[partner][QUANTITY])
                distribution = distribute_goods(best_diff, capacity)
                dict_response = dict()
                for i, partner in enumerate(list(partners)):
                    dict_response[partner] = distribution[i]
                response |= {
                    k: SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (
                            q,
                            self.awi.current_step,
                            self.best_price,
                            #  if best_diff > 1 else offers[k][UNIT_PRICE],
                        ),  #  self.best_price, offers[k][UNIT_PRICE]
                    )
                    for k, q in dict_response.items()
                    # k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    # for k in others
                }

            capacity = []
            for other in others:
                capacity.append(offers[other][QUANTITY])
            distribution = distribute_goods(best_diff, capacity)
            dict_response = dict()
            for i, other in enumerate(others):
                dict_response[other] = distribution[i]

            response |= {
                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                for k in partner_ids
            } | {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (
                        q,
                        self.awi.current_step,
                        # self.best_price if best_diff > 1 else offers[k][UNIT_PRICE],
                        self.best_price if best_diff > 1 else offers[k][UNIT_PRICE],
                    ),  #  self.best_price, offers[k][UNIT_PRICE]
                )
                for k, q in dict_response.items()
                # k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                # for k in others
            }

        return response

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

        self.verbose = False
        self.first = True
        self.days = 0
        self.expected_contract_quantity = [0] * 25
        self.expected_contract_step = [0] * 25

        if self.awi.level == 0:
            self.partners = self.awi.my_consumers
        else:
            self.partners = self.awi.my_suppliers

        self.partner_price_history = dict()

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

        self.negotiation_steps = 0

        if self.awi.level == 0:
            self.q = self.awi.current_exogenous_input_quantity
            self.p = self.awi.current_exogenous_input_price
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price
        else:
            self.q = self.awi.current_exogenous_output_quantity
            self.p = self.awi.current_exogenous_output_price
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price

        self.opponent_contract_history = dict()

        self.accumulated_profit = self.awi.current_score

        self.todays_contracts_num = 0
        self.todays_spent = 0
        self.contracts = []
        self.days += 1

        # if self.verbose:
        #     print(
        #         f" I am at level {self.awi.level} and I need {self.q} contracts. The min price is {self.min_price} and the max price is {self.max_price}. I have {self.awi.n_lines} lines."
        #     )
        #     print(
        #         f" Current disposal cost = {self.awi.current_disposal_cost}. Current shortfall cost = {self.awi.current_shortfall_penalty}"
        #     )
        #     print(f"My partners are {self.partners}")

    def step(self):
        """Called at at the END of every production step (day)"""
        # if self.verbose:
        #     print(f"Today, I'm {self.first} the first")
        uprice = (
            self.todays_spent / self.todays_contracts_num
            if not (self.todays_contracts_num == 0)
            else 0
        )
        if self.verbose:
            try:
                print(
                    f"Today I need {self.q} contracts, I managed to get {self.todays_contracts_num} contracts with the price of {uprice} each. Today I accumulated a profit of {self.ufun.from_contracts(self.contracts)}."
                )
                print(
                    f"Today's exorgenous contracts are at {self.p/self.q} for each of the {self.q} items"
                )
            except:
                pass  # print(f"Expected quantities: {self.expected_contract_quantity}")

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
        # self.todays_contracts += contract.agreement
        # print(f"On_negotiation_success{contract}")
        self.todays_contracts_num += contract.agreement["quantity"]
        self.todays_spent += (
            contract.agreement["unit_price"] * contract.agreement["quantity"]
        )
        self.contracts.append(contract)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([MatchingPennies], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
