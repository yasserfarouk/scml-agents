#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

# required for typing

# required for development
from scml.oneshot import *
from scml.std import *

# required for typing
from negmas import Contract, SAOResponse, ResponseType
from itertools import combinations, product

import random

import numpy as np

__all__ = ["Rchan"]


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
    randomness = False,
    concentrated_idx: list[int] = [],
) -> list[int]:
    """Distributes q values over n bins.

    Args:
        q: Quantity to distribute
        n: number of bins to distribute q over
        mx: Maximum allowed per bin. `None` for no limit
        equal: Try to make the values in each bins as equal as possible
        concentrated: If true, will try to concentrate offers in few bins. `mx` must be passed in this case
        allow_zero: Allow some bins to be zero even if that is not necessary
    """
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if randomness:
        per = 1
        r = Counter(choice(n, q))
        return [r.get(_, 0) + per for _ in range(n)]

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        concentrated_lst = sorted(lst, reverse=True)[: len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i, x in zip(concentrated_idx, concentrated_lst):
            lst.insert(i, x)
        # print(lst,concentrated_idx)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]

def powerset(iterable):
    """冪集合"""
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

########################################################################################################


class Rchan(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, if the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
        overmismatch_max: 相手から受けた提案の総量が自身の必要取引量を超える場合に許容する過剰量(n_linesに対する割合で付与)。
        undermismatching_min_selling: 自身が売り手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
        undermismatching_min_buying: 自身が買い手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.3,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.2,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        super().__init__(*args, **kwargs)

    def init(self):
        self.overordering_max = (
            self.overordering_max_selling
            if self.awi.my_suppliers == ["SELLER"]
            else self.overordering_max_buying
        )
        self.overmismatch_max_selling *= self.awi.n_lines
        self.overmismatch_max_buying *= self.awi.n_lines
        self.undermismatch_min_selling *= self.awi.n_lines
        self.undermismatch_min_buying *= self.awi.n_lines

        # 各ラウンドでの相手一人あたりの(擬似的な)提案個数
        self.rounds_ave_offered = (
            [self.awi.n_lines / len(self.awi.my_consumers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_consumers)] * 9
            + [1] * 10
            if self.awi.my_suppliers == ["SELLER"]
            else [self.awi.n_lines / len(self.awi.my_suppliers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_suppliers)] * 9
            + [1] * 10
        )

        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }

        self.system_scores = [0 for _ in range(self.awi.n_steps + 1)]
        self.system_scores_diff = [0 for _ in range(self.awi.n_steps + 1)]
        self.calculate_scores = [0 for _ in range(self.awi.n_steps + 1)]
        self.price_count = [[0,0] for _ in range(self.awi.n_steps + 1)]

        self.contract_quantities = [0 for _ in range(20)]
        self.contract_quantities_bystep = [ [0 for _ in range(20)] for _ in range(self.awi.n_steps + 1)]

        self.first_quantity = [0 for _ in range(20)]
        self.counter_quantity = [0 for _ in range(20)]
        self.accept_quantity = [0 for _ in range(20)]

        all_partners = (
            self.awi.my_consumers
            if self.awi.my_suppliers == ["SELLER"]
            else self.awi.my_suppliers
        )

        self.agent_count = {k:0 for k in all_partners}
        self.agent_ave_quantity = {k:0 for k in all_partners}
        self.agent_quantity = {k: [0 for _ in range(20)] for k in all_partners}

        self.first_attempt = {k: [0 for _ in range(20)] for k in all_partners}
        self.first_accept = {k: [0 for _ in range(20)] for k in all_partners}
        self.first_total = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.first_accept_ave = {
            k: 0.0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.first_accept_prob = {
            k: 0.0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.counter_attempt = {k: [0 for _ in range(20)] for k in all_partners}
        self.counter_accept = {k: [0 for _ in range(20)] for k in all_partners}
        self.accept = {k: [0 for _ in range(20)] for k in all_partners}
        self.counter_total = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }

        self.accept_total = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.first = 0
        self.counter = 0
        self.proposal_state = "first"

        self.complete_mission = [0 for _ in range(self.awi.n_steps + 1)]
        self.plus = [0 for _ in range(self.awi.n_steps + 1)]
        self.minus = [0 for _ in range(self.awi.n_steps + 1)]
        self.perfect = [0 for _ in range(self.awi.n_steps + 1)]

        self.first_accept_list = [[] for _ in range(self.awi.n_steps + 1)]

        self.preres = [[] for _ in range(self.awi.n_steps + 1)]
        self.opo_q = [0 for _ in range(20)]

        self.my_diff_plus = [0 for _ in range(40)]
        self.my_diff_minus = [0 for _ in range(40)]

        self.overordering_q = 2

        self.preoffer = None


        # print(self.awi.current_disposal_cost, self.awi.current_storage_cost, self.awi.current_shortfall_penalty)

        return super().init()

    def distribute_needs(
        self,
        t: float,
        mx: int | None = None,
        equal: bool | None = None,
        allow_zero: bool | None = None,
        concentrated: bool = False,
        concentrated_ids: list[str] = [],
    ) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        if equal is None:
            equal = self.equal_distribution
        if allow_zero is None:
            allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            # partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            # n_partners = len(partners)
            partners, n_partners = [], 0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            offering_quanitity = (
                int(needs * (1 + self._overordering_fraction(t)))
                if len(partners) > 1
                else needs
            )
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            n_partners,
                            mx=mx,
                            equal=equal,
                            concentrated=concentrated,
                            allow_zero=allow_zero,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            )
        return dist

    # def on_negotiation_success(self, contract: Contract, mechanism) -> None:
    #     super().on_negotiation_success(contract, mechanism)
    #     partner_id = [p for p in contract.partners if p != self.id][0]

    #     self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]


    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: SAONMI,
        state: SAOState,
    ) -> None:
        """
        Called whenever a negotiation ends without agreement.

        Args:
            partners: List of the partner IDs consisting from self and the opponent.
            annotation: The annotation of the negotiation including the seller ID,
                        buyer ID, and the product.
            mechanism: The `NegotiatorMechanismInterface` instance containing all information
                       about the negotiation.
            state: The final state of the negotiation of the type `SAOState`
                   including the agreement if any.
        """
        now = self.awi.current_step
        self.system_scores[now] = self.awi.current_score

        if( now > 0):
            self.system_scores_diff[now-1] = (
                self.awi.current_score
                - self.system_scores[now-1]
            )

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]

        self.contract_quantities[contract.agreement["quantity"]] += 1
        self.contract_quantities_bystep[self.awi.current_step][contract.agreement["quantity"]] += 1
        self.agent_count[partner_id] += 1
        self.agent_ave_quantity[partner_id] = (self.total_agreed_quantity[partner_id]) / self.agent_count[partner_id]
        self.agent_quantity[partner_id][contract.agreement["quantity"]] += 1


        # print("contract:", contract)
        if self.proposal_state == "first":
            if contract.mechanism_state.new_offerer_agents[0] == self.id:
                self.first_accept[partner_id][contract.agreement["quantity"]] += 1
                self.first_total[partner_id] += contract.agreement["quantity"]
                self.first_accept_list[self.awi.current_step].append({partner_id: contract.agreement["quantity"]})
                self.preres[self.awi.current_step].append(contract.agreement["quantity"])
                ave = 0.0
                for i in range(len(self.first_accept[partner_id])):
                    ave += i * self.first_accept[partner_id][i]
                self.first_accept_ave[partner_id] = ave / sum(self.first_accept[partner_id]) if sum(self.first_accept[partner_id]) > 0 else 0.0
                self.first_accept_prob[partner_id] = sum(self.first_accept[partner_id]) / (sum(self.first_attempt[partner_id]) - self.first_attempt[partner_id][0]) if (sum(self.first_attempt[partner_id]) - self.first_attempt[partner_id][0]) > 0 else 0.0
        else:
            if contract.mechanism_state.new_offerer_agents[0] == self.id:
                # print(contract)
                self.counter_accept[partner_id][contract.agreement["quantity"]] += 1
                self.counter_total[partner_id] += contract.agreement["quantity"]
        # if contract.agreement[""]

        now = self.awi.current_step
        level = self.awi.is_first_level
        exogenous_input_quantity=self.awi.current_exogenous_input_quantity
        exogenous_input_price=self.awi.current_exogenous_input_price
        exogenous_output_quantity=self.awi.current_exogenous_output_quantity
        exogenous_output_price=self.awi.current_exogenous_output_price

        if (self.awi.needed_sales < 1 and self.awi.needed_supplies < 1):
            self.complete_mission[now] = 1

        if (self.awi.needed_sales == 0 and self.awi.needed_supplies == 0):
            self.perfect[now] = 1
            self.minus[now] = 0
        else:
            if(self.awi.needed_sales > 0 or self.awi.needed_supplies > 0):
                self.minus[now] = 1
            else:
                self.plus[now] = 1
                self.minus[now] = 0


        total_quantity = self.price_count[now][1] + self.price_count[now][0]
        if(total_quantity != 0):
            q = exogenous_input_quantity if level else exogenous_output_quantity
            if( total_quantity > q):
                self.my_diff_plus[total_quantity - q] -= 1
            else:
                self.my_diff_minus[q - total_quantity] -= 1


        if contract.agreement["unit_price"] is self.awi.current_input_issues[UNIT_PRICE].max_value:
            self.price_count[now][1] = self.price_count[now][1] + contract.agreement["quantity"]
        else:
            self.price_count[now][0] = self.price_count[now][0] + contract.agreement["quantity"]

        total_quantity = self.price_count[now][1] + self.price_count[now][0]
        if(total_quantity != 0):
            q = exogenous_input_quantity if level else exogenous_output_quantity
            # print("total_quantity:", total_quantity, "q:", q)
            if( total_quantity > q):
                self.my_diff_plus[total_quantity - q] += 1
            else:
                self.my_diff_minus[q - total_quantity] += 1

        # print("price_count:", self.price_count[now])
        self.system_scores[now] = self.awi.current_score
        if( now > 0):
            self.system_scores_diff[now-1] = (
                self.awi.current_score
                - self.system_scores[now-1]
            )

        high, low = self.price_count[now][1], self.price_count[now][0]

        scores = self.cal_scores(high, low)

        self.calculate_scores[now] = scores
        # print(self.awi.current_step)
        # print("info")
        # print(self.preres)
        # print("complete_mission", sum(self.complete_mission) / (now+1))
        # print("plus", sum(self.plus) / (now+1))
        # print("minus", sum(self.minus) / (now+1))
        # print("perfect", sum(self.perfect) / (now+1))
        # print("wariai", self.first, self.counter)
        # print(self.awi.current_disposal_cost, self.awi.current_storage_cost, self.awi.current_shortfall_penalty)
        # print(high, low, self.awi.needed_sales, self.awi.needed_supplies, scores)
        # print("diff")
        # print(self.my_diff_plus)
        # print(self.my_diff_minus)
        # print("quantity")
        # print(self.contract_quantities)
        # print(self.first_quantity)
        # print(self.counter_quantity)
        # print(self.accept_quantity)
        # print("agent")
        # print(self.total_agreed_quantity)
        # print(self.agent_count)
        # print(self.agent_ave_quantity, sum(self.agent_ave_quantity.values()) / len(self.agent_ave_quantity))
        # print(self.agent_quantity)
        # print("opponent trend")
        # print(self.opo_q)
        # print(list(self.first_accept_list[now][0].values()))

        # print(self.first_accept_list[now], sum([list(d.values())[0] for d in self.first_accept_list[now]]))
        # # print("parsent")
        # for k in self.first_attempt.keys():
        #     print(k)
        #     print("first_attempt",self.first_attempt[k])
        #     print("first_accept",self.first_accept[k])
        #     print("counter_attempt",self.counter_attempt[k])
        #     print("counter_accept",self.counter_accept[k])
        #     print("accept",self.accept[k])
        #     print("agent_ave", self.first_accept_ave[k])
        #     print("accept_prob", sum(self.first_accept[k]) / (sum(self.first_attempt[k]) - self.first_attempt[k][0]) if(sum(self.first_attempt[k]) - self.first_attempt[k][0]) > 0 else 0.0)

        # print(self.first_total, sum(self.first_total.values()))
        # print(self.counter_total, sum(self.counter_total.values()))
        # print(self.accept_total, sum(self.accept_total.values()))


        # print(self.first_attempt)
        # print(self.first_accept)
        # print(self.counter_attempt)
        # print(self.counter_accept)
        # for i in range(self.awi.current_step + 1):
        #     print(self.contract_quantities_bystep[i])

    def cal_scores(self, high, low):

        now = self.awi.current_step
        level = self.awi.is_first_level
        exogenous_input_quantity=self.awi.current_exogenous_input_quantity
        exogenous_input_price=self.awi.current_exogenous_input_price
        exogenous_output_quantity=self.awi.current_exogenous_output_quantity
        exogenous_output_price=self.awi.current_exogenous_output_price

        if(level):
            sales = low + high

            q = min(self.awi.n_lines, exogenous_input_quantity)

            if(high + low > q):
                revenue = self.awi.current_output_issues[UNIT_PRICE].max_value * min(q, high) + self.awi.current_output_issues[UNIT_PRICE].min_value * max(0, q - high)
            else:
                revenue = self.awi.current_output_issues[UNIT_PRICE].max_value * high + self.awi.current_output_issues[UNIT_PRICE].min_value * low

            costs = exogenous_input_price
            prod_costs =  min(q, sales) * self.awi.profile.cost
            shortfall = max(0, sales - exogenous_input_quantity)
            shortfall_penalty = self.awi.current_shortfall_penalty * shortfall * self.awi.trading_prices[1]
            disposal = max(0, exogenous_input_quantity - sales)
            disposal_cost = disposal * self.awi.current_disposal_cost * self.awi.trading_prices[0]

            supplier_scores = revenue - (costs + prod_costs + shortfall_penalty + disposal_cost)
            # print(now, self.calculate_scores[now], high, low, prod_costs, disposal_cost, shortfall_penalty, revenue, costs)
            return supplier_scores
        
        else:
            supplies = low + high

            q = min(self.awi.n_lines, exogenous_output_quantity)

            if(high + low > q):
                costs = self.awi.current_input_issues[UNIT_PRICE].min_value * min(q, low) + self.awi.current_input_issues[UNIT_PRICE].max_value * max(0, q - low)
            else:
                costs = self.awi.current_input_issues[UNIT_PRICE].max_value * high + self.awi.current_input_issues[UNIT_PRICE].min_value * low

            if exogenous_output_quantity > 0:
                revenue = exogenous_output_price * (min((high + low), q) / exogenous_output_quantity)
            else:
                revenue = 0
                
            prod_costs =  min(q, supplies) * self.awi.profile.cost
            shortfall = max(0, exogenous_output_quantity - supplies)
            shortfall_penalty = self.awi.current_shortfall_penalty * shortfall * self.awi.trading_prices[2]
            disposal = max(0, supplies - exogenous_output_quantity)
            disposal_cost = disposal * self.awi.current_disposal_cost * self.awi.trading_prices[1]

            consumer_scores = revenue - (costs + prod_costs + shortfall_penalty + disposal_cost)
            # print(now, self.calculate_scores[now], high, low, prod_costs, disposal_cost, shortfall_penalty, revenue, costs)
            return consumer_scores


    def first_proposals(self):
        # print(self.awi.current_step, self.awi.is_first_level)
        self.first += 1
        self.proposal_state = "first"
        now = self.awi.current_step
        level = self.awi.is_first_level
        exogenous_input_quantity=self.awi.current_exogenous_input_quantity
        exogenous_input_price=self.awi.current_exogenous_input_price
        exogenous_output_quantity=self.awi.current_exogenous_output_quantity
        exogenous_output_price=self.awi.current_exogenous_output_price

        my_aim_quantity = self.awi.needed_sales if self.awi.my_suppliers == ["SELLER"] else self.awi.needed_supplies

    
        # just randomly distribute my needs over my partners (with best price for me).
        s, price = self._step_and_price(best_price=True)
        # my_negotiators = [p for p in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers) if p in self.negotiators.keys()]
        my_negotiators, not_negotiators = [], []
        if self.awi.my_suppliers == ["SELLER"]:
            for k in self.awi.my_consumers:
                if self.awi.is_bankrupt(k) or (
                    self.awi.current_step > min(self.awi.n_steps * 0.5, 20)
                    and self.first_total[k] == 0
                    # and self.total_agreed_quantity[k] == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            offering_quantity = (
                int(my_aim_quantity * (1 + self._overordering_fraction(0)))
                if len(my_negotiators) > 1
                else my_aim_quantity
            )
        else:
            for k in self.awi.my_suppliers:
                if self.awi.is_bankrupt(k) or (
                    self.awi.current_step > min(self.awi.n_steps * 0.5, 20)
                    and self.first_total[k] == 0
                    # and self.total_agreed_quantity[k] == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            offering_quantity = (
                int(my_aim_quantity * (1 + self._overordering_fraction(0)))
                # int(my_aim_quantity * (1 + 0.5))
                if len(my_negotiators) > 1
                else my_aim_quantity
            )
        now = self.awi.current_step
        self.preres[now].append(now)
        self.preres[now].append(my_aim_quantity)

        before = now - 10 if now - 10 >= 0 else 0

        plus = sum(self.plus[before:now]) / (now + 1 - before) if now > 0 else 0
        minus = sum(self.minus[before:now]) / (now + 1 - before) if now > 0 else 0
        perfect = sum(self.perfect[before:now]) / (now + 1 - before) if now > 0 else 0


        if(plus == max(plus, minus, perfect)):
            q = int(my_aim_quantity * 0.5) + 1
        elif(minus == max(plus, minus, perfect)):
            q = int(my_aim_quantity)
            s, price = self._step_and_price(best_price=False)
        else:
            if(plus < minus):
                q = int(my_aim_quantity * 0.8) + 1
            else:
                q = int(my_aim_quantity * 0.5) + 1

        agent_to_quantities = {
            agent: [
                i for i in range(q)
            ]
            for agent in my_negotiators
        }


        d = {}
        if len(my_negotiators) > 0:
            if (
                # self.awi.current_step > self.awi.n_steps * 0.5
                # and len(my_negotiators) > 0
                self.awi.current_step > 15
                and len(my_negotiators) > 0
            ):
                best_combo = [1 for _ in range(len(my_negotiators))]
                if level:
                    best_score = self.cal_scores(0, len(my_negotiators))
                    best_score = 0.0
                else:
                    best_score = self.cal_scores(len(my_negotiators), 0)
                    best_score = 0.0

                best_p = 0.0
                # best_q = my_aim_quantity

                precomputed_prob = {
                    agent: { (q): 0.0 for q in range(10)}
                    # agent: { (q): 0.0 for q in range(my_aim_quantity+1)}
                    for agent in my_negotiators
                }

                max_q = int(my_aim_quantity + 0.35)

                for agent in my_negotiators:
                    for q in range(my_aim_quantity+1):
                        # print("agent:", agent, "q:", q)
                        # print(self.first_accept)
                        # print(self.first_attempt)
                        a = self.first_accept[agent][q] + 1
                        b = self.first_attempt[agent][q] - self.first_accept[agent][q] + 1
                        # print(a, b)
                        p = np.random.beta(a, b)
                        precomputed_prob[agent][q] = p


                if len(my_negotiators) > 2:
                    
                    best_quantity = {
                        agent: 0 for agent in my_negotiators
                    }
                    concentrated_ids = sorted(
                        my_negotiators,
                        key=lambda x: self.first_accept_prob[x],
                        # key=lambda x: self.first_accept_ave[x],
                        # key=lambda x: self.first_total[x],
                        reverse=True,
                    )[:4]
                    # concentrated_ids = [id for id in concentrated_ids if self.first_accept_prob[id] > 0.4]
                    

                    quantity_options = [agent_to_quantities[agent] if agent in concentrated_ids else [best_quantity[agent]] for agent in my_negotiators]
                    # print("quantity_options:", quantity_options)

                    for q_combo in product(*quantity_options):  # 各エージェントに対する個数の全パターン
                        # print("q_combo:", q_combo)
                        total_quantity = sum(q_combo)
                        # print("total_quantity:", total_quantity)
                        if total_quantity > max_q:
                            continue
                        if level:
                            score = self.cal_scores(0, total_quantity)
                        else:
                            score = self.cal_scores(total_quantity, 0)
                        # print("score:", score)


                        # 成功確率は precomputed_prob[agent][q]
                        success_prob = 1.0
                        expected_value = 0.0
                        for agent in my_negotiators:
                            q = q_combo[my_negotiators.index(agent)]
                            expected_value += q * precomputed_prob[agent][q]
                            success_prob *= precomputed_prob[agent][q]


                        if (abs(expected_value - my_aim_quantity) < abs(best_score - my_aim_quantity)):
                            # if (expected_value > my_aim_quantity - 1):
                            best_score = expected_value
                            best_combo = q_combo
                            best_p = success_prob

                    if(plus == max(plus, minus, perfect)):
                        # self.overordering_q = self.overordering_q - 1
                        self.overordering_exp = self.overordering_exp * 0.7
                    elif(minus == max(plus, minus, perfect)):
                        # self.overordering_q = self.overordering_q + 1
                        self.overordering_exp = self.overordering_exp * 1.3
                    else:
                        if(plus < minus):
                            # self.overordering_q = self.overordering_q + 1
                            self.overordering_exp = self.overordering_exp * 1.1
                        else:
                            # self.overordering_q = self.overordering_q
                            self.overordering_exp = self.overordering_exp
                    

                    # amari = my_aim_quantity - sum(best_combo)
                    amari = (offering_quantity - best_score) * max(min(self.overordering_exp, 3), 0.3)

                    amari_agents = [agent for agent in my_negotiators if best_combo[my_negotiators.index(agent)] < 1]

                    # # if( ave_prob < 0.3):
                    # print(amari)
                    if amari > 0:
                        distribution = dict(
                            zip(
                                amari_agents,
                                distribute(
                                    amari,
                                    len(amari_agents),
                                    mx=self.awi.n_lines,
                                    # concentrated=True,
                                    # concentrated_idx=concentrated_idx,
                                    equal=True,
                                    allow_zero=False,
                                ),
                            )
                        )

                        # print(best_combo)
                        # print("distribution:", distribution)
                        best_combo = list(best_combo)
                        for i in range(len(best_combo)):
                            if best_combo[i] < 1:
                                best_combo[i] = distribution[my_negotiators[i]]
                    # print("best_combo:", best_combo)

                else:
                    quantity_options = [agent_to_quantities[agent] for agent in my_negotiators]
                    # print("quantity_options:", quantity_options)
                    for q_combo in product(*quantity_options):  # 各エージェントに対する個数の全パターン
                        # print("q_combo:", q_combo)
                        total_quantity = sum(q_combo)
                        # print("total_quantity:", total_quantity)
                        if total_quantity > max_q:
                            continue
                        if level:
                            score = self.cal_scores(0, total_quantity)
                        else:
                            score = self.cal_scores(total_quantity, 0)
                        # print("score:", score)


                        # 成功確率は precomputed_prob[agent][q]
                        expected_value = 0.0
                        success_prob = 1.0
                        for agent in my_negotiators:
                            q = q_combo[my_negotiators.index(agent)]
                            expected_value += q * precomputed_prob[agent][q]
                            success_prob *= precomputed_prob[agent][q]


                        if (expected_value > best_score):
                            best_score = expected_value
                            best_combo = q_combo
                            best_p = success_prob

                self.preres[now].append(best_score)
                self.preres[now].append(best_combo)
                self.preres[now].append(best_p)

                concentrated_ids = sorted(
                    my_negotiators,
                    key=lambda x: self.total_agreed_quantity[x],
                    reverse=True,
                )[:1]
                # distribution = self.distribute_needs(t=0,mx=self.awi.n_lines,allow_zero=False,concentrated=True,concentrated_ids=concentrated_ids)
                concentrated_idx = [
                    i for i, k in enumerate(my_negotiators) if k in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        my_negotiators,
                        distribute(
                            offering_quantity,
                            len(my_negotiators),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                            # equal=True,
                        ),
                    )
                )
                # print(best_combo, best_score, my_aim_quantity)
                for k, q in distribution.items():
                    # print(self.awi.current_input_issues)
                    distribution[k] = best_combo[my_negotiators.index(k)]


            else:
                distribution = dict(
                    zip(
                        my_negotiators,
                        # distribute(offering_quantity, len(my_negotiators),randomness=True,),
                        distribute(offering_quantity, len(my_negotiators),),
                        # distribute(offering_quantity, len(my_negotiators), equal=True, ),
                    )
                )
        
            for k, q in distribution.items():
                self.first_quantity[q] += 1
                self.first_attempt[k][q] += 1

            # print("distribution:", distribution)
            d |= {
                k: (q, s, price) if q >= 0 or self.awi.allow_zero_quantity else None
                for k, q in distribution.items()
            }
        d |= {k:  (1, s, price) for k in not_negotiators}
        # print("first proposals:", d)
        return d

    def counter_all(self, offers, states):
        self.counter += 1
        self.proposal_state = "counter"
        now = self.awi.current_step
        level = self.awi.is_first_level
        exogenous_input_quantity=self.awi.current_exogenous_input_quantity
        exogenous_input_price=self.awi.current_exogenous_input_price
        exogenous_output_quantity=self.awi.current_exogenous_output_quantity
        exogenous_output_price=self.awi.current_exogenous_output_price

        for k, v in offers.items():
            # if v[TIME] != now:
            #     # print("counter_all: not my offer", k, v)
            # else:
            #     if( v[QUANTITY] >= 0):
            #         self.opo_q[v[QUANTITY]] += 1
            if v[TIME] == now:
                if( v[QUANTITY] >= 0):
                    self.opo_q[v[QUANTITY]] += 1


        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(
                ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
            )
        ) 

        my_aim_quantity = self.awi.needed_sales if self.awi.my_suppliers == ["SELLER"] else self.awi.needed_supplies

        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                my_aim_quantity,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                my_aim_quantity,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            plus_best_diff, plus_best_expected_diff, plus_best_indx = (
                float("inf"),
                float("inf"),
                -1,
            )
            minus_best_diff, minus_best_expected_diff, minus_best_indx = (
                -float("inf"),
                -float("inf"),
                -1,
            )
            best_diff, best_indx = float("inf"), -1

            # print("offers:", offers) 
            # print("needs:", needs)
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff >= 0:  # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if is_selling:  # 売り手の場合は高かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                        else:  # 買い手の場合は安かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:  # 必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(
                            plist[minus_best_indx]
                        ):
                            if is_selling:  # 売り手の場合は高かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) > sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i
                            else:  # 買い手の場合は安かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) < sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i

                plus_scores = -float("inf")
                minus_scores = -float("inf")
                high, low = self.price_count[now][1], self.price_count[now][0]
                now_scores = self.cal_scores(high, low)
                now = self.awi.current_step
                level = self.awi.is_first_level
                exogenous_input_quantity=self.awi.current_exogenous_input_quantity
                exogenous_input_price=self.awi.current_exogenous_input_price
                exogenous_output_quantity=self.awi.current_exogenous_output_quantity
                exogenous_output_price=self.awi.current_exogenous_output_price

                # print(exogenous_input_quantity,exogenous_input_price,exogenous_output_quantity,exogenous_output_price)

                if(len(plist[plus_best_indx]) != 0):
                    high, low = self.price_count[now][1], self.price_count[now][0]
                    for id in plist[plus_best_indx]:
                        if level:
                            ind = offers[id][2] - self.awi.current_output_issues[UNIT_PRICE].min_value
                        else:
                            ind = offers[id][2] - self.awi.current_input_issues[UNIT_PRICE].min_value
                        if(ind == 0):
                            low = low + offers[id][0]
                        else:
                            high = high + offers[id][0]

                    plus_scores = self.cal_scores(high, low)


                if(len(plist[minus_best_indx]) != 0):
                    high, low = self.price_count[now][1], self.price_count[now][0]
                    for id in plist[minus_best_indx]:
                        quantity = self.price_count[now]
                        if level:
                            ind = offers[id][2] - self.awi.current_output_issues[UNIT_PRICE].min_value
                        else:
                            ind = offers[id][2] - self.awi.current_input_issues[UNIT_PRICE].min_value

                        if(ind == 0):
                            low = low + offers[id][0]
                        else:
                            high = high + offers[id][0]

                    minus_scores = self.cal_scores(high, low)

                avarage_calculate_scores = sum(self.calculate_scores) / (now + 1)  
            
            th_min_plus, th_max_plus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[plus_best_indx]).union(future_partners)),
                is_selling,
            )
            th_min_minus, th_max_minus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[minus_best_indx]).union(future_partners)),
                is_selling,
            )

            if th_min_minus <= minus_best_diff or plus_best_diff <= th_max_plus or (states[list(offers.keys())[0]].step > 5):
                if th_min_minus <= minus_best_diff and plus_best_diff <= th_max_plus:
                    if -minus_best_diff == plus_best_diff:
                        if is_selling:  # 売り手のときは、best_diff>0だとshortfall penaltyが発生するのでminus優先
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else:  # 買い手のときは、best_diff<0だとshortfall penaltyが発生するのでplus優先
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif -minus_best_diff < plus_best_diff:
                        # 自身が買い手で、かつ不足分を残りの相手へのCounterOfferで補えないときは、shortfall penaltyを防ぐためplus優先
                        if (
                            not is_selling
                            and len(
                                partners.difference(plist[minus_best_indx]).union(
                                    future_partners
                                )
                            )
                            == 0
                        ):
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                        else:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min_minus and plus_best_diff <= th_max_plus:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx

                best_scores = -float("inf")
                if best_indx != -1:
                    if(best_indx == plus_best_indx):
                        best_scores = plus_scores
                    elif(best_indx == minus_best_indx):
                        best_scores = minus_scores

                if(best_scores != -float("inf")):
                    max_scores = max(plus_scores, minus_scores)
                    average_calculate_scores = sum(self.calculate_scores) / (self.awi.current_step + 1)
                    # print(self.awi.current_step, best_scores, average_calculate_scores)
                    if(best_scores < max_scores):
                        # print(self.awi.current_step, "best_scores < max_scores", best_scores, max_scores, self.awi.needed_sales, self.awi.needed_supplies)
                        if(best_indx == plus_best_indx):
                            best_indx = minus_best_indx
                            best_diff = minus_best_diff
                            best_scores = minus_scores
                            # print("changing socres", best_scores, max_scores, self.awi.current_step, level)
                        elif(best_indx == minus_best_indx):
                            # print("changing socres??", best_scores, max_scores, self.awi.current_step, level)
                            now = self.awi.current_step
                            plus = sum(self.plus) / (now + 1) if now > 0 else 0
                            minus = sum(self.minus) / (now + 1) if now > 0 else 0
                            perfect = sum(self.perfect) / (now + 1) if now > 0 else 0
                            if(perfect == max([minus, plus, perfect]) and minus > plus):
                                best_indx = plus_best_indx
                                best_diff = plus_best_diff
                                best_scores = plus_scores
                                # print("changing socres", best_scores, max_scores, self.awi.current_step, level)


                    accept_q = sum(self.counter_total.values())

                    if(accept_q == 0):
                        if (minus_scores < plus_scores):
                            best_indx =  plus_best_indx
                            best_diff = plus_best_diff
                            best_scores = plus_scores
                        else:
                            best_indx = minus_best_indx
                            best_diff = minus_best_diff
                            best_scores = minus_scores

                        # print("changing socres", best_scores, max_scores, self.awi.current_step, level)

                accept_sum = 0
                for k in plist[best_indx]:
                    accept_sum += offers[k][0]
                
                partner_ids = plist[best_indx]

                partner_ids_tmp = list(partner_ids)
                test_b = True
                if(accept_sum < my_aim_quantity):
                    for p in partner_ids:
                        if offers[p][0] == 1:
                            if(self.counter_total[p] != 0):
                                if(test_b):
                                    partner_ids_tmp.remove(p)
                                    best_diff -= 1
                                    test_b = False

                if(accept_sum < my_aim_quantity):
                    for p in partner_ids:
                        if offers[p][0] == 1:
                            if(self.counter_total[p] == 0):
                                if(test_b):
                                    partner_ids_tmp.remove(p)
                                    test_b = False
                                    best_diff -= 1


                if(states[list(offers.keys())[0]].step < 5):

                    others = list(partners.difference(partner_ids).union(future_partners))
                    partner_ids = tuple(partner_ids_tmp)

                    others = list(partners.difference(partner_ids).union(future_partners))
                    # print("partner_ids:", partner_ids, "others:", others)

                    # print(states)
                else:
                    if(len(partner_ids) == 0):
                        if(now_scores < plus_scores):
                            partner_ids = plist[plus_best_indx]
                            best_diff = plus_best_diff
                        elif(now_scores < minus_scores):
                            partner_ids = plist[minus_best_indx]
                            best_diff = minus_best_diff
                        # print("jikannmoltatimasitayo")
                        # print(now_scores, plus_scores, minus_scores)
                    others = list(partners.difference(partner_ids).union(future_partners))
                    # print("partner_ids:", partner_ids, "others:", others)



   
                for k in partner_ids:
                    self.accept_quantity[offers[k][0]] += 1
                    self.accept[k][offers[k][0]] += 1
                    self.accept_total[k] += offers[k][0]
                    # print("accept:", k, offers[k][0], self.awi.current_step)


                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}

                if (
                    best_diff < 0 and len(others) > 0
                ):  # 必要量に足りないとき、CounterOfferで補う
                    s, p = self._step_and_price(best_price=True)
                    t = min(states[p].relative_time for p in others)
                    offering_quanitity = (
                        int(-best_diff * (1 + self._overordering_fraction(t)))
                        if len(others) > 1
                        else -best_diff
                    )
                    my_aim_quantity = offering_quanitity
                    # print(best_diff)
                    # print("offering_quanitity:", offering_quanitity)
                    if(offering_quanitity < 1):
                        offering_quanitity = 0

                    # agent_to_quantities = {
                    #     agent: [
                    #         i+1 for i in range(my_aim_quantity)
                    #     ]
                    #     for agent in others
                    # }
                    if self.awi.current_step > self.awi.n_steps * 0.5:
     
                        concentrated_ids = sorted(
                            others,
                            key=lambda x: self.counter_total[x],
                            reverse=True,
                        )[:1]
                        concentrated_idx = [
                            i for i, p in enumerate(others) if p in concentrated_ids
                        ]
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity,
                                    len(others),
                                    mx=self.awi.n_lines,
                                    # concentrated=True,
                                    # concentrated_idx=concentrated_idx,
                                    # equal=True
                                ),
                            )
                        )

                        # for k, q in distribution.items():
                        #     # print(self.awi.current_input_issues)
                        #     distribution[k] = best_combo[others.index(k)]


                    else:
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity, len(others), mx=self.awi.n_lines
                                ),
                            )
                        )

                        # distribution = dict(
                        #     zip(
                        #         others,
                        #         distribute(
                        #             offering_quanitity,
                        #             len(others),
                        #             mx=self.awi.n_lines,
                        #             equal=True
                        #         ),
                        #     )
                        # )
                    
                    # print(others)


                    if(self.preoffer == distribution):
                        if (states[list(offers.keys())[0]].step > 2):
                            # print("preoffer is same as distribution")
                            for k, q in distribution.items():
                                distribution[k] = max(1, distribution[k]-1)

                    self.preoffer = distribution

                    for k, q in distribution.items():
                        self.counter_quantity[q] += 1
                        self.counter_attempt[k][q] += 1

                    # print("distribution:", distribution)
                    response.update(
                        {
                            k: (
                                unneeded_response
                                if q == 0
                                else SAOResponse(ResponseType.REJECT_OFFER, (q, s, p))
                            )
                            for k, q in distribution.items()
                        }
                    )

                continue




            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            # distribution = self.distribute_needs(t)

            partners = partners.union(future_partners)
            partners = list(partners)
            offering_quanitity = (
                int(needs * (1 + self._overordering_fraction(t)))
                if len(partners) > 1
                else needs
            )

            my_aim_quantity = offering_quanitity


            if self.awi.current_step > self.awi.n_steps * 0.5 and len(partners) > 0:
              

                concentrated_ids = sorted(
                    partners, key=lambda x: self.counter_total[x], reverse=True
                )[:1]
                concentrated_idx = [
                    i for i, p in enumerate(partners) if p in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            len(partners),
                            mx=self.awi.n_lines,
                            # concentrated=True,
                            # concentrated_idx=concentrated_idx,
                            equal=True,
                        ),
                    )
                )

                delete_list = []
                for k, q in distribution.items():
                    if(self.counter_total[k] == 0):
                        delete_list.append(k)


                plist = list(powerset(delete_list))[::-1]

                high, low = self.price_count[now][1], self.price_count[now][0]
                now_scores = self.cal_scores(high, low)
                best_scores, best_indx = now_scores, -1

                level = self.awi.is_first_level
                for i, partner_ids in enumerate(plist):
                    high, low = self.price_count[now][1], self.price_count[now][0]

                    for id in partner_ids:
                        if level:
                            ind = offers[id][2] - self.awi.current_output_issues[UNIT_PRICE].min_value
                        else:
                            ind = offers[id][2] - self.awi.current_input_issues[UNIT_PRICE].min_value
                        if(ind == 0):
                            low = low + offers[id][0]
                        else:
                            high = high + offers[id][0]

                    scores = self.cal_scores(high, low)

                    if( scores > best_scores):
                        best_scores = scores
                        best_indx = i

                # print("best_indx:", best_indx, "best_scores:", best_scores, "plist:", plist[best_indx])
                
                for k in plist[best_indx]:
                    response.update(
                        {
                            k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        }
                    )
                    self.accept_quantity[offers[k][0]] += 1
                    self.accept[k][offers[k][0]] += 1
                    self.accept_total[k] += offers[k][0]
                    # print("sudden_accept:", k, offers[k][0], self.awi.current_step)
                    distribution.pop(k)

                # print("distribution:", distribution)


                # for k, q in distribution.items():
                #     # print(self.awi.current_input_issues)
                #     distribution[k] = best_combo[partners.index(k)]
            else:
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity, len(partners), mx=self.awi.n_lines, equal=True
                        ),
                    )
                )

            # print(partners)



            if(self.preoffer == distribution):
                if (states[list(offers.keys())[0]].step > 2):
                    pass # print("preoffer is same as distribution")
                    for k, q in distribution.items():
                        distribution[k] = max(1, distribution[k]-1)

            self.preoffer = distribution

            # print("distribution:", distribution)
            # print(offers)

            for k, q in distribution.items():
                self.counter_quantity[q] += 1
                self.counter_attempt[k][q] += 1

            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        # print("counter_all response:", response)
        return response

    # def _allowed_mismatch(self, r:float, n_others:int, is_selling:bool):
    #     return self.undermismatch_min * ((1-r)**self.mismatch_exp), self.overmismatch_max * (r**self.mismatch_exp)

    def _allowed_mismatch(self, r: float, n_others: int, is_selling: bool):
        #     if is_selling:
        #         # 入荷量が一定、過剰な販売契約が良くない
        #         # 一人平均3個くらいの算段でOK?
        #         th_min = - 3 * n_others
        #         th_max = self.overmismatch_max_selling * (r**self.mismatch_exp)
        #     else:
        #         # 不足が良くない、n_othersが多いほどth_minが小さくても(不足が多くても)良い
        #         # 一人平均1.5個くらいの算段でOK?
        #         th_min = - 1.5 * n_others
        #         th_max = self.overmismatch_max_buying * (r**self.mismatch_exp)
        #     return th_min,th_max
        undermismatch_min = (
            self.undermismatch_min_selling
            if is_selling
            else self.undermismatch_min_buying
        )
        overmismatch_max = (
            self.overmismatch_max_selling
            if is_selling
            else self.overmismatch_max_buying
        )
        return undermismatch_min * ((1 - r) ** self.mismatch_exp), overmismatch_max * (
            r ** (1 / self.mismatch_exp)
        )

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

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
            # return s, pmin if seller else pmax
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)
