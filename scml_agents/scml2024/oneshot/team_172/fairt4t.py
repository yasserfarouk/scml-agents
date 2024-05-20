#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""

from __future__ import annotations

from itertools import chain, combinations

# required for typing
from typing import Any

from numpy import random

# required for development
from scml.oneshot import QUANTITY, UNIT_PRICE, OneShotAWI, OneShotSyncAgent

# from scml_agents import get_agents
# from scml.oneshot import SCML2024OneShotWorld
# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

from numpy.random import choice
from collections import Counter

__all__ = ["FairT4T"]
# from scml_agents import get_agents


class FairT4T(OneShotSyncAgent):
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

    def distribute_needs(self) -> dict[str, int]:
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
            self.myneeds = needs
            # distribute my needs over my (remaining) partners.
            dist.update(dict(zip(partner_ids, self.distribute(needs, partners))))

        return dist

    def receive_offers(self, offers):
        # Update opponent_last_bid with the latest bids from negotiation partners
        for k, offer in offers.items():
            if offer is not None:
                self.opponents_last_bid[k] = offer  # [QUANTITY]

        # keep the utility from opponents' previous offer
        self.utility_opp_prev_offers = self.ufun.from_offers(list(offers.values()))

    def calculate_target_utility(self, offers):
        s = self.awi.current_step
        # estimate target utility to be able to decide on the offer to generate if the offers' utility is closest to target utility
        deltaU = (
            self.ufun.from_offers(list(offers.values())) - self.utility_opp_prev_offers
        )
        # if it is a positive chage, agent should concede as well.Hence it should reach at a target parameter
        self.target_utility = (
            self.utility_agent_prev_offer - deltaU * self.utility_parameter
        )

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()

        d = {}
        utility_new_offer = {}  # for each partner
        # print(f"Step: {s} , in proposal, and my needs {self.myneeds}")

        # Iterate over negotiation partners
        for partner_id, quantity in distribution.items():
            # Check if the partner has made any offers
            if partner_id in self.opponents_last_bid:
                # print(f"partner did made offer")

                # If the partner has made an offer, mirror their last bid
                last_bid = self.opponents_last_bid[partner_id]
                # print(f"Step {s} OFFER FROM partner partner{partner_id} distribute quantity: {quantity}  whose last bid was {last_bid[QUANTITY]}")
                # Calculate the threshold for mirroring the opponent's bid
                # by defining a cooperative factor, if we want to be more cooperative we will lower our threshold,
                # and increase it if we want to prioritize our gains
                cooperative_factor = 0.4

                # We are adjusting the threshold based on the cooperative factor
                adjusted_threshold = self.threshold * cooperative_factor
                threshold_quantity = adjusted_threshold * quantity
                if last_bid[QUANTITY] >= threshold_quantity:
                    d[partner_id] = last_bid
                # print(f"this bid is better imitate: {d[partner_id]}")
                else:
                    # If the opponent's bid does not meet the threshold, make a new offer
                    d[partner_id] = (quantity, s, p) if quantity > 0 else None
            # print(f"generate new one: {d[partner_id]}")
            else:
                # If the partner hasn't made an offer, make a random offer
                d[partner_id] = (quantity, s, p) if quantity > 0 else None
                # print(f"Step {s} no offer yet from  for partner{partner_id} offer: quantity {quantity} ")
            new_offer = d[partner_id]
            new_offer_filtered = {k: v for k, v in d.items() if v is not None}
            utility_new_offer[partner_id] = self.ufun.from_offers(new_offer_filtered)

        # after getting all the utility for the  upcoming offers, lets calculate target utility for each
        closest_to_target, pid = float("inf"), []
        for p, utility in utility_new_offer.items():
            # print(f"AND UTILITY with new offer from partner {p} is {utility}")
            # print(f"AND TARGET UTILITY at that time is {self.target_utility}")

            diff = abs(self.target_utility - utility)
            # Check if the utility of the new offer meets the target utility
            if diff == closest_to_target:
                pid.append(p)
            elif diff < closest_to_target:
                closest_to_target = diff
                pid.append(p)

            # print(f"closest_to_target: {closest_to_target} with partnners {pid}")

        d_accepted = {key: d[key] for key in pid if key in d}
        d_filtered = {k: v for k, v in d_accepted.items() if v is not None}
        # print("selected offers ", d_filtered)

        self.utility_upcoming_offer = self.ufun.from_offers(d_filtered)
        # print(f"at step {s}  utility agent upcoming offer is {self.utility_upcoming_offer} ")
        self.utility_agent_prev_offer = self.ufun.from_offers(d_filtered)

        return d_filtered

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """

        response = dict()

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
            # print("partners: ", len(partners))
            # If the set of offers is greater than upcoming offer from agent, accept them and end all
            # other negotiations

            # Apply ACnext strategy:
            # find the set of partners that gave me the best offer set
            # (i.e.  with utility better than the upcoming offer by me)
            plist = list(self.powerset(partners))

            max_u, best_indx = float("-inf"), -1
            for i, partner_ids in enumerate(plist):
                subset_offers = {}
                # print(f"for {i} partner ids {partner_ids}")
                for p in partner_ids:
                    subset_offers[p] = offers[p]

                utility_opp_off = self.ufun.from_offers(subset_offers)

                if max_u < utility_opp_off:
                    max_u, best_indx = utility_opp_off, i

            # print(f" max_utility_opp_off {max_u} while upcoming offer is {self.utility_upcoming_offer}")
            ac_next = self.alpha * max_u + self.beta >= self.utility_upcoming_offer
            offers_to_accept = {}

            if ac_next:
                partner_ids = plist[best_indx]
                others = partners.difference(partner_ids)
                for partner_id in partner_ids:
                    offers_to_accept[partner_id] = offers[partner_id]

                self.calculate_target_utility(offers_to_accept)
                self.receive_offers(offers_to_accept)
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            distribution = self.distribute_needs()
            response.update(
                {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                    )
                    for k, q in distribution.items()
                }
            )

        accepted_offer_count = self.calculate_accept_count(response)

        self.accepted_negotiations += accepted_offer_count
        # print(f"total accepted offers at step {self.awi.current_step} is {self.accepted_negotiations}")
        # print("acceptance rate :", self.calculate_acceptance_rate())
        # self.ind_utilities  = self.calculate_utility(response)
        # self.total_utility += self.calculate_utility(response)
        # print("Mean of ind utilities",self.total_utility/self.total_negotiations)
        return response

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self, *args, threshold=0.8, utility_threshold=0.5, **kwargs):
        super().init(*args, **kwargs)
        self.quantity_secured = 0
        self.threshold = threshold
        self.utility_parameter = utility_threshold
        self.opponents_last_bid = {}
        self.utility_agent_prev_offer = 0
        self.utility_opp_prev_offers = 0
        self.target_utility = 0
        self.utility_upcoming_offer = 0
        self.alpha = 1.2
        self.beta = 15
        self.myneeds = 0
        self.neg_quantity_history = [-1 for _ in range(self.awi.n_steps)]
        self.total_utility = 0
        self.ind_utilities = {}
        self.total_negotiations = 0
        self.accepted_negotiations = 0

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        super().before_step()
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

    def step(self):
        """Called at at the END of every production step (day)"""
        self.neg_quantity_history[self.awi.current_step] = self.quantity_secured
        # reset quantity_secured value for next step
        self.quantity_secured = 0

    def calculate_utility(self, response_dict):
        accepted_outcomes = []
        total_utility_fromaccepted = 0
        utility = {}
        for agent_id, response in response_dict.items():
            if response.response == ResponseType.ACCEPT_OFFER:
                accepted_outcomes.append(response.outcome)

        total_utility_fromaccepted = self.ufun.from_offers(accepted_outcomes)
        # print(f"individual utility at step {self.awi.current_step}: {total_utility_fromaccepted}")
        utility[self.awi.current_step] = total_utility_fromaccepted

        return total_utility_fromaccepted

    def calculate_accept_count(self, response_dict):
        accept_count = 0
        for agent_id, response in response_dict.items():
            if response.response == ResponseType.ACCEPT_OFFER:
                accept_count += 1
            self.total_negotiations += 1
        return accept_count

    def calculate_acceptance_rate(self):
        if self.total_negotiations == 0:
            return 0
        return self.accepted_negotiations / self.total_negotiations

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

    def is_seller(self, negotiator_id):
        return negotiator_id in self.awi.current_negotiation_details["sell"].keys()

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return mn + (mx - mn) * (r**4.0)

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

    def _needed(self, negotiator_id=None):
        return (
            self.awi.needed_sales
            if self.is_seller(negotiator_id)
            else self.awi.needed_supplies
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def distribute(self, q: int, n: int) -> list[int]:
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

    from itertools import chain, combinations

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([FairT4T], sys.argv[1] if len(sys.argv) > 1 else "oneshot")

    # winners = [
    #     get_agents(y, track="oneshot", winners_only=True, as_class=False)[1]
    #     for y in (2021, 2022, 2023)
    # ]
    #
    # world = SCML2024OneShotWorld(**SCML2024OneShotWorld.generate(
    #     agent_types=winners + [Group5], n_steps=50
    # ), construct_graphs=False, )
    #
    # world.run()
    #
    # scores = world.scores()
    #
    # # Map agent IDs to names
    # id_to_name = {agent.id: agent.type_name for agent in world.agents.values()}
    # # Sort agents based on scores for ranking
    # sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #
    # scores_by_type = {}
    #
    # # Group scores by agent type
    # for agent_id, score in scores.items():
    #     agent_type = id_to_name.get(agent_id, "Unknown")
    #     # Split the string by '.'
    #     parts = agent_type.split('.')
    #     # Take the last part of the split string
    #     agent_name = parts[-1]
    #     if agent_name not in scores_by_type:
    #         scores_by_type[agent_name] = []
    #     scores_by_type[agent_name].append(score)
    #
    # # Calculate statistics for each agent type
    # statistics_by_type = {}
    # for agent_type, agent_scores in scores_by_type.items():
    #     statistics_by_type[agent_type] = {
    #         "Mean": np.mean(agent_scores),
    #         "Std": np.std(agent_scores),
    #         "25%": np.percentile(agent_scores, 25),
    #         "75%": np.percentile(agent_scores, 75)
    #         # You can add more statistics here as needed
    #     }# Calculate mean
    #
    # df = pd.DataFrame.from_dict(statistics_by_type, orient='index')
    # print(df)
    #
    #
    # for rank, (agent_id, score) in enumerate(sorted_scores, start=1):
    #     agent_name = id_to_name.get(agent_id, "Unknown")  # Use "Unknown" if name is not found
    #     print(f"{rank}. {agent_name}: {score}")
