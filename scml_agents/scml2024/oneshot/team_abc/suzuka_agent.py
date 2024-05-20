import math

from negmas import ResponseType, Outcome, SAOState, SAOResponse
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotSyncAgent

__all__ = ["SuzukaAgent"]


class SuzukaAgent(OneShotSyncAgent):
    """Based on OneShotSyncAgent"""

    def init(self):
        """Initializes some values needed for the negotiations"""
        self.is_seller = True if self.awi.profile.level == 0 else False
        if self.awi.profile.level == 0:
            self.level = 0
        else:
            self.level = 1

        # renew
        self.compromise_point = 0.6
        self.pursue_points = 0.4
        self.max_compromise_need = 10
        self.final_needs_list = []
        self.acceptance_level = 0

    def before_step(self):
        self.secured = 0
        self.first_needs = (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
        )

        self.current_acceptance_level = self.acceptance_level

    def step(self):
        self.final_needs_list.append(abs(self.first_needs - self.secured))

        # renew acceptance level
        if sum(abs(x) for x in self.final_needs_list) > self.compromise_point * 10:
            self.acceptance_level = min(
                [self.acceptance_level + 2, self.max_compromise_need]
            )
            self.final_needs_list = []
        elif len(self.final_needs_list) >= 10:
            if sum(abs(x) for x in self.final_needs_list) < self.pursue_points * 10:
                self.acceptance_level = max([self.acceptance_level - 2, 0])
            self.final_needs_list = []

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def get_proposals(self, partners, needs):
        step = self.awi.current_step
        level = self.level
        n_of_partners = len(partners)

        offer_price = self.best_price()
        if n_of_partners == 1:
            propose_quantity = needs
        else:
            propose_quantity = math.ceil(needs / 2)

        partners_quantities = {partner: propose_quantity for partner in partners}

        proposal_dict = {
            partner: self.get_outcome(offer_price, quantity, step)
            for partner, quantity in partners_quantities.items()
        }
        return proposal_dict

    def first_proposals(self) -> dict[str, Outcome | None]:
        partners = set(self.partners())
        needs = self.needed_quantity()
        return self.get_proposals(partners, needs)  # type: ignore

    def counter_all(
        self, offers: dict[str, Outcome | None], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        current_step = list(states.values())[0].step
        needs = self.needed_quantity()
        all_partners = set(offers.keys())

        if needs <= 0:
            return {
                agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for agent in all_partners
            }

        else:
            m = len(all_partners)
            up_quantity_list = []
            voffers = list(offers.values())
            for i in range(m):
                if voffers[i] is None:
                    up_quantity_list.append([0, 0])
                    continue
                up_quantity_list.append(
                    [
                        voffers[i][UNIT_PRICE],  # type: ignore
                        voffers[i][QUANTITY],  # type: ignore
                    ]
                )
            sum_quantity = 0
            sum_price = 0
            compromised_false_point = 0
            decide_choice_number = 0
            disposal = self.awi.current_disposal_cost
            shortfall = self.awi.current_shortfall_penalty
            benefit_of_final_days = -10000

            # decide acceptance offers
            for choice_number in range(pow(2, m)):
                false_point = 0
                current_price = 0
                current_quantity = 0
                for i in range(m):  # culculate offers information
                    if (choice_number // pow(2, i)) % 2 == 1:
                        if (self.level == 1) and (
                            up_quantity_list[i][0] != self.best_price()
                        ):
                            false_point += up_quantity_list[i][1]
                        current_quantity += up_quantity_list[i][1]
                        current_price += up_quantity_list[i][0] * up_quantity_list[i][1]

                if current_step < 19:
                    if (
                        (false_point <= self.current_acceptance_level)
                        or (current_step == 18)
                    ) and (current_quantity <= needs):
                        if current_quantity > sum_quantity:
                            sum_quantity = current_quantity
                            sum_price = current_price
                            decide_choice_number = choice_number
                            compromised_false_point
                        elif current_quantity == sum_quantity:
                            previous_abs = abs(
                                sum_price - self.best_price() * sum_quantity
                            )
                            current_abs = abs(
                                current_price - self.best_price() * current_quantity
                            )
                            if current_abs < previous_abs:
                                sum_price = current_price
                                decide_choice_number = choice_number
                                compromised_false_point = false_point
                else:
                    if needs >= current_quantity:
                        if (self.level == 0) and (
                            benefit_of_final_days
                            < current_price - (needs - current_quantity) * disposal
                        ):
                            sum_quantity = current_quantity
                            sum_price = current_price
                            decide_choice_number = choice_number
                            compromised_false_point = false_point
                        elif (self.level == 1) and (
                            benefit_of_final_days
                            < -current_price - (needs - current_quantity) * shortfall
                        ):
                            sum_quantity = current_quantity
                            sum_price = current_price
                            decide_choice_number = choice_number
                            compromised_false_point = false_point
                    else:
                        if (self.level == 0) and (
                            benefit_of_final_days
                            < current_price - (needs - current_quantity) * shortfall
                        ):
                            sum_quantity = current_quantity
                            sum_price = current_price
                            decide_choice_number = choice_number
                            compromised_false_point = false_point
                        elif (self.level == 1) and (
                            benefit_of_final_days
                            < -current_price - (needs - current_quantity) * disposal
                        ):
                            sum_quantity = current_quantity
                            sum_price = current_price
                            decide_choice_number = choice_number
                            compromised_false_point = false_point

            self.current_acceptance_level -= compromised_false_point

            acceptable_partners = set()
            keys = list(offers.keys())
            for i in range(m):
                if (decide_choice_number // pow(2, i)) % 2 == 1:
                    acceptable_partners.add(keys[i])

            acceptance_dict: dict[str, SAOResponse] = {
                agent: SAOResponse(ResponseType.ACCEPT_OFFER, None)
                for agent in acceptable_partners
            }

            # decide reject or end to others
            if (needs == sum_quantity) or (
                len(all_partners - acceptable_partners) == 0
            ):
                end_partners = all_partners - acceptable_partners
                end_negotiation_dict: dict[str, SAOResponse] = {
                    agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for agent in end_partners
                }
                result = acceptance_dict | end_negotiation_dict
            else:
                reject_partners = all_partners - acceptable_partners
                agent_proposals = self.get_proposals(
                    reject_partners, needs - sum_quantity
                )
                reject_dict: dict[str, SAOResponse] = {
                    agent: SAOResponse(ResponseType.REJECT_OFFER, proposal)
                    for agent, proposal in agent_proposals.items()
                }
                result = acceptance_dict | reject_dict
            return result

    def partners(self) -> list[str]:
        return self.awi.my_consumers if self.is_seller else self.awi.my_suppliers

    def needed_quantity(self):
        return self.first_needs - self.secured

    def get_price_range(self) -> tuple[int, int]:
        if self.is_seller:
            price_issue = self.awi.current_output_issues[UNIT_PRICE]
        else:
            price_issue = self.awi.current_input_issues[UNIT_PRICE]
        return price_issue.min_value, price_issue.max_value

    def best_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mx
        else:
            return mn

    def worst_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mn
        else:
            return mx

    def get_outcome(self, unit_price: int, quantity: int, time: int) -> Outcome:
        offer = [0, 0, 0]
        offer[UNIT_PRICE] = unit_price
        offer[QUANTITY] = quantity
        offer[TIME] = time
        return tuple(offer)
