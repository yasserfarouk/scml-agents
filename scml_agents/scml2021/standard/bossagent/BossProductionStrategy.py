import copy

import numpy as np
from scml.scml2020.common import ANY_LINE, QUANTITY, TIME, UNIT_PRICE

from .helper import (
    calculate_contract_quantity,
    format_schedule,
    get_contract_buyer_sellers,
    get_unscheduled_total_pseudo_quantity,
    is_schedule_available,
    sort_negotiators_by_delivery,
)


class BossProductionStrategy:
    """
    Production strategy class.
    """

    def init(self):
        self.cumulative_signed_buyers = 0
        self.cumulative_signed_sellers = 0
        self.cumulative_produced_q = 0
        # Initialize other components.
        super().init()

    def on_contracts_finalized(self, signed, cancelled, rejectors):
        """
        Plan schedule when contracts are finalized.
        """

        """
			self.scheduled_buyer_contracts: {buyer_contract_id: {seller_contract_id: ((q, t, p), schedule), seller_contract2_id: ((q, t, p), schedule2) }}
			self.scheduled_seller_contracts: {seller_contract_id: {buyer_contract_id: ((q, t, p), schedule), buyer_contract2_id: ((q, t, p), schedule2) }}
			schedule: {step1: quantity, step2: quantity, step3: quantity}

			scheduled buyer contracts ve scheduled seller contracts defaultdict olmali. her yeni schedule productionda eklemeli, sifirdan yaratamayiz.


			signed: [contract1, contract2, contract3]
			cancelled:  [contract1, contract2, contract3]

			signed_buyers = {contract_1_id: (q, t, p), contract_2_id: (q, t, p)}
			signed_sellers = {contract_1_id: (q, t, p), contract_2_id: (q, t, p)}

			self.pseudo_buyer: {buyer_contract_id: (q, t, p) }
			self.pseudo_seller: {seller_contract_id: (q, t, p) }

			dispatch -> dispatch_partners[buyer_contract_id][seller_contract_id] = schedule -> schedule: {'1': 3, '2', 5, '3': 7}

			dispatch_partners = {buyer_contract_id: {seller_contract_id: schedule} }
		"""
        self.current_step = self.awi_caller.get_current_step()
        # Get the schedule.
        awi_schedule = self.awi.available_for_production(
            repeats=1,
            step=(self.current_step, self.max_number_of_steps),
            line=-1,
            override=False,
        )[0]
        formatted_awi_schedule = format_schedule(awi_schedule, self.max_number_of_steps)
        formatted_awi_schedule[self.current_step] = list(
            self.awi.state.commands[self.current_step]
        ).count(-1)

        # Get signed buyer and sellers from contracts.
        signed_buyers, signed_sellers = get_contract_buyer_sellers(self.id, signed)
        # Merge pseudo buyer & sellers with signed buyer & sellers.
        to_sort_buyers = {**self.pseudo_buyers, **signed_buyers}
        to_sort_sellers = {**self.pseudo_sellers, **signed_sellers}
        # Sort pseudo buyer & sellers by delivery date. Since we dont want to breach.
        sorted_buyers = sort_negotiators_by_delivery(to_sort_buyers)
        sorted_sellers = sort_negotiators_by_delivery(to_sort_sellers)
        # Send pseudo + signed ones to dispatch.
        dispatch_categories, dispatch_partners = self.dispatch_module.run_dispatch(
            sorted_buyers, sorted_sellers, formatted_awi_schedule
        )

        # Schedule partners buyers & sellers.
        for buyer_contract_id, seller_contract_details in dispatch_partners.items():
            for seller_contract_id, schedule in seller_contract_details.items():
                if (
                    seller_contract_id != "KEEP"
                ):  # If seller is keep, we dont need to schedule anything.
                    for schedule_step, quantity_to_produce in schedule.items():
                        if quantity_to_produce > 0:
                            if schedule_step != self.current_step:
                                steps, lines = self.awi.schedule_production(
                                    process=self.my_input_product,
                                    repeats=quantity_to_produce,
                                    step=schedule_step,
                                    line=-1,
                                    partial_ok=True,
                                    override=False,
                                    method="earliest",
                                )
                            else:
                                temp_commands = copy.deepcopy(
                                    self.awi.state.commands[self.current_step]
                                )
                                for q in range(quantity_to_produce):
                                    for index, command in enumerate(temp_commands):
                                        if command == -1:
                                            temp_commands[index] = self.my_input_product
                                            break
                                self.awi.set_commands(
                                    temp_commands, step=self.current_step
                                )

                seller_offer = to_sort_sellers[seller_contract_id]
                buyer_offer = to_sort_buyers[buyer_contract_id]
                # Create empty dict for buyer id if it does not exist.
                if buyer_contract_id not in self.scheduled_buyer_contracts.keys():
                    self.scheduled_buyer_contracts[buyer_contract_id] = {}
                # Create empty dict for seller id if it does not exist.
                if seller_contract_id not in self.scheduled_seller_contracts.keys():
                    self.scheduled_seller_contracts[seller_contract_id] = {}
                # Check whether we matched the same buyer & seller before, if so update, otherwise just create a new tuple.
                # Important for surplus sellers.
                # Check update or creating for scheduled buyer contracts.
                if (
                    seller_contract_id
                    not in self.scheduled_buyer_contracts[buyer_contract_id].keys()
                ):
                    self.scheduled_buyer_contracts[buyer_contract_id][
                        seller_contract_id
                    ] = (seller_offer, schedule)
                    self.scheduled_seller_contracts[seller_contract_id][
                        buyer_contract_id
                    ] = (buyer_offer, schedule)
                else:
                    current_offer = self.scheduled_buyer_contracts[buyer_contract_id][
                        seller_contract_id
                    ][0]
                    current_schedule = copy.deepcopy(
                        self.scheduled_buyer_contracts[buyer_contract_id][
                            seller_contract_id
                        ][1]
                    )
                    combined_steps = current_schedule.keys() | schedule.keys()
                    updated_schedule = {
                        step: current_schedule.get(step, 0) + schedule.get(step, 0)
                        for step in combined_steps
                    }
                    self.scheduled_buyer_contracts[buyer_contract_id][
                        seller_contract_id
                    ] = (seller_offer, updated_schedule)
                    self.scheduled_seller_contracts[seller_contract_id][
                        buyer_contract_id
                    ] = (buyer_offer, updated_schedule)

        # Pop fixed pseudo buyers & sellers from the dict.
        for green_contract_id in dispatch_categories["Green"].keys():
            if green_contract_id in self.pseudo_buyers:
                self.pseudo_buyers.pop(green_contract_id)
            elif green_contract_id in self.pseudo_sellers:
                self.pseudo_sellers.pop(green_contract_id)
        # Pop fixed pseudo buyers & sellers from the dict.
        for yellow_contract_id, offer_content in dispatch_categories["Yellow"].items():
            if yellow_contract_id in self.pseudo_buyers:
                self.pseudo_buyers.pop(yellow_contract_id)
            elif yellow_contract_id in self.pseudo_sellers:
                self.pseudo_sellers.pop(yellow_contract_id)
            self.pseudo_sellers[yellow_contract_id] = {}
            surplus_offer = (
                offer_content["Surplus"],
                offer_content["Offer"][1],
                offer_content["Offer"][2],
            )
            self.pseudo_sellers[yellow_contract_id] = surplus_offer
        # Add unscheduled signs to pseudo buyers & sellers.
        for red_contract_id, offer_content in dispatch_categories["Red"].items():
            if red_contract_id in list(to_sort_buyers.keys()):
                self.pseudo_buyers[red_contract_id] = copy.deepcopy(
                    offer_content["Offer"]
                )
            elif red_contract_id in list(to_sort_sellers.keys()):
                offer_delivery = max(self.current_step + 1, offer_content["Offer"][1])
                offer = (
                    offer_content["Offer"][0],
                    offer_delivery,
                    offer_content["Offer"][2],
                )
                self.pseudo_sellers[red_contract_id] = offer

        # Update nego history.
        for buyer, offer in signed_buyers.items():
            self.buyer_history_prices.append(offer[UNIT_PRICE])
            self.buyer_history_delivery_times.append(offer[TIME])
            self.buyer_history_quantities.append(offer[QUANTITY])

        for seller, offer in signed_sellers.items():
            self.seller_history_prices.append(offer[UNIT_PRICE])
            self.seller_history_delivery_times.append(offer[TIME])
            self.seller_history_quantities.append(offer[QUANTITY])

        self.update_constraints()

        # Get the schedule.
        awi_schedule = self.awi.available_for_production(
            repeats=1,
            step=(self.current_step, self.max_number_of_steps),
            line=-1,
            override=False,
        )[0]
        formatted_awi_schedule = format_schedule(awi_schedule, self.max_number_of_steps)

        self.cumulative_signed_buyers += calculate_contract_quantity(signed_buyers)
        self.cumulative_signed_sellers += calculate_contract_quantity(signed_sellers)

        self.cumulative_produced_q += 10 - formatted_awi_schedule[self.current_step]

        for signed_contract in signed:
            agent_sign = 1
            opp_sign = 1

            contract_partners = list(copy.deepcopy(signed_contract.partners))
            contract_partners.remove(self.id)
            # Get agent id from contract.
            agent_id = contract_partners[0]

            if signed_contract.mechanism_id is not None:
                self.nego_stats.set_negotiation_bid_sign(
                    self.current_step,
                    agent_id,
                    signed_contract.mechanism_id,
                    agent_sign,
                    opp_sign,
                )

        for contract_index, cancelled_contract in enumerate(cancelled):
            contract_partners = list(copy.deepcopy(cancelled_contract.partners))
            if self.id in rejectors[contract_index]:
                agent_sign = 0
            else:
                agent_sign = 1
            contract_partners.remove(self.id)
            # Get agent id from contract.
            agent_id = contract_partners[0]
            if agent_id in rejectors[contract_index]:
                opp_sign = 0
            else:
                opp_sign = 1

            if cancelled_contract.mechanism_id is not None:
                self.nego_stats.set_negotiation_bid_sign(
                    self.current_step,
                    agent_id,
                    cancelled_contract.mechanism_id,
                    agent_sign,
                    opp_sign,
                )

            if (
                agent_id in rejectors[contract_index]
                and agent_id in self.reject_acceptance_rate.keys()
                and agent_sign == 1
            ):
                if (
                    self.current_step
                    in self.reject_acceptance_rate[agent_id]["Reject"].keys()
                ):
                    self.reject_acceptance_rate[agent_id]["Reject"][
                        self.current_step
                    ] += cancelled_contract.agreement["quantity"]
                else:
                    self.reject_acceptance_rate[agent_id]["Reject"][
                        self.current_step
                    ] = cancelled_contract.agreement["quantity"]

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the
        agent to change what is to be produced in its factory on that step.
        """
        return commands
