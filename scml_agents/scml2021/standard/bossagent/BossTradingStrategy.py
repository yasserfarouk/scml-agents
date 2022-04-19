import copy
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

from negmas import Breach, Contract
from negmas.common import AgentMechanismInterface, MechanismState
from scml.scml2020.common import QUANTITY

from .helper import (
    calculate_current_keep_amount,
    calculate_total_keep_amount,
    format_schedule,
    get_contract_buyer_sellers,
    get_unscheduled_total_pseudo_quantity,
    is_schedule_available,
    schedule_dispatch_production,
    sort_buyers_by_price,
    sort_negotiators_by_delivery,
    sort_negotiators_by_descending_delivery,
    sort_sellers_by_price,
)


class BossTradingStrategy:
    def init(self):
        super().init()

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        # Sort pseudo buyer & sellers by delivery date. Since we dont want to breach.
        sorted_pseudo_buyers = sort_negotiators_by_delivery(self.pseudo_buyers)
        sorted_pseudo_sellers = sort_negotiators_by_delivery(self.pseudo_sellers)
        # Get unsigned buyer and sellers from contracts.
        buyers, sellers = get_contract_buyer_sellers(self.id, contracts)
        # Sort unsigned buyer and sellers by price.
        sorted_buyers = sort_buyers_by_price(buyers)
        sorted_sellers = sort_sellers_by_price(sellers)
        # Give prio to pseudo ones bc of breach.
        to_dispatch_buyers = sorted_pseudo_buyers + sorted_buyers
        to_dispatch_sellers = sorted_pseudo_sellers + sorted_sellers
        # Send pseudo + unsigned ones to dispatch.
        dispatch_categories, dispatch_partners = self.dispatch_module.run_dispatch(
            to_dispatch_buyers, to_dispatch_sellers, self.formatted_schedule
        )
        # Sign green, and yellow ones. Reject red ones.
        signs = []
        if self.awi_caller.is_first_level():
            # Sort the buyers by delivery.
            delivery_sorted_sellers = sort_negotiators_by_delivery(sellers)
            signs = [None] * len(contracts)
            # Get current schedule.
            awi_schedule = self.awi.available_for_production(
                repeats=1,
                step=(self.current_step + 1, self.max_number_of_steps - 1),
                line=-1,
                override=False,
            )[0]
            formatted_temp_schedule = format_schedule(
                awi_schedule, self.max_number_of_steps
            )
            # Get green agent ids.
            green_contract_ids = list(dispatch_categories["Green"].keys())
            yellow_contract_ids = list(dispatch_categories["Yellow"].keys())
            red_contract_ids = list(dispatch_categories["Red"].keys())
            # Current signed amount.
            current_signed_amount = 0
            # Get seller contract id.
            contract_ids = [c.id for c in contracts]
            # Temp schedule & sign partners (buyers & sellers).
            for buyer_contract_id, seller_contract_details in dispatch_partners.items():
                for seller_contract_id, schedule in seller_contract_details.items():
                    # If seller is keep, we dont need to schedule or sign anything.
                    if seller_contract_id != "KEEP":
                        for schedule_step, quantity_to_produce in schedule.items():
                            formatted_temp_schedule, _ = schedule_dispatch_production(
                                formatted_temp_schedule,
                                quantity_to_produce,
                                schedule_step,
                                schedule_step + 1,
                            )
                        if seller_contract_id in contract_ids:
                            seller_index = contract_ids.index(seller_contract_id)
                            # Sign seller contract.
                            signs[seller_index] = self.id
                # Sign the deal if the buyer is not keep.
                if buyer_contract_id != "KEEP" and buyer_contract_id in contract_ids:
                    # Get buyer contract id.
                    contract_ids = [c.id for c in contracts]
                    buyer_index = contract_ids.index(buyer_contract_id)
                    # Sign buyer contract.
                    signs[buyer_index] = self.id
            # Check signing for red buyers.
            for (contract_id, offer) in delivery_sorted_sellers:
                # Check if schedule is available or not.
                is_available, _ = is_schedule_available(
                    formatted_temp_schedule,
                    offer[0],
                    offer[1],
                    self.max_buyer_delivery_day,
                )
                # Get contract index.
                contract_index = contract_ids.index(contract_id)
                contract = contracts[contract_index]
                contract_partners = list(copy.deepcopy(contract.partners))
                contract_partners.remove(self.id)
                # Get agent id from contract.
                agent_id = contract_partners[0]
                # Check availability and sign feasible red buyer contracts.
                if (
                    is_available
                    and (contract_id in red_contract_ids)
                    and (agent_id == "SELLER")
                ):
                    # Check if this contract is feasible.
                    if (
                        contract.agreement["quantity"]
                        + current_signed_amount
                        + get_unscheduled_total_pseudo_quantity(self.pseudo_sellers)
                    ) + self.current_keep_amount <= self.max_seller_quantity:
                        formatted_temp_schedule, _ = schedule_dispatch_production(
                            formatted_temp_schedule,
                            offer[0],
                            offer[1],
                            self.max_buyer_delivery_day,
                        )
                        current_signed_amount += contract.agreement["quantity"]
                        signs[contract_index] = self.id
        elif self.awi_caller.is_last_level():
            # Sort the buyers by delivery.
            delivery_sorted_buyers = sort_negotiators_by_descending_delivery(buyers)
            signs = [None] * len(contracts)
            # Get current schedule.
            awi_schedule = self.awi.available_for_production(
                repeats=1,
                step=(self.current_step + 1, self.max_number_of_steps - 1),
                line=-1,
                override=False,
            )[0]
            formatted_temp_schedule = format_schedule(
                awi_schedule, self.max_number_of_steps
            )
            # Get green agent ids.
            green_contract_ids = list(dispatch_categories["Green"].keys())
            yellow_contract_ids = list(dispatch_categories["Yellow"].keys())
            red_contract_ids = list(dispatch_categories["Red"].keys())
            # Current signed amount.
            current_signed_amount = 0
            # Get seller contract id.
            contract_ids = [c.id for c in contracts]
            # Temp schedule & sign partners (buyers & sellers).
            for buyer_contract_id, seller_contract_details in dispatch_partners.items():
                for seller_contract_id, schedule in seller_contract_details.items():
                    # If seller is keep, we dont need to schedule or sign anything.
                    if seller_contract_id != "KEEP":
                        for schedule_step, quantity_to_produce in schedule.items():
                            formatted_temp_schedule, _ = schedule_dispatch_production(
                                formatted_temp_schedule,
                                quantity_to_produce,
                                schedule_step,
                                schedule_step + 1,
                            )
                        if seller_contract_id in contract_ids:
                            seller_index = contract_ids.index(seller_contract_id)
                            # Sign seller contract.
                            signs[seller_index] = self.id
                # Sign the deal if the buyer is not keep.
                if buyer_contract_id != "KEEP" and buyer_contract_id in contract_ids:
                    # Get buyer contract id.
                    contract_ids = [c.id for c in contracts]
                    buyer_index = contract_ids.index(buyer_contract_id)
                    # Sign buyer contract.
                    signs[buyer_index] = self.id
            # Check signing for red buyers.
            for (contract_id, offer) in delivery_sorted_buyers:
                # Check if schedule is available or not.
                is_available, _ = is_schedule_available(
                    formatted_temp_schedule,
                    offer[0],
                    self.min_seller_delivery_day,
                    offer[1],
                )
                # Get contract index.
                contract_index = contract_ids.index(contract_id)
                contract = contracts[contract_index]
                contract_partners = list(copy.deepcopy(contract.partners))
                contract_partners.remove(self.id)
                # Get agent id from contract.
                agent_id = contract_partners[0]
                # Check availability and sign feasible red buyer contracts.
                if (
                    is_available
                    and (contract_id in red_contract_ids)
                    and (agent_id == "BUYER")
                ):
                    # Check if this contract is feasible.
                    if (
                        contract.agreement["quantity"]
                        + get_unscheduled_total_pseudo_quantity(self.pseudo_buyers)
                        + current_signed_amount
                    ) <= self.max_buyer_quantity and contract.agreement[
                        "time"
                    ] >= self.min_buyer_delivery_day:
                        formatted_temp_schedule, _ = schedule_dispatch_production(
                            formatted_temp_schedule,
                            offer[0],
                            self.min_seller_delivery_day,
                            offer[1],
                        )
                        current_signed_amount += contract.agreement["quantity"]
                        signs[contract_index] = self.id
        else:
            # Iterate every contract again to sign (or not).
            for contract in contracts:
                contract_partners = list(copy.deepcopy(contract.partners))
                contract_partners.remove(self.id)
                # Get agent id from contract.
                agent_id = contract_partners[0]
                # Get green agent ids.
                green_contract_ids = list(dispatch_categories["Green"].keys())
                yellow_contract_ids = list(dispatch_categories["Yellow"].keys())
                # Check whether agent id is in green or yellow categories, or global buyer or seller
                if contract.id in (green_contract_ids + yellow_contract_ids):
                    # Sign it.
                    signs.append(self.id)
                else:  # Otherwise if agent is red or not buyer nor seller.
                    # Reject it.
                    signs.append(None)
        # Return signs
        return signs

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""
        # Get the contract id.
        executed_contract_id = contract.id

        if contract.id in self.pseudo_sellers.keys():
            # Get the schedule.
            awi_schedule = self.awi.available_for_production(
                repeats=1,
                step=(self.current_step, self.max_number_of_steps),
                line=-1,
                override=False,
            )[0]
            formatted_awi_schedule = format_schedule(
                awi_schedule, self.max_number_of_steps
            )
            formatted_awi_schedule[self.current_step] = list(
                self.awi.state.commands[self.current_step]
            ).count(-1)

            offer = self.pseudo_sellers[contract.id]
            # Check if schedule is available or not.
            is_available, schedule = is_schedule_available(
                formatted_awi_schedule, offer[0], offer[1], self.max_buyer_delivery_day
            )
            # Schedule production if schedule is available.
            if is_available:
                for schedule_step, quantity_to_produce in schedule.items():
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
                        self.awi.set_commands(temp_commands, step=self.current_step)

                if "KEEP" not in self.scheduled_buyer_contracts.keys():
                    self.scheduled_buyer_contracts["KEEP"] = {}
                # Create empty dict for seller id if it does not exist.
                if contract.id not in self.scheduled_seller_contracts.keys():
                    self.scheduled_seller_contracts[contract.id] = {}
                # Check whether we matched the same buyer & seller before, if so update, otherwise just create a new tuple.
                # Important for surplus sellers.
                # Check update or creating for scheduled buyer contracts.
                if contract.id not in self.scheduled_buyer_contracts["KEEP"].keys():
                    # FIX THIS PLEASE.
                    self.scheduled_buyer_contracts["KEEP"][contract.id] = (
                        offer,
                        schedule,
                    )
                    self.scheduled_seller_contracts[contract.id]["KEEP"] = (
                        (
                            offer[0],
                            self.max_buyer_delivery_day,
                            self.awi_caller.get_output_catalog_price(),
                        ),
                        schedule,
                    )
                else:
                    current_offer = self.scheduled_buyer_contracts["KEEP"][contract.id][
                        0
                    ]
                    updated_offer = (
                        current_offer[0] + offer[0],
                        current_offer[1],
                        current_offer[2],
                    )
                    current_schedule = copy.deepcopy(
                        self.scheduled_buyer_contracts["KEEP"][contract.id][1]
                    )
                    combined_steps = current_schedule.keys() | schedule.keys()
                    updated_schedule = {
                        step: current_schedule.get(step, 0) + schedule.get(step, 0)
                        for step in combined_steps
                    }
                    self.scheduled_buyer_contracts["KEEP"][contract.id] = (
                        updated_offer,
                        updated_schedule,
                    )
                    self.scheduled_seller_contracts[contract.id]["KEEP"] = (
                        (
                            offer[0],
                            self.max_buyer_delivery_day,
                            self.awi_caller.get_output_catalog_price(),
                        ),
                        updated_schedule,
                    )

                # Remove seller from pseudos.
                del self.pseudo_sellers[contract.id]

        # Check if we scheduled production etc. for this buyer.
        if executed_contract_id in self.scheduled_buyer_contracts.keys():
            # Check bankrupted agent's contracts.
            for (
                seller_contract_id,
                (seller_contract, schedule),
            ) in self.scheduled_buyer_contracts[executed_contract_id].items():
                # We pop executed buyer contract from the sellers.
                self.scheduled_seller_contracts[seller_contract_id].pop(
                    executed_contract_id
                )
                if not self.scheduled_seller_contracts[seller_contract_id]:
                    self.scheduled_seller_contracts.pop(seller_contract_id)
            # Pop the executed buyer from buyer contracts.
            self.scheduled_buyer_contracts.pop(executed_contract_id)
        # Pop executed contract id from pseudo buyers, if there is one.
        if executed_contract_id in self.pseudo_buyers.keys():
            offer = self.pseudo_buyers[executed_contract_id]
            self.pseudo_buyers.pop(executed_contract_id)

        self.update_constraints()

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution
        (i.e. resoluion is None)"""
        # Get the breached id.
        breached_contract_id = contract.id
        # # Check if we scheduled production etc. for this buyer.
        # if breached_contract_id in self.scheduled_buyer_contracts.keys():
        # 	# Check bankrupted agent's contracts.
        # 	for seller_contract_id, (seller_contract, schedule) in self.scheduled_buyer_contracts[breached_contract_id].items():
        # 		# We pop breached buyer contract from the sellers.
        # 		self.scheduled_seller_contracts[seller_contract_id].pop(breached_contract_id)
        # 		if not self.scheduled_seller_contracts[seller_contract_id]:
        # 			self.scheduled_seller_contracts.pop(seller_contract_id)
        # 	# Pop the breached buyer from buyer contracts.
        # 	self.scheduled_buyer_contracts.pop(breached_contract_id)
        # # Pop breached contract id from pseudo buyers, if there is one.
        # if breached_contract_id in self.pseudo_buyers.keys():
        # 	self.pseudo_buyers.pop(breached_contract_id)

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """
        scheduled_buyer_contracts: {buyer_contract_id: {seller_contract_id: ((q, t, p), schedule), seller_contract2_id: ((q, t, p), schedule2) }}
        scheduled_seller_contracts: {seller_contract_id: {buyer_contract_id: ((q, t, p), schedule), buyer_contract2_id: ((q, t, p), schedule2) }}
        schedule: {step1: quantity, step2: quantity, step3: quantity}

        pseudo_buyer: {buyer_contract_id: (q, t, p) }
        pseudo_seller: {seller_contract_id: (q, t, p) }
        """
        self.current_step = self.awi_caller.get_current_step()
        # Calculate keep amount.
        keep_amount = calculate_current_keep_amount(
            self.current_step,
            self.scheduled_buyer_contracts,
            self.awi_caller.get_inventory_output_prod_quantity(),
        )

        if len(contracts) != 0:
            if agent in self.my_suppliers:
                cancelled_schedule = defaultdict(int)
                # Iterate every contract in bankrupt contracts.
                for bankrupt_contract in contracts:
                    # Check if we scheduled production etc. for this seller.
                    if bankrupt_contract.id in self.scheduled_seller_contracts.keys():
                        # Check breached agent's contracts.
                        for (
                            buyer_contract_id,
                            (buyer_contract, schedule),
                        ) in self.scheduled_seller_contracts[
                            bankrupt_contract.id
                        ].items():
                            # Calculate total cancelled quantity from this seller, for this buyer.
                            cancelled_quantity = 0
                            # Iterate production days from the schedule in this contract to cancel production.
                            for scheduled_step, scheduled_quantity in schedule.items():
                                cancelled_schedule[scheduled_step] += scheduled_quantity
                                cancelled_quantity += scheduled_quantity
                            # Pop the bankrupted seller id from the buyer contracts too.
                            self.scheduled_buyer_contracts[buyer_contract_id].pop(
                                bankrupt_contract.id
                            )
                            # Check if we can companse the quantity with our keep quantity, so that we will add keep to the buyer's sellers in a way.
                            if keep_amount >= cancelled_quantity:
                                # Create empty dict for seller id if it does not exist.
                                if "KEEP" not in self.scheduled_seller_contracts.keys():
                                    self.scheduled_seller_contracts["KEEP"] = {}
                                # Set schedule just in case with cancelled q.
                                schedule = {-1: cancelled_quantity}
                                # Check whether we have keep in buyer schedule before.
                                if (
                                    "KEEP"
                                    not in self.scheduled_buyer_contracts[
                                        buyer_contract_id
                                    ].keys()
                                ):
                                    self.scheduled_buyer_contracts[buyer_contract_id][
                                        "KEEP"
                                    ] = (
                                        (
                                            cancelled_quantity,
                                            -1,
                                            self.seller_price_upper_bound,
                                        ),
                                        schedule,
                                    )
                                    self.scheduled_seller_contracts["KEEP"][
                                        buyer_contract_id
                                    ] = (buyer_contract, schedule)
                                else:
                                    current_offer = self.scheduled_buyer_contracts[
                                        buyer_contract_id
                                    ]["KEEP"][0]
                                    updated_offer = (
                                        current_offer[0] + cancelled_quantity,
                                        current_offer[1],
                                        current_offer[2],
                                    )
                                    schedule = {
                                        -1: current_offer[0] + cancelled_quantity
                                    }
                                    self.scheduled_buyer_contracts[buyer_contract_id][
                                        "KEEP"
                                    ] = (updated_offer, schedule)
                                    self.scheduled_seller_contracts["KEEP"][
                                        buyer_contract_id
                                    ] = (buyer_contract, schedule)
                            # If keep is not enough.
                            elif keep_amount < cancelled_quantity:
                                if keep_amount > 0:
                                    # Create empty dict for seller id if it does not exist.
                                    if (
                                        "KEEP"
                                        not in self.scheduled_seller_contracts.keys()
                                    ):
                                        self.scheduled_seller_contracts["KEEP"] = {}
                                    # Set schedule just in case with cancelled q.
                                    schedule = {-1: keep_amount}
                                    # Check whether we have keep in buyer schedule before.
                                    if (
                                        "KEEP"
                                        not in self.scheduled_buyer_contracts[
                                            buyer_contract_id
                                        ].keys()
                                    ):
                                        self.scheduled_buyer_contracts[
                                            buyer_contract_id
                                        ]["KEEP"] = (
                                            (
                                                keep_amount,
                                                -1,
                                                self.seller_price_upper_bound,
                                            ),
                                            schedule,
                                        )
                                        self.scheduled_seller_contracts["KEEP"][
                                            buyer_contract_id
                                        ] = (buyer_contract, schedule)
                                    else:
                                        current_offer = self.scheduled_buyer_contracts[
                                            buyer_contract_id
                                        ]["KEEP"][0]
                                        updated_offer = (
                                            current_offer[0] + keep_amount,
                                            current_offer[1],
                                            current_offer[2],
                                        )
                                        updated_schedule = {
                                            -1: current_offer[0] + keep_amount
                                        }
                                        self.scheduled_buyer_contracts[
                                            buyer_contract_id
                                        ]["KEEP"] = (updated_offer, updated_schedule)
                                        self.scheduled_seller_contracts["KEEP"][
                                            buyer_contract_id
                                        ] = (buyer_contract, updated_schedule)
                                    remaining_amount = cancelled_quantity - keep_amount
                                else:
                                    remaining_amount = cancelled_quantity
                                # Add this buyer contract as pseudo.
                                self.pseudo_buyers[buyer_contract_id] = (
                                    remaining_amount,
                                    buyer_contract[1],
                                    buyer_contract[2],
                                )
                        # Pop the bankrupted agent from scheduled dict.
                        self.scheduled_seller_contracts.pop(bankrupt_contract.id)
                    # Pop bankrupt contract id from pseudo sellers, if there is one.
                    if bankrupt_contract.id in self.pseudo_sellers.keys():
                        self.pseudo_sellers.pop(bankrupt_contract.id)

                for cancelled_step, cancelled_q in cancelled_schedule.items():
                    # Cancel production for that scheduled day.
                    # TODO: CHECK WHETHER OR NOT THIS IS LEGIT.
                    steps, _ = self.awi.schedule_production(
                        process=-1,
                        repeats=cancelled_q,
                        step=cancelled_step,
                        line=-1,
                        partial_ok=True,
                        override=True,
                        method="earliest",
                    )
            else:
                cancelled_schedule = defaultdict(int)
                # Iterate every contract in bankrupt contracts.
                for bankrupt_contract in contracts:
                    # Check if we scheduled production etc. for this buyer.
                    if bankrupt_contract.id in self.scheduled_buyer_contracts.keys():
                        # Check bankrupted agent's contracts.
                        for (
                            seller_contract_id,
                            (seller_contract, schedule),
                        ) in self.scheduled_buyer_contracts[
                            bankrupt_contract.id
                        ].items():
                            # Check if the buyer contains any keep amount in it, if it does, we dont need to move schedules.
                            if seller_contract_id != "KEEP":
                                total_cancelled_q = 0
                                # Iterate production days from the schedule in this contract to cancel production.
                                for (
                                    scheduled_step,
                                    scheduled_quantity,
                                ) in schedule.items():
                                    if scheduled_step >= self.current_step:
                                        cancelled_schedule[
                                            scheduled_step
                                        ] += scheduled_quantity
                                        total_cancelled_q += scheduled_quantity

                                if (
                                    seller_contract_id not in self.pseudo_sellers
                                    and total_cancelled_q > 0
                                ):
                                    self.pseudo_sellers[seller_contract_id] = (
                                        total_cancelled_q,
                                        seller_contract[1],
                                        seller_contract[2],
                                    )
                                elif (
                                    seller_contract_id in self.pseudo_sellers
                                    and total_cancelled_q > 0
                                ):
                                    current_pseudo_offer = self.pseudo_sellers[
                                        seller_contract_id
                                    ]
                                    updated_pseudo_offer = (
                                        current_pseudo_offer[0] + total_cancelled_q,
                                        current_pseudo_offer[1],
                                        current_pseudo_offer[2],
                                    )
                                    self.pseudo_sellers[
                                        seller_contract_id
                                    ] = updated_pseudo_offer

                                self.scheduled_seller_contracts[seller_contract_id].pop(
                                    bankrupt_contract.id
                                )

                        # Pop the bankrupted buyer from buyer contracts.
                        self.scheduled_buyer_contracts.pop(bankrupt_contract.id)
                    # Pop bankrupt contract id from pseudo buyers, if there is one.
                    if bankrupt_contract.id in self.pseudo_buyers.keys():
                        self.pseudo_buyers.pop(bankrupt_contract.id)

                for cancelled_step, cancelled_q in cancelled_schedule.items():
                    # Cancel production for that scheduled day.
                    steps, _ = self.awi.schedule_production(
                        process=-1,
                        repeats=cancelled_q,
                        step=cancelled_step,
                        line=-1,
                        partial_ok=True,
                        override=True,
                        method="earliest",
                    )

        total_c_q = 0
        for contract in contracts:
            total_c_q += contract.agreement["quantity"]

        # Add agent to the bankrupt list.
        self.bankrupt_agents.append(agent)

        self.update_constraints()

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""
        # Get opponent agent id that we negotiated with. 0th index is our agent id.
        agent_ids = copy.deepcopy(mechanism.agent_names)
        agent_ids.remove(self.id)
        self.nego_result_stats[agent_ids[0]]["Disagreement"] += 1

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""
        # Get opponent agent id that we negotiated with. 0th index is our agent id.
        agent_ids = copy.deepcopy(mechanism.agent_names)
        agent_ids.remove(self.id)
        self.nego_result_stats[agent_ids[0]]["Agreement"] += 1
