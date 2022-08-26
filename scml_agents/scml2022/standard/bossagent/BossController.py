from negmas import SAOSyncController

from scml.scml2020 import TIME, QUANTITY, UNIT_PRICE
from negmas import ResponseType, outcome_is_valid
from negmas.sao import SAOResponse
from typing import Dict

import math
import copy
from collections import defaultdict

# Custom imports.
from .helper import (
    sort_buyers_by_price,
    sort_sellers_by_price,
    sort_negotiators_by_delivery,
)


class SyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent
    needs and with lowest price.

    Args:
            is_seller: Are we trying to sell (or to buy)?
            parent: The agent from which we will access `needed` and `secured` arrays
            price_weight: The importance of price in utility calculation
            utility_threshold: Accept anything with a relative utility above that
            time_threshold: Accept anything with a positive utility when we are that close
                                            to the end of the negotiation
    """

    def __init__(
        self,
        *args,
        production_cost,
        parent: "BossBusinessPlanner",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__parent = parent
        self.production_cost = production_cost
        self.my_suppliers = copy.deepcopy(self.__parent.my_suppliers)
        self.my_consumers = copy.deepcopy(self.__parent.my_consumers)

    # =========================================
    #        Initialization (Every Step)
    # =========================================

    def set_limitations(
        self,
        current_step,
        max_number_of_steps,
        formatted_schedule,
        inventory_output_product,
        seller_price_upper_bound,
        buyer_price_lower_bound,
    ):
        """
        Every step set limitations of the controller.
        """
        self.current_step = current_step
        self.max_number_of_steps = max_number_of_steps
        self.formatted_schedule = copy.deepcopy(formatted_schedule)
        self.inventory_output_product = inventory_output_product
        # Keep accepted buyer and sellers.
        self.accepted_buyer_list = []
        self.accepted_seller_list = []
        self.accepted_negotiator_ids = []
        self.ended_negotiator_ids = []
        self.buyer_prices = {}
        self.seller_prices = {}
        self.nego_history = defaultdict(dict)
        self.seller_price_upper_bound = seller_price_upper_bound
        self.buyer_price_lower_bound = buyer_price_lower_bound

    # =========================================
    #            Offer Validation
    # =========================================

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        """Is this a valid offer for that negotiation"""
        issues = self.negotiators[negotiator_id][0].ami.issues
        return outcome_is_valid(offer, issues)

    # =========================================
    #               First proposals
    # =========================================

    def first_offer(self, negotiator_id: str) -> "Outcome":
        """
        Finds the first offer for this given negotiator. By default it will be the best offer
        Args:
                negotiator_id: The ID of the negotiator
        Returns:
                The first offer to use.
        Remarks:
                Default behavior is to use the ufun defined for the controller if any then try the ufun
                defined for the negotiator. If neither exists, the first offer will be None.
        """

        negotiator, _ = self.negotiators.get(negotiator_id, (None, None))

        if negotiator is None or negotiator.ami is None:
            return None

        issues = negotiator.ami.issues

        partner_id = self.partner_agent_ids(negotiator_id)[0]
        if partner_id in self.my_suppliers:  # If current proposer agent is seller.
            self.nego_history[negotiator.ami.id][0] = {
                "agentID": self.__parent.id,
                "q": int(issues[QUANTITY].max_value),
                "t": int(issues[TIME].min_value),
                "p": int(issues[UNIT_PRICE].min_value + 1),
            }
            return (
                issues[QUANTITY].max_value,
                issues[TIME].min_value,
                issues[UNIT_PRICE].min_value + 1,
            )
        elif partner_id in self.my_consumers:  # If current proposer agent is buyer.
            self.nego_history[negotiator.ami.id][0] = {
                "agentID": self.__parent.id,
                "q": int(issues[QUANTITY].max_value),
                "t": int(issues[TIME].min_value),
                "p": int(issues[UNIT_PRICE].max_value - 1),
            }
            return (
                issues[QUANTITY].max_value,
                issues[TIME].max_value,
                issues[UNIT_PRICE].max_value - 1,
            )
        else:
            return None

    # =========================================
    #          Main Negotiation Logic
    # =========================================

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, "SAOState"]
    ) -> Dict[str, "SAOResponse"]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
                offers: Maps negotiator IDs to offers.
                        Example format: {'8a8816d7-4b98-446a-b5d1-1714e7015b8e': (19, 7, 20), '090f1fcc-5ae3-43ea-93aa-845d473295da': (19, 7, 20)}
                        Where Q, T, P is int!
                states: Maps negotiator IDs to offers AT the time the offers were made.
                        Example format: {'8a8816d7-4b98-446a-b5d1-1714e7015b8e': SAOState(running=True, waiting=False, started=True, step=18, time=1.3397322999999979, relative_time=0.9,
                        broken=False, timedout=False, agreement=None, results=None, n_negotiators=2, has_error=False, error_details='', current_offer=(19, 7, 20),
                        current_proposer='04Bos@2', current_proposer_agent='02BCS@1', n_acceptances=0, new_offers=[], new_offerer_agents=[])}

        Remarks:
                - The response type CANNOT be WAIT.

        """
        # Initialize buyers and sellers for dispatch algorithm.
        sellers = {}
        buyers = {}
        # Reset responses, we will fill by each category.
        responses = {}
        # Categorize buyer and sellers.
        for negotiator_id, state in states.items():
            partner_id = self.partner_agent_ids(negotiator_id)[0]
            negotiator, _ = self.negotiators.get(negotiator_id, (None, None))
            self.nego_history[negotiator.ami.id][
                2 * (states[negotiator_id].step) - 1
            ] = {
                "agentID": partner_id,
                "q": int(offers[negotiator_id][QUANTITY]),
                "t": int(offers[negotiator_id][TIME]),
                "p": int(offers[negotiator_id][UNIT_PRICE]),
            }
            if partner_id in self.my_suppliers:  # If current proposer agent is seller.
                sellers[negotiator_id] = offers[negotiator_id]
            elif partner_id in self.my_consumers:  # If current proposer agent is buyer.
                buyers[negotiator_id] = offers[negotiator_id]
        # Get negotiating buyer and seller ids.
        buyer_ids = list(buyers.keys())
        seller_ids = list(sellers.keys())
        # Sort the buyer and sellers. Sorting returns them as list of tuples
        sorted_buyer_list = sort_buyers_by_price(
            buyers
        )  # E.g. [(('agent1', (10, 20, 5), ('agent2', (5, 10, 3))]
        sorted_seller_list = sort_sellers_by_price(
            sellers
        )  # E.g. [(('agent1', (10, 20, 5), ('agent2', (5, 10, 3))]
        # Sort pseudo buyer & sellers by delivery time.
        sorted_pseudo_buyers = sort_negotiators_by_delivery(self.__parent.pseudo_buyers)
        sorted_pseudo_sellers = sort_negotiators_by_delivery(
            self.__parent.pseudo_sellers
        )
        # Add & give priority to the accepted buyer & sellers, since we want to resolve them first.
        complete_buyer_list = (
            sorted_pseudo_buyers + self.accepted_buyer_list + sorted_buyer_list
        )
        complete_seller_list = (
            sorted_pseudo_sellers + self.accepted_seller_list + sorted_seller_list
        )
        # Get dispatch result from the module.
        dispatch_result_categories, _ = self.__parent.dispatch_module.run_dispatch(
            complete_buyer_list, complete_seller_list, self.formatted_schedule
        )
        # Get each category from dispatch result.
        green_negotiators = dispatch_result_categories["Green"]
        yellow_negotiators = dispatch_result_categories["Yellow"]
        red_negotiators = dispatch_result_categories["Red"]
        # Otherwise, we should call negotiation methods, and create responses for each negotiator.
        # Set responses for each category.
        responses = self.set_green_category_responses(
            responses, states, green_negotiators, buyer_ids, seller_ids
        )
        responses = self.set_yellow_category_responses(
            responses, states, yellow_negotiators, buyer_ids, seller_ids
        )
        responses = self.set_red_category_responses(
            responses, states, red_negotiators, buyer_ids, seller_ids
        )
        # Return the responses.
        return responses

    # =========================================
    #           Negotiation Callback
    # =========================================

    def on_negotiation_end(self, negotiator_id: str, state: "MechanismState") -> None:
        """Update accepted buyer and sellers when it is accepted."""
        super().on_negotiation_end(negotiator_id, state)
        if self.partner_agent_ids(negotiator_id):
            partner_id = self.partner_agent_ids(negotiator_id)[0]
            negotiator, _ = self.negotiators.get(negotiator_id, (None, None))
            if state.agreement:
                self.nego_history[negotiator.ami.id]["Acceptance"] = {
                    "q": int(state.current_offer[QUANTITY]),
                    "t": int(state.current_offer[TIME]),
                    "p": int(state.current_offer[UNIT_PRICE]),
                    "accept": 1,
                    "agent_sign": 0,
                    "opp_sign": 0,
                }
                if partner_id in self.my_consumers:
                    self.accepted_buyer_list.append(
                        (negotiator_id, state.current_offer)
                    )
                elif partner_id in self.my_suppliers:
                    self.accepted_seller_list.append(
                        (negotiator_id, state.current_offer)
                    )
                # Add to accepted negotiators either way.
                self.accepted_negotiator_ids.append(negotiator_id)
                self.__parent.reject_acceptance_rate[partner_id][
                    "Acceptance"
                ] += state.current_offer[QUANTITY]
            else:
                self.nego_history[negotiator.ami.id]["Acceptance"] = {}
                self.ended_negotiator_ids.append(negotiator_id)

            mechanism_nego_history = copy.deepcopy(self.nego_history[negotiator.ami.id])
            self.__parent.nego_stats.set_negotiation_bid_history(
                self.current_step, partner_id, negotiator.ami.id, mechanism_nego_history
            )

    # =========================================
    #       Dispatch Category Responses
    # =========================================

    def set_green_category_responses(
        self, before_responses, states, green_negotiators, buyer_ids, seller_ids
    ):
        """
        Set responses for GREEN negotiation categories.
        NOT UPDATED WITH MULTIPLE RED ISSUES.
        """
        pseudo_ids = list(self.__parent.pseudo_buyers.keys()) + list(
            self.__parent.pseudo_sellers.keys()
        )

        # Copy the beforehand responses and add into them.
        responses = copy.deepcopy(before_responses)

        for negotiator_id, offer_content in green_negotiators.items():
            # Check acceptance conditions.
            if (
                negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids)
                and states[negotiator_id].step >= 19
            ):
                responses[negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            elif negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids):
                # Check whether this negotiator's negotiation is accepted or not, we do not send request to finished nego.
                if negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids):
                    issues = self.negotiators[negotiator_id][0].ami.issues
                    relative_time = states[negotiator_id].relative_time
                    offer = offer_content["Offer"]
                    # Check if the negotiator is buyer.
                    if negotiator_id in buyer_ids:
                        if negotiator_id in self.buyer_prices.keys():
                            self.buyer_prices[negotiator_id].append(offer[UNIT_PRICE])
                        else:
                            self.buyer_prices[negotiator_id] = []
                            self.buyer_prices[negotiator_id].append(offer[UNIT_PRICE])
                        # offer_quantity = min(self.boulware_proportion_tactic(issues[QUANTITY].max_value, offer[QUANTITY], relative_time), issues[QUANTITY].max_value)
                        # offer_time = min(self.boulware_proportion_tactic(issues[TIME].max_value, offer[TIME], relative_time), issues[TIME].max_value)
                        offer_price = self.boulware_proportion_tactic(
                            issues[UNIT_PRICE].max_value,
                            max(self.buyer_prices[negotiator_id]),
                            relative_time,
                        )
                    # Check if negotiator is seller.
                    elif negotiator_id in seller_ids:
                        if negotiator_id in self.seller_prices.keys():
                            self.seller_prices[negotiator_id].append(offer[UNIT_PRICE])
                        else:
                            self.seller_prices[negotiator_id] = []
                            self.seller_prices[negotiator_id].append(offer[UNIT_PRICE])
                        # offer_quantity = min(self.boulware_proportion_tactic(issues[QUANTITY].max_value, offer[QUANTITY], relative_time), issues[QUANTITY].max_value)
                        # offer_time = max(self.boulware_inverse_proportion_tactic(offer[TIME], issues[TIME].min_value, relative_time), issues[TIME].min_value)
                        offer_price = self.boulware_inverse_proportion_tactic(
                            min(self.seller_prices[negotiator_id]),
                            issues[UNIT_PRICE].min_value,
                            relative_time,
                        )
                    # Negotiator should be either buyer or seller, end negotiation otherwise.
                    else:
                        # End the negotiation.
                        responses[negotiator_id] = SAOResponse(
                            ResponseType.END_NEGOTIATION, None
                        )
                    # Check whether generated offer is valid or not.
                    if not self.is_valid(
                        negotiator_id, (offer[QUANTITY], offer[TIME], offer_price)
                    ):
                        # End the negotiation.
                        responses[negotiator_id] = SAOResponse(
                            ResponseType.END_NEGOTIATION, None
                        )
                    else:
                        if offer_price == offer[UNIT_PRICE]:
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.ACCEPT_OFFER, None
                            )
                        else:
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(offer[QUANTITY]),
                                "t": int(offer[TIME]),
                                "p": int(offer_price),
                            }
                            # Add our offer to responses with correspoding negotiator id.
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (offer[QUANTITY], offer[TIME], offer_price),
                            )
        return responses

    def set_yellow_category_responses(
        self, before_responses, states, yellow_negotiators, buyer_ids, seller_ids
    ):
        """
        Set responses for YELLOW negotiation categories.
        NOT UPDATED WITH MULTIPLE RED ISSUES.
        """
        pseudo_ids = list(self.__parent.pseudo_buyers.keys()) + list(
            self.__parent.pseudo_sellers.keys()
        )

        # Copy the beforehand responses and add into them.
        responses = copy.deepcopy(before_responses)

        for negotiator_id, offer_content in yellow_negotiators.items():
            # Check whether this negotiator's negotiation is accepted or not, we do not send request to finished nego.
            if (
                negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids)
                and offer_content["Surplus"] <= 20
                and states[negotiator_id].step >= 19
            ):
                responses[negotiator_id] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            elif negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids):
                # Check whether this negotiator's negotiation is accepted or not, we do not send request to finished nego.
                if negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids):
                    issues = self.negotiators[negotiator_id][0].ami.issues
                    relative_time = states[negotiator_id].relative_time
                    offer = offer_content["Offer"]
                    if negotiator_id in seller_ids:
                        if negotiator_id in self.seller_prices.keys():
                            self.seller_prices[negotiator_id].append(offer[UNIT_PRICE])
                        else:
                            self.seller_prices[negotiator_id] = []
                            self.seller_prices[negotiator_id].append(offer[UNIT_PRICE])

                        if offer_content["Target"]:
                            offer_q = min(
                                offer_content["Target"], issues[QUANTITY].max_value
                            )
                        else:
                            offer_q = max(
                                offer[QUANTITY] - offer_content["Surplus"],
                                issues[QUANTITY].min_value,
                            )

                        offer_price = self.boulware_inverse_proportion_tactic(
                            min(self.seller_prices[negotiator_id]),
                            issues[UNIT_PRICE].min_value,
                            relative_time,
                        )
                        # Check whether generated offer is valid or not.
                        if not self.is_valid(
                            negotiator_id, (offer_q, offer[TIME], offer_price)
                        ):
                            # End the negotiation.
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.END_NEGOTIATION, None
                            )
                        else:
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(offer_q),
                                "t": int(offer[TIME]),
                                "p": int(offer_price),
                            }
                            # Add our offer to responses with correspoding negotiator id.
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (offer_q, offer[TIME], offer_price),
                            )
                    else:
                        # End negotiation for now.
                        responses[negotiator_id] = SAOResponse(
                            ResponseType.END_NEGOTIATION, None
                        )
        return responses

    def set_red_category_responses(
        self, before_responses, states, red_negotiators, buyer_ids, seller_ids
    ):
        """
        Set responses for RED negotiation categories.
        We will start with getting every issue from the negotiation strategy,
        will change the red issue one with the red_issue_target.
        """
        pseudo_ids = list(self.__parent.pseudo_buyers.keys()) + list(
            self.__parent.pseudo_sellers.keys()
        )

        # Copy the beforehand responses and add into them.
        responses = copy.deepcopy(before_responses)

        for negotiator_id, offer_content in red_negotiators.items():
            # Check whether this negotiator's negotiation is accepted or not, we do not send request to finished nego.
            if negotiator_id not in (pseudo_ids + self.accepted_negotiator_ids):
                # Get issues with boundries from mechanism.
                issues = self.negotiators[negotiator_id][0].ami.issues
                # Get current negotiation time.
                relative_time = states[negotiator_id].relative_time
                # Get opponent's offer.
                offer = offer_content["Offer"]
                # Get the REASONS (DICT) why this negotiator is red according to the dispatch result.
                red_issues = offer_content["Issues"]
                # Check if the red negotiator is a buyer.
                if negotiator_id in buyer_ids:
                    # Get initial offer values from the negotiation tactic.
                    offer_quantity = offer[QUANTITY]
                    offer_time = offer[TIME]
                    offer_price = self.boulware_proportion_tactic(
                        issues[UNIT_PRICE].max_value,
                        self.buyer_price_lower_bound,
                        relative_time,
                    )
                    # Iterate over red issues, and fix them with target values.
                    for red_issue, red_issue_target in red_issues.items():
                        # If red issue is no seller.
                        if "No Seller" in red_issues.keys():
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    issues[QUANTITY].min_value,
                                    issues[TIME].max_value,
                                    issues[UNIT_PRICE].max_value,
                                ),
                            )
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(issues[QUANTITY].min_value),
                                "t": int(issues[TIME].max_value),
                                "p": int(issues[UNIT_PRICE].max_value),
                            }
                            break
                        # Check the red issue and update the target value and check negotiation boundries.
                        if red_issue == "Price":
                            if red_issue_target <= issues[UNIT_PRICE].max_value:
                                buyer_lower_price = max(
                                    red_issue_target, self.buyer_price_lower_bound
                                )
                            else:
                                buyer_lower_price = self.buyer_price_lower_bound
                            offer_price = self.boulware_proportion_tactic(
                                issues[UNIT_PRICE].max_value,
                                buyer_lower_price,
                                relative_time,
                            )
                        elif red_issue == "Volume":
                            buyer_upper_quantity = min(
                                red_issue_target, issues[QUANTITY].max_value
                            )
                            offer_quantity = self.boulware_proportion_tactic(
                                buyer_upper_quantity, offer[QUANTITY], relative_time
                            )
                        elif red_issue == "Delivery":
                            if (
                                red_issue_target <= issues[TIME].max_value
                                and red_issue_target >= issues[TIME].min_value
                            ):
                                buyer_lower_time = red_issue_target
                                offer_time = self.boulware_proportion_tactic(
                                    issues[TIME].max_value,
                                    buyer_lower_time,
                                    relative_time,
                                )
                            else:
                                offer_time = red_issue_target
                        else:
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    issues[QUANTITY].min_value,
                                    issues[TIME].max_value,
                                    issues[UNIT_PRICE].max_value,
                                ),
                            )
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(issues[QUANTITY].min_value),
                                "t": int(issues[TIME].max_value),
                                "p": int(issues[UNIT_PRICE].max_value),
                            }
                            break
                # Check if the red negotiator is a seller.
                elif negotiator_id in seller_ids:
                    # Get initial offer values from the negotiation tactic.
                    offer_quantity = self.boulware_proportion_tactic(
                        issues[QUANTITY].max_value, offer[QUANTITY], relative_time
                    )
                    offer_time = self.boulware_inverse_proportion_tactic(
                        offer[TIME], issues[TIME].min_value, relative_time
                    )
                    offer_price = self.boulware_inverse_proportion_tactic(
                        self.seller_price_upper_bound,
                        max(issues[UNIT_PRICE].min_value, 1),
                        relative_time,
                    )
                    # Iterate over red issues, and fix them with target values.
                    for red_issue, red_issue_target in red_issues.items():
                        # If there is no buyer.
                        if "No Buyer" in red_issues.items():
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    issues[QUANTITY].min_value,
                                    issues[TIME].min_value,
                                    issues[UNIT_PRICE].min_value,
                                ),
                            )
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(issues[QUANTITY].min_value),
                                "t": int(issues[TIME].min_value),
                                "p": int(issues[UNIT_PRICE].min_value),
                            }
                            break
                        # Check the red issue and update the target value and check negotiation boundries.
                        if red_issue == "Price":
                            if red_issue_target >= issues[UNIT_PRICE].min_value:
                                seller_upper_price = min(
                                    red_issue_target, self.seller_price_upper_bound
                                )
                            else:
                                seller_upper_price = self.seller_price_upper_bound
                            offer_price = self.boulware_proportion_tactic(
                                seller_upper_price,
                                issues[UNIT_PRICE].min_value,
                                relative_time,
                            )
                        elif red_issue == "Volume":
                            seller_upper_quantity = min(
                                red_issue_target, issues[QUANTITY].max_value
                            )
                            offer_quantity = self.boulware_proportion_tactic(
                                seller_upper_quantity, offer[QUANTITY], relative_time
                            )
                        elif red_issue == "Delivery":
                            if (
                                red_issue_target <= issues[TIME].max_value
                                and red_issue_target >= issues[TIME].min_value
                            ):
                                seller_upper_time = red_issue_target
                                offer_time = self.boulware_inverse_proportion_tactic(
                                    seller_upper_time,
                                    issues[TIME].min_value,
                                    relative_time,
                                )
                            else:
                                offer_time = red_issue_target
                        else:
                            responses[negotiator_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    issues[QUANTITY].min_value,
                                    issues[TIME].min_value,
                                    issues[UNIT_PRICE].min_value,
                                ),
                            )
                            negotiator, _ = self.negotiators.get(
                                negotiator_id, (None, None)
                            )
                            self.nego_history[negotiator.ami.id][
                                2 * states[negotiator_id].step
                            ] = {
                                "agentID": self.__parent.id,
                                "q": int(issues[QUANTITY].min_value),
                                "t": int(issues[TIME].min_value),
                                "p": int(issues[UNIT_PRICE].min_value),
                            }
                            break
                else:
                    # End negotiation for now.
                    responses[negotiator_id] = SAOResponse(
                        ResponseType.END_NEGOTIATION, None
                    )

                # If we havent already response to this negotiator, create one.
                if negotiator_id not in responses.keys():
                    # Check whether offer is valid or not.
                    if not self.is_valid(
                        negotiator_id, (offer_quantity, offer_time, offer_price)
                    ):
                        # We are ending negotiation, since it is not valid (it should be somehow).
                        responses[negotiator_id] = SAOResponse(
                            ResponseType.END_NEGOTIATION, None
                        )
                    else:
                        negotiator, _ = self.negotiators.get(
                            negotiator_id, (None, None)
                        )
                        self.nego_history[negotiator.ami.id][
                            2 * states[negotiator_id].step
                        ] = {
                            "agentID": self.__parent.id,
                            "q": int(offer_quantity),
                            "t": int(offer_time),
                            "p": int(offer_price),
                        }
                        # Add our offer to responses with correspoding negotiator id.
                        responses[negotiator_id] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (offer_quantity, offer_time, offer_price),
                        )

        return responses

    # =========================================
    #      Time-based negotiation tactics
    # =========================================

    def boulware_proportion_tactic(self, upper_bound, lower_bound, current_time):
        # Set upper bound to min 1, since we do not want division 0.
        upper_bound = max(1, upper_bound)
        P0 = 1.0
        P2 = lower_bound / (upper_bound * 1.0)
        P1 = (P0 + P2) / 2

        normalized_target = (
            ((1 - current_time) ** 2) * P0
            + 2 * (1 - current_time) * current_time * P1
            + (current_time**2) * P2
        )
        target = math.floor(normalized_target * upper_bound)

        return int(target)

    def boulware_inverse_proportion_tactic(
        self, upper_bound, lower_bound, current_time
    ):
        """
        Seller price and quantity should be inverse proportion with price and quantity.
        Basically we want to buy cheapest and shortest delivery time.
        """
        # Set upper bound to min 1, since we do not want division 0.
        upper_bound = max(1, upper_bound)
        P0 = 1.0
        P2 = lower_bound / upper_bound * 1.0
        P1 = (P0 + P2) / 2

        normalized_target = (
            ((1 - current_time) ** 2) * P0
            + 2 * (1 - current_time) * current_time * P1
            + (current_time**2) * P2
        )
        target = math.floor(lower_bound / normalized_target)

        return int(target)
