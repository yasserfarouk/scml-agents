import copy
from collections import defaultdict

from .helper import (
    buyer_closest_available_delivery,
    is_schedule_available,
    most_available_amount_in_schedule,
    schedule_dispatch_production,
    seller_closest_available_delivery,
)


class BossDispatch:
    def __init__(self, production_cost):
        # Keep production cost.
        self.production_cost = production_cost

    # =====================
    #   Dispatch Algorithm
    # =====================

    def run_dispatch(self, buyers, sellers, formatted_schedule):
        """
        Dispatch algorithm for buyers and sellers.

        Keep the global categories, after dispatch algorithm finishes, we select nego strategy from this categories.
        Yellow has one specific situation, where surplus amount is also should be considered.
        Green and red structure as follows: 'Green': {agentId: {'Offer': Offer}}, e.g.'Green': {'Agent1': {'Offer': (1, 3, 5)}}.
        Where offer is (volume, time, price).
        Yellow has specific condition where surplus amount is should be considered.
        Therefore yellow structure as follows: 'Yellow': {agentId: {'Offer': Offer, 'Surplus': int}},
        e.g. 'Yellow': {'Agent1': {'Offer': (1, 3, 5), 'Surplus': 10, 'Target': 30}}
        Red has specific condition where a buyer or seller couldnt meet the requirements to turn green. So, we will keep the reason with offer and agent id.
        e.g. 'Red': {'Agent1': {'Offer': (1, 3, 5), 'Issues': {'Price': 'Target', 'Volume': 'Target', 'Delivery': 'Target}}}
        Buyer can be red with 4 conditions: 'Price', 'Volume', 'Delivery', 'No Seller'
        Seller can be red with 4 conditions: 'Price', 'Volume', 'Delivery', 'No Buyer'
        """

        # Get a copy of categories, and schedule.
        dispatch_categories = {"Green": {}, "Yellow": {}, "Red": {}}
        dispatch_schedule = copy.deepcopy(formatted_schedule)
        dispatch_partners = (
            {}
        )  # {buyer_contract_id: {seller_contract_id: schedule} }, schedule: {'1': 3, '2', 5, '3': 7}

        if len(buyers) == 0:  # If there are no buyer, all sellers are red.
            for seller in sellers:
                # Get id and offer.
                seller_id = seller[0]
                seller_offer = seller[1]
                dispatch_categories["Red"][seller_id] = {
                    "Offer": seller_offer,
                    "Issues": {"No Buyer": "No Buyer"},
                }

            return dispatch_categories, dispatch_partners

        if len(sellers) == 0:
            for buyer in buyers:
                # Get id and offer.
                buyer_id = buyer[0]
                buyer_offer = buyer[1]
                dispatch_categories["Red"][buyer_id] = {
                    "Offer": buyer_offer,
                    "Issues": {"No Seller": "No Seller"},
                }

            return dispatch_categories, dispatch_partners

        # We should run algorithm buyer by buyer.
        for buyer_index, buyer in enumerate(buyers):
            # Get id and offer.
            buyer_id = buyer[0]
            buyer_offer = buyer[1]
            # Get offer content.
            buyer_volume = buyer_offer[0]
            buyer_delivery_time = buyer_offer[1]
            buyer_price = buyer_offer[2]

            buyer_paid_partial_volume = 0  # We should keep paid (hypothetically) volume, in case of 1 seller cannot provide enough mats.
            buyer_paid_partial_price = 0  # We should also keep paid price (hypothetically) for partial volume, in case of 1 seller cannot provide enough mats.

            # Take the deepcopy of the local dispatch categories. If buyer is green, this temp will be applied to the local.
            temp_dispatch_categories = copy.deepcopy(dispatch_categories)

            # Take the deepcopy of the local dispatch schedule at the start. If buyer is green, this temp will be applied to the local.
            temp_dispatch_schedule = copy.deepcopy(dispatch_schedule)

            # Initialize buyer's partners.
            buyer_partners = {}

            # Keep a flag for seller, so that we can manage to increase the quantity of it.
            is_nego_with_yellow = False

            for seller in sellers:
                # Get id and offer.
                seller_id = seller[0]
                seller_offer = seller[1]
                # Get offer content.
                seller_volume = seller_offer[0]
                seller_delivery_time = seller_offer[1]
                seller_price = seller_offer[2]
                # We should skip if the seller is already green.
                if (seller_id not in dispatch_categories["Green"].keys()) or (
                    seller_id in dispatch_categories["Yellow"].keys()
                    and buyers[buyer_index - 1][0]
                    in dispatch_categories["Green"].keys()
                ):
                    if (
                        seller_id in dispatch_categories["Yellow"].keys()
                        and buyers[buyer_index - 1][0]
                        in dispatch_categories["Green"].keys()
                    ):
                        is_nego_with_yellow = True
                    # Calculate the remaining volume for the buyer.
                    buyer_remaining_partial_volume = (
                        buyer_volume - buyer_paid_partial_volume
                    )
                    if (
                        seller_id in dispatch_categories["Yellow"].keys()
                    ):  # If seller is yellow.
                        seller_volume = dispatch_categories["Yellow"][seller_id][
                            "Surplus"
                        ]  # Update the available seller volume with surplus amount.
                    # Check if this seller can manage to fill remaining volume.
                    if (
                        buyer_remaining_partial_volume <= seller_volume
                    ):  # Check if seller can supply the demand with a surplus (>= 0).
                        seller_new_remaining_partial_volume = (
                            seller_volume - buyer_remaining_partial_volume
                        )
                        # Calculate profit.
                        profit = (
                            (buyer_price - self.production_cost) * buyer_volume
                        ) - (
                            buyer_paid_partial_price
                            + (buyer_remaining_partial_volume * seller_price)
                        )
                        # Check if schedule is available or not.
                        is_available, _ = is_schedule_available(
                            temp_dispatch_schedule,
                            buyer_remaining_partial_volume,
                            seller_delivery_time,
                            buyer_delivery_time,
                        )
                        # Check whether this buyer is profitable and schedule is ok.
                        if profit > 0 and is_available:
                            if (
                                seller_new_remaining_partial_volume == 0
                            ):  # If we supply the buyer's demand with profit and exact same amount.
                                temp_dispatch_categories["Yellow"].pop(
                                    seller_id, None
                                )  # If seller is yellow, we can pop seller from yellow.
                                temp_dispatch_categories["Red"].pop(
                                    seller_id, None
                                )  # If seller is red, we can pop seller from red.
                                # We add seller to the green list.
                                temp_dispatch_categories["Green"][seller_id] = {
                                    "Offer": seller_offer
                                }
                            else:  # If we supply the buyer's demand with profit and surplus.
                                temp_dispatch_categories["Red"].pop(
                                    seller_id, None
                                )  # If seller is red, we can pop seller from red.
                                # We add (or update) the seller to the yellow group.
                                temp_dispatch_categories["Yellow"][seller_id] = {
                                    "Offer": seller_offer,
                                    "Surplus": seller_new_remaining_partial_volume,
                                    "Target": None,
                                }
                            # We will put seller to the green either way.
                            temp_dispatch_categories["Green"][buyer_id] = {
                                "Offer": buyer_offer
                            }
                            # Update global categories with modifications.
                            dispatch_categories = copy.deepcopy(
                                temp_dispatch_categories
                            )
                            # Update the temp schedule.
                            (
                                temp_dispatch_schedule,
                                partner_schedule,
                            ) = schedule_dispatch_production(
                                temp_dispatch_schedule,
                                buyer_remaining_partial_volume,
                                seller_delivery_time,
                                buyer_delivery_time,
                            )
                            # Update the global schedule with the temp one.
                            dispatch_schedule = copy.deepcopy(temp_dispatch_schedule)
                            # TODO: We will create the partners buyer - seller, since this buyer is green now.
                            # Add this seller to buyer's partners.
                            buyer_partners[seller_id] = partner_schedule
                            # Update the global partners, since we found the both of them green.
                            dispatch_partners[buyer_id] = copy.deepcopy(buyer_partners)
                            # We can break the loop here, since we do not need anything anymore for this buyer.
                            break
                        else:  # This buyer is not profitable or delivery time (or quantity) is problematic.
                            # We do not need to apply temp to the global, since we cannot fill this buyer with profit.
                            # We also do not need to schedule anything, since we are not going to produce for this buyer (at this step).
                            # Buyer should be turned into red. Seller should be turned into red if it is not yellow previously.
                            # We also need to check the cause of the redness.
                            # Create issues dict, since we can keep more than 1 issue at a time.
                            buyer_issues = {}
                            seller_issues = {}
                            if profit <= 0:  # Check if problem is price.
                                buyer_issue = "Price"
                                buyer_issue_target = (
                                    (
                                        (
                                            buyer_paid_partial_price
                                            + (
                                                buyer_remaining_partial_volume
                                                * seller_price
                                            )
                                        )
                                        / (buyer_volume * 1.0)
                                    )
                                    + self.production_cost
                                    + 1
                                )
                                buyer_issues[buyer_issue] = buyer_issue_target
                            if (
                                not is_available
                            ):  # Otherwise check delivery time, since seller can fill this buyer to green.
                                buyer_issue_target = buyer_closest_available_delivery(
                                    temp_dispatch_schedule,
                                    buyer_remaining_partial_volume,
                                    seller_delivery_time,
                                    buyer_delivery_time,
                                )
                                if buyer_issue_target != -1:
                                    buyer_issue = "Delivery"
                                    buyer_issues[buyer_issue] = buyer_issue_target
                                else:
                                    buyer_issue = "Volume"
                                    buyer_issues[
                                        buyer_issue
                                    ] = most_available_amount_in_schedule(
                                        temp_dispatch_schedule,
                                        seller_delivery_time,
                                        buyer_delivery_time,
                                    )
                            # Add buyer to the red list, with problematic issue and the target for offer.
                            dispatch_categories["Red"][buyer_id] = {
                                "Offer": buyer_offer,
                                "Issues": buyer_issues,
                            }
                            # If seller is not yellow before, we should add it to the red.
                            if seller_id not in dispatch_categories["Yellow"].keys():
                                if profit <= 0:
                                    seller_issue = "Price"
                                    seller_issue_target = (
                                        (
                                            (buyer_price - self.production_cost)
                                            * buyer_volume
                                        )
                                        - buyer_paid_partial_price
                                    ) / (buyer_remaining_partial_volume * 1.0)
                                    seller_issues[seller_issue] = seller_issue_target
                                if not is_available:
                                    seller_issue_target = (
                                        seller_closest_available_delivery(
                                            temp_dispatch_schedule,
                                            buyer_remaining_partial_volume,
                                            buyer_delivery_time,
                                        )
                                    )
                                    if seller_issue_target != -1:
                                        seller_issue = "Delivery"
                                        seller_issues[
                                            seller_issue
                                        ] = seller_issue_target
                                    else:
                                        seller_issue = "Volume"
                                        seller_issues[
                                            seller_issue
                                        ] = most_available_amount_in_schedule(
                                            temp_dispatch_schedule,
                                            seller_delivery_time,
                                            buyer_delivery_time,
                                        )
                                dispatch_categories["Red"][seller_id] = {
                                    "Offer": seller_offer,
                                    "Issues": seller_issues,
                                }
                    else:  # Seller cannot fill the remaining volume, therefore buyer is still will be yellow (still need some mats).
                        # Calculate new remaining partial (and paid) volume. Apply it if the seller is profitable.
                        buyer_new_remaining_partial_volume = (
                            buyer_remaining_partial_volume - seller_volume
                        )
                        buyer_new_paid_partial_volume = (
                            buyer_volume - buyer_new_remaining_partial_volume
                        )
                        # Calculate profit.
                        profit = (
                            (buyer_price - self.production_cost)
                            * buyer_new_paid_partial_volume
                        ) - (buyer_paid_partial_price + (seller_volume * seller_price))
                        # Check if schedule is available or not.
                        is_available, _ = is_schedule_available(
                            temp_dispatch_schedule,
                            seller_volume,
                            seller_delivery_time,
                            buyer_delivery_time,
                        )
                        # Check whether this buyer is profitable and schedule is ok.
                        if (
                            profit > 0 and is_available
                        ):  # This buyer is still profitable.
                            # Since dispatch did not finish for this buyer, we put seller to the temp green category.
                            temp_dispatch_categories["Yellow"].pop(
                                seller_id, None
                            )  # If seller is yellow, we can pop seller from yellow.
                            temp_dispatch_categories["Red"].pop(
                                seller_id, None
                            )  # If seller is red, we can pop seller from red.
                            # Put seller to the temp green.
                            temp_dispatch_categories["Green"][seller_id] = {
                                "Offer": seller_offer
                            }
                            # Update the temp schedule.
                            (
                                temp_dispatch_schedule,
                                partner_schedule,
                            ) = schedule_dispatch_production(
                                temp_dispatch_schedule,
                                seller_volume,
                                seller_delivery_time,
                                buyer_delivery_time,
                            )
                            # Update the buyer's paid partial volume with new one.
                            buyer_paid_partial_volume = buyer_new_paid_partial_volume
                            # Update the buyer's partial paid price.
                            buyer_paid_partial_price += seller_volume * seller_price
                            # TODO: We will create the partners buyer - seller, since this buyer is green now.
                            # Add this seller to buyer's partners.
                            buyer_partners[seller_id] = partner_schedule
                            # Continue with the next seller.
                            continue
                        else:  # Even with partial pricing, this buyer is not profitable. Therefore we can move buyer to the red group.
                            # We do not need to apply temp to the global, since we cannot fill this buyer with profit.
                            # We also do not need to schedule anything, since we are not going to produce for this buyer (at this step).
                            # Buyer should be turned into red. Seller should be turned into red if it is not yellow previously.
                            # We also need to check the cause of the redness.
                            # Create issues dict, since we can keep more than 1 issue at a time.
                            buyer_issues = {}
                            seller_issues = {}
                            if profit <= 0:  # Check if problem is price.
                                buyer_issue = "Price"
                                buyer_issue_target = (
                                    (
                                        (
                                            buyer_paid_partial_price
                                            + (seller_volume * seller_price)
                                        )
                                        / (buyer_new_paid_partial_volume * 1.0)
                                    )
                                    + self.production_cost
                                    + 1
                                )
                                buyer_issues[buyer_issue] = buyer_issue_target
                            if (
                                not is_available
                            ):  # Otherwise its volume, since seller can fill this buyer to green.
                                buyer_issue_target = buyer_closest_available_delivery(
                                    temp_dispatch_schedule,
                                    seller_volume,
                                    seller_delivery_time,
                                    buyer_delivery_time,
                                )
                                if buyer_issue_target != -1:
                                    buyer_issue = "Delivery"
                                    buyer_issues[buyer_issue] = buyer_issue_target
                                else:
                                    buyer_issue = "Volume"
                                    buyer_issues[
                                        buyer_issue
                                    ] = most_available_amount_in_schedule(
                                        temp_dispatch_schedule,
                                        seller_delivery_time,
                                        buyer_delivery_time,
                                    )
                            # Add buyer to the red list, with problematic issue and the target for offer.
                            dispatch_categories["Red"][buyer_id] = {
                                "Offer": buyer_offer,
                                "Issues": buyer_issues,
                            }
                            # If seller is not yellow before, we should add it to the red.
                            if seller_id not in dispatch_categories["Yellow"].keys():
                                if profit <= 0:
                                    seller_issue = "Price"
                                    seller_issue_target = (
                                        (
                                            (buyer_price - self.production_cost)
                                            * buyer_new_paid_partial_volume
                                        )
                                        - buyer_paid_partial_price
                                    ) / (seller_volume * 1.0)
                                    seller_issues[seller_issue] = seller_issue_target
                                if not is_available:
                                    seller_issue_target = (
                                        seller_closest_available_delivery(
                                            temp_dispatch_schedule,
                                            seller_volume,
                                            buyer_delivery_time,
                                        )
                                    )
                                    if seller_issue_target != -1:
                                        seller_issue = "Delivery"
                                        seller_issues[
                                            seller_issue
                                        ] = seller_issue_target
                                    else:
                                        seller_issue = "Volume"
                                        seller_issues[
                                            seller_issue
                                        ] = most_available_amount_in_schedule(
                                            temp_dispatch_schedule,
                                            seller_delivery_time,
                                            buyer_delivery_time,
                                        )
                                dispatch_categories["Red"][seller_id] = {
                                    "Offer": seller_offer,
                                    "Issues": seller_issues,
                                }

            ### TODO: CHECK HERE.

            # If every seller is finished and buyer is not in any categories (in global), meaning that it is not fullfilled.
            if (
                buyer_id not in dispatch_categories["Red"]
                and buyer_id not in dispatch_categories["Green"]
            ):
                # We should mark buyer as red, and volume should be partial volume that is added until now.
                if buyer_paid_partial_volume != 0:
                    dispatch_categories["Red"][buyer_id] = {
                        "Offer": buyer_offer,
                        "Issues": {"Volume": buyer_paid_partial_volume},
                    }
                else:
                    dispatch_categories["Red"][buyer_id] = {
                        "Offer": buyer_offer,
                        "Issues": {"No Seller": "No Seller"},
                    }

            if is_nego_with_yellow and buyer_id in dispatch_categories["Red"]:
                if "Issues" in dispatch_categories["Red"][buyer_id].keys():
                    if (
                        "Volume"
                        in dispatch_categories["Red"][buyer_id]["Issues"].keys()
                    ):
                        yellow_id = list(dispatch_categories["Yellow"].keys())[0]
                        dispatch_categories["Yellow"][yellow_id][
                            "Target"
                        ] = dispatch_categories["Yellow"][yellow_id]["Offer"][0] + (
                            buyer_volume - buyer_paid_partial_volume
                        )

        # If every buyer is finished and seller is not in any categories (in global), meaning that it is not fulfill anybody, problem is no buyer.
        for seller in sellers:
            # Get id and offer.
            seller_id = seller[0]
            seller_offer = seller[1]
            # Get offer content.
            seller_volume = seller_offer[0]
            # Add no buyer status if we could not put seller into any category, meaning that no buyer left.
            if (
                seller_id not in dispatch_categories["Red"]
                and seller_id not in dispatch_categories["Yellow"].keys()
                and seller_id not in dispatch_categories["Green"].keys()
            ):
                dispatch_categories["Red"][seller_id] = {
                    "Offer": seller_offer,
                    "Issues": {"No Buyer": "No Buyer"},
                }

        # Return final dispatch categories.
        return dispatch_categories, dispatch_partners
