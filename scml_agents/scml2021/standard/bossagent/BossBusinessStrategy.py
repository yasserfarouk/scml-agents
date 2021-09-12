import copy
import math
from statistics import mean

from .helper import (
    get_negotiable_agent_rate,
    get_unscheduled_total_pseudo_quantity,
    most_available_amount_in_schedule,
)


class BossBusinessStrategy:
    """
    Core business strategy.
    """

    def __init__(self, parent):
        """
        Initialize necessary variables for bound calculations. Also format the schedule that comes from AWI.
        """
        self.__parent = parent

    def update_limitations(self):
        self.current_step = self.__parent.current_step  # Get the current step.
        self.max_number_of_steps = (
            self.__parent.max_number_of_steps
        )  # Get the negotiation step limit.
        self.formatted_schedule = (
            self.__parent.formatted_schedule
        )  # formatted schedule (AWI).
        self.normalized_time = self.__parent.normalized_time
        self.saturation_rate = self.set_saturation_rate()
        self.my_production_cost = self.__parent.my_production_cost
        self.NEGO_WINDOW = (
            self.__parent.NEGO_WINDOW
        )  # Constant number that determines production day window.
        self.production_level = self.__parent.my_input_product

    # =====================
    #   Helper functions
    # =====================

    def set_saturation_rate(self):
        """
        Helper function that calculates saturation rate at normalized time [0, 1].
        """
        return 1.0 * self.normalized_time

    def estimate_production_day(self, max_seller_quantity, start_date, finish_date):
        """
        Helper function that estimates THE production finish day, given schedule and max_seller_quantity,
        from (current_step + MAX_BUYER_DELIVERY_TIME) to (current_step + 1) (from latest day to earliest, last day excluded).
        """
        temp_schedule = copy.deepcopy(self.formatted_schedule)

        for i in reversed(range(start_date, finish_date)):
            possible_amount_at_step = temp_schedule[i]
            if possible_amount_at_step > 0:
                if max_seller_quantity - possible_amount_at_step > 0:
                    max_seller_quantity -= possible_amount_at_step
                    temp_schedule[i] = 0
                else:
                    return i  # Return the day that we can finish this production in.

        # If we cant produce anything (schedule is full), we simply dont need to do anything this step.
        return -1

    def estimate_production_quantity(self, schedule, start_date, finish_date):
        """
        Helper function that calculates max available production quantity between start_date, and finish_date (excluded) from schedule.
        """
        temp_schedule = copy.deepcopy(self.formatted_schedule)

        max_available_quantity = 0

        for i in range(start_date, finish_date):
            max_available_quantity += temp_schedule[i]

        return max_available_quantity

    # =====================
    #        Getters
    # =====================

    def get_buyer_and_seller_prices(
        self,
        seller_history_prices,
        buyer_history_prices,
        input_catalog_price,
        output_catalog_price,
    ):
        """
        Set seller & buyer price bounds for step t.
        """
        consumer_agent_negotiable_rate, _ = get_negotiable_agent_rate(
            self.__parent.my_consumers,
            self.__parent.reject_acceptance_rate,
            self.current_step,
            self.__parent.bankrupt_agents,
        )
        supplier_agent_negotiable_rate, _ = get_negotiable_agent_rate(
            self.__parent.my_suppliers,
            self.__parent.reject_acceptance_rate,
            self.current_step,
            self.__parent.bankrupt_agents,
        )

        if self.__parent.awi_caller.is_first_level():
            # Set buyer lower and upper prices.
            buyer_price_lower_bound = math.ceil(
                output_catalog_price
                * (1.25 - 0.5 * (1 - consumer_agent_negotiable_rate))
            )
            buyer_price_upper_bound = 2 * buyer_price_lower_bound
            # These 2 dont matter.
            seller_price_lower_bound = input_catalog_price
            seller_price_upper_bound = input_catalog_price
        elif self.__parent.awi_caller.is_last_level():
            # Set seller lower and upper prices.
            seller_price_lower_bound = math.floor(
                input_catalog_price * (0.5 + (1 - supplier_agent_negotiable_rate) * 0.5)
            )
            seller_price_upper_bound = input_catalog_price + 1
            # These 2 dont matter.
            buyer_price_upper_bound = output_catalog_price
            buyer_price_lower_bound = output_catalog_price
        else:
            # Set seller lower and upper prices.
            seller_price_lower_bound = math.floor(
                input_catalog_price * (0.5 + (1 - supplier_agent_negotiable_rate) * 0.5)
            )
            seller_price_upper_bound = input_catalog_price + 1
            # Set buyer lower and upper prices.
            buyer_price_lower_bound = math.ceil(
                output_catalog_price
                * (1.25 - 0.5 * (1 - consumer_agent_negotiable_rate))
            )
            buyer_price_upper_bound = math.ceil(buyer_price_lower_bound * 2)
        # Return the boundries, that will be used for sending requests.
        return (
            seller_price_upper_bound,
            seller_price_lower_bound,
            buyer_price_upper_bound,
            buyer_price_lower_bound,
        )

    def get_seller_and_buyer_quantity(
        self,
        balance,
        balance_keep_rate,
        input_catalog_price,
        seller_history_prices,
        seller_history_delivery_times,
    ):
        """Set buying and selling quantity for step t."""
        consumer_agent_negotiable_rate, _ = get_negotiable_agent_rate(
            self.__parent.my_consumers,
            self.__parent.reject_acceptance_rate,
            self.current_step,
            self.__parent.bankrupt_agents,
        )
        supplier_agent_negotiable_rate, _ = get_negotiable_agent_rate(
            self.__parent.my_suppliers,
            self.__parent.reject_acceptance_rate,
            self.current_step,
            self.__parent.bankrupt_agents,
        )

        if self.__parent.awi_caller.is_first_level():
            # Check available q in the schedule with nego window.
            available_production_quantity = self.estimate_production_quantity(
                self.formatted_schedule,
                self.min_seller_delivery_day,
                self.max_buyer_delivery_day + 1,
            )
            max_seller_quantity = math.ceil(
                available_production_quantity * 0.8 * consumer_agent_negotiable_rate
            )
            min_seller_quantity = 1
            max_buyer_quantity = math.ceil(
                (available_production_quantity * 0.8) / len(self.__parent.my_consumers)
            )
            min_buyer_quantity = 1
        elif self.__parent.awi_caller.is_last_level():
            # Check available q in the schedule with nego window.
            available_production_quantity = self.estimate_production_quantity(
                self.formatted_schedule,
                self.min_buyer_delivery_day,
                self.max_buyer_delivery_day + 1,
            )
            max_seller_quantity = math.ceil(
                (available_production_quantity * 0.5) / len(self.__parent.my_suppliers)
            )
            min_seller_quantity = 1
            max_buyer_quantity = math.ceil(
                available_production_quantity * 0.5 * supplier_agent_negotiable_rate
            )
            min_buyer_quantity = 1
        else:
            # Check available q in the schedule with nego window.
            available_production_quantity = self.estimate_production_quantity(
                self.formatted_schedule,
                self.min_buyer_delivery_day,
                self.max_buyer_delivery_day + 1,
            )
            max_seller_quantity = math.ceil(
                (available_production_quantity * 0.5) / len(self.__parent.my_suppliers)
            )
            min_seller_quantity = 1
            max_buyer_quantity = math.ceil(
                (available_production_quantity * 0.5) / len(self.__parent.my_consumers)
            )
            max_buyer_quantity = min(max_buyer_quantity, max_seller_quantity)
            min_buyer_quantity = 1

        return (
            max_seller_quantity,
            min_seller_quantity,
            max_buyer_quantity,
            min_buyer_quantity,
        )

    def get_buyer_and_seller_day(self):
        # Set start and finish date.
        if self.__parent.awi_caller.is_first_level():
            self.min_seller_delivery_day = min(
                self.max_number_of_steps - 1, self.current_step + 1
            )
            self.max_seller_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_seller_delivery_day + self.NEGO_WINDOW,
            )
            self.min_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.current_step + 1 + self.production_level,
            )
            self.max_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_buyer_delivery_day + self.NEGO_WINDOW,
            )
        elif self.__parent.awi_caller.is_last_level():
            self.min_seller_delivery_day = min(
                self.max_number_of_steps - 1, self.current_step + 1
            )
            self.max_seller_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_seller_delivery_day + self.NEGO_WINDOW,
            )
            self.min_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.current_step + 1 + self.production_level,
            )
            self.max_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_seller_delivery_day + self.NEGO_WINDOW,
            )
        else:
            self.min_seller_delivery_day = min(
                self.max_number_of_steps - 1, self.current_step + 1
            )
            self.max_seller_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_seller_delivery_day + self.NEGO_WINDOW,
            )
            self.min_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.current_step + 1 + self.production_level,
            )
            self.max_buyer_delivery_day = min(
                self.max_number_of_steps - 1,
                self.min_buyer_delivery_day + self.NEGO_WINDOW - 1,
            )

        return (
            self.max_seller_delivery_day,
            self.min_seller_delivery_day,
            self.max_buyer_delivery_day,
            self.min_buyer_delivery_day,
        )
