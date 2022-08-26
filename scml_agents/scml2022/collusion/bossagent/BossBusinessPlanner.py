from negmas import AgentMechanismInterface
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

import copy
from collections import defaultdict

from .BossDispatch import BossDispatch
from .BossNegoStats import BossNegoStats
from .BossController import SyncController
from .BossBusinessStrategy import BossBusinessStrategy
from .BossBusinessAnalytics import BossBusinessAnalytics


from .helper import (
    format_schedule,
    calculate_total_keep_amount,
    calculate_current_keep_amount,
    seller_closest_available_delivery,
    buyer_closest_available_delivery,
    is_balance_available,
    is_schedule_available,
    get_agent_reject_rate,
    get_pseudo_seller_contracts_with_quota,
    get_pseudo_buyer_contracts_with_quota,
)


from typing import List, Dict, Optional, Any

import math


class BossBusinessPlanner:
    """
    Boss negotiation strategy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        # Create planner and negotiation related modules.
        self.nego_controller = SyncController(
            production_cost=self.my_production_cost, parent=self
        )
        # Initialize module that will keep the history.
        self.nego_stats = BossNegoStats(parent=self)
        # Initialize business strategy.
        self.business_strategy = BossBusinessStrategy(self)
        # Initiailize analytics module.
        self.business_analytics = BossBusinessAnalytics(parent=self)
        # Initialize dispatch module.
        self.dispatch_module = BossDispatch(self.my_production_cost)
        # Create an empty list where we keep track of which agents we are going to make a negotiation, so that there wont be any duplicates.
        self.negotiation_list = defaultdict(list)
        # Negotiation partner quota.
        self.nego_quota = self.awi.settings["n_concurrent_negs_between_partners"]
        # Get current step.
        self.update_constraints()
        # Init other components.
        super().init()

    def before_step(self):
        self.update_constraints()
        self.business_analytics.update_stats()

    def update_constraints(self):
        """
        The method is called before step function is called.
        We are setting boundries and getting relative informations here.
        """
        # Get max and current number of steps.
        self.current_step = self.awi_caller.get_current_step()
        # Check whether we gonna negotiate this step or not. We wont negotiate last NEGO_WINDOW steps with SELLERS.
        self.is_negotiating = True
        # Check if the step is last step or not.
        if self.current_step != self.max_number_of_steps:
            # Get schedule between (current_step + 1, max_number_of_steps)
            self.schedule = self.awi.available_for_production(
                repeats=1,
                step=(self.current_step + 1, self.max_number_of_steps),
                line=-1,
                override=False,
            )[0]
            # Get the formatted schedule for later usages.
            self.formatted_schedule = format_schedule(
                self.schedule, self.max_number_of_steps
            )
            # Get current balance.
            self.current_balance = self.awi_caller.get_balance()
            inventory_output_product = (
                self.awi_caller.get_inventory_output_prod_quantity()
            )
            # Get input and output catalog prices.
            self.input_catalog_price = self.awi_caller.get_input_catalog_price()
            self.output_catalog_price = self.awi_caller.get_output_catalog_price()
            # Get normalized time.
            self.normalized_time = self.awi_caller.get_relative_time()
            # Update the keep.
            self.KEEP_AMOUNT = math.ceil(self.KEEP_AMOUNT * (1 - self.normalized_time))
            # Initialize business strategy that will set the limitations of sending & accepting negotiation requests.
            self.business_strategy.update_limitations()
            # We need to get delivery day bounds from strategy.
            (
                self.max_seller_delivery_day,
                self.min_seller_delivery_day,
                self.max_buyer_delivery_day,
                self.min_buyer_delivery_day,
            ) = self.business_strategy.get_buyer_and_seller_day()
            # We need to get price bounds from strategy.
            (
                self.seller_price_upper_bound,
                self.seller_price_lower_bound,
                self.buyer_price_upper_bound,
                self.buyer_price_lower_bound,
            ) = self.business_strategy.get_buyer_and_seller_prices(
                self.seller_history_prices,
                self.buyer_history_prices,
                self.input_catalog_price,
                self.output_catalog_price,
            )
            # We need to get quantity bounds from strategy.
            (
                self.max_seller_quantity,
                self.min_seller_quantity,
                self.max_buyer_quantity,
                self.min_buyer_quantity,
            ) = self.business_strategy.get_seller_and_buyer_quantity(
                self.current_balance,
                self.BALANCE_KEEP_RATE,
                self.input_catalog_price,
                self.seller_history_prices,
                self.seller_history_delivery_times,
            )
            # Set limitations of the negotiation controller (SYNController).
            self.nego_controller.set_limitations(
                self.current_step,
                self.max_number_of_steps,
                self.formatted_schedule,
                inventory_output_product,
                self.seller_price_upper_bound,
                self.buyer_price_lower_bound,
            )
            # TODO: CHECK HERE AGAIN. Update max delivery dates to max negotiating date that we are doing.
            # Get total keep amount (producted + scheduled).
            self.total_keep_amount = calculate_total_keep_amount(
                self.current_step,
                self.scheduled_buyer_contracts,
                self.awi_caller.get_inventory_output_prod_quantity(),
            )
            # Get current keep amount (producted).
            self.current_keep_amount = calculate_current_keep_amount(
                self.current_step,
                self.scheduled_buyer_contracts,
                self.awi_caller.get_inventory_output_prod_quantity(),
            )
            # Create pseudo keep buyer if we lack keep amount (producted + scheduled).
            if (self.total_keep_amount < self.KEEP_AMOUNT) and (not self.pseudo_buyers):
                to_buy_amount = self.KEEP_AMOUNT - self.total_keep_amount
                self.pseudo_buyers["KEEP"] = (
                    to_buy_amount,
                    self.max_buyer_delivery_day,
                    self.awi_caller.get_output_catalog_price(),
                )
                if "KEEP" in self.pseudo_sellers.keys():
                    self.pseudo_sellers.pop("KEEP")
            # Create pseudo buyer if we have more than enough keep amount (producted).
            elif self.current_keep_amount > self.KEEP_AMOUNT:
                to_sell_amount = self.current_keep_amount - self.KEEP_AMOUNT
                self.pseudo_sellers["KEEP"] = (
                    to_sell_amount,
                    -1,
                    self.seller_price_upper_bound,
                )
                if "KEEP" in self.pseudo_buyers.keys():
                    self.pseudo_buyers.pop("KEEP")

    def step(self):
        """
        This method is called to process the day, so we are sending our negotiation requests here.
        """
        super().step()

        if self.awi_caller.is_first_level():
            # Iterate & send requests to buyers.
            for buyer_id in self.my_consumers:
                # Get agent reject rate.
                agent_reject_rate = get_agent_reject_rate(
                    buyer_id, self.reject_acceptance_rate, self.current_step
                )
                # Keep list of agent answers.
                opp_answer_list = []
                # First send pseudo_sellers as negotiation requests if there is any.
                if (
                    self.pseudo_sellers
                    and buyer_id not in self.bankrupt_agents
                    and agent_reject_rate != -1
                ):
                    if "KEEP" in self.pseudo_sellers and len(self.pseudo_sellers) > 1:
                        temp_pseudo_sellers = copy.deepcopy(self.pseudo_sellers)
                        temp_pseudo_sellers.pop("KEEP")
                        formatted_pseudos = get_pseudo_seller_contracts_with_quota(
                            self.formatted_schedule,
                            temp_pseudo_sellers,
                            self.nego_quota - 2,
                            self.current_step,
                            self.max_buyer_delivery_day,
                        )
                        formatted_pseudos.append((-1, self.pseudo_sellers["KEEP"][0]))
                    else:
                        formatted_pseudos = get_pseudo_seller_contracts_with_quota(
                            self.formatted_schedule,
                            self.pseudo_sellers,
                            self.nego_quota - 1,
                            self.current_step,
                            self.max_buyer_delivery_day,
                        )

                    for (offer_t, offer_q) in formatted_pseudos:
                        if offer_t != -1:  # Check if the contract is keep.
                            buyer_delivery_lower = buyer_closest_available_delivery(
                                self.formatted_schedule,
                                offer_q,
                                offer_t,
                                self.max_buyer_delivery_day,
                            )
                            if buyer_delivery_lower > self.current_step + 1:
                                # If it is ok, send the negotiation request with boundries that are determined in before step.
                                if buyer_delivery_lower >= self.max_buyer_delivery_day:
                                    time_interval = buyer_delivery_lower
                                else:
                                    time_interval = (
                                        buyer_delivery_lower,
                                        self.max_buyer_delivery_day,
                                    )
                                is_accepted = self.awi.request_negotiation(
                                    is_buy=False,
                                    product=self.my_output_product,
                                    quantity=(1, offer_q),
                                    unit_price=(
                                        self.buyer_price_lower_bound,
                                        self.buyer_price_upper_bound,
                                    ),
                                    time=time_interval,
                                    partner=buyer_id,
                                    negotiator=self.nego_controller.create_negotiator(),
                                )
                                # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                                agent_action = 1 if is_accepted else 0
                                opp_answer_list.append(agent_action)
                        else:
                            # If it is ok, send the negotiation request with boundries that are determined in before step.
                            is_accepted = self.awi.request_negotiation(
                                is_buy=False,
                                product=self.my_output_product,
                                quantity=(1, offer_q),
                                unit_price=(
                                    self.buyer_price_lower_bound,
                                    self.buyer_price_upper_bound,
                                ),
                                time=(
                                    self.current_step + 1,
                                    self.max_buyer_delivery_day,
                                ),
                                partner=buyer_id,
                                negotiator=self.nego_controller.create_negotiator(),
                            )
                            # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                            agent_action = 1 if is_accepted else 0
                            opp_answer_list.append(agent_action)
                else:
                    opp_answer_list.append(-1)
                # Add respond history.
                self.nego_stats.set_opponent_respond_history(
                    buyer_id, self.current_step, opp_answer_list
                )

        if self.awi_caller.is_middle_level():
            # Iterate & send requests to buyers.
            for buyer_id in self.my_consumers:
                # Get agent reject rate.
                agent_reject_rate = get_agent_reject_rate(
                    buyer_id, self.reject_acceptance_rate, self.current_step
                )
                # Add opp answer list.
                opp_answer_list = []
                # First send pseudo_sellers as negotiation requests if there is any.
                if (
                    self.pseudo_sellers
                    and buyer_id not in self.bankrupt_agents
                    and agent_reject_rate != -1
                ):
                    if "KEEP" in self.pseudo_sellers and len(self.pseudo_sellers) > 1:
                        temp_pseudo_sellers = copy.deepcopy(self.pseudo_sellers)
                        temp_pseudo_sellers.pop("KEEP")
                        formatted_pseudos = get_pseudo_seller_contracts_with_quota(
                            self.formatted_schedule,
                            temp_pseudo_sellers,
                            self.nego_quota - 2,
                            self.current_step,
                            self.max_buyer_delivery_day,
                        )
                        formatted_pseudos.append((-1, self.pseudo_sellers["KEEP"][0]))
                    else:
                        formatted_pseudos = get_pseudo_seller_contracts_with_quota(
                            self.formatted_schedule,
                            self.pseudo_sellers,
                            self.nego_quota - 1,
                            self.current_step,
                            self.max_buyer_delivery_day,
                        )

                    for (offer_t, offer_q) in formatted_pseudos:
                        if offer_t != -1:  # Check if the contract is keep.
                            buyer_delivery_lower = buyer_closest_available_delivery(
                                self.formatted_schedule,
                                offer_q,
                                offer_t,
                                self.max_buyer_delivery_day,
                            )
                            if buyer_delivery_lower > self.current_step + 1:
                                # If it is ok, send the negotiation request with boundries that are determined in before step.
                                if buyer_delivery_lower >= self.max_buyer_delivery_day:
                                    time_interval = buyer_delivery_lower
                                else:
                                    time_interval = (
                                        buyer_delivery_lower,
                                        self.max_buyer_delivery_day,
                                    )
                                is_accepted = self.awi.request_negotiation(
                                    is_buy=False,
                                    product=self.my_output_product,
                                    quantity=(1, offer_q),
                                    unit_price=(
                                        self.buyer_price_lower_bound,
                                        self.buyer_price_upper_bound,
                                    ),
                                    time=time_interval,
                                    partner=buyer_id,
                                    negotiator=self.nego_controller.create_negotiator(),
                                )
                                # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                                agent_action = 1 if is_accepted else 0
                                opp_answer_list.append(agent_action)
                        else:
                            # If it is ok, send the negotiation request with boundries that are determined in before step.
                            is_accepted = self.awi.request_negotiation(
                                is_buy=False,
                                product=self.my_output_product,
                                quantity=(1, offer_q),
                                unit_price=(
                                    self.buyer_price_lower_bound,
                                    self.buyer_price_upper_bound,
                                ),
                                time=(
                                    self.current_step + 1,
                                    self.max_buyer_delivery_day,
                                ),
                                partner=buyer_id,
                                negotiator=self.nego_controller.create_negotiator(),
                            )
                            # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                            agent_action = 1 if is_accepted else 0
                            opp_answer_list.append(agent_action)

                if (
                    0 <= agent_reject_rate <= 0.75
                    and buyer_id not in self.bankrupt_agents
                ):
                    # If it is ok, send the negotiation request with boundries that are determined in before step.
                    is_accepted = self.awi.request_negotiation(
                        is_buy=False,
                        product=self.my_output_product,
                        quantity=(self.min_buyer_quantity, self.max_buyer_quantity),
                        unit_price=(
                            self.buyer_price_lower_bound,
                            self.buyer_price_upper_bound,
                        ),
                        time=(self.min_buyer_delivery_day, self.max_buyer_delivery_day),
                        partner=buyer_id,
                        negotiator=self.nego_controller.create_negotiator(),
                    )
                    # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                    agent_action = 1 if is_accepted else 0
                    opp_answer_list.append(agent_action)
                    # Add respond history.
                    self.nego_stats.set_opponent_respond_history(
                        buyer_id, self.current_step, opp_answer_list
                    )

            # Iterate & send requests to sellers.
            for seller_id in self.my_suppliers:
                # Get agent reject rate.
                agent_reject_rate = get_agent_reject_rate(
                    seller_id, self.reject_acceptance_rate, self.current_step
                )
                # First send pseudo_buyers as negotiation requests if there is any.
                if (
                    self.pseudo_buyers
                    and seller_id not in self.bankrupt_agents
                    and agent_reject_rate != -1
                ):
                    formatted_pseudos = (
                        formatted_pseudos
                    ) = get_pseudo_buyer_contracts_with_quota(
                        self.formatted_schedule, self.pseudo_buyers, self.nego_quota - 1
                    )
                    for offer_t, offer_q in formatted_pseudos:
                        seller_delivery_upper = seller_closest_available_delivery(
                            self.formatted_schedule, offer_q, offer_t
                        )
                        if (
                            seller_delivery_upper != -1
                            and seller_delivery_upper < offer_t
                        ):
                            # If it is ok, send the negotiation request with boundries that are determined in before step.
                            is_accepted = self.awi.request_negotiation(
                                is_buy=True,
                                product=self.my_input_product,
                                quantity=(1, offer_q),
                                unit_price=(
                                    self.seller_price_lower_bound,
                                    self.seller_price_upper_bound,
                                ),
                                time=(
                                    self.min_seller_delivery_day,
                                    seller_delivery_upper,
                                ),
                                partner=seller_id,
                                negotiator=self.nego_controller.create_negotiator(),
                            )
                            # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                            agent_action = 1 if is_accepted else 0
                            opp_answer_list.append(agent_action)

                if (
                    0 <= agent_reject_rate <= 0.75
                    and seller_id not in self.bankrupt_agents
                ):
                    # If it is ok, send the negotiation request with boundries that are determined in before step.
                    is_accepted = self.awi.request_negotiation(
                        is_buy=True,
                        product=self.my_input_product,
                        quantity=(self.min_seller_quantity, self.max_seller_quantity),
                        unit_price=(
                            self.seller_price_lower_bound,
                            self.seller_price_upper_bound,
                        ),
                        time=(
                            self.min_seller_delivery_day,
                            self.max_seller_delivery_day,
                        ),
                        partner=seller_id,
                        negotiator=self.nego_controller.create_negotiator(),
                    )
                    # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                    agent_action = 1 if is_accepted else 0
                    opp_answer_list.append(agent_action)

                    # Add respond history.
                    self.nego_stats.set_opponent_respond_history(
                        seller_id, self.current_step, opp_answer_list
                    )

        # Check if we are negotiating before sending any requests to sellers.
        if self.awi_caller.is_last_level():
            # Iterate & send requests to sellers.
            for seller_id in self.my_suppliers:
                # Opp answer list.
                opp_answer_list = []
                # Get agent reject rate.
                agent_reject_rate = get_agent_reject_rate(
                    seller_id, self.reject_acceptance_rate, self.current_step
                )
                # First send pseudo_buyers as negotiation requests if there is any.
                if (
                    self.pseudo_buyers
                    and seller_id not in self.bankrupt_agents
                    and agent_reject_rate != -1
                ):
                    formatted_pseudos = get_pseudo_buyer_contracts_with_quota(
                        self.formatted_schedule, self.pseudo_buyers, self.nego_quota - 1
                    )
                    for offer_t, offer_q in formatted_pseudos:
                        seller_delivery_upper = seller_closest_available_delivery(
                            self.formatted_schedule, offer_q, offer_t
                        )
                        if (
                            seller_delivery_upper != -1
                            and seller_delivery_upper < offer_t
                        ):
                            # If it is ok, send the negotiation request with boundries that are determined in before step.
                            is_accepted = self.awi.request_negotiation(
                                is_buy=True,
                                product=self.my_input_product,
                                quantity=(1, offer_q),
                                unit_price=(
                                    self.seller_price_lower_bound,
                                    self.seller_price_upper_bound,
                                ),
                                time=(
                                    self.min_seller_delivery_day,
                                    seller_delivery_upper,
                                ),
                                partner=seller_id,
                                negotiator=self.nego_controller.create_negotiator(),
                            )
                            # Also keep in history that we did send request to this agent at this step, and keep the respond of the opponent.
                            agent_action = 1 if is_accepted else 0
                            opp_answer_list.append(agent_action)

                # Add respond history.
                self.nego_stats.set_opponent_respond_history(
                    seller_id, self.current_step, opp_answer_list
                )

    def get_agent_risk_score(self, agent_id):
        """
        Gets agent id, returns most frequent risk score of the agent.
        """
        agent_reports = self.awi.reports_of_agent(agent_id)

        if agent_reports is not None:
            latest_report_step = list(agent_reports.keys())[-1]
            breach_prob = agent_reports[latest_report_step].breach_prob
            breach_level = agent_reports[latest_report_step].breach_level
            agent_risk_score = (
                breach_level + breach_prob * (latest_report_step % 5) * 0.5
            )  ## TODO: CHANGE THIS LATER TO PUBLISH FREQ.
        else:
            # If there is no history already, set agent risk score to 0.
            agent_risk_score = 0.0

        return agent_risk_score

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        """
        Check whether the partner is negotiable or not.
        """

        if initiator not in self.bankrupt_agents:
            agent_reject_rate = get_agent_reject_rate(
                initiator, self.reject_acceptance_rate, self.current_step
            )
            # Check whether requester is seller or buyer.
            if self.id == annotation["seller"] and agent_reject_rate != -1:
                # Check whether requester's negotiation boundries are negotiable.
                if (
                    (self.buyer_price_lower_bound <= issues[UNIT_PRICE].max_value)
                    and (self.min_buyer_delivery_day <= issues[TIME].max_value)
                    and (self.max_buyer_quantity >= issues[QUANTITY].max_value)
                    and (
                        is_schedule_available(
                            self.formatted_schedule,
                            issues[QUANTITY].min_value,
                            self.min_seller_delivery_day,
                            issues[TIME].max_value,
                        )
                    )
                    and (
                        agent_reject_rate <= 0.5
                        or self.current_step <= 10
                        or self.pseudo_sellers
                    )
                    and (
                        initiator
                        not in self.negotiation_list[self.awi_caller.get_current_step()]
                    )
                ):
                    # Add to the nego list.
                    self.negotiation_list[self.awi_caller.get_current_step()].append(
                        initiator
                    )
                    # Add to agent respond.
                    self.nego_stats.set_agent_respond_history(
                        initiator, self.current_step, 1
                    )
                    # Accept the negotiation by sending a negotiator.
                    return self.nego_controller.create_negotiator()
                else:
                    # Add to agent respond.
                    self.nego_stats.set_agent_respond_history(
                        initiator, self.current_step, 0
                    )
                    # Reject the negotiation by sending None.
                    return None
            # If requester is seller.
            elif agent_reject_rate != -1:
                # Check whether requester's negotiation boundries are negotiable.
                if (
                    (self.seller_price_upper_bound >= issues[UNIT_PRICE].min_value)
                    and (self.max_seller_delivery_day >= issues[TIME].min_value)
                    and (self.max_seller_quantity >= issues[QUANTITY].max_value)
                    and (
                        is_balance_available(
                            self.current_balance,
                            self.my_production_cost,
                            issues[QUANTITY].min_value,
                            issues[UNIT_PRICE].min_value,
                        )
                    )
                    and (
                        is_schedule_available(
                            self.formatted_schedule,
                            issues[QUANTITY].min_value,
                            issues[TIME].min_value,
                            self.max_buyer_delivery_day,
                        )
                    )
                    and (
                        get_agent_reject_rate(
                            initiator, self.reject_acceptance_rate, self.current_step
                        )
                        <= 0.5
                        or self.current_step <= 10
                        or self.pseudo_buyers
                    )
                    and (
                        initiator
                        not in self.negotiation_list[self.awi_caller.get_current_step()]
                    )
                ):
                    # Add to agent respond.
                    self.nego_stats.set_agent_respond_history(
                        initiator, self.current_step, 1
                    )
                    # Add to the nego list.
                    self.negotiation_list[self.awi_caller.get_current_step()].append(
                        initiator
                    )
                    # Accept the negotiation by sending a negotiator.
                    return self.nego_controller.create_negotiator()
                else:
                    # Add to agent respond.
                    self.nego_stats.set_agent_respond_history(
                        initiator, self.current_step, 0
                    )
                    # Reject the negotiation by sending None.
                    return None
            else:
                # Add to agent respond.
                self.nego_stats.set_agent_respond_history(
                    initiator, self.current_step, 0
                )
                return None
