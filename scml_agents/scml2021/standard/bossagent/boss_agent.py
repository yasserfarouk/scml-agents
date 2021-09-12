# required for typing
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

from negmas import Contract
from scml.scml2020 import SCML2020Agent

# Boss modules.
from .BossAWICaller import BossAWICaller
from .BossBusinessPlanner import BossBusinessPlanner
from .BossProductionStrategy import BossProductionStrategy
from .BossTradingStrategy import BossTradingStrategy

__all__ = [
    "CharliesAgent",
]


class CharliesAgent(
    BossBusinessPlanner, BossTradingStrategy, BossProductionStrategy, SCML2020Agent
):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        # Initialize AWI caller.
        self.awi_caller = BossAWICaller(parent=self)
        # AWI related things.
        self.my_input_product = self.awi_caller.get_input_product()
        self.my_output_product = self.awi_caller.get_output_product()
        self.my_suppliers = self.awi_caller.get_suppliers()
        self.my_consumers = self.awi_caller.get_consumers()
        self.num_of_suppliers = len(self.my_suppliers)
        self.num_of_consumers = len(self.my_consumers)
        self.my_competitors = self.awi_caller.get_competitors()
        self.my_production_cost = self.awi_caller.get_production_cost()
        self.max_number_of_steps = self.awi_caller.get_max_step()

        self.alias = self.id.split("@")[0][2:]

        self.is_supplier_collusion_mode = False
        self.is_consumer_collusion_mode = False

        for supplier in self.my_suppliers:
            if supplier.split("@")[0][2:] == self.alias:
                self.is_supplier_collusion_mode = True
                break

        for consumer in self.my_consumers:
            if consumer.split("@")[0][2:] == self.alias:
                self.is_consumer_collusion_mode = True
                break

        # Initialize constraints.
        if self.awi_caller.is_first_level() or self.awi_caller.is_last_level():
            self.NEGO_WINDOW = self.awi.settings["exogenous_horizon"]
        else:
            self.NEGO_WINDOW = math.ceil(
                self.max_number_of_steps * 0.4
            )  # Window size of the negotiation.

        if self.awi_caller.is_first_level():
            self.KEEP_AMOUNT = 0
        else:
            self.KEEP_AMOUNT = 0

        self.BALANCE_KEEP_RATE = 0.97

        # Pseudo buyer & seller negotiator list. That will keep contractid (since it is unique) - offer pairs.
        self.pseudo_buyers = {}
        self.pseudo_sellers = {}

        # Custom history.
        self.buyer_history_prices = []
        self.seller_history_prices = []
        self.buyer_history_delivery_times = []
        self.seller_history_delivery_times = []
        self.buyer_history_quantities = []
        self.seller_history_quantities = []

        # Keep bankrupts.
        self.bankrupt_agents = []

        # Initialize scheduled variables and pseudo.
        self.scheduled_buyer_contracts = defaultdict()
        self.scheduled_seller_contracts = defaultdict()
        self.pseudo_buyers = {}
        self.pseudo_sellers = {}

        # Keep custom stats.
        self.reject_acceptance_rate = (
            {}
        )  # Key is agent id, value is {'Acceptance': int, 'Reject': int}.

        for supplier in self.my_suppliers:
            self.reject_acceptance_rate[supplier] = {"Acceptance": 0, "Reject": {}}

        for consumer in self.my_consumers:
            self.reject_acceptance_rate[consumer] = {"Acceptance": 0, "Reject": {}}

        self.nego_result_stats = {}

        for supplier in self.my_suppliers:
            self.nego_result_stats[supplier] = {"Agreement": 0, "Disagreement": 0}

        for consumer in self.my_consumers:
            self.nego_result_stats[consumer] = {"Agreement": 0, "Disagreement": 0}

        super().init()

    # Step should include negotiation strategy etc.
    def step(self):
        """Called at every production step by the world"""
        super().step()
