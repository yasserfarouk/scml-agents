class BossBusinessAnalytics:
    def __init__(self, parent):
        # Get parent class to call AWI.
        self.__parent = parent
        # AWI Static Nego calls.
        # Bunlari initialize edip icerde kendi namingini olusturabilirsin.
        self.__parent.my_input_product
        self.__parent.my_output_product
        self.__parent.my_suppliers
        self.__parent.my_consumers
        self.__parent.my_competitors
        self.__parent.max_number_of_steps
        self.__parent.awi_caller.get_production_level()
        self.__parent.awi_caller.get_production_cost()

    def update_stats(self):
        # Nego stat history.
        # self.__parent.nego_stats.get_agent_bid_history()
        # self.__parent.nego_stats.get_opponent_bid_history()
        # self.__parent.nego_stats.get_opponent_respond_history()
        # self.__parent.nego_stats.get_agent_respond_history()
        # self.__parent.nego_stats.get_nego_acceptance_history()
        # self.__parent.nego_stats.get_nego_sign_history()
        # self.__parent.nego_stats.get_nego_breach_history()
        # AWI Dynamic Nego calls.
        self.__parent.awi_caller.get_current_step()  # Returns day number (integer), e.g. 3rd day.
        self.__parent.awi_caller.get_relative_time()  # Normalized negotiation time between [0, 1].
        self.__parent.awi_caller.get_balance()
        self.__parent.awi_caller.get_current_inventory()
        self.__parent.awi_caller.get_inventory_input_prod_quantity()
        self.__parent.awi_caller.get_inventory_output_prod_quantity()
        self.__parent.awi_caller.get_input_catalog_price()
        self.__parent.awi_caller.get_output_catalog_price()
        # Agent reports
        my_first_supplier_id = self.__parent.awi_caller.get_suppliers()[0]
        self.__parent.awi_caller.get_agent_all_reports(
            agentID=my_first_supplier_id
        )  # Returns all history until current step.
        self.__parent.awi_caller.get_latest_agent_report(
            agentID=my_first_supplier_id
        )  # Returns latest possible report of the agent.
