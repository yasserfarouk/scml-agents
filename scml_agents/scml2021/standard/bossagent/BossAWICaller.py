class BossAWICaller:
    def __init__(self, parent):
        self.__parent = parent

    # =========================================
    #      Negotiation Static Information
    # =========================================

    def is_last_level(self):
        return self.__parent.awi.is_last_level

    def is_first_level(self):
        return self.__parent.awi.is_first_level

    def is_middle_level(self):
        return self.__parent.awi.is_middle_level

    def get_max_step(self):
        return self.__parent.awi.n_steps

    def get_input_product(self):
        return self.__parent.awi.my_input_product

    def get_output_product(self):
        return self.__parent.awi.my_output_product

    def get_suppliers(self):
        return self.__parent.awi.my_suppliers

    def get_consumers(self):
        return self.__parent.awi.my_consumers

    def get_competitors(self):
        return self.__parent.awi.all_consumers[self.get_input_product()]

    def get_production_level(self):
        """
        Return my input product index, since it is same with the level.
        """
        return self.__parent.awi.my_input_product

    def get_production_cost(self):
        return self.__parent.awi.profile.costs[0][self.get_input_product()]

    # =========================================
    #      Negotiation Dynamic Information
    # =========================================

    def get_current_step(self):
        return self.__parent.awi.current_step

    def get_relative_time(self):
        return self.__parent.awi.relative_time

    def get_state(self):
        return self.__parent.awi.state

    def get_balance(self):
        return self.__parent.awi.state.balance

    def get_current_inventory(self):
        return self.__parent.awi.current_inventory

    def get_inventory_input_prod_quantity(self):
        return self.__parent.awi.current_inventory[self.get_input_product()]

    def get_inventory_output_prod_quantity(self):
        return self.__parent.awi.current_inventory[self.get_output_product()]

    def get_input_catalog_price(self):
        return self.__parent.awi.catalog_prices[self.get_input_product()]

    def get_output_catalog_price(self):
        return self.__parent.awi.catalog_prices[self.get_output_product()]

    def get_agent_all_reports(self, agentID):
        return self.__parent.awi.reports_of_agent(agentID)

    def get_latest_agent_report(self, agentID):
        agent_reports = self.__parent.awi.reports_of_agent(agentID)
        if agent_reports is not None:
            latest_report_step = list(agent_reports.keys())[-1]
            return agent_reports[latest_report_step]
        else:
            return {}

    # def get_input_catalog_prices(self):
