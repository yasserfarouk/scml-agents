import numpy as np
from sklearn import linear_model as lm


class BossBusinessAnalytics:
    def __init__(self, parent):
        # Get parent class to call AWI.
        self.level_info = {}  # update
        self.__parent = parent
        # AWI Static Nego calls.
        # Bunlari initialize edip icerde kendi namingini olusturabilirsin.
        self.my_input_product = self.__parent.my_input_product
        self.my_output_product = self.__parent.my_output_product
        self.my_suppliers = self.__parent.my_suppliers
        self.my_consumers = self.__parent.my_consumers
        self.my_competitors = self.__parent.my_competitors
        self.max_number_of_steps = self.__parent.max_number_of_steps
        self.production_level = self.__parent.awi_caller.get_production_level()
        self.production_cost = self.__parent.awi_caller.get_production_cost()

    def update_stats(self):  # for reference
        # Nego stat history.
        # nego bid history.
        opponent_respond_history = (
            self.__parent.nego_stats.get_opponent_respond_history()
        )
        agent_respond_history = self.__parent.nego_stats.get_agent_respond_history()

        # AWI Dynamic Nego calls.

        current_step = (
            self.__parent.awi_caller.get_current_step()
        )  # Returns day number (integer), e.g. 3rd day.
        relative_time = (
            self.__parent.awi_caller.get_relative_time()
        )  # Normalized negotiation time between [0, 1].
        my_balance = self.__parent.awi_caller.get_balance()  # bizim para
        current_inventory = (
            self.__parent.awi_caller.get_current_inventory()
        )  # list -> [0, 0, 3, 5] index prod levellari gosteriyor
        inventory_input_prod_quantity = (
            self.__parent.awi_caller.get_inventory_input_prod_quantity()
        )  # bizim
        inventory_output_prod_quantity = (
            self.__parent.awi_caller.get_inventory_output_prod_quantity()
        )  # bizim
        input_catalog_price = self.__parent.awi_caller.get_input_catalog_price()  # this
        output_catalog_price = (
            self.__parent.awi_caller.get_output_catalog_price()
        )  # this

        # Agent reports

        my_first_supplier_id = self.__parent.awi_caller.get_suppliers()[0]
        all_reports = self.__parent.awi_caller.get_agent_all_reports(
            agentID=my_first_supplier_id
        )  # Returns all history until current step.
        latest_report = self.__parent.awi_caller.get_latest_agent_report(
            agentID=my_first_supplier_id
        )  # Returns latest possible report of the agent.

    def pred_cash(self, agent_id, step_to_pred):

        current_step = self.__parent.awi_caller.get_current_step()
        all_reports = self.__parent.awi.reports_of_agent(agent_id)

        if step_to_pred == 0:
            return 0

        cash = []
        # Append first cash value in the report to cash list
        report_key = list(all_reports.keys())[0]  # 0. report 1. gunü
        cash.append(all_reports[report_key].cash)

        if current_step < 6:
            for t in range(1, current_step + 1):
                cash.append(cash[-1])

        else:
            for t in range(0, current_step - 5):  # current step: 16 -> t en son: 10
                prev_cash = all_reports[int(t - (t % 5))].cash
                next_cash = all_reports[int((t + 5) - (t % 5))].cash
                val_to_subtract = (prev_cash - next_cash) / 5
                softened = cash[-1] - val_to_subtract
                cash.append(softened)
            for t in range(current_step - 5, current_step):  # t -> 11, 12, 13, 14, 15
                softened = cash[-1] - val_to_subtract
                cash.append(softened)

        x = np.array(range(1, len(cash) + 1))
        x = x.reshape(len(x), 1)
        regressor = lm.LinearRegression(fit_intercept=True).fit(x, np.array(cash))
        x_pred = np.array(range(current_step, step_to_pred + 1))

        x_pred = x_pred.reshape(len(x_pred), 1)
        y_pred = regressor.predict(x_pred)

        pred_cash = y_pred[-1]

        return pred_cash

    def pred_bankruptcy_day(self, agent_id):

        current_step = self.__parent.awi_caller.get_current_step()
        all_reports = self.__parent.awi.reports_of_agent(agent_id)

        # Append first cash value in the report to cash list
        cash = []
        report_key = list(all_reports.keys())[0]
        cash.append(all_reports[report_key].cash)

        if current_step < 6:
            for t in range(1, current_step + 1):
                cash.append(cash[-1])

        else:
            for t in range(0, current_step - 5):  # current step: 16 -> t en son: 10
                prev_cash = all_reports[int(t - (t % 5))].cash
                next_cash = all_reports[int((t + 5) - (t % 5))].cash
                val_to_subtract = (prev_cash - next_cash) / 5
                softened = cash[-1] - val_to_subtract
                cash.append(softened)

            for t in range(current_step - 5, current_step):  # t -> 11, 12, 13, 14, 15
                softened = cash[-1] - val_to_subtract
                cash.append(softened)

        x = np.array(range(1, len(cash) + 1))
        x = x.reshape(len(x), 1)
        regressor = lm.LinearRegression(fit_intercept=True).fit(x, np.array(cash))

        intercept = regressor.intercept_
        slope = regressor.coef_
        # Yi = intercept + (slope * Xi)
        # print(f'Slope: {slope}, Intercept: {intercept}')
        if slope != 0:
            predicted_day = (0 - intercept) / slope
        else:
            predicted_day = (0 - intercept) / (slope - 0.001)

        return int(predicted_day)

    def pred_issues(self, agent_id):
        agent_role = ""
        if agent_id in self.my_consumers:
            agent_role = "seller"
        elif agent_id in self.my_suppliers:
            agent_role = "buyer"

        negotiation_bid_history = self.__parent.nego_stats.negotiation_bid_history

        accepted_p = []
        accepted_q = []
        accepted_t = []

        for day in negotiation_bid_history[agent_role].keys():
            # print(f"================ DAY{day} ================")
            for agent in negotiation_bid_history[agent_role][day].keys():
                # print(f"--> AgentID: {agent}")
                if agent == agent_id:
                    contracts = list(
                        negotiation_bid_history[agent_role][day][agent].values()
                    )
                    # print(f"----> Current agent is our agent! For day {day}, {agent} has {len(contracts)} contracts.")
                    for contract in contracts:
                        # print(f"------>> Contract: Has {len(contract.keys()) - 1} negotiation rounds")
                        if ("accept" in list(contract["Acceptance"].keys())) and (
                            contract["Acceptance"]["accept"] == 1
                        ):
                            accepted_p.append(contract["Acceptance"]["p"])
                            accepted_q.append(contract["Acceptance"]["q"])
                            accepted_t.append(contract["Acceptance"]["t"])

        if len(accepted_p) == 0:
            return -1, -1, -1

        # Predict Q
        x_q = np.array(range(len(accepted_q)))
        x_q = x_q.reshape(len(x_q), 1)
        regressor_q = lm.LinearRegression(fit_intercept=True).fit(
            x_q, np.array(accepted_q)
        )
        x_pred_q = np.array(len(accepted_q))
        x_pred_q = x_pred_q.reshape(
            1, 1
        )  # x_pred_q = x_pred_q.reshape(len(x_pred_q), 1)
        q_pred = regressor_q.predict(x_pred_q)

        # Predict P
        x_p = np.array(range(len(accepted_p)))
        x_p = x_q.reshape(len(x_p), 1)
        regressor_p = lm.LinearRegression(fit_intercept=True).fit(
            x_p, np.array(accepted_p)
        )
        x_pred_p = np.array(len(accepted_p))
        x_pred_p = x_pred_p.reshape(1, 1)
        p_pred = regressor_p.predict(x_pred_p)

        # Predict t
        x_t = np.array(range(len(accepted_t)))
        x_t = x_t.reshape(len(x_t), 1)
        regressor_t = lm.LinearRegression(fit_intercept=True).fit(
            x_t, np.array(accepted_t)
        )
        x_pred_t = np.array(len(accepted_t))
        x_pred_t = x_pred_t.reshape(1, 1)
        t_pred = regressor_t.predict(x_pred_t)

        return int(q_pred), int(p_pred), int(t_pred)
