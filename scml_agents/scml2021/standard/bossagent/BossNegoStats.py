import copy
import json
from collections import defaultdict

import pandas as pd


class BossNegoStats:
    def __init__(self, parent):
        """
        Keeps track of the negotiation history.
        """
        self.__parent = parent

        self.negotiation_bid_history = {}
        self.agent_respond_history = {}
        self.opponent_respond_history = {}

        self.negotiation_bid_history["buyer"] = {}
        self.agent_respond_history["buyer"] = {}
        self.opponent_respond_history["buyer"] = {}
        self.negotiation_bid_history["seller"] = {}
        self.agent_respond_history["seller"] = {}
        self.opponent_respond_history["seller"] = {}

        for step in range(0, self.__parent.max_number_of_steps):
            self.negotiation_bid_history["buyer"][step] = {}
            self.agent_respond_history["buyer"][step] = {}
            self.opponent_respond_history["buyer"][step] = {}
            self.negotiation_bid_history["seller"][step] = {}
            self.agent_respond_history["seller"][step] = {}
            self.opponent_respond_history["seller"][step] = {}
            for buyer_id in self.__parent.my_consumers:
                self.negotiation_bid_history["buyer"][step][buyer_id] = {}
                self.agent_respond_history["buyer"][step][buyer_id] = []
                self.opponent_respond_history["buyer"][step][buyer_id] = []
            for seller_id in self.__parent.my_suppliers:
                self.negotiation_bid_history["seller"][step][seller_id] = {}
                self.agent_respond_history["seller"][step][seller_id] = []
                self.opponent_respond_history["seller"][step][seller_id] = []

    # =====================================
    #   Agent Bid History Getter & Setter
    # =====================================

    def set_negotiation_bid_history(
        self, step, agentID, mechanism_id, negotiation_history
    ):
        # print("My consumers: ", self.__parent.my_consumers)
        # print("My suppliers: ", self.__parent.my_suppliers)
        # print("Buyer Nego bid history: ", self.negotiation_bid_history['buyer'][step])
        # print("Seller nego bid history: ", self.negotiation_bid_history['seller'][step])
        if agentID in self.__parent.my_consumers:
            self.negotiation_bid_history["buyer"][step][agentID][
                mechanism_id
            ] = copy.deepcopy(negotiation_history)
        else:
            self.negotiation_bid_history["seller"][step][agentID][
                mechanism_id
            ] = copy.deepcopy(negotiation_history)

    def set_negotiation_bid_sign(
        self, step, agentID, mechanism_id, agent_sign, opponent_sign
    ):
        try:
            if agentID in self.__parent.my_consumers:
                self.negotiation_bid_history["buyer"][step][agentID][mechanism_id][
                    "Acceptance"
                ]["agent_sign"] = agent_sign
            else:
                self.negotiation_bid_history["seller"][step][agentID][mechanism_id][
                    "Acceptance"
                ]["opponent_sign"] = opponent_sign
        except:
            return

        # Save it as json.
        self.save_dict(
            dict_to_save=self.negotiation_bid_history,
            file_name="negotiation_bid_history",
        )

    def get_negotiation_bid_history(self, step, agentID):
        if agentID in self.__parent.my_consumers:
            return self.negotiation_bid_history["buyer"][step][agentID]
        else:
            return self.negotiation_bid_history["seller"][step][agentID]

    # ============================================
    #   Opponent Respond History Getter & Setter
    # ============================================

    def set_opponent_respond_history(self, agentID, step, action_list):
        """
        Sets opponent's respond history given its agentID and step with the action in that time step.
        Where action is 0: rejection, 1: acceptance, -1: we did not send that turn.
        """
        if agentID in self.__parent.my_consumers:
            self.opponent_respond_history["buyer"][step][agentID] = copy.deepcopy(
                action_list
            )
        else:
            self.opponent_respond_history["seller"][step][agentID] = copy.deepcopy(
                action_list
            )

        # Save it as json.
        self.save_dict(
            dict_to_save=self.opponent_respond_history,
            file_name="opponent_respond_history",
        )

    def get_opponent_respond_history(self):
        """
        Return opponent's respond history given its agentID and which simulation step.
        """
        return self.opponent_respond_history

    # ============================================
    #    Agent Respond History Getter & Setter
    # ============================================

    def set_agent_respond_history(self, agentID, step, action_list):
        """
        Sets our agent's respond history given opponent's agentID and step with the action in that time step.
        Where action is 0: rejection, 1: acceptance, -1: we did not send that turn.
        """
        if agentID in self.__parent.my_consumers:
            self.agent_respond_history["buyer"][step][agentID] = copy.deepcopy(
                action_list
            )
        else:
            self.agent_respond_history["seller"][step][agentID] = copy.deepcopy(
                action_list
            )
        # Save it as json.
        self.save_dict(
            dict_to_save=self.agent_respond_history, file_name="agent_respond_history"
        )

    def get_agent_respond_history(self):
        """
        Return agent's respond history given opponent's agentID and which simulation step.
        """
        return self.agent_respond_history

    # ============================================
    #    			JSON File Saver
    # ============================================

    def save_dict(self, dict_to_save, file_name):
        default_dict = self.default_to_regular(dict_to_save)
        # Save dict as json.
        # json_file_path = self.__parent.agent_log_folder_name + '/' + file_name + '.json'
        # with open(json_file_path, 'w') as f:
        # 	try:
        # 		json.dump(default_dict, f, indent=4)
        # 	except Exception as e:
        # 		print(f"JSON DUMP ERROR: {e}")

        # Save dict as csv.
        # csv_file_path = self.__parent.agent_log_folder_name + '/' + file_name + '.csv'
        # df = pd.read_json(json.dumps(dict))
        # df.to_csv(csv_file_path, index=False)

    def default_to_regular(self, d):
        """Converts default dict to regular dict."""
        if isinstance(d, defaultdict):
            d = {k: self.default_to_regular(v) for k, v in d.items()}
        return d
