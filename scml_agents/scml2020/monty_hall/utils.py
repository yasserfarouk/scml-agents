from typing import Any, Dict, List, Optional, Tuple, Union

from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    SAONegotiator,
)

from .nvm_lib2.nvm_lib2 import NVMLib2


class AgentProfile:
    def __init__(self):
        self.id = None
        self.initial_balance = None

        self.input_product = None
        self.output_product = None

        self.n_lines = None  # production capacity
        self.n_processes = None  # n_products - 1
        self.n_products = None
        self.n_steps = None

        self.supplier_list = None
        self.consumer_list = None

        self.supplier_matrix = None  # first index is product
        self.consumer_matrix = None  # first index is product
        self.catalog_price_list = None

        self.process = None  # is equal to input product
        self.production_cost = None

        self.agents = None

        self.last_day = None


class AgentPlan:
    def getNVMPlan(
        self,
        n_lines,
        n_processes,
        n_steps,
        process: int,
        production_cost,
        current_time,
        current_inventory: int,
    ):
        num_periods = 5
        self.horizon = num_periods
        self.nvm = NVMLib2(
            mpnvp_number_of_periods=num_periods,
            mpnvp_quantities_domain_size=20,
            game_length=n_steps,
            input_product_index=process,
            output_product_index=process + 1,
            num_intermediate_products=n_processes,
            production_cost=production_cost,
            current_inventory=current_inventory,
            current_time=current_time,
        )
        time = 0
        if isinstance(current_time, set):
            time = current_time.pop()
        else:
            time = current_time

        # print("---------------TIME:" + str(time))

        nvm_sol = self.nvm.create_NVM_plan()
        # clear buy, sell, and produce plans
        self.buy_plan = []
        self.sell_plan = []
        self.produce_plan = []
        if nvm_sol is None:
            # print("NVM PLAN IS NONE -- THIS SHOULD NOT BE HAPPENING")
            self.buy_plan = [0] * num_periods
            self.sell_plan = [0] * num_periods
            self.produce_plan = [0] * num_periods
        else:
            # print(nvm_sol)
            # print("----------FULL BUY PLAN:" + str(nvm_sol.get_buy_plan()))
            # print("----------FULL SELL PLAN:" + str(nvm_sol.get_sell_plan()))
            # print("----------FULL PRODUCE PLAN:" + str(nvm_sol.get_production_plan()))

            self.buy_plan.append(nvm_sol.get_buy_plan_at(0))
            # print('---------BUY PLAN:' + str(self.buy_plan[0]))
            # self.sell_plan.append(nvm_sol.get_sell_plan_at(2))
            self.sell_plan.append(nvm_sol.get_sell_plan_at(0))
            # print('---------SELL PLAN:' + str(self.sell_plan[0]))
            # self.produce_plan.append(nvm_sol.get_production_plan_at(1))
            self.produce_plan.append(nvm_sol.get_production_plan_at(0))
            # print('---------PRODUCE PLAN:' + str(self.produce_plan[0]))

            # #print("BUY PLAN LENGTH" + str(len(self.buy_plan)))

    def __init__(self):
        self.target_input = None  # How much input I want to possess at each step
        self.expected_input = None
        self.min_sell_price = None
        self.available_money = None
        self.available_output = None
        self.true_input = None
        self.buy_plan = None
        self.sell_plan = None
        self.produce_plan = None
        self.NVM = None
        self.horizon = None


class BuyPlan:
    def __init__(
        self,
        agent,
        target_input,
        target_output,
        production_capacity,
        contract_inputs=0,
        excess_inputs=0,
    ):
        self.agent = agent
        self.target_input = target_input
        self.target_output = target_output
        self.production_capacity = production_capacity

        self.contract_inputs = contract_inputs  # inputs received from that day
        self.excess_inputs = (
            excess_inputs  # inputs which are left unused from previous day
        )

    def get_total(self):
        return self.contract_inputs + self.excess_inputs

    def get_needed(self):
        input_space = max(self.target_input - self.get_total(), 0)
        output_space = max(self.target_output - self.agent.plan.available_output, 0)
        return min(input_space, output_space)

    def get_excess(self):  # inputs which could be used tomorrow
        return max(
            self.get_total() - self.production_capacity, 0
        )  # Assumes we can always afford production cost


class AgentStatistics:
    def __init__(self):
        self.agent = None

        self.buy_neg_agreed_count = 0
        self.buy_neg_reject_count = 0

        self.sell_neg_agreed_count = 0
        self.sell_neg_reject_count = 0

        self.buy_both_reject = 0
        self.buy_agent_reject = 0
        self.buy_partner_reject = 0
        self.buy_both_accept = 0

        self.sell_both_reject = 0
        self.sell_agent_reject = 0
        self.sell_partner_reject = 0
        self.sell_both_accept = 0

        self.default_agent_names = ["BUYER", "SELLER"]

    def step(self):
        params = [
            self.agent.id,
            self.agent.get_balance(),
            self.agent.plan.available_money,
            self.agent.get_balance_change(),
            self.agent.get_input_inventory(),
            self.agent.get_output_inventory(),
            self.agent.plan.available_output,
            self.buy_neg_agreed_count,
            self.buy_neg_reject_count,
            self.sell_neg_agreed_count,
            self.sell_neg_reject_count,
            self.buy_both_reject,
            self.buy_agent_reject,
            self.buy_partner_reject,
            self.buy_both_reject,
            self.sell_both_reject,
            self.sell_agent_reject,
            self.sell_partner_reject,
            self.sell_both_reject,
        ]
        report = """
        ############# {}  STAT INFO #############
        Current Balance = {}  Available Balance = {}    Balance Change = {}
        Input Count = {}   Output Count = {}  Available Output = {}
        Buy: Agreed Neg = {}     Rejected Neg = {}
        Sell: Agreed Neg = {}     Rejected Neg = {}
        Buy: Both Reject = {}   Agent Reject = {}   Partner Reject = {}     Both Accept = {}
        Sell: Both Reject = {}   Agent Reject = {}   Partner Reject = {}    Both Accept = {}
        """.format(
            *params
        )

    #        #print(report)

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:

        is_buy = annotation["is_buy"]
        if is_buy:
            self.buy_neg_reject_count += 1
        else:
            self.sell_neg_reject_count += 1

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        is_buy = contract.annotation["is_buy"]
        if is_buy:
            self.buy_neg_agreed_count += 1
        else:
            self.sell_neg_agreed_count += 1

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # Note = BUYER and SELLER contracts are received without negotiation
        for contract in signed:
            annotation = contract.annotation

            if (
                annotation["buyer"] in self.default_agent_names
                or annotation["seller"] in self.default_agent_names
            ):
                continue

            if annotation["is_buy"]:
                self.buy_both_accept += 1
            else:
                self.sell_both_accept += 1

        for i in range(len(cancelled)):
            contract = cancelled[i]
            annotation = contract.annotation

            if (
                annotation["buyer"] in self.default_agent_names
                or annotation["seller"] in self.default_agent_names
            ):
                continue

            if contract.annotation["is_buy"]:
                if len(rejectors[i]) == 2:
                    self.buy_both_reject += 1
                elif self.agent.id in rejectors[i]:
                    self.buy_agent_reject += 1
                else:
                    self.buy_partner_reject += 1
            else:  # is sell
                if len(rejectors[i]) == 2:
                    self.sell_both_reject += 1
                elif self.agent.id in rejectors[i]:
                    self.sell_agent_reject += 1
                else:
                    self.sell_partner_reject += 1

    def print_supply_chain(self):
        for line in self.agent.data.supplier_matrix:
            continue
            # print(line)
        # print(self.agent.data.consumer_matrix[-1])
