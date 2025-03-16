"""
**Submitted to ANAC 2021 SCML**
*Authors* Koki Katagiri <k.katagiri.733@stn.nitech.ac.jp>

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition. This version will use subcomponents. Please refer to the
[game description](http://www.yasserm.com/scml/scml2021.pdf) for all the
callbacks and subcomponents available.

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu


To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/myagent.py

1. Install a venv (recommended)
>> python3 -m venv .venv

2. Activate the venv (required if you installed a venv)
On Linux/Mac:
    >> source .venv/bin/activate
On Windows:
    >> \\.venv\\Scripts\activate.bat

3. Update pip just in case (recommended)

>> pip install -U pip wheel

4. Install SCML

>> pip install scml

5. [Optional] Install last year's agents for STD/COLLUSION tracks only

>> pip install scml-agents

6. Run the script with no parameters (assuming you are )

>> python /{path-to-this-file}/myagent.py

You should see a short tournament running and results reported.
"""

# required for development
from typing import Any, Dict, List, Optional

from negmas import ResponseType, SAOState
from negmas.common import AgentMechanismInterface, MechanismState
from negmas.negotiators import Controller
from negmas.sao.negotiators import SAONegotiator
from negmas.situated import Contract
from scml.scml2020 import SCML2020Agent
from scml.scml2020.components.production import SupplyDrivenProductionStrategy

__all__ = [
    "ArtisanKangaroo",
]


class MyTradingStrategy:
    def init(self):
        # stack for output contracts both agents signed
        self.output_contracts_stack = []
        # stack for input contracts both agents signed
        self.input_contracts_stack = []

    # add signed contracts to stack
    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # update input and output stacks
        for contract in signed:
            # make contract object for stack
            copy_contract = {
                "id": str(contract.id),
                "quantity": int(contract.agreement["quantity"]),
                "time": int(contract.agreement["time"]),
                "unit_price": contract.agreement["unit_price"],
                "partner": contract.partners,
                "correspond": [],
            }

            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                # append to output_contracts_stack because you are a seller
                self.output_contracts_stack.append(copy_contract)
            else:
                # append to output_contracts_stack because you are a buyer
                self.input_contracts_stack.append(copy_contract)

        # sort contracts_stack in time order
        self.input_contracts_stack = sorted(
            self.input_contracts_stack, key=lambda contract: contract["time"]
        )
        self.output_contracts_stack = sorted(
            self.output_contracts_stack, key=lambda contract: contract["time"]
        )

        # check if each input contract can apply to some output contracts
        for input_contract in self.input_contracts_stack:
            for output_contract in self.output_contracts_stack:
                if input_contract["quantity"] > 0:
                    if output_contract["quantity"] > 0:
                        # decide how many input products in this input contract can be used to this output contract
                        use_quantity = min(
                            input_contract["quantity"], output_contract["quantity"]
                        )
                        # update each contract quantity and correspond contracts
                        # only if the agent has capacity to produce output products until the delivery time of the output contract
                        if self.use_available_capacity(
                            input_contract["time"],
                            output_contract["time"],
                            use_quantity,
                            False,
                        ):
                            input_contract["quantity"] = (
                                input_contract["quantity"] - use_quantity
                            )
                            input_contract["correspond"].append(output_contract["id"])
                            output_contract["quantity"] = (
                                output_contract["quantity"] - use_quantity
                            )
                            output_contract["correspond"].append(input_contract["id"])
                            # break inner for statement if you used all the input products in this input contract
                            if input_contract["quantity"] <= 0:
                                break

        # update provisional capacity with quantity
        # provisional capacity will be used in sign_all_contracts()
        for capacity in self.produce_capacity:
            capacity["provisional"] = int(capacity["quantity"])

    # return how many input products will be required until given step
    def get_required_quantity(self, time):
        target_contracts = list(
            filter(
                lambda contract: contract["time"] > time, self.output_contracts_stack
            )
        )
        total_quantity = 0
        for i in range(len(target_contracts)):
            total_quantity = total_quantity + target_contracts[i]["quantity"]

        return total_quantity

    # return how many input products will be surplus in given step
    def get_surplus_quantity(self, time):
        # assumes that you can produce output products within two days since you bought the input products
        target_contracts = list(
            filter(
                lambda contract: contract["time"] + 2 < time, self.input_contracts_stack
            )
        )
        total_quantity = 0
        for i in range(len(target_contracts)):
            total_quantity = total_quantity + target_contracts[i]["quantity"]

        return total_quantity

    # check if the agent can produce output products from input arrival time to output ship time
    # is_provisional should be set True in sign_all_contracts()
    def is_available_capacity(
        self, input_arrival_time, output_ship_time, required_quantity, is_provisional
    ) -> bool:
        if input_arrival_time >= output_ship_time:
            return False

        if is_provisional:
            key = "provisional"
        else:
            key = "quantity"

        available_capacity = list(
            filter(
                lambda capacity: (capacity["time"] > input_arrival_time + 1)
                and (capacity["time"] < output_ship_time)
                and (capacity[key] > 0),
                self.produce_capacity,
            )
        )
        available_capacity = sorted(
            available_capacity, key=lambda capacity: capacity["time"]
        )
        available_capacity_sum = sum(capacity[key] for capacity in available_capacity)

        return available_capacity_sum > required_quantity

    # update the agent's capacity to produce output products from input arrival time to output ship time
    # is_provisional should be set True in sign_all_contracts()
    def use_available_capacity(
        self, input_arrival_time, output_ship_time, required_quantity, is_provisional
    ) -> bool:
        if input_arrival_time >= output_ship_time:
            return False

        if is_provisional:
            key = "provisional"
        else:
            key = "quantity"

        available_capacity = list(
            filter(
                lambda capacity: (capacity["time"] > input_arrival_time + 1)
                and (capacity["time"] < output_ship_time)
                and (capacity[key] > 0),
                self.produce_capacity,
            )
        )
        available_capacity = sorted(
            available_capacity, key=lambda capacity: capacity["time"]
        )
        available_capacity_sum = sum(capacity[key] for capacity in available_capacity)

        if available_capacity_sum < required_quantity:
            return False

        for capacity in available_capacity:
            use_quantity = min(required_quantity, capacity[key])
            required_quantity = required_quantity - use_quantity
            capacity[key] = capacity[key] - use_quantity

            if required_quantity == 0:
                return True

        return False


class MyNegotiator(SAONegotiator):
    def __init__(
        self,
        name: Optional[str] = None,
        parent: Controller = None,
        owner: "Agent" = None,
        id: Optional[str] = None,
        is_seller=None,
        current_inventory=None,
        my_input_product=None,
        my_output_product=None,
        n_lines=None,
        n_step=None,
        current_step=None,
        catalog_prices=None,
        costs=None,
        is_first_level=None,
        is_last_level=None,
        target_quantity=None,
        target_time=None,
        target_unitprice=None,
        profit_width=None,
        awi=None,
        collude=False,
    ):
        super().__init__(name=name, ufun=None, parent=None, owner=owner, id=id)
        self.__end_negotiation = False
        self.is_seller = is_seller
        self.last_offer = None
        self.current_inventory = current_inventory
        self.my_input_product = my_input_product
        self.my_output_product = my_output_product
        self.n_lines = n_lines
        self.n_step = n_step
        self.current_step = current_step
        self.catalog_prices = catalog_prices
        self.costs = costs
        self.is_first_level = is_first_level
        self.is_last_level = is_last_level
        self.target_quantity = target_quantity
        self.target_time = target_time
        self.target_unitprice = target_unitprice
        self.last_unitprice = None
        self.profit_width = profit_width
        self.awi = awi
        self.collude = collude
        self.propose_quantity = 0
        self.propose_time = 0
        self.propose_unitprice = 0

        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})

    def propose_(self, state: SAOState, dest: str | None = None) -> Optional["Outcome"]:
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        proposal = self.propose(state=state, dest=dest)
        return proposal

    def respond_(self, state: SAOState, source="") -> "ResponseType":
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        return self.respond(state=state)

    def respond(self, state: SAOState, source="") -> "ResponseType":
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        else:
            # update last_offer with partner's offer
            self.last_offer = offer

        # collusion strategy
        # accept the offer if self.collude is True which means the partner of this negotiation is the same class
        if self.collude:
            return ResponseType.ACCEPT_OFFER

        # reject if given quantity is zero
        if offer[0] <= 0:
            return ResponseType.REJECT_OFFER
        # reject if given quantity is more than max production capacity
        if offer[0] > (self.n_step - self.current_step - 1) * self.n_lines:
            return ResponseType.REJECT_OFFER

        # reject if given time is later than n_step-1
        if offer[1] > self.n_step - 1:
            return ResponseType.REJECT_OFFER
        # reject if given time does not satisfy target_time
        if offer[1] < self.target_time[0]:
            return ResponseType.REJECT_OFFER
        if offer[1] > self.target_time[1]:
            return ResponseType.REJECT_OFFER

        # in first 40% of the simulation steps, reject if given unitprice does not satisfy target_unitprice
        if self.current_step < self.n_step * 0.4:
            if self.is_seller:
                if offer[2] < self.target_unitprice:
                    return ResponseType.REJECT_OFFER
            else:
                if offer[2] > self.target_unitprice:
                    return ResponseType.REJECT_OFFER

        # check if given unit price satisfies profitable condition
        if self.is_seller:
            if self.profit_width > 0:
                if offer[2] < self.catalog_prices[self.my_output_product] - min(
                    self.profit_width // 5, 2
                ):
                    return ResponseType.REJECT_OFFER
            else:
                if (
                    offer[2]
                    <= self.catalog_prices[self.my_output_product]
                    - self.profit_width
                    + 1
                ):
                    return ResponseType.REJECT_OFFER
        else:
            if self.profit_width > 0:
                if offer[2] > self.catalog_prices[self.my_input_product] + min(
                    self.profit_width // 5, 2
                ):
                    return ResponseType.REJECT_OFFER
            else:
                if (
                    offer[2]
                    >= self.catalog_prices[self.my_input_product]
                    + self.profit_width
                    - 1
                ):
                    return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def propose(
        self, state: MechanismState, dest: str | None = None
    ) -> Optional["Outcome"]:
        # initialize last_offer
        if self.last_offer == None:
            self.last_offer = (
                self.target_quantity,
                (self.target_time[0] + self.target_time[1]) // 2,
                self.target_unitprice,
            )

        # collusion strategy
        # propose the offer if self.collude is True which means the partner of this negotiation is the same class
        # the unit price issue is its catalog price
        if self.collude:
            if self.is_seller:
                return (
                    self.target_quantity,
                    (self.target_time[0] + self.target_time[1]) // 2,
                    self.catalog_prices[self.my_output_product],
                )
            else:
                return (
                    self.target_quantity,
                    (self.target_time[0] + self.target_time[1]) // 2,
                    self.catalog_prices[self.my_input_product],
                )

        self.propose_quantity = self.target_quantity
        # propose_time will be the delivery time which the partner is offering
        self.propose_time = self.last_offer[1]
        self.propose_unitprice = self.target_unitprice

        return (self.propose_quantity, self.propose_time, self.propose_unitprice)

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]


class ArtisanKangaroo(
    SupplyDrivenProductionStrategy,
    MyTradingStrategy,
    SCML2020Agent,
):
    # make buy/sell requests to all supplies/consumers
    def request(
        self,
        is_sell,
        qvalues,
        uvalues,
        tvalues,
        target_quantity,
        target_time,
        target_unitprice,
    ):
        if target_quantity <= 0:
            return

        if is_sell:
            partners = self.awi.my_consumers
            product = self.awi.my_output_product
        else:
            partners = self.awi.my_suppliers
            product = self.awi.my_input_product

        for partner in partners:
            self.awi.request_negotiation(
                is_buy=not is_sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=MyNegotiator(
                    is_seller=is_sell,
                    current_inventory=self.awi.current_inventory,
                    my_input_product=self.awi.my_input_product,
                    my_output_product=self.awi.my_output_product,
                    n_lines=self.awi.n_lines,
                    n_step=self.awi.n_steps,
                    current_step=self.awi.current_step,
                    catalog_prices=self.awi.catalog_prices,
                    costs=self.awi.profile.costs[0],
                    is_first_level=self.is_first_level,
                    is_last_level=self.is_last_level,
                    target_quantity=target_quantity,
                    target_time=target_time,
                    target_unitprice=target_unitprice,
                    profit_width=self.profit_width,
                    awi=self.awi,
                ),
            )

    def init(self):
        super().init()
        # whether you are in the first level or the last level
        self.is_first_level = self.awi.my_input_product == 0
        self.is_last_level = self.awi.my_output_product == self.awi.n_processes

        # expected profit if you trade and produce one product at catalog_prices
        self.profit_width = self.awi.catalog_prices[self.awi.my_output_product] - (
            self.awi.catalog_prices[self.awi.my_input_product]
            + self.awi.profile.costs[0][self.awi.my_input_product]
        )
        # default value for target_quantity
        self.default_target_quantity = self.awi.n_lines

        # list of input unit price objects to manage input unit prices to negotiate
        self.input_unitprices = [
            {
                "start_time": 0,
                "end_time": self.awi.n_steps // 5,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_input_product] * 0.8
                ),
                "prev_success": 0,
            },
            {
                "start_time": (self.awi.n_steps // 5) + 1,
                "end_time": (self.awi.n_steps // 5) * 2,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_input_product] * 0.8
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 2) + 1,
                "end_time": (self.awi.n_steps // 5) * 3,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_input_product] * 0.7
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 3) + 1,
                "end_time": (self.awi.n_steps // 5) * 4,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_input_product] * 0.7
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 4) + 1,
                "end_time": (self.awi.n_steps // 5) * 5,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_input_product] * 0.6
                ),
                "prev_success": 0,
            },
        ]

        # list of output unit price objects to manage output unit prices to negotiate
        self.output_unitprices = [
            {
                "start_time": 0,
                "end_time": self.awi.n_steps // 5,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_output_product] * 1.4
                ),
                "prev_success": 0,
            },
            {
                "start_time": (self.awi.n_steps // 5) + 1,
                "end_time": (self.awi.n_steps // 5) * 2,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_output_product] * 1.3
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 2) + 1,
                "end_time": (self.awi.n_steps // 5) * 3,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_output_product] * 1.2
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 3) + 1,
                "end_time": (self.awi.n_steps // 5) * 4,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_output_product] * 1.1
                ),
                "prev_success": 0,
            },
            {
                "start_time": ((self.awi.n_steps // 5) * 4) + 1,
                "end_time": (self.awi.n_steps // 5) * 5,
                "unitprice": int(
                    self.awi.catalog_prices[self.awi.my_output_product] * 1.0
                ),
                "prev_success": 0,
            },
        ]

        # list of produce_capacity objects to manage production capacity at each time step
        # provisional value is used in sign_all_contracts()
        self.produce_capacity = []
        for i in range(self.awi.n_steps):
            capacity = {
                "time": i,
                "quantity": self.awi.n_lines,
                "provisional": self.awi.n_lines,
            }
            self.produce_capacity.append(capacity)
        self.produce_capacity[0]["quantity"] = 0
        self.produce_capacity[len(self.produce_capacity) - 1]["quantity"] = 0

    def step(self):
        super().step()

        # update input_unitprices and output_unitprices based on how many contracts were concluded in last step
        for unitprice_obj in self.input_unitprices:
            change = max(
                int(self.awi.catalog_prices[self.awi.my_input_product] * 0.1), 3
            )
            if unitprice_obj["prev_success"] <= 0:
                unitprice_obj["unitprice"] = min(
                    unitprice_obj["unitprice"] + (change // 3),
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + min(self.profit_width // 5, 2),
                )
            else:
                unitprice_obj["unitprice"] = unitprice_obj["unitprice"] - change

            unitprice_obj["prev_success"] = 0

        for unitprice_obj in self.output_unitprices:
            change = max(
                int(self.awi.catalog_prices[self.awi.my_output_product] * 0.1), 3
            )
            if unitprice_obj["prev_success"] <= 0:
                unitprice_obj["unitprice"] = max(
                    unitprice_obj["unitprice"] - (change // 3),
                    self.awi.catalog_prices[self.awi.my_output_product]
                    - min(self.profit_width // 5, 2),
                )
            else:
                unitprice_obj["unitprice"] = unitprice_obj["unitprice"] + change
            unitprice_obj["prev_success"] = 0

        # request 5 negotiations to each partner
        # behave differently according to agent's level
        if self.is_first_level:
            if self.awi.current_step + 5 < self.awi.n_steps - 1:
                start_time = self.awi.current_step + 5
                end_time = self.awi.n_steps - 1
                interval = ((end_time - start_time) // 5) + 1

                if end_time <= start_time:
                    start_time = 0
                    end_time = self.awi.n_steps - 1
                    interval = 2

                for target_time in range(start_time, end_time, interval):
                    target_quantity = min(
                        self.default_target_quantity,
                        self.get_surplus_quantity(target_time),
                    )

                    output_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= target_time)
                                and (target_time <= unitprice_obj["end_time"])
                            ),
                            self.output_unitprices,
                        )
                    )
                    if len(output_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_output_product
                        ]
                    else:
                        target_unitprice = output_unitprice[0]["unitprice"]

                    sell_qvalues = (1, target_quantity)
                    sell_tvalues = (target_time, min(target_time + interval, end_time))
                    sell_uvalues = (
                        int(
                            self.awi.catalog_prices[self.awi.my_output_product]
                            - self.profit_width
                        ),
                        target_unitprice + 1,
                    )

                    self.request(
                        True,
                        sell_qvalues,
                        sell_uvalues,
                        sell_tvalues,
                        target_quantity,
                        sell_tvalues,
                        target_unitprice,
                    )

        elif self.is_last_level:
            if self.awi.current_step + 2 < self.awi.n_steps - 1:
                start_time = self.awi.current_step + 2
                end_time = (
                    self.awi.n_steps // 2
                    if (self.awi.current_step < 0.3 * self.awi.n_steps)
                    else self.awi.n_steps - 1
                )
                interval = ((end_time - start_time) // 5) + 1

                if end_time <= start_time:
                    start_time = 0
                    end_time = self.awi.n_steps - 1
                    interval = 2

                for target_time in range(start_time, end_time, interval):
                    target_quantity = self.default_target_quantity

                    input_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= target_time)
                                and (target_time <= unitprice_obj["end_time"])
                            ),
                            self.input_unitprices,
                        )
                    )
                    if len(input_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_input_product
                        ]
                    else:
                        target_unitprice = input_unitprice[0]["unitprice"]

                    buy_qvalues = (1, self.awi.n_lines)
                    buy_tvalues = (target_time, min(target_time + interval, end_time))
                    buy_uvalues = (
                        target_unitprice - 1,
                        self.awi.catalog_prices[self.awi.my_input_product]
                        + self.profit_width,
                    )

                    self.request(
                        False,
                        buy_qvalues,
                        buy_uvalues,
                        buy_tvalues,
                        target_quantity,
                        buy_tvalues,
                        target_unitprice,
                    )

        else:
            if self.awi.current_step + 2 < self.awi.n_steps - 1:
                start_time = self.awi.current_step + 2
                end_time = (
                    self.awi.n_steps // 2
                    if (self.awi.current_step < 0.3 * self.awi.n_steps)
                    else self.awi.n_steps - 1
                )
                interval = ((end_time - start_time) // 5) + 1

                if end_time <= start_time:
                    start_time = 0
                    end_time = self.awi.n_steps - 1
                    interval = 2

                for target_time in range(start_time, end_time, interval):
                    target_quantity = self.default_target_quantity

                    input_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= target_time)
                                and (target_time <= unitprice_obj["end_time"])
                            ),
                            self.input_unitprices,
                        )
                    )
                    if len(input_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_input_product
                        ]
                    else:
                        target_unitprice = input_unitprice[0]["unitprice"]

                    buy_qvalues = (1, self.awi.n_lines)
                    buy_tvalues = (target_time, min(target_time + interval, end_time))
                    buy_uvalues = (
                        target_unitprice - 1,
                        self.awi.catalog_prices[self.awi.my_input_product]
                        + self.profit_width,
                    )

                    self.request(
                        False,
                        buy_qvalues,
                        buy_uvalues,
                        buy_tvalues,
                        target_quantity,
                        buy_tvalues,
                        target_unitprice,
                    )

            if self.awi.current_step + 2 < self.awi.n_steps - 1:
                start_time = self.awi.current_step + 2
                end_time = self.awi.n_steps - 1
                interval = ((end_time - start_time) // 5) + 1

                if end_time <= start_time:
                    start_time = 0
                    end_time = self.awi.n_steps - 1
                    interval = 2

                for target_time in range(start_time, end_time, interval):
                    target_quantity = min(
                        self.default_target_quantity,
                        self.get_surplus_quantity(target_time),
                    )

                    output_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= target_time)
                                and (target_time <= unitprice_obj["end_time"])
                            ),
                            self.output_unitprices,
                        )
                    )
                    if len(output_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_output_product
                        ]
                    else:
                        target_unitprice = output_unitprice[0]["unitprice"]

                    sell_qvalues = (1, self.awi.n_lines)
                    sell_tvalues = (target_time, min(target_time + interval, end_time))
                    sell_uvalues = (
                        int(
                            self.awi.catalog_prices[self.awi.my_output_product]
                            - self.profit_width
                        ),
                        target_unitprice + 1,
                    )

                    self.request(
                        True,
                        sell_qvalues,
                        sell_uvalues,
                        sell_tvalues,
                        target_quantity,
                        sell_tvalues,
                        target_unitprice,
                    )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        def get_agent_name(name):
            agent_name = ""
            for c in name.split("@")[0]:
                try:
                    n = int(c)
                except Exception:
                    agent_name += c

            return agent_name

        try:
            # collusion strategy
            # if initiator's class is the same, return MyNegotiator in Collude Mode
            if get_agent_name(self.name) in initiator:
                is_seller = annotation["seller"] == self.id
                target_quantity = issues[0].max_value
                target_time = (issues[1].min_value, issues[1].max_value)
                target_unitprice = 0

                return MyNegotiator(
                    is_seller=is_seller,
                    current_inventory=self.awi.current_inventory,
                    my_input_product=self.awi.my_input_product,
                    my_output_product=self.awi.my_output_product,
                    n_lines=self.awi.n_lines,
                    n_step=self.awi.n_steps,
                    current_step=self.awi.current_step,
                    catalog_prices=self.awi.catalog_prices,
                    costs=self.awi.profile.costs[0],
                    is_first_level=self.is_first_level,
                    is_last_level=self.is_last_level,
                    target_quantity=target_quantity,
                    target_time=target_time,
                    target_unitprice=target_unitprice,
                    profit_width=self.profit_width,
                    awi=self.awi,
                    collude=True,
                )
            else:
                is_seller = annotation["seller"] == self.id

                target_time = (issues[1].min_value, issues[1].max_value)
                middle_time = (target_time[0] + target_time[1]) // 2

                target_quantity = self.default_target_quantity
                if target_quantity < issues[0].min_value:
                    target_quantity = issues[0].min_value
                elif target_quantity > issues[0].max_value:
                    target_quantity = issues[0].max_value

                if is_seller:
                    output_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= middle_time)
                                and (middle_time <= unitprice_obj["end_time"])
                            ),
                            self.output_unitprices,
                        )
                    )[0]
                    if len(output_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_output_product
                        ]
                    else:
                        target_unitprice = iniput_unitprice[0]["unitprice"]
                else:
                    input_unitprice = list(
                        filter(
                            lambda unitprice_obj: (
                                (unitprice_obj["start_time"] <= middle_time)
                                and (middle_time <= unitprice_obj["end_time"])
                            ),
                            self.input_unitprices,
                        )
                    )[0]
                    if len(input_unitprice) == 0:
                        target_unitprice = self.awi.catalog_prices[
                            self.awi.my_input_product
                        ]
                    else:
                        target_unitprice = input_unitprice[0]["unitprice"]

                if self.is_first_level:
                    if is_seller:
                        target_quantity = min(
                            self.default_target_quantity,
                            self.get_surplus_quantity(target_time[0]),
                        )
                    else:
                        return None
                elif self.is_last_level:
                    if is_seller:
                        return None

                return MyNegotiator(
                    is_seller=is_seller,
                    awi=self.awi,
                    current_inventory=self.awi.current_inventory,
                    my_input_product=self.awi.my_input_product,
                    my_output_product=self.awi.my_output_product,
                    n_lines=self._awi.n_lines,
                    n_step=self.awi.n_steps,
                    current_step=self.awi.current_step,
                    catalog_prices=self.awi.catalog_prices,
                    costs=self.awi.profile.costs[0],
                    is_first_level=self.is_first_level,
                    is_last_level=self.is_last_level,
                    target_quantity=target_quantity,
                    target_time=target_time,
                    target_unitprice=target_unitprice,
                    profit_width=self.profit_width,
                )

        except Exception:
            return None

        return None

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        results = [None] * len(contracts)

        # tmp_ means that it is the stack of contracts which aren't concluded yet
        tmp_output_contracts_stack = []
        tmp_input_contracts_stack = []
        # all_ means that it is the stack of both contracts which were concluded and aren't concluded yet
        all_input_contracts_stack = []
        all_output_contracts_stack = []
        # stack for contracts which have corresponding contracts
        used_input_contracts_stack = []
        used_output_contracts_stack = []

        # initialize tmp_ stacks
        tmp_contracts = zip(contracts, range(len(contracts)))
        for tmp_contract in tmp_contracts:
            copy_contract = {
                "id": str(tmp_contract[0].id),
                "quantity": int(tmp_contract[0].agreement["quantity"]),
                "time": int(tmp_contract[0].agreement["time"]),
                "unit_price": int(tmp_contract[0].agreement["unit_price"]),
                "partner": tmp_contract[0].partners,
                "index": int(tmp_contract[1]),
                "prov_correspond": [],
                "tmp": 1,
            }
            if tmp_contract[0].annotation["seller"] == self.id:
                tmp_output_contracts_stack.append(copy_contract)
            else:
                copy_contract["use_quantity"] = 0
                tmp_input_contracts_stack.append(copy_contract)

        # initialize all_input_stacks
        for tmp_input_contract in tmp_input_contracts_stack:
            all_input_contracts_stack.append(tmp_input_contract)

        for contract in self.input_contracts_stack:
            if contract["quantity"] <= 0:
                continue
            copy_contract = {
                "id": str(contract["id"]),
                "quantity": int(contract["quantity"]),
                "time": int(contract["time"]),
                "unit_price": int(contract["unit_price"]),
                "partner": contract["partner"],
                "index": None,
                "prov_correspond": [],
                "tmp": 0,
                "use_quantity": 0,
            }
            all_input_contracts_stack.append(copy_contract)

        # initialize all_output_stacks
        for tmp_output_contract in tmp_output_contracts_stack:
            all_output_contracts_stack.append(tmp_output_contract)
        for contract in self.output_contracts_stack:
            if contract["quantity"] <= 0:
                continue
            copy_contract = {
                "id": str(contract["id"]),
                "quantity": int(contract["quantity"]),
                "time": int(contract["time"]),
                "unit_price": int(contract["unit_price"]),
                "partner": contract["partner"],
                "index": None,
                "prov_correspond": [],
                "tmp": 0,
            }
            all_output_contracts_stack.append(copy_contract)

        # sort all_ stacks in tmp and time order so that contracts which have already conclued will be used first
        all_input_contracts_stack = sorted(
            all_input_contracts_stack,
            key=lambda contract: (contract["tmp"], contract["time"]),
        )
        all_output_contracts_stack = sorted(
            all_output_contracts_stack,
            key=lambda contract: (contract["tmp"], contract["time"]),
        )

        # check if each input contract can apply to some output contracts
        for all_input_contract in all_input_contracts_stack:
            for all_output_contract in all_output_contracts_stack:
                # assumes that you can produce output products within two days since you bought the input products
                if all_input_contract["quantity"] > 0:
                    if all_output_contract["quantity"] > 0:
                        use_quantity = min(
                            all_input_contract["quantity"],
                            all_output_contract["quantity"],
                        )
                        if self.is_available_capacity(
                            all_input_contract["time"],
                            all_output_contract["time"],
                            use_quantity,
                            True,
                        ):
                            # decide how many input products in this input contract can be used to this output contract
                            all_output_contract_id = all_output_contract["id"]
                            all_input_contract["quantity"] = (
                                all_input_contract["quantity"] - use_quantity
                            )
                            all_input_contract[all_output_contract_id] = use_quantity
                            all_input_contract["prov_correspond"].append(
                                all_output_contract["id"]
                            )
                            all_output_contract["quantity"] = (
                                all_output_contract["quantity"] - use_quantity
                            )
                            all_output_contract["prov_correspond"].append(
                                all_input_contract
                            )

                            # break inner for statement if you used all the input products in this input contract
                            if all_input_contract["quantity"] <= 0:
                                break

        # used_output_contracts_stack contains output contracts which have zero planning quantity
        used_output_contracts_stack = list(
            filter(
                lambda contract: contract["quantity"] <= 0, all_output_contracts_stack
            )
        )

        # sign the output contract in used_output_contracts_stack only if valid production is possible
        for output_contract in used_output_contracts_stack:
            output_contract_id = output_contract["id"]

            # check if it is possible to produce output products from its corresponding input contract
            is_possible = True
            for prov_correspond_input_contract in output_contract["prov_correspond"]:
                if prov_correspond_input_contract["index"] != None:
                    if not self.is_available_capacity(
                        prov_correspond_input_contract["time"],
                        output_contract["time"],
                        prov_correspond_input_contract[output_contract_id],
                        True,
                    ):
                        is_possible = False

            # sign contracts if it's possible
            if is_possible:
                for prov_correspond_input_contract in output_contract[
                    "prov_correspond"
                ]:
                    if prov_correspond_input_contract["index"] != None:
                        if self.use_available_capacity(
                            prov_correspond_input_contract["time"],
                            output_contract["time"],
                            prov_correspond_input_contract[output_contract_id],
                            True,
                        ):
                            results[prov_correspond_input_contract["index"]] = self.id
                            all_input_contracts_stack = list(
                                filter(
                                    lambda contract: contract["id"]
                                    != prov_correspond_input_contract["id"],
                                    all_input_contracts_stack,
                                )
                            )
                            used_input_contracts_stack.append(
                                prov_correspond_input_contract
                            )

                if output_contract["index"] != None:
                    results[output_contract["index"]] = self.id

                all_output_contracts_stack = list(
                    filter(
                        lambda contract: contract["id"] != output_contract["id"],
                        all_output_contracts_stack,
                    )
                )

        # filter and sort all_ stacks
        all_input_contracts_stack = list(
            filter(lambda contract: contract["quantity"] > 0, all_input_contracts_stack)
        )
        all_output_contracts_stack = list(
            filter(
                lambda contract: contract["quantity"] > 0, all_output_contracts_stack
            )
        )
        all_input_contracts_stack = list(
            filter(
                lambda contract: contract["index"] != None, all_input_contracts_stack
            )
        )
        all_output_contracts_stack = list(
            filter(
                lambda contract: contract["index"] != None, all_output_contracts_stack
            )
        )
        all_input_contracts_stack = sorted(
            all_input_contracts_stack,
            key=lambda contract: (contract["unit_price"], contract["time"]),
        )
        all_output_contracts_stack = sorted(
            all_output_contracts_stack,
            key=lambda contract: (-contract["time"], contract["unit_price"]),
        )

        sold_count = 0
        bought_count = 0

        # sign output contracts as stock contracts
        stock_output_contracts = []
        for output_contract in all_output_contracts_stack:
            id = output_contract["id"]
            step = output_contract["time"]
            quantity = output_contract["quantity"]
            partner = output_contract["partner"]

            if quantity <= 0:
                continue
            if (
                step > self.awi.n_steps - 1 - self.awi.n_lines
                or step < self.awi.current_step
            ):
                continue

            if self.is_first_level:
                if step < 0.1 * self.awi.n_steps:
                    continue

                neighbor_output_contracts = list(
                    filter(
                        lambda contract: (
                            (contract["time"] >= step - self.awi.n_lines)
                            and (contract["time"] < step + self.awi.n_lines)
                        ),
                        self.output_contracts_stack,
                    )
                )
                neighbor_stock_output_contracts = list(
                    filter(
                        lambda contract: (
                            (contract["time"] >= step - self.awi.n_lines)
                            and (contract["time"] < step + self.awi.n_lines)
                        ),
                        stock_output_contracts,
                    )
                )

                total_quantity = 0
                for neighbor_output_contract in neighbor_output_contracts:
                    total_quantity = (
                        total_quantity + neighbor_output_contract["quantity"]
                    )
                for neighbor_stock_output_contract in neighbor_stock_output_contracts:
                    total_quantity = (
                        total_quantity + neighbor_stock_output_contract["quantity"]
                    )

                if total_quantity + quantity > self.awi.n_lines:
                    continue
            elif self.is_last_level:
                if step < 0.15 * self.awi.n_steps:
                    continue
                if (
                    self.get_required_quantity(self.awi.n_steps - 1)
                    > 3 * self.awi.n_lines
                ):
                    continue
            else:
                if step < self.awi.n_steps * 0.5:
                    continue
                if quantity > self.get_surplus_quantity(step) - sold_count:
                    continue

            stock_output_contracts.append(output_contract)

            if self.is_available_capacity(
                output_contract["time"] - self.awi.n_lines,
                output_contract["time"],
                output_contract["quantity"],
                True,
            ):
                sold_count = sold_count + quantity
                results[output_contract["index"]] = self.id

        # sign input contracts as stock contracts
        stock_input_contracts = []
        for input_contract in all_input_contracts_stack:
            id = input_contract["id"]
            step = input_contract["time"]
            quantity = input_contract["quantity"]
            partner = input_contract["partner"]

            if quantity <= 0:
                continue
            if (
                step > self.awi.n_steps - 1 - self.awi.n_lines
                or step < self.awi.current_step
            ):
                continue
            if step > self.awi.n_steps * 0.35:
                continue

            max_stock_count = (
                int(0.2 * self.awi.n_steps) * self.awi.n_lines
                if self.profit_width > 0
                else (self.awi.n_lines * 2) // 3
            )
            if (
                self.get_surplus_quantity(self.awi.n_steps - 1)
                + bought_count
                + quantity
                > max_stock_count
            ):
                continue

            stock_input_contracts.append(input_contract)

            if self.is_available_capacity(
                input_contract["time"],
                input_contract["time"] + self.awi.n_lines,
                input_contract["quantity"],
                True,
            ):
                bought_count = bought_count + quantity
                results[input_contract["index"]] = self.id

        return results

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        is_seller = contract.annotation["seller"] == self.id

        # update prev_success which means the number of successful negotiations
        if is_seller:
            for output_unitprice in self.output_unitprices:
                if (
                    output_unitprice["start_time"] <= contract.agreement["time"]
                    and contract.agreement["time"] <= output_unitprice["end_time"]
                ):
                    output_unitprice["prev_success"] = (
                        output_unitprice["prev_success"] + 1
                    )
        else:
            for input_unitprice in self.input_unitprices:
                if (
                    input_unitprice["start_time"] <= contract.agreement["time"]
                    and contract.agreement["time"] <= input_unitprice["end_time"]
                ):
                    input_unitprice["prev_success"] = (
                        input_unitprice["prev_success"] + 1
                    )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        # update secured quantities of contracts with the bankrupt agent
        for contract in contracts:
            is_seller = contract.annotation["seller"] == self.id

            if is_seller:
                affected_input_contracts = list(
                    filter(
                        lambda affected_contract: (
                            contract.id in affected_contract["correspond"]
                        ),
                        self.input_contracts_stack,
                    )
                )
                for affected_input_contract in affected_input_contracts:
                    affected_input_contract["quantity"] += contract.agreement[
                        "quantity"
                    ]
            else:
                affected_output_contracts = list(
                    filter(
                        lambda affected_contract: (
                            contract.id in affected_contract["correspond"]
                        ),
                        self.output_contracts_stack,
                    )
                )
                for affected_output_contract in affected_output_contracts:
                    affected_output_contract["quantity"] += contract.agreement[
                        "quantity"
                    ]
