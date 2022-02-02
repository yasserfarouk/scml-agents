import functools
import math
import pickle
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pprint import pformat
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from dana_neg_algo import DanasController, DanasNegotiator
from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Breach,
    Contract,
    Issue,
    Negotiator,
    SAONegotiator,
    UtilityFunction,
    num_outcomes,
)
from negmas.helpers import get_class
from scml import SCML2020World
from scml.scml2020 import (
    AWI,
    NO_COMMAND,
    DecentralizingAgent,
    IndDecentralizingAgent,
    PredictionBasedTradingStrategy,
    SCML2020Agent,
)
from scml.scml2020.common import TIME
from scml.scml2020.components.prediction import ExecutionRatePredictionStrategy
from scml.scml2020.components.production import (
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
)
from scml.scml2020.services.controllers import StepController, SyncController


class OmerProductionStrategyAgent(ProductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_omer_production_args(self):
        if not hasattr(self, "_production_classifier"):
            self._production_classifier, self._vectorizer = pickle.load(
                open("production_binary_classifier.pkl", "rb")
            )
        if not hasattr(self, "omer_input"):
            self.omer_input = 0
        if not hasattr(self, "omer_output"):
            self.omer_output = 0
        if not hasattr(self, "_features"):
            self._features = [
                {
                    "n_agents": len(
                        "|".join(["|".join(x) for x in self.awi.all_consumers]).split(
                            "|"
                        )
                    )
                    - 1,
                    "step": -1,
                    "steps_to_end": self.awi.state.n_steps - self.awi.current_step,
                    "relative_step": 0,
                    "input_product": self.awi.my_input_product,
                    "output_product": self.awi.my_input_product + 1,
                    "inventory_input": self.awi.state.inventory[
                        self.awi.my_input_product
                    ],
                    "inventory_output": self.awi.state.inventory[
                        self.awi.my_input_product + 1
                    ],
                    "price_in": self.awi.catalog_prices[self.awi.my_input_product],
                    "price_out": self.awi.catalog_prices[self.awi.my_input_product + 1],
                    "n_buy": 0,
                    "n_buy_price_mean": 0,
                    "n_buy_price_std": 0,
                    "n_buy_quantity_mean": 0,
                    "n_buy_quantity_std": 0,
                    "n_buy_time_mean": 0,
                    "n_sell": 0,
                    "n_sell_price_mean": 0,
                    "n_sell_price_std": 0,
                    "n_sell_quantity_mean": 0,
                    "n_sell_quantity_std": 0,
                    "n_sell_time_mean": 0,
                }
            ]

    def _mean(self, l):
        return np.mean(l) if len(l) > 0 else 0

    def _std(self, l):
        return np.std(l) if len(l) > 0 else 0

    def _save_feature_state(self, signed):
        # print(self.awi.current_step)
        signed_sell = [x for x in signed if x.annotation["seller"] == self.id]
        signed_buy = [x for x in signed if x.annotation["seller"] != self.id]

        vec = {
            "n_agents": len(
                "|".join(["|".join(x) for x in self.awi.all_consumers]).split("|")
            )
            - 1,
            "step": self.awi.current_step,
            "steps_to_end": self.awi.state.n_steps - self.awi.current_step,
            "relative_step": str(self.awi.relative_time),
            "input_product": self.awi.my_input_product,
            "output_product": self.awi.my_input_product + 1,
            "inventory_input": self.awi.state.inventory[self.awi.my_input_product],
            "inventory_output": self.awi.state.inventory[self.awi.my_input_product + 1],
            "price_in": self.awi.catalog_prices[self.awi.my_input_product],
            "price_out": self.awi.catalog_prices[self.awi.my_input_product + 1],
            "n_buy": len(signed_buy),
            "n_buy_price_mean": self._mean(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_buy
                    if contract
                ]
            ),
            "n_buy_price_std": self._std(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_buy
                    if contract
                ]
            ),
            "n_buy_quantity_mean": self._mean(
                [contract.agreement["quantity"] for contract in signed_buy if contract]
            ),
            "n_buy_quantity_std": self._std(
                [contract.agreement["quantity"] for contract in signed_buy if contract]
            ),
            "n_buy_time_mean": self._mean(
                [
                    contract.agreement["time"] - self.awi.current_step
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell": len(signed_sell),
            "n_sell_price_mean": self._mean(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell_price_std": self._std(
                [
                    contract.agreement["unit_price"]
                    for contract in signed_sell
                    if contract
                ]
            ),
            "n_sell_quantity_mean": self._mean(
                [contract.agreement["quantity"] for contract in signed_sell if contract]
            ),
            "n_sell_quantity_std": self._std(
                [contract.agreement["quantity"] for contract in signed_sell if contract]
            ),
            "n_sell_time_mean": self._mean(
                [
                    contract.agreement["time"] - self.awi.current_step
                    for contract in signed_sell
                    if contract
                ]
            ),
        }
        self._features.append(vec)

    def _produce_by_prediction(self):
        instance = {}
        for (prev_k, prev_v), (next_k, next_v) in zip(
            self._features[-2].items(), self._features[-1].items()
        ):
            instance["prev_" + prev_k] = prev_v
            instance["next_" + next_k] = next_v
        to_prodeuce = self._production_classifier.predict(
            self._vectorizer.transform(instance)
        ).item()

        if to_prodeuce:
            steps, _ = self.awi.schedule_production(
                process=self.awi.my_input_product,
                repeats=1,
                step=(self.awi.current_step + 1, self.awi.current_step + 1),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range["prediction" + str(self.awi.current_step)] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                False,
            )

    def on_contract_executed(self, contract: Contract) -> None:
        self._load_omer_production_args()
        super().on_contract_executed(contract)
        if self.id == contract.annotation["seller"]:
            self.omer_output += contract.agreement["quantity"]
        else:
            self.omer_input += contract.agreement["quantity"]

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        self._load_omer_production_args()
        super().on_contract_breached(contract, breaches, resolution)
        if self.id == contract.annotation["seller"]:
            self.omer_output += int(
                contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches))
            )
        else:
            self.omer_input += int(
                contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches))
            )

    def step(self):
        super().step()
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)

    def on_contracts_finalized(
        self: "SCML2020Agent",
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        self._load_omer_production_args()
        self._save_feature_state(signed)
        super().on_contracts_finalized(signed, cancelled, rejectors)
        self._produce_by_prediction()
        latest = self.awi.n_steps - 2
        earliest_production = self.awi.current_step
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            if step > latest + 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            input_product = contract.annotation["product"]
            steps, _ = self.awi.schedule_production(
                process=input_product,
                repeats=contract.agreement["quantity"],
                step=(step, latest),
                line=-1,
                partial_ok=True,
            )
            self.schedule_range[contract.id] = (
                min(steps) if len(steps) > 0 else -1,
                max(steps) if len(steps) > 0 else -1,
                is_seller,
            )


class DrorMeanERPStrategy(ExecutionRatePredictionStrategy):
    """
    Predicts that the there is a fixed execution rate that does not change for all partners

    Args:
        execution_fraction: The expected fraction of any contract's quantity to be executed

    Provides:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract

    Hooks Into:
        - `internal_state`
        - `init`
        - `on_contract_executed`
        - `on_contract_breached`

    """

    def __init__(self, *args, execution_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_fraction = execution_fraction
        self._total_quantity = None
        self.buy_rates = {}
        self.sell_rates = {}
        self.sell_total_quantities = {}
        self.buy_total_quantities = {}

    def predict_quantity(self, contract: Contract):
        return contract.agreement["quantity"] * self._execution_fraction

    def init(self):
        super().init()
        self._total_quantity = max(1, self.awi.n_steps * self.awi.n_lines // 10)
        for item in self.awi.my_consumers:
            self.sell_total_quantities[item] = {
                "rate": self._execution_fraction,
                "quantity": self._total_quantity,
            }
        for item in self.awi.my_suppliers:
            self.buy_total_quantities[item] = {
                "rate": self._execution_fraction,
                "quantity": self._total_quantity,
            }

    @property
    def internal_state(self):
        state = super().internal_state
        state.update({"execution_fraction": self._execution_fraction})
        return state

    def update_dicts(self, contract, q):
        if self.id == contract.annotation["buyer"]:
            # buy agreement
            seller = self.buy_total_quantities.get(contract.annotation["seller"])
            if seller:
                seller_old_quantity = seller["quantity"]
                seller["quantity"] += q
                seller["rate"] = (seller["rate"] * seller_old_quantity + q) / seller[
                    "quantity"
                ]
        else:
            buyer = self.buy_total_quantities.get(contract.annotation["buyer"])
            if buyer:
                buyer_old_quantity = buyer["quantity"]
                buyer["quantity"] += q
                buyer["rate"] = (buyer["rate"] * buyer_old_quantity + q) / buyer[
                    "quantity"
                ]

    def on_contract_executed(self, contract: Contract) -> None:
        super().on_contract_executed(contract)
        old_total = self._total_quantity
        q = contract.agreement["quantity"]
        self._total_quantity += q
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
        self.update_dicts(contract, q)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        old_total = self._total_quantity
        q = contract.agreement["quantity"] * max(b.level for b in breaches)
        self._total_quantity += q
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
        self.update_dicts(contract, q)


class DrorNegotiationManager:
    """A negotiation manager is a component that provides negotiation control functionality to an agent

    Args:
        horizon: The number of steps in the future to consider for selling outputs.

    Provides:
        - `start_negotiations` An easy to use method to start a set of buy/sell negotiations

    Requires:
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Abstract:
        - `respond_to_negotiation_request`

    Hooks Into:
        - `init`
        - `step`
        - `on_contracts_finalized`
        - `respond_to_negotiation_request`
    """

    def __init__(self, *args, horizon=5, negotiate_on_signing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._horizon = horizon
        self._negotiate_on_signing = negotiate_on_signing

    def init(self):
        # set horizon to exogenous horizon
        if self._horizon is None:
            self._horizon = self.awi.bb_read("settings", "exogenous_horizon")
            if self._horizon is None:
                self._horizon = 5

        # let other component work. We init last here as we do not depend on any other components for this method.
        super().init()

    def start_negotiations(
        self,
        product: int,
        quantity: int,
        unit_price: int,
        step: int,
        partners: List[str] = None,
    ) -> None:
        """
        Starts a set of negotiations to buy/sell the product with the given limits

        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            step: The maximum/minimum time for buy/sell
            partners: A list of partners to negotiate with

        Remarks:

            - This method assumes that product is either my_input_product or my_output_product

        """
        awi: AWI
        awi = self.awi  # type: ignore
        is_seller = product == self.awi.my_output_product
        if quantity < 1 or unit_price < 1 or step < awi.current_step + 1:
            awi.logdebug_agent(
                f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{step})"
            )
            return
        # choose ranges for the negotiation agenda.
        qvalues = (1, quantity)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(step, is_seller, tvalues)
        if tvalues[0] > tvalues[1]:
            return
        if partners is None:
            if is_seller:
                partners = awi.my_consumers
            else:
                partners = awi.my_suppliers
        self._start_negotiations(
            product, is_seller, step, qvalues, uvalues, tvalues, partners
        )

    def step(self):
        self.awi.logdebug_agent(f"Enter step:\n{pformat(self.internal_state)}")
        super().step()
        """Generates buy and sell negotiations as needed"""
        s = self.awi.current_step
        if s == 0:
            # in the first step, generate buy/sell negotiations for horizon steps in the future
            last = min(self.awi.n_steps - 1, self._horizon + 2)
            for step in range(1, last):
                self._generate_negotiations(step, False)
                self._generate_negotiations(step, True)
        else:
            # generate buy and sell negotiations to secure inputs/outputs the step after horizon steps
            nxt = s + self._horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self._generate_negotiations(nxt, False)
            self._generate_negotiations(nxt, True)
        self.awi.logdebug_agent(f"End step:\n{pformat(self.internal_state)}")

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        if self._negotiate_on_signing:
            steps = (self.awi.current_step + 1, self.awi.n_steps)
            in_pre = self.target_quantities(steps, False)
            out_pre = self.target_quantities(steps, True)
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if (not self._negotiate_on_signing) or (in_pre is None) or (out_pre is None):
            return
        # calculating inputs and outputs before on_contract_finalized and
        # after, if changed negotiate on diff
        inputs_needed = self.target_quantities(steps, False)
        outputs_needed = self.target_quantities(steps, True)
        if (inputs_needed is None) or (outputs_needed is None):
            return
        inputs_needed -= in_pre
        outputs_needed -= out_pre
        for s in np.nonzero(inputs_needed)[0]:
            if inputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_input_product,
                inputs_needed[s],
                self.acceptable_unit_price(s, False),
                s,
            )
        for s in np.nonzero(outputs_needed)[0]:
            if outputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_output_product,
                outputs_needed[s],
                self.acceptable_unit_price(s, True),
                s,
            )

    def _generate_negotiations(self, step: int, sell: bool) -> None:
        """Generates all the required negotiations for selling/buying for the given step"""
        product = self.awi.my_output_product if sell else self.awi.my_input_product
        quantity = self.target_quantity(step, sell)
        unit_price = self.acceptable_unit_price(step, sell)

        if quantity <= 0 or unit_price <= 0:
            return
        self.start_negotiations(
            product=product,
            step=step,
            quantity=min(self.awi.n_lines * (step - self.awi.current_step), quantity),
            unit_price=unit_price,
        )

    def _urange(self, step, is_seller, time_range):
        if is_seller:
            cprice = self.awi.catalog_prices[self.awi.my_output_product]
            return cprice, 2 * cprice

        cprice = self.awi.catalog_prices[self.awi.my_input_product]
        return 1, cprice

    def _trange(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self._horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """
        Returns the target quantity to negotiate about for each step in the range given (beginning included and ending
        excluded) for buying/selling

        Args:
            steps: Simulation step
            sell: Sell or buy
        """
        return np.array([self.target_quantity(s, sell) for s in range(*steps)])

    @abstractmethod
    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        """
        Actually start negotiations with the given agenda

        Args:
            product: The product to negotiate about.
            sell: If true, this is a sell negotiation
            step: The step
            qvalues: the range of quantities
            uvalues: the range of unit prices
            tvalues: the range of times
            partners: partners
        """
        pass

    @abstractmethod
    def target_quantity(self, step: int, sell: bool) -> int:
        """
        Returns the target quantity to sell/buy at a given time-step

        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement target_quantity")

    @abstractmethod
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """
        Returns the maximum/minimum acceptable unit price for buying/selling at the given time-step

        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement acceptable_unit_price")

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        raise ValueError("You must implement respond_to_negotiation_request")


@dataclass
class ControllerInfo:
    """Keeps a record of information about one of the controllers used by the agent"""

    controller: StepController
    time_step: int
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    expected: int
    done: bool = False


# TODO DrorMeanERPStrategy
class DrorStepNegotiationManager(DrorMeanERPStrategy, DrorNegotiationManager):
    """
    A negotiation manager that controls a controller and another for selling for every timestep

    Args:
        negotiator_type: The negotiator type to use to manage all negotiations
        negotiator_params: Paramters of the negotiator

    Provides:
        - `all_negotiations_concluded`

    Requires:
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Hooks Into:
        - `init`
        - `respond_to_negotiation_request`

    """

    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        # negotiator_type: Union[SAONegotiator, str] = DanasNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # TODO
        #       super().__init__(*args, execution_fraction=0.3, **kwargs)
        super().__init__(*args, **kwargs)
        # super().__init__(*args, execution_fraction=0.5, **kwargs)
        # save construction parameters
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        # attributes that will be read during init() from the AWI
        # -------------------------------------------------------
        self.buyers = self.sellers = None
        """Buyer controllers and seller controllers. Each of them is responsible of covering the
        needs for one step (either buying or selling)."""

    def init(self):
        super().init()

        # initialize one controller for buying and another for selling for each time-step
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.awi.logdebug_agent(f"Initialized\n{pformat(self.internal_state)}")

    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        if sell:
            # TODO:
            execution_fraction = mean(
                [item[1].get("rate") for item in self.sell_total_quantities.items()]
            )
            expected_quantity = int(math.floor(qvalues[1] * execution_fraction))
            # expected_quantity = int(math.floor(qvalues[1] *self._execution_fraction))
        else:
            # TODO
            execution_fraction = mean(
                [item[1].get("rate") for item in self.buy_total_quantities.items()]
            )
            expected_quantity = int(math.floor(qvalues[1] * execution_fraction))
            # expected_quantity = int(math.floor(qvalues[1] *self._execution_fraction))

        # negotiate with everyone
        controller = self.add_controller(
            sell, qvalues[1], uvalues, expected_quantity, step
        )
        self.awi.loginfo_agent(
            f"Requesting {'selling' if sell else 'buying'} negotiation "
            f"on u={uvalues}, q={qvalues}, t={tvalues}"
            f" with {str(partners)} using {str(controller)}"
        )
        self.awi.request_negotiations(
            is_buy=not sell,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,
            controller=controller,
            extra=dict(controller_index=step, is_seller=sell),
        )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:

        # find negotiation parameters
        is_seller = annotation["seller"] == self.id
        tmin, tmax = issues[TIME].min_value, issues[TIME].max_value + 1
        # find the time-step for which this negotiation should be added
        step = max(0, tmin - 1) if is_seller else min(self.awi.n_steps - 1, tmax + 1)
        # find the corresponding controller.
        controller_info: ControllerInfo
        controller_info = self.sellers[step] if is_seller else self.buyers[step]
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        target = self.target_quantities((tmin, tmax + 1), is_seller).sum()
        if target <= 0:
            return None
        self.awi.loginfo_agent(
            f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
            f"({num_outcomes(mechanism.issues)})"
        )
        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.add_controller(
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                int(target),
                step,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator()

    def all_negotiations_concluded(
        self, controller_index: int, is_seller: bool
    ) -> None:
        """Called by the `StepController` to affirm that it is done negotiating for some time-step"""
        info = (
            self.sellers[controller_index]
            if is_seller
            else self.buyers[controller_index]
        )
        info.done = True
        c = info.controller
        if c is None:
            return
        quantity = c.secured
        target = c.target
        time_range = info.time_range
        if is_seller:
            controllers = self.sellers
        else:
            controllers = self.buyers

        self.awi.logdebug_agent(
            f"Killing Controller {str(controllers[controller_index].controller)}"
        )
        controllers[controller_index].controller = None
        if quantity <= target:
            self._generate_negotiations(step=controller_index, sell=is_seller)
            return

    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> StepController:
        if is_seller and self.sellers[step].controller is not None:
            return self.sellers[step].controller
        if not is_seller and self.buyers[step].controller is not None:
            return self.buyers[step].controller
        controller = StepController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            step=step,
            urange=urange,
            product=self.awi.my_output_product
            if is_seller
            else self.awi.my_input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
            horizon=self._horizon,
            negotiations_concluded_callback=functools.partial(
                self.__class__.all_negotiations_concluded, self
            ),
            parent_name=self.name,
            awi=self.awi,
        )
        # controller = DanasController(
        #     is_seller=is_seller,
        #     target_quantity=target,
        #     negotiator_type=self.negotiator_type,
        #     negotiator_params=self.negotiator_params,
        #     step=step,
        #     urange=urange,
        #     product=self.awi.my_output_product
        #     if is_seller
        #     else self.awi.my_input_product,
        #     partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
        #     horizon=self._horizon,
        #     negotiations_concluded_callback=functools.partial(
        #         self.__class__.all_negotiations_concluded, self
        #     ),
        #     parent_name=self.name,
        #     awi=self.awi,
        # )
        if is_seller:
            assert self.sellers[step].controller is None
            self.sellers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        else:
            assert self.buyers[step].controller is None
            self.buyers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        return controller

    def _get_controller(self, mechanism) -> StepController:
        neg = self._running_negotiations[mechanism.id]
        return neg.negotiator.parent


class DrorStepAgent(
    DrorStepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    # OmerProductionStrategyAgent,
    SCML2020Agent,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_cost_factor = 1.45
        # def __init__(

    #         self,
    #         *args,
    #         production_cost_factor,
    #         **kwargs,
    # ):
    #     super().__init__(*args, **kwargs)
    #     self.production_cost_factor = production_cost_factor

    def set_param(self, production_cost_factor):
        # self.production_cost_factor = production_cost_factor  #1.03 # TODO=1.45
        self._execution_fraction = production_cost_factor

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return self.production_cost_factor * (
                production_cost + self.input_cost[step]
            )
        return (self.output_price[step] - production_cost) / self.production_cost_factor

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class DrorOmer(
    DrorStepNegotiationManager,
    PredictionBasedTradingStrategy,
    # SupplyDrivenProductionStrategy,
    OmerProductionStrategyAgent,
    SCML2020Agent,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_cost_factor = 1.45
        # def __init__(

    #         self,
    #         *args,
    #         production_cost_factor,
    #         **kwargs,
    # ):
    #     super().__init__(*args, **kwargs)
    #     self.production_cost_factor = production_cost_factor

    def set_param(self, production_cost_factor):
        # self.production_cost_factor = production_cost_factor  #1.03 # TODO=1.45
        self._execution_fraction = production_cost_factor

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return self.production_cost_factor * (
                production_cost + self.input_cost[step]
            )
        return (self.output_price[step] - production_cost) / self.production_cost_factor

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


def return_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return scores
