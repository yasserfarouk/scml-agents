import numpy as np
from scml.scml2020 import (
    TIME,
    QUANTITY,
    UNIT_PRICE,
    TradingStrategy,
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
    DemandDrivenProductionStrategy,
    AWI,
    FactoryState,
)
from negmas import (
    ResponseType,
    outcome_is_valid,
    Outcome,
    MechanismState,
    UtilityFunction,
    Negotiator,
    SAOController,
)
from negmas.sao import SAOResponse, SAOSyncController
from scml.scml2020.components.trading import NoTradingStrategy
from typing import List, Dict, Optional, Tuple, Any, Union

__all__ = ["IntegratedNegotiationManager", "SyncController", "ControllerUFun"]


class IntegratedNegotiationManager(ReactiveTradingStrategy):
    """My negotiation strategy

    Args:
        price_weight: The relative importance of price in the utility calculation.
        utility_threshold: The fraction of maximum utility above which all offers will be accepted.
        time_threshold: The fraction of the negotiation time after which any valid offers will be accepted.
        time_range: The time-range for each controller as a fraction of the number of simulation steps

    Hooks Into:
        - `init`
        - `step`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.


    """

    def __init__(
        self,
        *args,
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.controllers: Dict[bool, SyncController] = {
            False: SyncController(
                is_seller=False,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
            True: SyncController(
                is_seller=True,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
        }
        self._current_end = -1
        self._current_start = -1

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        pass

    def request_negotiations(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        controller: Optional[SAOController],
        negotiators: List[Negotiator] = None,
        partners: List[str] = None,
        extra: Dict[str, Any] = None,
    ) -> bool:
        return None
        return self.awi.request_negotiations(
            is_buy,
            product,
            quantity,
            unit_price,
            time,
            controller,
            negotiators,
            partners,
            extra,
        )

    def step(self):
        super().step()

        awi: AWI = self.awi
        factory_state: FactoryState = awi.state

        # find the range of steps about which we plan to negotiate
        step = awi.current_step
        self._current_start = step + 1
        self._current_end = min(
            awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return

        # for seller, needed, secured, product in [
        #     (False, self.inputs_needed, self.inputs_secured, awi.my_input_product),
        #     (
        #         True,
        #         self.outputs_needed,
        #         self.outputs_secured,
        #         awi.my_output_product,
        #     ),
        # ]:
        #     # find the maximum amount needed at any time-step in the given range
        #     needs = np.max(
        #         needed[self._current_start : self._current_end]
        #         - secured[self._current_start : self._current_end]
        #     )
        #     if needs < 1:
        #         continue
        #
        #     # set a range of prices
        #     if seller:
        #         # for selling set a price that is at least the catalog price
        #         min_price = awi.catalog_prices[product]
        #         price_range = (min_price, 2 * min_price)
        #     else:
        #         # for buying sell a price that is at most the catalog price
        #         price_range = (0, awi.catalog_prices[product])
        #     awi.request_negotiations(
        #         not seller,
        #         product,
        #         (1, needs),
        #         price_range,
        #         time=(self._current_start, self._current_end),
        #         controller=self.controllers[seller],
        #     )

        input_demand: List[int] = self.inputs_needed - self.outputs_needed
        production_cost: float = min(awi.profile.costs[0])
        current_marginal_material_inventory: int = awi.state.inventory[
            awi.my_input_product
        ]
        current_marginal_product_inventory: int = awi.state.inventory[
            awi.my_output_product
        ]
        for i_step, demand_quantity in enumerate(input_demand):
            if i_step < awi.current_step:
                # 過去の分
                continue
            if i_step > awi.current_step + self.time_horizon * awi.n_steps:
                # horizonより先
                continue
            if i_step == awi.current_step:
                # 現在のターンに出したい => 在庫あまりの場合のみ
                marginal_product: int = input_demand[
                    awi.current_step
                ] + current_marginal_product_inventory
                if marginal_product > 0:
                    self.request_negotiations(
                        is_buy=False,
                        product=awi.my_input_product,
                        quantity=(1, marginal_product),
                        unit_price=(0, awi.catalog_prices[awi.my_output_product] * 10),
                        time=i_step,
                        controller=self.controllers[True],
                        partners=awi.my_consumers,
                    )
                continue
            if i_step == awi.current_step + 1:
                # 次のターンに出したい => lineが空いているか確認してから発注
                current_blank_line: int = min(
                    sum(factory_state.commands[awi.current_step] == -1),
                    current_marginal_material_inventory,
                )
                assert current_blank_line >= 0
                self.request_negotiations(
                    is_buy=False,
                    product=awi.my_input_product,
                    quantity=(1, current_blank_line),
                    unit_price=(
                        awi.catalog_prices[awi.my_input_product] + production_cost,
                        awi.catalog_prices[awi.my_output_product] * 10,
                    ),
                    time=i_step,
                    controller=self.controllers[True],
                    partners=awi.my_consumers,
                )
                current_marginal_material_inventory -= current_blank_line
                continue
            if demand_quantity < 0:
                # 必要な量が0 => 材料あまり => 売りたい
                selling_quantity: int = demand_quantity
                current_blank_line: int = min(
                    sum(factory_state.commands[i_step - 1] == -1),
                    current_marginal_material_inventory,
                )
                assert current_blank_line >= 0
                self.request_negotiations(
                    is_buy=False,
                    product=awi.my_output_product,
                    quantity=(1, selling_quantity + current_blank_line),
                    unit_price=(
                        awi.catalog_prices[awi.my_input_product] + production_cost,
                        awi.catalog_prices[awi.my_output_product] * 4,
                    ),
                    time=i_step,
                    controller=self.controllers[True],
                    partners=awi.my_consumers,
                )
                # self.request_negotiations(is_buy=True, product=awi.my_input_product, quantity=(1, selling_quantity),
                #                           unit_price=(0, awi.catalog_prices[awi.my_input_product] * 2),
                #                           time=i_step, controller=self.controllers[False], partners=awi.my_suppliers)
                current_marginal_material_inventory -= current_blank_line
                continue
            if demand_quantity > 0:
                self.request_negotiations(
                    is_buy=True,
                    product=awi.my_input_product,
                    quantity=(1, demand_quantity),
                    unit_price=(0, awi.catalog_prices[awi.my_input_product] * 2),
                    time=i_step,
                    controller=self.controllers[False],
                    partners=awi.my_suppliers,
                )
                # self.request_negotiations(is_buy=False, product=awi.my_output_product, quantity=(1, demand_quantity),
                #                           unit_price=(0, awi.catalog_prices[awi.my_output_product]*3),
                #                           time=i_step, controller=self.controllers[False], partners=awi.my_suppliers)
                continue
            self.request_negotiations(
                is_buy=True,
                product=awi.my_input_product,
                quantity=(1, 10),
                unit_price=(0, awi.catalog_prices[awi.my_input_product] * 2),
                time=i_step,
                controller=self.controllers[False],
                partners=awi.my_suppliers,
            )
            self.request_negotiations(
                is_buy=False,
                product=awi.my_output_product,
                quantity=(1, 10),
                unit_price=(
                    awi.catalog_prices[awi.my_input_product] + production_cost,
                    awi.catalog_prices[awi.my_output_product] * 3,
                ),
                time=i_step,
                controller=self.controllers[False],
                partners=awi.my_suppliers,
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List["Issue"],
        annotation: Dict[str, Any],
        mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        controller = self.controllers[not annotation["is_buy"]]
        if controller is None:
            return None
        return controller.create_negotiator()


class ControllerUFun(UtilityFunction):
    """A utility function for the controller"""

    def __init__(self, controller=None):
        super().__init__(outcome_type=tuple)
        self.controller = controller

    def eval(self, offer: "Outcome"):
        return self.controller.utility(offer)

    def xml(self, issues):
        pass


class SyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent
    needs and with lowest price.

    Args:
        is_seller: Are we trying to sell (or to buy)?
        parent: The agent from which we will access `needed` and `secured` arrays
        price_weight: The importance of price in utility calculation
        utility_threshold: Accept anything with a relative utility above that
        time_threshold: Accept anything with a positive utility when we are that close
                        to the end of the negotiation
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.utility_function = ControllerUFun(controller=self)

    def utility(self, offer: "Outcome") -> float:
        """A simple utility function

        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract

        """

        # get my needs and secured amounts arrays
        if self._is_seller:
            _needed, _secured = (
                self.__parent.outputs_needed,
                self.__parent.inputs_needed + self.__parent.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.__parent.inputs_needed,
                self.__parent.outputs_needed + self.__parent.inputs_secured,
            )

        # invalide offers have no utility
        if offer is None:
            return -1000

        # offers for contracts that can never be executed have no utility
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0

        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]
        return self._price_weight * price + (1 - self._price_weight) * q

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        """Is this a valid offer for that negotiation"""
        issues = self.negotiators[negotiator_id][0].ami.issues
        return outcome_is_valid(offer, issues)

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, "SAOState"]
    ) -> Dict[str, "SAOResponse"]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.

        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array([self.utility(o) for o in offers.values()])

        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.first_proposals()

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            return {
                k: SAOResponse(ResponseType.REJECT_OFFER, best_proposals[k])
                for k in offers.keys()
            }

        relative_time = min(_.relative_time for _ in states.values())

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility
            >= self._utility_threshold * self.utility(best_proposals[best_partner])
            or relative_time > self._time_threshold
        ):
            responses = {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    best_offer if self.is_valid(k, best_offer) else best_proposals[k],
                )
                for k in offers.keys()
            }
            responses[best_partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return responses

        # send the best offer to everyone else and try to improve it
        responses = {
            k: SAOResponse(
                ResponseType.REJECT_OFFER,
                best_offer if self.is_valid(k, best_offer) else best_proposals[k],
            )
            for k in offers.keys()
        }
        responses[best_partner] = SAOResponse(
            ResponseType.REJECT_OFFER, best_proposals[best_partner]
        )
        return responses

    def on_negotiation_end(self, negotiator_id: str, state: "MechanismState") -> None:
        """Update the secured quantities whenever a negotiation ends"""
        if state.agreement is None:
            return

        q, t = state.agreement[QUANTITY], state.agreement[TIME]
        if self._is_seller:
            self.__parent.outputs_secured[t] += q
        else:
            self.__parent.inputs_secured[t] += q
