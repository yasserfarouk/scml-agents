import functools
from typing import Any, Dict, Optional, Tuple, Union

from dana_neg_algo import DanasController, DanasNegotiator
from negmas import SAONegotiator
from run_tournament import run
from scml.scml2020 import (
    PredictionBasedTradingStrategy,
    SCML2020Agent,
    StepNegotiationManager,
    SupplyDrivenProductionStrategy,
)
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.components.negotiation import ControllerInfo
from scml.scml2020.services import StepController


class MyNegotiationManager(StepNegotiationManager):
    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = DanasNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            negotiator_type=negotiator_type,
            negotiator_params=negotiator_params,
            **kwargs,
        )

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
        controller = DanasController(
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


class MyAgent(
    _NegotiationCallbacks,  # as in  DecentralizingAgent
    MyNegotiationManager,
    PredictionBasedTradingStrategy,  # as in  DecentralizingAgent
    SupplyDrivenProductionStrategy,  # as in  DecentralizingAgent
    SCML2020Agent,
):
    pass


if __name__ == "__main__":
    # a = MyAgent()
    # a.sign_all_contracts()
    run(agent=MyAgent)
