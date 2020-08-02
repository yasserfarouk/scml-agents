from collections import defaultdict
from typing import Tuple, List, Callable, Any, Dict

import numpy
from negmas import (
    SAONegotiator,
    AgentWorldInterface,
    NonlinearHyperRectangleUtilityFunction,
)
from negmas.helpers import instantiate
from scml.scml2020.services import StepController


class Controller(StepController):
    def __init__(
        self,
        *args,
        is_seller: bool,
        negotiator_type: SAONegotiator,
        negotiator_params: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            is_seller=is_seller,
            negotiator_type=negotiator_type,
            negotiator_params=negotiator_params,
            **kwargs,
        )
        if is_seller:
            self.ufun = NonlinearHyperRectangleUtilityFunction(
                hypervolumes=[
                    {0: (1.0, 2.0), 1: (1.0, 2.0)},
                    {0: (1.4, 2.0), 2: (2.0, 3.0)},
                ],
                mappings=[2.0, lambda x: 2 * x[2] + x[0]],
                f=lambda x: numpy.exp(x),
            )
        else:
            self.ufun = NonlinearHyperRectangleUtilityFunction(
                hypervolumes=[
                    {0: (1.0, 2.0), 1: (1.0, 2.0)},
                    {0: (1.4, 2.0), 2: (2.0, 3.0)},
                ],
                mappings=[2.0, lambda x: 2 * x[2] + x[0]],
                f=lambda x: numpy.exp(x),
            )
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)
