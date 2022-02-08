import os
import sys

sys.path.append(os.path.dirname(__file__))

from typing import Any, List

import numpy as np
import torch
from hyperparameters import *
from negmas import Issue, UtilityFunction, Value
from negmas.outcomes import Outcome
from scml.scml2020 import QUANTITY, TIME, UNIT_PRICE
from utility_model import (
    UtilityModel,
    load_buyer_utility_model,
    load_seller_utiltiy_model,
)


class MyUtilityFunction(UtilityFunction):
    def __init__(
        self,
        *argv,
        _is_seller=True,
        parent=None,
        manager=None,
        price_weight=0.7,
        unnecessary_percentage_threshold=0.45,
        neg_parent=None,
        **kwargs
    ):
        super().__init__(self, *argv, **kwargs)
        self.outcome_type = Any
        self.reserved_value = BAD_UTILITY_THRESHOLD
        self._is_seller = _is_seller
        self.__parent = parent
        if manager is None:
            self.manager = parent
        else:
            self.manager = manager
        self._price_weight = price_weight
        self.unnecessary_percentage_threshold = unnecessary_percentage_threshold
        self.neg_parent = neg_parent

    def update_neg_parent(self, negotiator):
        self.neg_parent = negotiator

    def xml(self, issues: List[Issue]) -> str:
        """Converts the function into a well formed XML string preferrably in GENIUS format.

        If the output has with </objective> then discount factor and reserved value should also be included
        If the output has </utility_space> it will not be appended in `to_xml_str`

        """
        return ""

    def eval(self, offer: Outcome) -> Value:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - You cannot return None from overriden __call__() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return A Value not a float for real-valued utilities for the benefit of inspection code.
            - Return the reserved value if the offer was None

        Returns:
            Value: The utility_function value which may be a distribution. If `None` it means the
                          utility_function value cannot be calculated.
        """
        if offer is None:
            return self.reserved_value
        # get my needs and secured amounts arrays
        if self._is_seller:
            _needed, _secured = (
                self.manager.outputs_needed,
                self.manager.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.manager.inputs_needed,
                self.manager.inputs_secured,
            )

        # invalide offers have no utility
        if offer is None:
            return BAD_UTILITY_THRESHOLD

        # offers for contracts that can never be executed have no utility
        t = offer[TIME]
        if t < self.manager.awi.current_step or t > self.manager.awi.n_steps - 1:
            return BAD_UTILITY_THRESHOLD

        # offers that exceed my needs have no utility (that can be improved)
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            to_produce = _needed[offer[TIME]] - _secured[t]
            unnecessary_percentage = 1 - (to_produce / offer[QUANTITY])
            if unnecessary_percentage > self.unnecessary_percentage_threshold:
                return BAD_UTILITY_THRESHOLD

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs
        price = offer[UNIT_PRICE] if self._is_seller else -offer[UNIT_PRICE]

        util = self._price_weight * price + (1 - self._price_weight) * q
        return util


class MyRLUtilityFunction(UtilityFunction):
    def __init__(
        self,
        *argv,
        _is_seller=True,
        parent=None,
        manager=None,
        _load_model=False,
        neg_parent=None,
        **kwargs
    ):
        super().__init__(self, *argv, **kwargs)
        self.outcome_type = Any
        self._is_seller = _is_seller
        self.__parent = parent
        if manager is None:
            self.manager = parent
        else:
            self.manager = manager
        if _load_model:
            self.model = self.load()
        else:
            self.model = self.create_model()
        self.neg_parent = neg_parent

    def xml(self, issues: List[Issue]) -> str:
        """Converts the function into a well formed XML string preferrably in GENIUS format.

        If the output has with </objective> then discount factor and reserved value should also be included
        If the output has </utility_space> it will not be appended in `to_xml_str`

        """
        return ""

    def load(self):
        if self._is_seller:
            return load_seller_utiltiy_model()
        return load_buyer_utility_model()

    def create_model(self):
        return UtilityModel(self._is_seller)

    def update_neg_parent(self, negotiator):
        self.neg_parent = negotiator

    def predict_outcome(self, data):
        to_features = np.array(data)
        to_features = torch.from_numpy(to_features).float()
        if len(to_features) != 15:
            x = 0
        return self.model(to_features)

    def __call__(self, offer: Outcome) -> Value:
        return self.eval(offer)

    def eval(self, offer: Outcome) -> Value:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - You cannot return None from overriden __call__() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return A Value not a float for real-valued utilities for the benefit of inspection code.
            - Return the reserved value if the offer was None

        Returns:
            Value: The utility_function value which may be a distribution. If `None` it means the
                          utility_function value cannot be calculated.
        """
        if offer is None:
            return self.reserved_value

        # offers for contracts that can never be executed have no utility
        t = offer[OFFER_STATE_STEP]
        if t < self.manager.awi.current_step or t > self.manager.awi.n_steps - 1:
            return BAD_UTILITY_THRESHOLD

        return self.predict_outcome(offer)
