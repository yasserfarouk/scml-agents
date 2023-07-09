import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Contract,
    Issue,
    Negotiator,
    ResponseType,
    SAONegotiator,
    ToughNegotiator,
)
from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.helpers import get_class
from scipy.stats import linregress
from scml.scml2020 import (
    AWI,
    Failure,
    MovingRangeNegotiationManager,
    SCML2020Agent,
    SCML2020World,
    SupplyDrivenProductionStrategy,
    TradingStrategy,
)
from scml.scml2020.components.negotiation import (
    IndependentNegotiationsManager,
    NegotiationManager,
    StepNegotiationManager,
)
from scml.scml2020.services.controllers import StepController
from sklearn.linear_model import LinearRegression


class ModifiedAspirationAgent(AspirationNegotiator):
    def respond(self, state):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun_max is None or self.ufun_min is None:
            self.on_preferences_changed(
                [PreferencesChange(PreferencesChangeType.General)]
            )

        if self.ufun is None or self.ufun_max is None or self.ufun_min is None:
            return ResponseType.REJECT_OFFER

        u = self.ufun(offer)

        slope = None
        if self.owner is not None:
            slope = self.owner.register_negotiation_bids(self.name, (offer, u))

        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER

        if slope > 0:
            self.ufun_min -= (slope - 0.2) % (0 - 0.2 + 1) + 0.2

        asp = (
            self.utility_at(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )

        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER

        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION

        return ResponseType.REJECT_OFFER


class MyNegotiationManager(StepNegotiationManager):
    def __init__(
        self,
        *args,
        negotiator_type=ModifiedAspirationAgent,
        negotiator_params=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else {"owner": self}
        )

        self.controller_models = {}

        self.buyers = self.sellers = None

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        if is_seller:
            return LinearUtilityFunction((0, 1, 100))
        return LinearUtilityFunction((0, -1, -100))

    def acceptable_unit_price(self, step: int, sell: bool):
        if sell:
            if step / self.awi.n_steps < 0.2:
                return self.output_price[step] * 1.3
            if step / self.awi.n_steps < 0.8:
                return self.output_price[step]
            else:
                return self.output_price[step] * 0.8

        return np.mean(
            [
                self.awi.catalog_prices[self.awi.my_output_product],
                self.output_price[step],
            ]
        )

    def target_quantity(self, step: int, sell: bool):
        """A fixed target quantity of 33 my production capacity"""
        if step >= self.awi.n_steps:
            return self.outputs_needed[step - 1] if sell else 0

        if step / self.awi.n_steps < 0.2:
            return (self.outputs_needed[step] - self.outputs_secured[step]) * 2
        elif step / self.awi.n_steps < 0.8:
            return self.outputs_needed[step] - self.outputs_secured[step]
        else:
            return (self.outputs_needed[step] - self.outputs_secured[step]) * 0.5

    def _urange(self, step, sell, time_range):
        unit_price = self.acceptable_unit_price(step, sell)

        if sell:
            if step / self.awi.n_steps < 0.2:
                return unit_price * 0.8, unit_price * 2
            elif step / self.awi.n_steps < 0.8:
                return unit_price, unit_price * 2
            else:
                return unit_price * 1.2, unit_price * 2

        return 1, unit_price

    def all_negotiations_concluded(self, controller_index, is_seller):
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

        controllers[controller_index].controller = None
        if quantity <= target:
            self._generate_negotiations(step=controller_index, sell=is_seller)
            return

        super().all_negotiations_concluded(controller_index, is_seller)

    def step(self):
        super().step()

        self.inputs_needed[:-1] = self.expected_outputs[1:]
        self.outputs_needed[1:] = self.expected_inputs[:-1]

        current_state = self.awi.current_step
        total_state = self.awi.n_steps

    def register_negotiation_bids(self, aid, bid_tuple):
        agent_dir = (
            self.controller_models.get(aid)
            if self.controller_models.get(aid) is not None
            else {}
        )
        step = self.awi.current_step
        bids_list = agent_dir.get(step) if agent_dir.get(step) is not None else []
        agent_dir[step] = bids_list.append(bid_tuple)

        x = list(range(1, len(bids_list) + 1))
        y = [a[1] for a in bids_list]

        lr = LinearRegression().fit(
            np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
        )
        slope = lr.coef_

        self.controller_models[aid] = agent_dir
        return slope
