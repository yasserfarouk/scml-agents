import logging

from scml.scml2020 import SCML2020Agent, SCML2020World, RandomAgent, DecentralizingAgent
from scml.scml2020.components.production import (
    DemandDrivenProductionStrategy,
    ProductionStrategy,
    SupplyDrivenProductionStrategy,
)
from scml.scml2020.components.trading import (
    PredictionBasedTradingStrategy,
    ReactiveTradingStrategy,
)
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.services.controllers import SyncController
from scml.scml2020.components.negotiation import StepNegotiationManager
from .pred_trade import ConsumeDrivenPredictionBasedTradingStrategy

from scml.scml2020 import TIME, QUANTITY, UNIT_PRICE
from negmas import ResponseType, outcome_is_valid
from negmas.sao import SAOResponse
from typing import List, Dict, Optional, Tuple, Any
from negmas import SAOSyncController
from negmas import AspirationNegotiator
import numpy as np


class ProtectedSyncController(SyncController):
    def __init__(self, *args, **kwargs):
        self.get_avg_p = kwargs["get_avg_p"]
        del kwargs["get_avg_p"]
        self.__parent = kwargs["parent"]
        super().__init__(*args, **kwargs)

    def propose(self, negotiator_id: str, state):
        # if there are no proposals yet, get first proposals
        if len(self.proposals) == 0:
            self.proposals = self.first_proposals()
        # get the saved proposal if it exists and return it
        proposal = self.proposals.get(negotiator_id, None)
        # if some proposal was there, delete it to force the controller to get a new one
        # if proposal is not None:
        #    self.proposals[negotiator_id] = None
        # if the proposal that was there was None, just offer the best offer
        # proposal = self.first_offer(negotiator_id)
        return proposal

    def best_proposal(self, nid):
        try:
            return super().best_proposal(nid)
        except TypeError:
            return None, -1000

    def utility(self, offer: Tuple[int, int, int], max_price: int) -> float:
        """A simple utility function

        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract

        """
        production_cost = np.max(
            self.__parent.awi.profile.costs[:, self.__parent.awi.my_input_product]
        )
        if self._is_seller:
            _needed, _secured = (
                self.__parent.outputs_needed,
                self.__parent.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.__parent.inputs_needed,
                self.__parent.inputs_secured,
            )
        if offer is None:
            return -1000
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0

        if self._is_seller:
            price = offer[UNIT_PRICE] - (self.get_avg_p() + int(production_cost)) * 1.2
        else:
            price = max_price - offer[UNIT_PRICE]
        if price < 0:
            return -1000
        return self._price_weight * price + (1 - self._price_weight) * q

    def my_first_proposals(self, offers):
        return {nid: self.best_proposal(nid)[0] for nid in offers.keys()}

    def counter_all(self, offers, states):
        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array(
            [
                self.utility(
                    o, self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value
                )
                for nid, o in offers.items()
            ]
        )

        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.my_first_proposals(offers)
        good_proposals = self.good_neg_offers(offers)

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            prep_resond = {
                k: SAOResponse(ResponseType.REJECT_OFFER, good_proposals[k][0])
                for k in offers.keys()
            }
            return prep_resond

        relative_time = min(_.relative_time for _ in states.values())

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility >= self._utility_threshold * self._best_utils[best_partner]
            or relative_time > self._time_threshold
        ):
            responses = {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,  # good_proposals[k][0]
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
                best_offer if self.is_valid(k, best_offer) else good_proposals[k][0],
            )
            for k in offers.keys()
        }
        responses[best_partner] = SAOResponse(
            ResponseType.REJECT_OFFER, best_proposals[best_partner]
        )
        return responses

    def good_neg_proposal(self, nid):
        negotiator = self.negotiators[nid][0]
        if negotiator.ami is None:
            return None, -1000
        utils = np.array(
            [
                self.utility(_, negotiator.ami.issues[UNIT_PRICE].max_value)
                for _ in negotiator.ami.outcomes
            ]
        )
        good_offers = []
        for i in range(len(utils)):
            if utils[i] > 0:
                good_offers.append((negotiator.ami.outcomes[i], utils[i]))

        if len(good_offers) == 0:
            return None, -1000

        production_cost = int(
            np.max(
                self.__parent.awi.profile.costs[:, self.__parent.awi.my_input_product]
            )
        )
        n = len(good_offers)
        lowest_offer = good_offers[0]
        for i in range(n):
            if (lowest_offer[0][2] < good_offers[i][0][2]) or (
                (lowest_offer[0][0] < good_offers[i][0][0])
                and (lowest_offer[0][2] == good_offers[i][0][2])
            ):
                lowest_offer = good_offers[i]

        return lowest_offer

    def good_neg_offers(self, offers):
        return {nid: self.good_neg_proposal(nid) for nid in self.offers.keys()}

    def on_negotiation_end(self, negotiator_id, state):
        pass


class MyAsp(AspirationNegotiator):
    def __init__(self, *args, **kwargs):
        self.manager = kwargs["MyManager"]
        del kwargs["MyManager"]
        self.manager.good_buy_price = 0
        super().__init__(*args, **kwargs)

    def my_acceptable_unit_price(self, step: int, sell: bool):
        production_cost = np.max(
            self.manager.awi.profile.costs[:, self.manager.awi.my_input_product]
        )
        expected_inventory = sum(self.manager.inputs_secured[0:step]) - sum(
            self.manager.outputs_secured[0:step]
        )
        self.manager.awi.n_lines
        alpha = expected_inventory / self.manager.awi.n_lines
        if sell:
            if alpha < 0:
                beta = 1.2
            elif alpha <= 4:
                beta = 1
            elif alpha <= 8:
                beta = 1 - alpha * (0.1 / 8)
            else:
                beta = 0.9
            return (production_cost + self.manager.input_cost[step]) * beta

        if alpha > 2 and alpha < 10:
            beta = (6 - alpha / 2) / 5
        elif alpha > 10:
            beta = 0.1
        else:
            beta = 1
        if self.manager.avg_o_q < 5:
            p = (self.manager.output_price[0] - production_cost) * beta
        else:
            p = (self.manager.avg_o_p * 0.9 - production_cost) * beta
        return p

    def _is_seller(self):
        return self._utility_function.weights[2] > 0

    def _price_ok(self, offer):
        ok_price = self.my_acceptable_unit_price(offer[1], self._is_seller())
        if self._is_seller():
            return offer[2] >= ok_price
        else:
            return offer[2] <= ok_price

    def respond(self, state, offer):
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER
        production_cost = np.max(
            self.manager.awi.profile.costs[:, self.manager.awi.my_input_product]
        )
        ok_ufunc = self._utility_function(
            (1, 1, self.manager.output_price[0] - production_cost)
        )
        asp = (
            self.aspiration(state.relative_time)
            * ((self.ufun_max * 0.3 + ok_ufunc * 0.7) - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value and self._price_ok(offer):
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def return_good_offer(self, offer):
        if self._price_ok(offer):
            return offer
        else:
            return None

    def propose(self, state):
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if self.ufun_max < self.reserved_value:
            return None
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        if self.presorted:
            if len(self.ordered_outcomes) < 1:
                return None
            for i, (u, o) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.return_good_offer(self.ordered_outcomes[i][1])
                    if self.randomize_offer:
                        return random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    return self.return_good_offer(self.ordered_outcomes[i - 1][1])
            if self.randomize_offer:
                return random.sample(self.ordered_outcomes, 1)[0][1]
            return self.return_good_offer(self.ordered_outcomes[-1][1])
        else:
            if asp >= 0.99999999999 and self.best_outcome is not None:
                return self.best_outcome
            if self.randomize_offer:
                return outcome_with_utility(
                    ufun=self._utility_function,
                    rng=(asp, float("inf")),
                    issues=self._ami.issues,
                )
            tol = self.tolerance
            for _ in range(self.n_trials):
                rng = self.ufun_max - self.ufun_min
                mx = min(asp + tol * rng, self.__last_offer_util)
                outcome = outcome_with_utility(
                    ufun=self._utility_function, rng=(asp, mx), issues=self._ami.issues,
                )
                if outcome is not None:
                    break
                tol = math.sqrt(tol)
            else:
                outcome = (
                    self.best_outcome
                    if self.__last_offer is None
                    else self.__last_offer
                )
            self.__last_offer_util = self.utility_function(outcome)
            self.__last_offer = outcome
            return outcome


class StepBuyBestSellNegManager(StepNegotiationManager):
    """
    Based on StepNegotiationManager for buying and SyncController for selling
    """

    def __init__(
        self,
        *args,
        price_weight=0.95,
        utility_threshold=0.5,
        time_threshold=0.7,  # 0.5,
        time_horizon=0.4,  # 0.2,
        **kwargs,
    ):
        kwargs["negotiator_type"] = "scml_agents.scml2020.team_25.mixed_neg.MyAsp"
        kwargs["negotiator_params"] = {"MyManager": self}
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self.sell_controller = ProtectedSyncController(
            is_seller=True,
            parent=self,
            price_weight=self._price_weight,
            time_threshold=self._time_threshold,
            utility_threshold=self._utility_threshold,
            get_avg_p=self.get_avg_p,
        )
        self._current_end = -1
        self._current_start = -1

    def _mygenerate_negotiations(self, step, sell):
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

    def _generate_negotiations(self, step: int, sell: bool):
        return

    def start_negotiations(
        self, product, quantity, unit_price, step, partners=None,
    ):
        awi: AWI
        awi = self.awi  # type: ignore
        is_seller = product == self.awi.my_output_product
        if quantity < 1 or unit_price < 1 or step < awi.current_step + 1:
            # awi.logdebug_agent(
            #     f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{step})"
            # )
            return
        # choose ranges for the negotiation agenda.
        qvalues = (1, quantity)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(step, is_seller, tvalues)
        if is_seller:
            uvalues = unit_price, uvalues[1]
        else:
            uvalues = uvalues[0], unit_price
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

    def stepBuy(self):
        if self._horizon == 5:
            self._horizon = int(self.awi.n_steps / 10)
        s = self.awi.current_step
        if s == 0:
            # in the first step, generate buy negotiations for horizon steps in the future
            last = min(self.awi.n_steps - 1, self._horizon + 2)
            for step in range(1, last):
                self._mygenerate_negotiations(step, False)
        else:
            # generate buy negotiations to secure inputs/outputs the step after horizon steps
            nxt = s + self._horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self._mygenerate_negotiations(nxt, False)
        self.updateBuyNegotiations()

    def updateBuyNegotiations(self):
        """
        For each time step: if there is no controller and there is amount needed
        create controller. otherwise, if there is a difference between controller
        target and needed target -> update controller target
        """
        s = self.awi.current_step
        for t in range(s, s + self._horizon):
            ctrlr = self.buyers[s]
            if ctrlr is None:
                self._mygenerate_negotiations(step, False)
            elif ctrlr.target < self.inputs_needed[s]:
                ctrlr.target = self.inputs_needed[s] - self.inputs_secured[s]

    def stepSell(self):
        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return

        needed = self.outputs_needed
        secured = self.outputs_secured
        product = self.awi.my_output_product

        # find the maximum amount needed at any time-step in the given range
        needs = np.max(
            needed[self._current_start : self._current_end]
            - secured[self._current_start : self._current_end]
        )

        # set a range of prices
        min_price = np.min(
            [
                self.acceptable_unit_price(t, True)
                for t in range(self._current_start, self._current_end)
            ]
        )
        max_price = 2 * self.awi.catalog_prices[self.awi.my_output_product]

        # self.controllers[seller].clear()
        self.awi.request_negotiations(
            False,
            product,
            (1, needs),
            (int(min_price), int(max_price)),
            time=(self._current_start, self._current_end),
            controller=self.sell_controller,
        )

    def step(self):
        # if self.awi.current_step == 0:
        #    production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        #    print("production_cost = " + str(production_cost))
        super().step()
        """Generates buy and sell negotiations as needed"""
        self.stepBuy()
        self.stepSell()

    def sell_respond_to_negotiation_request(
        self, initiator, issues, annotation, mechanism,
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        controller = self.sell_controller
        if controller is None:
            return None
        return controller.create_negotiator()

    def buy_respond_to_negotiation_request(
        self, initiator, issues, annotation, mechanism,
    ):

        # find negotiation parameters
        is_seller = annotation["seller"] == self.id
        tmin, tmax = issues[TIME].min_value, issues[TIME].max_value + 1
        # find the time-step for which this negotiation should be added
        step = max(0, tmin - 1) if is_seller else min(self.awi.n_steps - 1, tmax + 1)
        # find the corresponding controller.
        controller_info: ControllerInfo
        controller_info = self.sellers[step] if is_seller else self.buyers[step]
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        tmp = self.target_quantities((tmin, tmax + 1), is_seller)
        if len(tmp) > 0:
            target = tmp.max()
        else:
            return None
        if target <= 0:
            return None
        # self.awi.loginfo_agent(
        #     f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
        #     f"({Issue.num_outcomes(mechanism.issues)})"
        # )
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

    def respond_to_negotiation_request(
        self, initiator, issues, annotation, mechanism,
    ):
        is_seller = annotation["seller"] == self.id
        if is_seller:
            return self.sell_respond_to_negotiation_request(
                initiator, issues, annotation, mechanism
            )
        else:
            return self.buy_respond_to_negotiation_request(
                initiator, issues, annotation, mechanism
            )


class MixedNegAgent(
    _NegotiationCallbacks,
    StepBuyBestSellNegManager,
    # ConsumeDrivenPredictionBasedTradingStrategy,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass
    # def trade_prediction_init(self):
    #    super().trade_prediction_init()
    #    self.expected_inputs = 0 * self.expected_inputs

    # def init(self):
    #    super().init()
    #    threshold = int(self.awi.n_steps/2)
    #    self.inputs_needed[threshold:] = self.inputs_needed[threshold:]*0


if __name__ == "__main__":
    from collections import defaultdict

    def show_agent_scores(world):
        scores = defaultdict(list)
        for aid, score in world.scores().items():
            scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
        scores = {k: sum(v) / len(v) for k, v in scores.items()}
        plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
        plt.show()

    import matplotlib.pyplot as plt

    world = SCML2020World(
        **SCML2020World.generate(
            [MixedNegAgent, DecentralizingAgent],
            n_steps=30,
            n_processes=2,
            n_agents_per_process=2,
            log_stats_every=1,
        ),
        construct_graphs=True,
    )
    world.run_with_progress()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()
    show_agent_scores(world)
