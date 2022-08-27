import copy
import cProfile
import inspect
import math
import pathlib
import random
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from time import sleep
from typing import Callable, Collection, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from negmas import (
    AgentMechanismInterface,
    Outcome,
    PolyAspiration,
    ResponseType,
    SAOResponse,
)
from negmas.outcomes import Issue
from negmas.preferences import UtilityFunction, normalize
from scml.oneshot import OneShotAgent, OneShotSyncAgent, OneShotUFun
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

from .bilat_ufun import (
    BilatUFun,
    BilatUFunAvg,
    BilatUFunDummy,
    BilatUFunMarginalTable,
    BilatUFunUniform,
)
from .model import *
from .model import (
    Model,
    ModelCheating,
    ModelDisagreement,
    ModelRandomPoint,
    ModelUniform,
    realistic,
)
from .model_data import *
from .model_data import construct_test_cols
from .model_empirical import ModelEmpirical
from .negotiation_history import BilateralHistory, SCMLHistory, WorldInfo
from .offer import Offer
from .outcome_distr import (
    OutcomeDistr,
    OutcomeDistrMarginal,
    OutcomeDistrPoint,
    OutcomeDistrRandom,
    OutcomeDistrTable,
    OutcomeDistrUniform,
)
from .simulator import BilatSimulator
from .spaces import *
from .strategy import *
from .strategy import Strategy, StrategyAspiration
from .ufun_calc import UFunCalc

# import matplotlib
# from matplotlib import animation
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

__all__ = [
    "GoldfishParetoEmpiricalGodfatherAgent",
    # "MinDisagreementGodfatherAgent",
    # "MinEmpiricalGodfatherAgent",
    # "AspirationUniformGodfatherAgent",
    # "NonconvergentGodfatherAgent",
    # "ParetoEmpiricalGodfatherAgent",
    # "SlowGoldfish",
    # "HardnosedGoldfishGodfatherAgent",
    # "HardnosedGoldfishBiggerAgent",
    # "HardnosedGoldfishSmallerAgent",
    # "SoftnosedGoldfishGodfatherAgent",
    # "QuickLearningGodfather",
    # "MediumLearningGodfather",
    # "SlowLearningGodfather",
    # "ZooGodfather",
    # "TrainingCollectionGodfatherAgent",
    # "ChristopherTheGoldfishAgent",
]


class GodfatherAgent(OneShotSyncAgent):
    enable_vis = False
    enable_training_collection = False

    def __init__(
        self,
        model_type=None,
        strategy_type=None,
        num_iter_consistency=1,
        num_sample_outcomes=1,
        *args,
        **kwargs,
    ) -> None:
        warnings.filterwarnings("ignore")
        self._model_type = model_type
        self._strategy_type = strategy_type
        self._num_iter_consistency = num_iter_consistency
        self._num_sample_outcomes = num_sample_outcomes
        self._f_opponents_initialized = False
        self._final_estufuns = []
        self._final_outcomes = []

        self._ufun_hist = []

        self._counter_idx = -1
        self._step_idx = 0
        self._current_call = 0
        self._call_in_progress = None

        self.enable_logging = False
        self.enable_safety_checks = False
        self.enable_profiling = False

        self.vis_path = "visualizations"

        super().__init__(*args, **kwargs)
        self.__asp = PolyAspiration(1.0, "boulware")

    def _enter_call(self):
        caller = None
        try:
            caller = inspect.stack()[1][3]
        except Exception:
            pass
        was_in_progress = self._call_in_progress is not None
        if was_in_progress:
            warnings.warn(
                "entering {} while call {} already in progress; time since step {}".format(
                    caller,
                    self._call_in_progress,
                    datetime.now() - self.step_start_time,
                )
            )
        self._call_in_progress = caller
        self._current_call += 1
        return self._current_call, was_in_progress

    def _exit_call(self, sthg=None):
        self._call_in_progress = None
        return sthg

    def _call_is_current(self, call_idx: int):
        caller = inspect.stack()[1][3]
        if call_idx != self._current_call:
            warnings.warn(
                "aborting {} bc no longer current call; time since step {}".format(
                    caller, datetime.now() - self.step_start_time
                )
            )
        return call_idx == self._current_call

    def _vis_consistency_criterion(self):
        # days 20, 40, 60, 80, at negotiation rounds 0 and 15
        return (
            self.enable_vis
            and self._step_idx in [0, 2, 20, 40, 60, 80]
            and self._counter_idx % 5 == 0
        )

    def init(self) -> None:
        super().init()
        world_info = WorldInfo(
            n_negotiation_rounds=self.awi.settings["neg_n_steps"],
            my_level=self.awi.level,
            n_competitors=self.awi.n_competitors,
            n_partners=len(
                self.awi.my_consumers if self.awi.level == 0 else self.awi.my_suppliers
            ),
        )
        self._history = SCMLHistory(world_info)
        self.log("initing new Godfather agent")
        self.step_start_time = datetime.now()

    def _get_model(self, opp_id: str) -> Model:
        """Will need to be overloaded based on model type"""
        if self._model_type:
            strat = self._get_strategy(opp_id)
            return self._model_type(opp_id, strategy_self=strat)
        else:
            raise Exception("No model type specified")

    def _get_strategy(self, opp_id: str) -> Strategy:
        """Will need to be overloaded based on strategy type"""
        if self._strategy_type:
            return self._strategy_type()
        else:
            raise Exception("No strategy type specified")

    def _init_opponents(self) -> None:
        self._f_opponents_initialized = True
        opp_ids = self.negotiators.keys()
        self.log("init opponents", list(opp_ids))
        self._models: Dict[str, Model] = {
            opp_id: self._get_model(opp_id) for opp_id in opp_ids
        }
        self._strategies: Dict[str, Strategy] = {
            opp_id: self._get_strategy(opp_id) for opp_id in opp_ids
        }
        for opp_id in opp_ids:
            self._history.register_agent(opp_id, self._get_outcome_space(opp_id))

    def step(self) -> None:
        call_idx, was_in_progress = self._enter_call()
        self.step_start_time = datetime.now()

        if was_in_progress:
            # before proceeding, clean up
            for neg_id in self.negotiators.keys():
                neg = self._history.negotiator_histories(neg_id)[-1]
                if not neg.is_ended():
                    neg.fail()

        if self.enable_safety_checks:
            assert [hs[-1].outcome() for hs in self._history.negotiations.values()]

        if self.enable_vis:
            self._vis_negotiations()

        self.stored_estufuns = self._final_estufuns
        self._final_estufuns = []
        self._final_outcomes = []

        # important: we have to guarantee that self.estufuns
        # is always from the current round (bc offer space could've changed)
        if hasattr(self, "estufuns"):
            del self.estufuns

        self._step_idx += 1
        self._counter_idx = -1
        self._history.next_round()
        self.log("step complete!")

        self._exit_call()

    def on_negotiation_failure(self, partners, annotation, nmi, state):
        """Updates negotiation history"""
        call_idx, _ = self._enter_call()

        opp_id = self._get_opp_id_from_nmi(nmi)
        self.log(f"failing negotiation with {opp_id}")
        self._history.fail(opp_id)

        self._exit_call()

    def on_negotiation_success(self, contract, nmi):
        """Updates negotiation history"""
        call_idx, _ = self._enter_call()

        opp_id = self._get_opp_id_from_nmi(nmi)
        self.log(f"succeeding negotiation with {opp_id}")
        my_offers = self._history.current_negotiator_history(opp_id).my_offers()
        if my_offers and my_offers[-1] == Moves.ACCEPT:
            pass  # we already recorded this move in counter_all, so no need to record now
        else:
            self._history.opponent_move(opp_id, Moves.ACCEPT)

        self._exit_call()

    def first_proposals(self) -> Dict[str, Tuple[int, ...]]:
        """Decide a first proposal on every negotiation. (Defers to _counter.)"""
        call_idx, _ = self._enter_call()

        first_proposals_time = datetime.now()

        # self.log('first proposals beginning')
        if not self._f_opponents_initialized:
            # cProfile.runctx("self._init_opponents()", globals(), locals(),
            #     filename="profiling/init_opponents_stats")
            self._init_opponents()
        # self.log("current negotiation histories", {k: self._history.current_negotiator_history(k) for k in self.negotiators.keys()})

        # type annotation to pinky promise that we won't make any non-offer moves...
        move_dict: Dict[str, Offer] = self._counter_profile(self.negotiators.keys())  # type: ignore

        # update negotiation histories with our moves
        for opp_id, move in move_dict.items():
            # self.log("first move", move, "against", opp_id)
            self._history.my_move(opp_id, move)

        # self.log("first prop time", datetime.now() - first_proposals_time)
        return self._exit_call(
            {
                opp_id: move.to_negmas(self.awi.current_step)
                for opp_id, move in move_dict.items()
            }
        )

    def counter_all(
        self, offers: Dict[str, Tuple[int, int, int]], states
    ) -> Dict[str, SAOResponse]:
        call_idx, _ = self._enter_call()
        # self.log("negotiation time: ", self.get_nmi(list(offers.keys())[0]).state.time)

        counter_time = datetime.now()

        if not hasattr(self, "_models"):
            warnings.warn(
                "{}: Should not be running counter_all before first_proposals (at time {}). Is this Yasser's problem?".format(
                    self.name, self.awi.current_step
                )
            )
            self.first_proposals()  # run and throw away, just to keep things on track
            # # disabled because on reflection this is a bad strategy
            # # makes our agent less predictable
            # self._history.no_first_proposals(opp_id)

        # format data nicely
        nice_offers: Dict[str, Offer] = {
            opp_id: Offer(offer[UNIT_PRICE], offer[QUANTITY])
            for opp_id, offer in offers.items()
        }

        if not self._call_is_current(call_idx):
            return {}  # abort

        # update negotiation histories with opponent moves
        for opp_id, offer in nice_offers.items():
            self._history.opponent_move(opp_id, offer)
        # self.log("counter_all: incoming opponent moves:", nice_offers)

        if self.enable_safety_checks:
            assert all(
                self._history.current_negotiator_history(j).whose_turn() == 1
                for j in nice_offers.keys()
            )

        move_dict = self._counter_profile(nice_offers.keys())
        # self.log("counter_all: responding with moves:", move_dict)

        if not self._call_is_current(call_idx):
            return {}  # abort

        # update negotiation histories with our moves
        for opp_id, move in move_dict.items():
            self._history.my_move(opp_id, move)

        # self.log("counter time", datetime.now() - counter_time)
        return self._exit_call(
            {
                opp_id: MoveSpace.move_to_sao_response(move, self.awi.current_step)
                for opp_id, move in move_dict.items()
            }
        )

    def _counter_profile(self, opponents_to_counter):
        if not self.enable_profiling:
            return self._counter(opponents_to_counter)
        else:
            self.counter_start_time = datetime.now()
            cProfile.runctx(
                "self._counter_profile_res = self._counter(opponents_to_counter)",
                globals(),
                locals(),
                filename="profiling/counter_stats",
            )
            return self._counter_profile_res  # type: ignore

    def _counter(self, opponents_to_counter: Iterable[str]) -> Dict[str, Move]:
        """Respond to a set of offers given the negotiation state of each."""
        # self.log("counter_all > _counter: begin")
        self._counter_idx += 1
        opponents_all = self.negotiators.keys()

        # generate consistent outcomes
        outcomes: Dict[str, OutcomeDistr] = {
            j: self.initial_outcome_distr(j) for j in opponents_all
        }
        estufuns: Dict[str, BilatUFun] = {}

        outcome_history: List[Dict[str, OutcomeDistr]] = []
        estufun_history: List[Dict[str, BilatUFun]] = []
        if self._vis_consistency_criterion():
            outcome_history = [copy.deepcopy(outcomes)]

        for _ in range(self._num_iter_consistency):
            for j in opponents_all:
                estufuns[j] = self._est_bilat_ufun(j, outcomes)
                outcomes[j] = self._models[j](
                    estufuns[j], self._history.negotiator_histories(j)
                )

            if self._vis_consistency_criterion():
                estufun_history.append(copy.deepcopy(estufuns))
            if self._vis_consistency_criterion():
                outcome_history.append(copy.deepcopy(outcomes))

        self.estufuns = estufuns  # publish estufuns so other agents can cheat

        if self.enable_vis:
            self._final_estufuns.append(copy.deepcopy(estufuns))
            self._final_outcomes.append(copy.deepcopy(outcomes))
            self.ex_quant = (
                self.awi.current_exogenous_input_quantity
                if self._is_selling()
                else self.awi.current_exogenous_output_quantity
            )
        if self.enable_training_collection:
            for j in opponents_all:
                h = self._history.current_negotiator_history(j)
                h.register_prediction_point(estufuns[j])

        # self.log("counter_all > _counter: visualizing")
        if self._vis_consistency_criterion():
            self._vis_consistency(outcome_history, estufun_history)

        # make moves
        moves = {}
        # self.log("counter_all > _counter: making moves")
        for j in opponents_to_counter:
            if self.enable_safety_checks:
                assert not self._history.current_negotiator_history(
                    j
                ).is_ended(), "history for j ended {}".format(
                    self._history.current_negotiator_history(j)
                )
            estufuns[j] = self._est_bilat_ufun(j, outcomes)  # one last update
            moves[j] = self._strategies[j](
                estufuns[j], copy.deepcopy(self._history.negotiator_histories(j))
            )

        # self.log("counter_all > _counter: responding moves")
        return moves

    def _est_bilat_ufun(
        self, j: str, outcome_distrs: Dict[str, OutcomeDistr]
    ) -> Optional[BilatUFun]:
        """Estimates the bilateral utility function for negotiation j, wrt other outcomes predicted"""

        offer_space = self._get_offer_space(j)
        outcome_space = self._get_outcome_space(j)
        all_offers = offer_space.offer_set()
        all_outcomes = outcome_space.outcome_set()
        util_table: Dict[Outcome, float] = {}  # outcome -> utility

        # failed speedup (caused some problem I couldn't track down)
        # if self._history.current_negotiator_history(j).is_ended():
        #     return BilatUFunDummy(offer_space)  # dummy ufun

        for outcome in all_outcomes:
            util_table[outcome] = 0

        ufun_calc = UFunCalc(self.ufun, self._is_selling())
        for k in range(self._num_sample_outcomes):
            # draw outcomes
            outcomes_other_negs = {
                i: distr.sample() for i, distr in outcome_distrs.items() if i != j
            }
            self.update_util_table(
                ufun_calc, util_table, outcomes_other_negs, all_outcomes
            )

        for outcome in all_outcomes:
            util_table[outcome] /= self._num_sample_outcomes

        return BilatUFunMarginalTable(offer_space, util_table)

    def update_util_table(
        self,
        ufun_calc: UFunCalc,
        util_table: Dict[Outcome, float],
        outcomes_not_j: Dict[str, Outcome],
        possible_outcomes_j_xx: Collection[Outcome],
    ) -> None:
        """Updates (adds to) a utility table values given outcomes of other negotiations"""
        # from here on in, offers are always in negmas form (Tuple[int, int, int])
        # unless indicated by the suffix _xx
        offers_not_j = [
            o.to_negmas(self.awi.current_step)
            for o in list(outcomes_not_j.values())
            if isinstance(o, Offer)
        ]
        ufun_calc.set_persistent_offers(offers_not_j)

        for outcome_j_xx in possible_outcomes_j_xx:
            offer_j = (
                outcome_j_xx.to_negmas(self.awi.current_step)
                if isinstance(outcome_j_xx, Offer)
                else None
            )
            util: float = ufun_calc.ufun_from_offer(offer_j)  # type: ignore

            if self.ufun.current_balance < 0:
                warnings.warn(
                    "We are going bankrupt; util calculation not guaranteed to be accurate"
                )

            if self.enable_safety_checks:
                util_ref: float = self.ufun.from_offers(
                    tuple(offers_not_j + [offer_j]),
                    tuple([self._is_selling()] * (len(offers_not_j) + 1)),
                )  # type: ignore
                assert abs(util - util_ref) < 0.01, "util {}, util_ref {}".format(
                    util, util_ref
                )

            util_table[outcome_j_xx] += util

            # approx_offers = []
            # tot_quant = sum(o.quantity for o in offers)
            # if tot_quant != 0:
            #     avg_price = sum(o.price * o.quantity for o in offers) / tot_quant
            #     approx_offers = [Offer(unit_price=avg_price, quantity=tot_quant)]

            # approx_util = self._my_ufun(approx_offers)
            # real_util = self._my_ufun(offers)

            # self.log("diff: ", offers, approx_offers)
            # self.log(abs(approx_util - real_util), ">>>", approx_util, real_util)
            # util_1 = self._my_ufun(offers)
            # util_2 = self._my_ufun(approx_offers)
            # #util_3 = helpers.ufun_from_offer(self.ufun, approx_offers[0].to_negmas(self.awi.current_step), self._is_selling())
            # util_3 = helpers.ufun_from_offers(self.ufun, offers, [self._is_selling()] * len(offers))
            # self.log("offers:", offers, "offers_approx:", approx_offers)
            # self.log("util:", util_1, "util_approx:", util_2)
            # self.log("diff", abs(util_1 - util_2), abs(util_1 - util_3))
            # util_table[outcome_j] += util_1

    def initial_outcome_distr(self, j: str) -> OutcomeDistr:
        """Initialize outcome distrs with empirical model."""
        model_empirical = ModelEmpirical(j, strategy_self=self._get_strategy(j))
        return model_empirical(
            BilatUFunUniform(self._get_offer_space(j)),  # dummy ufun
            self._history.negotiator_histories(j),
            enable_realistic_checks=False,
        )  # bc of dummy ufun

    # =====================================
    #              HELPERS
    # =====================================

    def _log_negotiation_state(self, offers, counter_proposals):
        """Log stuff so we can see if things are working"""
        self.awi.logdebug(
            "Round {}, ~ percent round complete {}".format(
                self.awi.current_step, self._approx_percent_round_complete()
            )
        )
        self.awi.logdebug(
            "Looking to {}. Units desired: {}, budget factor {}, total: {}".format(
                ("sell" if self._is_selling() else "buy"),
                self._quantity_desired(),
                self._quantity_budget_factor(),
                self._quantity_desired() * self._quantity_budget_factor(),
            )
        )
        self.awi.logdebug(
            "Concession factor {} (0 = full concession, 1 = no concession)".format(
                self._exp_th(self._approx_percent_round_complete())
            )
        )
        self.awi.logdebug("Received offers:")
        for k, v in sorted(offers.items()):
            self.awi.logdebug(f"{self._get_opp_id_from_neg_id(k)} {v}")
        self.awi.logdebug("Counter-offers:\n")
        for k, v in sorted(counter_proposals.items()):
            self.awi.logdebug(f"{self._get_opp_id_from_neg_id(k)} {v}")

    def _log_history(self):
        self.awi.logdebug(
            f"SCML History\n=============\n{self._history}\n================"
        )

    def _vis_negotiations(self):
        """Visualize all the negotiations this round in their entirety"""
        if self._is_selling():
            return  # limit visualizations to buyers so they don't overwrite each other
        opponents_all = sorted(self.negotiators.keys())

        matplotlib.use("agg")
        plt.clf()
        fig, ax = plt.subplots(len(opponents_all), 2)

        def step_ufun_animation(tup):
            rd, (final_estufuns, final_outcomes) = tup

            for idx, o in enumerate(opponents_all):
                ax1 = ax[idx, 0]
                ax2 = ax[idx, 1]

                if final_estufuns and final_outcomes:
                    if ax1.collections and ax1.collections[-1].colorbar:
                        ax1.collections[-1].colorbar.remove()
                    if ax2.collections and ax2.collections[-1].colorbar:
                        ax2.collections[-1].colorbar.remove()
                    final_estufuns[o].vis(fig, ax1)
                    final_outcomes[o].vis(fig, ax2)
                else:
                    # last round, show outcomes
                    rd = "FIN"
                    # draw final outcomes (green stars on top of graph)
                    neg = self._history.negotiator_histories(o)[-1]
                    if neg.is_ended():
                        outcome = neg.outcome()
                        ax2.scatter(
                            outcome.price, outcome.quantity, marker=(5, 1), c="g"
                        )

                u_q = [
                    (
                        self.ufun.from_offers(
                            (outcome.to_negmas(0),), (self._is_selling(),)
                        ),
                        outcome.quantity,
                    )
                    for outcome in self._get_outcome_space(o).outcome_set()
                ]
                u_q.sort()
                best_quant = u_q[-1][1]

                ax1.set_title(f"ufuns: opp {o} round {rd}")
                ax1.set_yticks([self.ex_quant, best_quant])
                ax1.set_yticklabels(
                    [
                        f"ex. q = {self.ex_quant}",
                        f"best_q = {best_quant}",
                    ]
                )
                ax2.set_title(f"outcomes: opp {o} round {rd}")
                ax2.set_yticks([self.ex_quant])
                ax2.set_yticklabels([f"ex. q = {self.ex_quant}"])

        file_id = f"{self.__class__.__name__}.day_{self._step_idx}.final"
        writer = animation.writers["ffmpeg"](fps=1, bitrate=6400)
        anim = animation.FuncAnimation(
            fig,
            step_ufun_animation,
            enumerate(
                zip(self._final_estufuns + [None], self._final_outcomes + [None])
            ),
            repeat=False,
        )
        anim.save(f"{self.vis_path}/{file_id}.ufun_progression.mp4", writer=writer)
        plt.close()

        # GRAPH NEGOTIATION TRACE

        nrows = math.ceil(len(opponents_all) / 2)
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 24))
        axes_1d = np.ravel(axes)
        for opp, ax in zip(opponents_all, axes_1d):
            h = self._history.current_negotiator_history(opp)
            my_offers = [m for m in h.my_offers() if type(m) == Offer]
            opp_offers = [m for m in h.opp_offers() if type(m) == Offer]

            if my_offers:
                prev_move = my_offers[0]
                ax.scatter([prev_move.price], [prev_move.quantity], c="g")
                for o in my_offers[1:]:
                    dp = o.price - prev_move.price
                    dq = o.quantity - prev_move.quantity
                    ax.arrow(
                        prev_move.price,
                        prev_move.quantity,
                        dp,
                        dq,
                        head_width=0.2,
                        color="g",
                    )
                    prev_move = o

            if opp_offers:
                prev_move = opp_offers[0]
                ax.scatter([prev_move.price], [prev_move.quantity], c="r")
                for o in opp_offers[1:]:
                    dp = o.price - prev_move.price
                    dq = o.quantity - prev_move.quantity
                    ax.arrow(
                        prev_move.price,
                        prev_move.quantity,
                        dp,
                        dq,
                        head_width=0.2,
                        color="r",
                    )
                    prev_move = o

            ax.set_title(f"negotiation with {opp}")
            ax.set_xlabel("Price")
            os = self._get_offer_space(opp)
            plt.xlim(0, os.max_p + 1)
            ax.set_ylabel("Quantity")
            plt.ylim(0, os.max_q + 1)

        plt.savefig(f"{self.vis_path}/{file_id}.traces.png")
        plt.close()

        with open(f"{self.vis_path}/{file_id}.traces.txt", "w") as f:
            for opp in opponents_all:
                print("trace", self._history.negotiator_histories(opp)[-1].moves)
                f.write(
                    "{}\n{}\n\n".format(
                        opp, self._history.negotiator_histories(opp)[-1].moves
                    )
                )

    def _vis_consistency(
        self,
        outcome_history: List[Dict[str, OutcomeDistr]],
        estufun_history: List[Dict[str, BilatUFun]],
    ) -> None:
        if self._is_selling():
            return  # limit visualizations to sellers so they don't overwrite each other
        opponents_all = sorted(self.negotiators.keys())

        # calculate ufun distances
        diffs = []
        for i in range(len(outcome_history) - 1):
            outcomes = outcome_history[i + 1]
            old_outcomes = outcome_history[i]
            diffs.append(
                max(outcomes[j].distance(old_outcomes[j]) for j in opponents_all)
            )  # type: ignore
        # self.log("diff progression:", diffs)

        # animations
        writer = animation.writers["ffmpeg"](fps=1, bitrate=6400)
        file_id = "{}.day_{}.counter_{}".format(
            self.__class__.__name__, self._step_idx, self._counter_idx
        )

        matplotlib.use("agg")
        plt.clf()
        fig, axes = plt.subplots(len(opponents_all), 1)

        def step_ufun_animation(tup):
            iteration_idx, estufuns = tup
            iteration_idx += 0.5  # because ufuns are updated half a step after outcomes
            for idx, o in enumerate(opponents_all):
                ax1 = axes[idx]
                if ax1.collections and ax1.collections[-1].colorbar:
                    ax1.collections[-1].colorbar.remove()

                estufuns[o].vis(fig, ax1)

                u_q = [
                    (
                        self.ufun.from_offers(
                            (outcome.to_negmas(0),), (self._is_selling(),)
                        ),
                        outcome.quantity,
                    )
                    for outcome in self._get_outcome_space(o).outcome_set()
                ]
                u_q.sort()
                best_quant = u_q[-1][1]

                ax1.set_title(f"opp {o} iteration # {iteration_idx}")
                ax1.set_yticks([self.ex_quant, best_quant])
                ax1.set_yticklabels(
                    [
                        f"ex. q = {self.ex_quant}",
                        f"best_q = {best_quant}",
                    ]
                )

        anim = animation.FuncAnimation(
            fig, step_ufun_animation, enumerate(estufun_history), repeat=False
        )
        anim.save(f"{self.vis_path}/{file_id}.ufuns_convergence.mp4", writer=writer)

        def step_outcome_animation(tup):
            iteration_idx, outcomes = tup
            for idx, opp in enumerate(opponents_all):
                ax1 = axes[idx]
                outcomes[opp].vis(fig, ax1)
                ax1.set_title(f"opp {opp} iteration # {iteration_idx}")

        anim = animation.FuncAnimation(
            fig, step_ufun_animation, enumerate(outcome_history), repeat=False
        )
        anim.save(
            f"{self.vis_path}/{file_id}.outcomes_convergence.mp4",
            writer=writer,
        )
        plt.close()

        # 3d ufun plot of final ufun
        estufun = estufun_history[-1][opponents_all[0]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        estufun.vis_3d(fig, ax)
        plt.savefig(f"{self.vis_path}/{file_id}.ufun_vis_3d.png")
        plt.close()

    # =====================================
    #             UTILITIES
    # =====================================

    def log(self, *args):
        if self.enable_logging:
            print(self.name, ": ", sep="", end="", file=sys.stderr)
            print(*args, file=sys.stderr)
            self.awi.logdebug_agent(" ".join([str(a) for a in args]))

    def _my_ufun(self, offers: List[Offer]) -> float:
        is_selling = [self._is_selling()] * len(offers)
        offers_negmas = [o.to_negmas(self.awi.current_step) for o in offers]
        util: float = self.ufun.from_offers(tuple(offers_negmas), tuple(is_selling))
        # util_2 = helpers.ufun_from_offer(self.ufun, offer, self._is_selling())
        # self.log("diff", util, util2)
        return util

    def _is_selling(self) -> bool:
        if self.awi.profile.input_product != 0 and self.awi.profile.input_product != 1:
            raise Exception(
                "Input product {} is not 0 or 1 (operating outside OneShot?)".format(
                    self.awi.input_product
                )
            )
        return self.awi.profile.input_product == 0

    def _get_opp_id_from_contract(self, contract) -> str:
        return (
            contract.annotation["buyer"]
            if contract.annotation["product"] == self.awi.my_output_product
            else contract.annotation["seller"]
        )

    def _get_opp_id_from_nmi(self, nmi) -> str:
        return (
            nmi.annotation["buyer"] if self._is_selling() else nmi.annotation["seller"]
        )

    def _get_opp_id_from_neg_id(self, negotiator_id: str) -> str:
        return self._get_opp_id_from_nmi(self.get_nmi(negotiator_id))

    def _get_offer_space(self, neg_id: str) -> OfferSpace:
        nmi = self.get_nmi(neg_id)
        q = nmi.issues[QUANTITY]
        p = nmi.issues[UNIT_PRICE]
        return OfferSpace(
            min_p=p.min_value,
            max_p=p.max_value,
            min_q=q.min_value,
            max_q=q.max_value,
            reserve=Offer(0, 0),
        )

    def _get_outcome_space(self, neg_id: str) -> OutcomeSpace:
        return OutcomeSpace(self._get_offer_space(neg_id))


class MinDisagreementGodfatherAgent(GodfatherAgent):
    """Simple instantiation of Godfather agent with a strategy that always
    bids the minimum price and quantity and never accepts offers, and a
    negotiation model that always predicts disagreement."""

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelDisagreement,
            strategy_type=StrategyMin,
            num_iter_consistency=3,
            num_sample_outcomes=1,
        )

    # def initial_outcome_distr(self, opp_id: str) -> OutcomeDistr:
    #     return OutcomeDistrPoint(self._get_outcome_space(opp_id), Offer(5, 5))


class MinEmpiricalGodfatherAgent(GodfatherAgent):
    """Always bids the minimum price and quantity and never accepts offers,
    plus an empirical negotiation model"""

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyMin,
            num_iter_consistency=3,
            num_sample_outcomes=1,
        )


class AspirationUniformGodfatherAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelUniform,
            strategy_type=StrategyAspiration,
            num_iter_consistency=3,
            num_sample_outcomes=100,
        )


class NonconvergentGodfatherAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelRandomPoint,
            strategy_type=StrategyAspiration,
            num_iter_consistency=7,
            num_sample_outcomes=50,
        )


class ParetoEmpiricalGodfatherAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategySimpleParetoAspiration,
            num_iter_consistency=3,
            num_sample_outcomes=15,
        )


class GoldfishParetoEmpiricalGodfatherAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyGoldfishParetoAspiration,
            num_iter_consistency=1,
            num_sample_outcomes=25,
        )


class SlowGoldfish(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyGoldfishParetoAspiration,
            num_iter_consistency=30,
            num_sample_outcomes=150,
        )


class HardnosedGoldfishGodfatherAgent(GodfatherAgent):
    enable_vis = False

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyHardnosedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )


class HardnosedGoldfishBiggerAgent(GodfatherAgent):
    enable_vis = False

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyHardnosedBigger,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )


class HardnosedGoldfishSmallerAgent(GodfatherAgent):
    enable_vis = False

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyHardnosedSmaller,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )


class SoftnosedGoldfishGodfatherAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategySoftnosedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=15,
        )


class QuickLearningGodfather(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Model:
        return self._strategy_type(0.5, 0.75)

    def _get_model(self, opp_id: str) -> Model:
        strat = self._get_strategy(opp_id)
        return ModelEmpirical(opp_id, strategy_self=strat, prior_bias=2)


class MediumLearningGodfather(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Model:
        return self._strategy_type(0.5, 0.75)

    def _get_model(self, opp_id: str) -> Model:
        strat = self._get_strategy(opp_id)
        return ModelEmpirical(opp_id, strategy_self=strat, prior_bias=10)


class SlowLearningGodfather(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Model:
        return self._strategy_type(0.5, 0.75)

    def _get_model(self, opp_id: str) -> Model:
        strat = self._get_strategy(opp_id)
        return ModelEmpirical(opp_id, strategy_self=strat, prior_bias=25)


class TestGodfatherHard(GodfatherAgent):
    asp_ex = 4
    pct = 1

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Strategy:
        return self._strategy_type(self.asp_ex, self.pct)


class TestGodfatherSoft(GodfatherAgent):
    asp_ex = 1
    pct = 0.5

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Strategy:
        return self._strategy_type(self.asp_ex, self.pct)


class TestGodfatherRegAsp(GodfatherAgent):
    asp_ex = 1

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedRegAsp,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Strategy:
        return self._strategy_type(self.asp_ex)


class ZooGodfather(GodfatherAgent):
    enable_vis = False

    def __init__(self) -> None:
        pop = ["hard"] * 10 + ["soft"] * 10 + ["randpareto"] * 20 + ["regasp"] * 15
        self.strategy_name = random.sample(pop, k=1)[0]
        self.prior_bias = random.sample([1, 5, 10, 10, 10, 20, 30], k=1)[0]
        if self.strategy_name == "hard":
            self._strategy_type = StrategyParameterizedGoldfish
            self.asp_ex = 4
            self.pct = 1
        elif self.strategy_name == "soft":
            self._strategy_type = StrategyParameterizedGoldfish
            self.asp_ex = 1
            self.pct = 0.5
        elif self.strategy_name == "randpareto":
            self._strategy_type = StrategyParameterizedGoldfish
            self.asp_ex = random.uniform(0.25, 4)
            self.pct = random.uniform(0, 1)
        elif self.strategy_name == "regasp":
            self._strategy_type = StrategyParameterizedRegAsp
            self.asp_ex = random.uniform(0.25, 4)
        else:
            raise RuntimeError(f"Wrong strategy {self.strategy_name} in ZooGodfather")
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=self._strategy_type,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_model(self, opp_id: str) -> Model:
        strat = self._get_strategy(opp_id)
        return ModelEmpirical(opp_id, strategy_self=strat, prior_bias=self.prior_bias)

    def _get_strategy(self, opp_id: str) -> Strategy:
        if self._strategy_type == StrategyParameterizedGoldfish:
            return self._strategy_type(self.asp_ex, self.pct)
        elif StrategyParameterizedGoldfish:
            return self._strategy_type(self.asp_ex)
        else:
            raise RuntimeError("No strategy type for RobustGoldfishGodfather")


class TrainingCollectionGodfatherAgent(GodfatherAgent):
    enable_vis = False

    def __init__(self) -> None:
        super().__init__(
            model_type=ModelEmpirical,
            strategy_type=StrategyParameterizedGoldfish,
            num_iter_consistency=1,
            num_sample_outcomes=20,
        )

    def _get_strategy(self, opp_id: str) -> Model:
        return self._strategy_type(0.5, 0.75)


class ChristopherTheGoldfishAgent(GodfatherAgent):
    def __init__(self) -> None:
        super().__init__(
            model_type=None,
            strategy_type=StrategyGoldfishParetoAspiration,
            num_iter_consistency=2,
            num_sample_outcomes=30,
        )

    def _get_model(self, opp_id: str) -> Model:
        return ModelChris(opp_id, StrategyGoldfishParetoAspiration(), self)


class CheatingGodfatherAgent(GodfatherAgent):
    def __init__(self, strat) -> None:
        self.strat = strat
        super().__init__(
            strategy_type=strat,
            num_iter_consistency=3,
            num_sample_outcomes=1,  # point estimate
        )

    def get_my_id(self, opp_id: str):
        """Gets agent's own id. Probably not the best method, but oh well."""
        nmi = self.get_nmi(opp_id)
        our_ids = nmi.agent_ids
        my_id_list = [i for i in our_ids if i != opp_id]
        if len(my_id_list) != 1:
            raise RuntimeError("Couldn't get my negotiator id from nmi")
        return my_id_list[0]

    def opp_ufun_getter(self, opp_id: str) -> Callable[[], Optional[BilatUFun]]:
        opponent = self.awi._world.agents[opp_id]
        my_id = self.get_my_id(opp_id)

        def getter():
            if not hasattr(opponent, "estufuns") or my_id not in opponent.estufuns:
                return None
            else:
                return opponent.estufuns[my_id]

        return getter

    def _get_model(self, opp_id: str) -> Model:
        return ModelCheating(
            ufun_getter=self.opp_ufun_getter(opp_id),
            strategy_self=self.strat(),
            strategy_opp=self.strat(),
        )

    def _get_strategy(self, opp_id: str) -> Strategy:
        return self.strat()


class CheatingSPAGodfatherAgent(CheatingGodfatherAgent):
    def __init__(self) -> None:
        super().__init__(strat=StrategySimpleParetoAspiration)


class CheatingGPAGodfatherAgent(CheatingGodfatherAgent):
    def __init__(self) -> None:
        super().__init__(strat=StrategyGoldfishParetoAspiration)


class CheatingCPAGodfatherAgent(CheatingGodfatherAgent):
    def __init__(self) -> None:
        super().__init__(strat=StrategyCheatingParetoAspiration)

    def _get_model(self, opp_id: str) -> Model:
        opponent = self.awi._world.agents[opp_id]
        my_id = self.get_my_id(opp_id)

        return ModelCheating(
            ufun_getter=self.opp_ufun_getter(opp_id),
            strategy_self=self.strat(ufun_getter=self.opp_ufun_getter(opp_id)),
            strategy_opp=self.strat(ufun_getter=opponent.opp_ufun_getter(my_id)),
        )

    def _get_strategy(self, opp_id: str) -> Strategy:
        return self.strat(ufun_getter=self.opp_ufun_getter(opp_id))


class ModelChris(Model):
    HEADERS_LIST = "trace 0 price,trace 0 quant,trace 1 price,trace 1 quant,trace 2 price,trace 2 quant,trace 3 price,trace 3 quant,trace 4 price,trace 4 quant,trace 5 price,trace 5 quant,trace 6 price,trace 6 quant,trace 7 price,trace 7 quant,trace 8 price,trace 8 quant,trace 9 price,trace 9 quant,trace 10 price,trace 10 quant,trace 11 price,trace 11 quant,trace 12 price,trace 12 quant,trace 13 price,trace 13 quant,trace 14 price,trace 14 quant,trace 15 price,trace 15 quant,trace 16 price,trace 16 quant,trace 17 price,trace 17 quant,trace 18 price,trace 18 quant,trace 19 price,trace 19 quant,trace 20 price,trace 20 quant,trace 21 price,trace 21 quant,trace 22 price,trace 22 quant,trace 23 price,trace 23 quant,trace 24 price,trace 24 quant,trace 25 price,trace 25 quant,trace 26 price,trace 26 quant,trace 27 price,trace 27 quant,trace 28 price,trace 28 quant,trace 29 price,trace 29 quant,trace 30 price,trace 30 quant,trace 31 price,trace 31 quant,trace 32 price,trace 32 quant,trace 33 price,trace 33 quant,trace 34 price,trace 34 quant,trace 35 price,trace 35 quant,trace 36 price,trace 36 quant,trace 37 price,trace 37 quant,trace 38 price,trace 38 quant,trace 39 price,trace 39 quant,trace 40 price, trace 40 quant, day,n_level,n_opp_level,my_layer_size,opp_layer_size,competitiveness,ufun_param_0,ufun_param_1,ufun_param_2,ufun_param_3,ufun_param_4,ufun_param_5,ufun_param_6,ufun_param_7,ufun_param_8,ufun_param_9,empirical_distr_q_0,empirical_distr_q_1,empirical_distr_q_2,empirical_distr_q_3,empirical_distr_q_4,empirical_distr_q_5,empirical_distr_q_6,empirical_distr_q_7,empirical_distr_q_8,empirical_distr_q_9,empirical_distr_q_10,empirical_distr_p"
    feature_names = [
        " day",
        " trace 40 quant",
        "competitiveness",
        "empirical_distr_p",
        "empirical_distr_q_0",
        "empirical_distr_q_1",
        "empirical_distr_q_10",
        "empirical_distr_q_2",
        "empirical_distr_q_3",
        "empirical_distr_q_4",
        "empirical_distr_q_5",
        "empirical_distr_q_6",
        "empirical_distr_q_7",
        "empirical_distr_q_8",
        "empirical_distr_q_9",
        "my_layer_size",
        "n_level",
        "n_opp_level",
        "opp_layer_size",
        "trace 0 price",
        "trace 0 quant",
        "trace 1 price",
        "trace 1 quant",
        "trace 10 price",
        "trace 10 quant",
        "trace 11 price",
        "trace 11 quant",
        "trace 12 price",
        "trace 12 quant",
        "trace 13 price",
        "trace 13 quant",
        "trace 14 price",
        "trace 14 quant",
        "trace 15 price",
        "trace 15 quant",
        "trace 16 price",
        "trace 16 quant",
        "trace 17 price",
        "trace 17 quant",
        "trace 18 price",
        "trace 18 quant",
        "trace 19 price",
        "trace 19 quant",
        "trace 2 price",
        "trace 2 quant",
        "trace 20 price",
        "trace 20 quant",
        "trace 21 price",
        "trace 21 quant",
        "trace 22 price",
        "trace 22 quant",
        "trace 23 price",
        "trace 23 quant",
        "trace 24 price",
        "trace 24 quant",
        "trace 25 price",
        "trace 25 quant",
        "trace 26 price",
        "trace 26 quant",
        "trace 27 price",
        "trace 27 quant",
        "trace 28 price",
        "trace 28 quant",
        "trace 29 price",
        "trace 29 quant",
        "trace 3 price",
        "trace 3 quant",
        "trace 30 price",
        "trace 30 quant",
        "trace 31 price",
        "trace 31 quant",
        "trace 32 price",
        "trace 32 quant",
        "trace 33 price",
        "trace 33 quant",
        "trace 34 price",
        "trace 34 quant",
        "trace 35 price",
        "trace 35 quant",
        "trace 36 price",
        "trace 36 quant",
        "trace 37 price",
        "trace 37 quant",
        "trace 38 price",
        "trace 38 quant",
        "trace 39 price",
        "trace 39 quant",
        "trace 4 price",
        "trace 4 quant",
        "trace 40 price",
        "trace 5 price",
        "trace 5 quant",
        "trace 6 price",
        "trace 6 quant",
        "trace 7 price",
        "trace 7 quant",
        "trace 8 price",
        "trace 8 quant",
        "trace 9 price",
        "trace 9 quant",
        "ufun_param_0",
        "ufun_param_1",
        "ufun_param_2",
        "ufun_param_3",
        "ufun_param_4",
        "ufun_param_5",
        "ufun_param_6",
        "ufun_param_7",
        "ufun_param_8",
        "ufun_param_9",
    ]

    def __init__(self, opp_id: str, strategy_self: Strategy, agent: "GodfatherAgent"):
        self.agent = agent
        self.models_q = {}
        self.models_p = {}
        for i in range(40):
            self.models_q[i] = xgb.Booster()
            self.models_p[i] = xgb.Booster()
            model_dir = pathlib.Path(__file__).parent / "models"
            self.models_p[i].load_model(f"{model_dir}/p_model_{i}.json")
            self.models_q[i].load_model(f"{model_dir}/q_model_{i}.json")

        super().__init__(opp_id, strategy_self)

    def predict_quantity(
        self, trace_len: int, row: List[float], headers: List[str]
    ) -> List[float]:
        if trace_len > 20:
            trace_len = 20
        if trace_len % 2 != 0:
            trace_len -= 1

        cols = self.HEADERS_LIST.split(",")
        row = pd.DataFrame([row], columns=cols)
        row = row[self.feature_names]
        dmatrix = xgb.DMatrix(row)

        model_q = self.models_q[trace_len]
        q = model_q.predict(dmatrix)
        q_probs = q.tolist()[0]
        # q = [q[i]/sum(q) for i in q]
        if not abs(sum(q_probs) - 1) < 0.01:
            warnings.warn("q_probs don't sum to 1")
            q_probs = [p / sum(q_probs) for p in q_probs]
        return q_probs

    def predict_price(
        self, trace_len: int, row: List[float], headers: List[str]
    ) -> int:
        cols = self.HEADERS_LIST.split(",")
        row = pd.DataFrame([row], columns=cols)
        row = row[self.feature_names]
        dmatrix = xgb.DMatrix(row)

        model_p = self.models_p[trace_len]
        ps = model_p.predict(dmatrix)
        p = int(round(ps[0]))
        return p

    def predict_quantity_similarity(self):
        raise NotImplementedError

    @realistic
    def __call__(
        self, self_ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> OutcomeDistr:
        len_trace = len(histories[-1].moves)
        ufun_params = self_ufun.poly_fit_3rd_correct()
        row, headers = construct_test_cols(
            day_idx=self.agent._step_idx,
            agent=self.agent,
            histories=histories,
            opp_type=self._neg_id[2:5],
            offer_space=self_ufun.offer_space,
            len_trace=len_trace,
            ufun_params=ufun_params,
        )

        q_probs = self.predict_quantity(len_trace, row, headers)
        p_est = self.predict_price(len_trace, row, headers)

        complete_histories = [h for h in histories if h.is_ended()]
        model_info = len(complete_histories) + 1
        total_weight = model_info + 5
        prior_weight = 5 / total_weight
        model_weight = model_info / total_weight

        # print("round", len(complete_histories))
        # print("model", model_weight)
        # print("prior", prior_weight)

        for i in range(len(q_probs)):
            q_probs[i] = q_probs[i] * model_weight

        q_probs[0] += prior_weight

        return OutcomeDistrMarginal(
            outcome_space=self_ufun.outcome_space, q_probs=q_probs, p_est=p_est
        )
