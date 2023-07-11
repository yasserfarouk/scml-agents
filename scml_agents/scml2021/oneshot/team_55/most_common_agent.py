#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition (one-shot track).
Game Description is available at:
http://www.yasserm.com/scml/scml2021oneshot.pdf

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.oneshot.OneShotAWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu

To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/most_common_agent.py

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

>> python /{path-to-this-file}/most_common_agent.py

You should see a short tournament running and results reported.

"""

# required for running tournaments and printing
import collections
import logging
import time

# required for typing
import types
from typing import Any, Optional, Type, Union

import numpy as np
from negmas import ControlledNegotiator, Outcome, ResponseType
from negmas.helpers import humanize_time
from negmas.sao import SAOState

# required for development
from scml import QUANTITY, UNIT_PRICE, GreedyOneShotAgent
from scml.oneshot import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from tabulate import tabulate

from .worker_agents import AdaptiveAgent, BetterAgent, LearningAgent, SimpleAgent

__all__ = []


class ZilberanBackup(OneShotAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details

    """

    DEBUG = True
    ALL_BASE_AGENTS = [
        GreedyOneShotAgent,
        SimpleAgent,
        BetterAgent,
        AdaptiveAgent,
        LearningAgent,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_agents = [agent(*args, **kwargs) for agent in self.ALL_BASE_AGENTS]

        word_blacklist = ["span", "negotiator"]
        not_overridden_methods = [
            "add_method",
            "generate_method_which_call_inner_agents",
            "propose",
            "respond",
            "connect_to_oneshot_adapter",
            "connect_to_2021_adapter",
            "make_ufun",
            "on_preferences_changed",
        ]

        for attribute in dir(self):
            is_private = "_" == attribute[:1]
            is_builtin = "__" == attribute[:2]
            is_in_blacklist = any([word in attribute for word in word_blacklist])
            is_forbidden = attribute in not_overridden_methods

            try:
                attribute_object = getattr(self, attribute)
                if (
                    isinstance(attribute_object, types.MethodType)
                    and not is_forbidden
                    and not is_builtin
                    and not is_private
                    and not is_in_blacklist
                ):
                    if self.DEBUG:
                        self.awi.logdebug(f"Patching {attribute}")

                    method_with_proxy = self.generate_method_which_call_inner_agents(
                        attribute, self.base_agents
                    )
                    self.add_method(attribute, method_with_proxy)

                else:
                    if self.DEBUG:
                        self.awi.logdebug(f"Skipped because illegal: {attribute}")

            except AttributeError:
                pass

    def init(self):
        for agent in self.base_agents:
            agent._awi = self._awi
            agent._negotiators = self._negotiators

    def generate_method_which_call_inner_agents(self, method_name, agents):
        def modified_method(*args, **kwargs):
            for agent in agents:
                if hasattr(agent, method_name):
                    try:
                        getattr(agent, method_name)(*args[1:], **kwargs)

                    except TypeError as e:
                        raise ValueError(f"{args} {kwargs}")

            if hasattr(self, method_name):
                try:
                    getattr(super(), method_name)(*args[1:], **kwargs)

                except TypeError as e:
                    raise ValueError(f"{args} {kwargs}")

        modified_method.__name__ = method_name
        return modified_method

    @classmethod
    def add_method(cls, name, func):
        return setattr(cls, name, types.MethodType(func, cls))

    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        """Called when the agent is asking to propose in one negotiation"""
        proposes = [agent.propose(negotiator_id, state) for agent in self.base_agents]

        # TODO: Run KMEANS with Elbow
        propose = [-1] * 3
        propose[UNIT_PRICE] = sum(propose[UNIT_PRICE] for propose in proposes) / len(
            proposes
        )
        propose[QUANTITY] = sum(propose[QUANTITY] for propose in proposes) // len(
            proposes
        )
        return propose

    def respond(self, negotiator_id: str, state: SAOState) -> ResponseType:
        """Called when the agent is asked to respond to an offer"""
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        responds = [
            agent.respond(negotiator_id, state, offer) for agent in self.base_agents
        ]
        return collections.Counter(responds).most_common()[0][0]


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=2,
    n_configs=1,  # =10  # =2
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are oneshot, std,
                     collusion).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    if competition == "oneshot":
        competitors = [
            MyAgent,
            RandomOneShotAgent,
            BetterAgent,
            AdaptiveAgent,
            LearningAgent,
        ]
    else:
        from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

        competitors = [
            MyAgent,
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
        ]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2021_std
    elif competition == "collusion":
        runner = anac2021_collusion
    else:
        runner = anac2021_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        log_screen_level=logging.ERROR,
        log_to_screen=True,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
