import random
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
from scml.oneshot import *
from scml.scml2020 import is_system_agent

__all__ = [
    'try_agent',
    'try_agents',
    'analyze_contracts',
    'print_type_scores',
    'print_agent_scores',
]


def try_agent(agent_type, n_processes=2, draw=True):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    return try_agents([RandomOneShotAgent, agent_type], n_processes, draw=draw)


def try_agents(
        agent_types,
        n_processes=2,
        n_trials=1,
        steps=50,
        draw=True,
        agent_params=None
):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    for _ in range(n_trials):
        p = n_processes if isinstance(n_processes, int) else random.randint(*n_processes)
        world = SCML2023OneShotWorld(
            **SCML2023OneShotWorld.generate(
                agent_types,
                agent_params=agent_params,
                n_steps=steps,
                n_processes=p,
                random_agent_types=True,
            ),
            construct_graphs=True,
        )
        world.run()

        all_scores = world.scores()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            key = aid if n_trials == 1 else f"{aid}@{world.id[:4]}"
            agent_scores[key] = (
                agent.type_name.split(':')[-1].split('.')[-1],
                all_scores[aid],
                '(bankrupt)' if world.is_bankrupt[aid] else ''
            )
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            type_ = agent.type_name.split(':')[-1].split('.')[-1]
            type_scores[type_] += all_scores[aid]
            counts[type_] += 1
    type_scores = {k: v / counts[k] if counts[k] else v for k, v in type_scores.items()}
    if draw:
        world.draw(
            what=["contracts-concluded"],
            steps=(0, world.n_steps - 1),
            together=True, ncols=1, figsize=(20, 20)
        )
        plt.show()

    return world, agent_scores, type_scores


def analyze_contracts(world):
    """
    Analyzes the contracts signed in the given world
    """
    import pandas as pd
    data = pd.DataFrame.from_records(world.saved_contracts)
    return data.groupby(["seller_name", "buyer_name"])[["quantity", "unit_price"]].mean()


def print_agent_scores(agent_scores):
    """
    Prints scores of individiual agent instances
    """
    for aid, (type_, score, bankrupt) in agent_scores.items():
        pass # print(f"Agent {aid} of type {type_} has a final score of {score} {bankrupt}")


def print_type_scores(type_scores):
    """Prints scores of agent types"""
    pprint(sorted(tuple(type_scores.items()), key=lambda x: -x[1]))
