import os
import sys

sys.path.append(os.path.dirname(__file__))

from collections import defaultdict

from matplotlib import pyplot as plt
from negmas import save_stats


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


def plot(world):
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()
    save_stats(world, world.log_folder)
    show_agent_scores(world)
