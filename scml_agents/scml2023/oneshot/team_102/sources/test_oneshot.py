import os, sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(__file__))

import datetime
from typing import Dict, Any

import pandas as pd
from scml.utils import anac2023_oneshot
from scml_agents.agents import get_agents

from agent import RLIndAgent, RLSyncAgent
from tutorials.example_agents_oneshot import (
    GreedyAgent,
    SimpleAgent,
    BetterAgent,
    AdaptiveAgent,
    LearningAgent,
    SyncAgent,
    MyOneShotDoNothing,
    DeepSimpleAgent,
    SimpleSingleAgreementAgent,
    GreedyIndNeg,
    SimpleSyncAgent,
    MySyncAgent,
)
from tutorials.run_example import (
    try_agents,
    print_agent_scores,
    print_type_scores,
    analyze_contracts,
)

# 現在時間
now = datetime.datetime.now()


def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec.tolist()


def my_try_agent(agent_type, n_processes=2):
    return try_agents(
        [
            SimpleAgent,
            BetterAgent,
            AdaptiveAgent,
            LearningAgent,
            SyncAgent,
            MyOneShotDoNothing,
            DeepSimpleAgent,
            SimpleSingleAgreementAgent,
            GreedyIndNeg,
            agent_type
        ],
        n_processes=n_processes,
        n_trials=1,
        steps=50
    )


def test_agent(agent_type):
    world, ascores, tscores = my_try_agent(agent_type)

    print_agent_scores(ascores)

    # エージェントタイプごとのスコアの表示
    print_type_scores(tscores)

    # 契約の分析
    pd.set_option('display.max_columns', None)
    pass # print(analyze_contracts(world))


def test_agent_on_tournaments(
        agent_type,
        agent_param: Dict[str, Any] = None,
        n_config: int = 30,
        n_runs_per_world: int = 1,
        n_steps: int = 100,
        parallelism: str = "serial",
):
    non_competitors = [
        # GreedyAgent,
        SimpleAgent,
        # BetterAgent,
        AdaptiveAgent,
        LearningAgent,
        # SyncAgent,
        # MySyncAgent,
        # MyOneShotDoNothing,
        # DeepSimpleAgent,
        # SimpleSingleAgreementAgent,
        # GreedyIndNeg,
        # RLSyncAgent,
    ]
    competitors = [agent_type] + non_competitors
    competitor_params = None
    if agent_param:
        competitor_params = [dict(controller_params=agent_param)] + [dict()] * len(non_competitors)

    pass # print(competitors)
    run_tournaments_oneshot(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        n_config=n_config,
        n_steps=n_steps,
        n_runs_per_world=n_runs_per_world,
        parallelism=parallelism,
    )


def run_tournaments_oneshot(
        competitors: list,
        competitor_params: list = None,
        non_competitors: list = None,
        tournament_path: str = "results/tournaments",
        log_folder: str = "results/tournaments/logs",
        log_file_name: str = f"{now.date()}-{now.time()}_log.txt",
        n_config: int = 30,
        n_runs_per_world: int = 1,
        n_steps: int = 100,
        parallelism: str = "serial",
):
    results = anac2023_oneshot(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        tournament_path=tournament_path,
        log_folder=log_folder,
        log_file_name=log_file_name,
        n_configs=n_config,
        # n_competitors_per_world=None,
        n_runs_per_world=n_runs_per_world,
        n_steps=n_steps,
        n_processes=2,
        # print_exceptions=True,
        # compact=True,
        # no_logs=True,
        # verbose=False,
        parallelism=parallelism,
    )
    pass # print(results.winners)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pass # print(results.score_stats)


TRY = 'try'
TOURNAMENT = 'tournament'
TEST = 'test'
RL = 'rl'
RL_TEST = 'rl_test'

if __name__ == '__main__':
    AGENT = RLIndAgent
    mode = RL_TEST

    if mode == TRY:
        # 軽いテスト
        test_agent(AGENT)

    elif mode == TOURNAMENT:
        # 重めのテスト
        test_agent_on_tournaments(AGENT)

    elif mode == TEST:
        # 去年のエージェントとのテスト
        tournament_types = get_agents(
            version=2022,
            track="oneshot",
            finalists_only=True,
            top_only=5,
            as_class=True,
        )
        pass # print(tournament_types)
        run_tournaments_oneshot(tournament_types)

    elif mode == RL:
        params = {
            'model_name': 'Ind_PPO_I-MD-AM_I-MD-OM_128-128.pth',
            'update_model': True,
            'save_model': True,
            'tb_log': True,
            'lr_actor': 3e-5,
            'lr_critic': 1e-4,
        }

        test_agent_on_tournaments(
            AGENT,
            params,
            n_steps=100,
            n_config=30,
            n_runs_per_world=3,
            parallelism='serial',
        )

        # plot rewards
        plt.figure()
        plt.plot(RLSyncAgent.step_rewards)
        plt.plot(moving_average(RLSyncAgent.step_rewards, 100))
        plt.show()

    elif mode == RL_TEST:
        test_agent_on_tournaments(
            AGENT,
            n_steps=100,
            n_config=5,
            n_runs_per_world=1,
            parallelism='parallel',
        )


