import pandas as pd
from scml.oneshot import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.utils import anac2023_oneshot

# トーナメントの実行（Standard, Collusion)

pd.options.display.float_format = '{:,.2f}'.format


def shorten_names(results):
    # just make agent types more readable
    results.score_stats.agent_type = results.score_stats.agent_type.str.split(".").str[-1]
    results.kstest.a = results.kstest.a.str.split(".").str[-1]
    results.kstest.b = results.kstest.b.str.split(".").str[-1]
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(".").str[-1]
    results.scores.agent_type = results.scores.agent_type.str.split(".").str[-1]
    results.winners = [_.split(".")[-1] for _ in results.winners]
    return results


def main():
    tournament_types = [RandomOneShotAgent, SyncRandomOneShotAgent, GreedyOneShotAgent, GreedySingleAgreementAgent]
    # may take a long time
    results = anac2023_oneshot(
        competitors=tournament_types,
        n_configs=5,  # number of different configurations to generate
        n_runs_per_world=1,  # number of times to repeat every simulation (with agent assignment)
        n_steps=10,  # number of days (simulation steps) per simulation
        print_exceptions=True,
    )
    results = shorten_names(results)

    # 勝者の表示
    pass # print("winner:" + str(results.winners))

    # 結果詳細
    pass # print("score_stats")
    pass # print(results.score_stats)

    pass # print("\nkstest")
    pass # print(results.kstest)

    pass # print("\ntotal_scores")
    pass # print(results.total_scores)

    # agent typeごとのスコア
    pass # print("\nscores of agent types")
    pass # print(results.scores.loc[:, ["agent_name", "agent_type", "score"]].head())


if __name__ == '__main__':
    main()