from scml.std import (
    ANACStdContext,
    StrongSupplierStdContext,
    StrongConsumerStdContext,
    StrongMiddleManStdContext,
    GreedyStdAgent,
    SyncRandomStdAgent,
    RandomStdAgent,
    GreedySyncAgent,
    SingleAgreementAspirationAgent,
)
from scml.runner import WorldRunner

from balanced_greedy.balanced_greedy_std_agent import BalancedGreedyStdAgent


# --- 評価設定 ---------------------------------------------------
# CONFIGS を増やすほど結果が安定するが時間がかかる。
# 普段の確認は 5、しっかり測りたいときは 10〜20。
CONFIGS = 10
REPS = 2
STEPS = 50

# 本番に近づけるため、複数の立場(Context)で測る。
# 各 Context で自分のエージェントがどう振る舞うかを見る。
CONTEXTS = {
    "ANAC (general)": ANACStdContext,
    "StrongSupplier": StrongSupplierStdContext,
    "StrongConsumer": StrongConsumerStdContext,
    #"StrongMiddleMan": StrongMiddleManStdContext,
}

# 本番の相手に近づけるため、組み込みエージェントを多めに混ぜる。
OPPONENTS = [
    GreedyStdAgent,
    SyncRandomStdAgent,
    RandomStdAgent,
    GreedySyncAgent,
    SingleAgreementAspirationAgent,
]


def evaluate_in_context(name, context_cls):
    print(f"\n############################################")
    print(f"### Context: {name}")
    print(f"############################################")

    context = context_cls(
        n_steps=STEPS,
        world_params=dict(construct_graphs=False),
    )

    runner = WorldRunner(
        context,
        n_configs=CONFIGS,
        n_repetitions=REPS,
        save_worlds=False,
    )

    # まず自分のエージェント、続いて相手たちを同条件で評価
    agents = [BalancedGreedyStdAgent] + OPPONENTS
    for agent in agents:
        print(f"  Evaluating {agent.__name__} ...")
        runner(agent)

    print(f"\n--- Score Summary ({name}) ---")
    print(runner.score_summary())


if __name__ == "__main__":
    for name, ctx in CONTEXTS.items():
        evaluate_in_context(name, ctx)