# このスクリプトは、指定されたエージェントクラスを使ってSCMLのシミュレーションを実行し、結果を図として保存するためのものです。

# 必要なモジュールのインポート（引数解析、動的インポート、ファイルパス操作など）
import argparse
import importlib
import os
import sys
# SCML環境とシミュレーション実行用クラスのインポート
from scml.std import *
from scml.runner import WorldRunner
# コマンドライン引数の設定（エージェントクラス名を受け取る）
parser = argparse.ArgumentParser(description="Run SCML simulation with specified agent")
parser.add_argument("agent_class", type=str, help="Agent class name in myagent module (e.g., SotaAgent)")
args = parser.parse_args()

# 上位ディレクトリをパスに追加してmyagentパッケージをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# myagent配下のすべてのモジュールからエージェントクラスを検索する準備
import pkgutil
import myagent

# 指定された名前のエージェントクラスをmyagent内のどこかから探して取得する関数
def find_agent_class(agent_class_name):
    for loader, module_name, is_pkg in pkgutil.iter_modules(myagent.__path__):
        module = importlib.import_module(f"myagent.{module_name}")
        if hasattr(module, agent_class_name):
            return getattr(module, agent_class_name)
    raise ImportError(f"Agent class '{agent_class_name}' not found in any myagent module.")

AgentClass = find_agent_class(args.agent_class)

# シミュレーション環境の設定（交渉ステップ数、プロセス数など）
context = ANACStdContext(
    n_steps=10, n_processes=3, world_params=dict(construct_graphs=True)
)
# エージェント単体でのシミュレーションを行うランナーの初期化
single_agent_runner = WorldRunner(
    context, n_configs=4, n_repetitions=1, save_worlds=True
)
# すべてのエージェントを制御するモード（今回は未使用だが拡張用に定義）
full_market_runner = WorldRunner.from_runner(
    single_agent_runner, control_all_agents=True
)

# 指定されたエージェントでシミュレーションを実行
single_agent_runner(AgentClass)
# 結果の可視化（ワールド構造、交渉の進行、スコアなど）
world_fig, _ = single_agent_runner.draw_worlds_of(AgentClass)
plots_fig, _ = single_agent_runner.plot_stats(agg=False)
stats_fig, _ = single_agent_runner.plot_stats(stats="score")

# 出力先ディレクトリ（myagent/fig/{エージェント名}）を作成
output_dir = os.path.join("myagent", "fig", args.agent_class)
os.makedirs(output_dir, exist_ok=True)

world_fig.savefig(os.path.join(output_dir, "world_fig.png"))
plots_fig.savefig(os.path.join(output_dir, "plots_fig.png"))
stats_fig.savefig(os.path.join(output_dir, "stats_fig.png"))

# シミュレーション結果のスコアを出力
pass # print(single_agent_runner.score_summary())