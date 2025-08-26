import json
import os

# 假设 P_tester_vary_ptoday.py 在项目根目录
# SCML_initial/P_tester_vary_ptoday.py
# LitaAgentP 在 SCML_initial/litaagent_std/litaagent_p.py
from .litaagent_std.litaagent_p import LitaAgentP
from scml.std import SCML2024StdWorld, RandomStdAgent, SyncRandomStdAgent
from scml.oneshot import GreedyOneShotAgent
import matplotlib.pyplot as plt

# 定义配置文件路径 (在项目根目录)
CONFIG_FILE_PATH = "agent_config.json"

# 定义要测试的 _ptoday 值列表
ptoday_values_to_test = [0.3, 0.5, 0.7, 0.9, 1.0]

# 存储每个 ptoday 值对应的 LitaAgentP 的各项得分
results_summary = {}

# --- 主测试循环 ---
for ptoday_val in ptoday_values_to_test:
    pass  # print(f"\n{'=' * 20} 正在测试 LitaAgentP, _ptoday = {ptoday_val} {'=' * 20}")

    # 1. 写入配置文件
    config_data = {"ptoday": ptoday_val}
    try:
        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(config_data, f)
        pass  # print(f"已将配置: {config_data} 写入到 {CONFIG_FILE_PATH}")
    except IOError:
        pass  # print(f"错误：无法写入配置文件 {CONFIG_FILE_PATH}: {e}")
        continue  # 如果无法写入配置，则跳过此次测试

    # 2. 设置并运行 SCML 世界
    # 确保 LitaAgentP 是参与者之一
    agent_types_for_run = [
        LitaAgentP,
        RandomStdAgent,
        SyncRandomStdAgent,
        GreedyOneShotAgent,
    ]

    world_params = SCML2024StdWorld.generate(
        agent_types=agent_types_for_run,
        n_steps=25,  # 可以根据需要调整仿真步数
    )

    world = SCML2024StdWorld(**world_params)

    pass  # print(f"开始为 _ptoday = {ptoday_val} 运行仿真...")
    world.run_with_progress()
    _, _ = world.draw()

    # Run the tournament
    world.run_with_progress()  # may take few minutes

    # Plot the results
    world.plot_stats("n_negotiations", ylegend=1.25)
    world.draw(figsize=(300, 300))
    plt.show()

    world.plot_stats("storage_cost", ylegend=1.25)
    world.draw(figsize=(300, 300))
    plt.show()

    world.plot_stats("inventory_penalized", ylegend=1.25)
    world.draw(figsize=(300, 300))
    plt.show()

    world.plot_stats("productivity", ylegend=1.25)
    world.draw(figsize=(300, 300))
    plt.show()

    world.plot_stats("score", ylegend=1.25)
    world.draw(figsize=(300, 300))
    plt.show()

    world.plot_stats(figsize=(10, 10))
    world.draw(figsize=(300, 300))
    plt.show()


# 最后清理配置文件
if os.path.exists(CONFIG_FILE_PATH):
    try:
        os.remove(CONFIG_FILE_PATH)
        pass  # print(f"\n已清理配置文件: {CONFIG_FILE_PATH}")
    except OSError:
        pass  # print(f"错误：无法清理配置文件 {CONFIG_FILE_PATH}: {e}")
