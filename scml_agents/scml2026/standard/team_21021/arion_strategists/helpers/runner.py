import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from negmas.helpers import humanize_time
from tabulate import tabulate

from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.std import SCML2024StdWorld
from scml.std.agents import GreedyStdAgent, SyncRandomStdAgent
from scml.utils import anac2024_oneshot, anac2024_std


def run(
    competitors=tuple(),
    competition="oneshot",
    n_steps=10,
    n_configs=2,
):
    if competition == "oneshot":
        competitors = list(competitors) + [RandomOneShotAgent, SyncRandomOneShotAgent]
    else:
        competitors = list(competitors) + [SyncRandomStdAgent, GreedyStdAgent]

    start = time.perf_counter()
    runner = anac2024_std if competition == "std" else anac2024_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
    )
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(".").str[-1]  # type: ignore
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


def _metric_series(world, metric: str, agent_name: str) -> list[float]:
    return list(world.stats.get(f"{metric}_{agent_name}", []))


def _collect_world_metrics(world) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for agent in world.non_system_agents:
        name = str(agent.name)
        agent_type = str(getattr(agent, "short_type_name", type(agent).__name__))
        score_series = _metric_series(world, "score", name)
        storage_series = _metric_series(world, "storage_cost", name)
        shortfall_series = _metric_series(world, "shortfall_penalty", name)
        rows.append(
            dict(
                agent_type=agent_type,
                agent_name=name,
                final_score=float(score_series[-1]) if score_series else float("nan"),
                shortfall_penalty_total=float(sum(shortfall_series)),
                storage_cost_total=float(sum(storage_series)),
            )
        )
    return rows


def _summarize(results_df: pd.DataFrame, group_col: str = "agent_type") -> pd.DataFrame:
    grouped = results_df.groupby(group_col)
    summary = pd.DataFrame(
        {
            "runs": grouped["final_score"].count(),
            "score_mean": grouped["final_score"].mean(),
            "score_std": grouped["final_score"].std(ddof=0),
            "score_p10": grouped["final_score"].quantile(0.10),
            "score_min": grouped["final_score"].min(),
            "score_max": grouped["final_score"].max(),
            "shortfall_mean": grouped["shortfall_penalty_total"].mean(),
            "storage_mean": grouped["storage_cost_total"].mean(),
        }
    ).reset_index()
    return summary.sort_values("score_mean", ascending=False)


def benchmark_std(
    competitors=tuple(),
    n_steps=30,
    n_configs=4,
    seeds=(11, 22, 33),
):
    start = time.perf_counter()
    base_competitors = list(competitors) + [SyncRandomStdAgent, GreedyStdAgent]
    seen = set()
    final_competitors = []
    for c in base_competitors:
        key = f"{c.__module__}.{c.__name__}"
        if key in seen:
            continue
        seen.add(key)
        final_competitors.append(c)

    all_rows: list[dict[str, float | int | str]] = []
    world_counter = 0
    for seed in seeds:
        for cfg_idx in range(n_configs):
            world_counter += 1
            local_seed = int(seed + cfg_idx * 1009)
            random.seed(local_seed)
            np.random.seed(local_seed)
            world = SCML2024StdWorld(
                **SCML2024StdWorld.generate(
                    agent_types=final_competitors,
                    n_steps=n_steps,
                    no_logs=True,
                )
            )
            world.run()
            world_rows = _collect_world_metrics(world)
            for row in world_rows:
                row["world_id"] = world_counter
                row["seed"] = seed
                row["config_idx"] = cfg_idx
                all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    summary = _summarize(results_df, group_col="agent_type")
    print(
        f"Benchmark finished: {len(seeds)} seeds x {n_configs} configs = "
        f"{len(seeds) * n_configs} worlds"
    )
    print(tabulate(summary, headers="keys", tablefmt="psql", showindex=False))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    return results_df, summary


def compare_strategies(
    n_steps=15,
    n_configs=2,
    seeds=(11, 22),
):
    """Benchmark each ArionAgent strategy variant against the same baselines."""
    from arion_strategists.arion_agent import STRATEGY_VARIANTS

    start = time.perf_counter()
    rows: list[dict] = []
    for name, cls in STRATEGY_VARIANTS.items():
        print(f"\n=== Strategy: {name} ({cls.__name__}) ===")
        try:
            results_df, summary = benchmark_std(
                competitors=(cls,),
                n_steps=n_steps,
                n_configs=n_configs,
                seeds=seeds,
            )
        except MemoryError as exc:
            print(f"SKIP {name}: MemoryError during simulation ({exc})")
            continue
        except Exception as exc:
            print(f"SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        for _, srow in summary.iterrows():
            rows.append(
                {
                    "strategy": name,
                    "agent_type": srow["agent_type"],
                    "runs": int(srow["runs"]),
                    "score_mean": float(srow["score_mean"]),
                    "shortfall_mean": float(srow["shortfall_mean"]),
                    "storage_mean": float(srow["storage_mean"]),
                }
            )

    cmp_df = pd.DataFrame(rows)
    arion_only = cmp_df[cmp_df["agent_type"].str.contains("Arion", na=False)]
    if arion_only.empty:
        arion_only = cmp_df

    pivot = (
        arion_only.groupby("strategy", as_index=False)
        .agg(
            score_mean=("score_mean", "mean"),
            shortfall_mean=("shortfall_mean", "mean"),
            storage_mean=("storage_mean", "mean"),
        )
        .sort_values("score_mean", ascending=False)
    )
    print("\n=== Strategy comparison (Arion variants) ===")
    print(tabulate(pivot, headers="keys", tablefmt="psql", showindex=False))
    best = str(pivot.iloc[0]["strategy"]) if not pivot.empty else "hybrid"
    print(f"\nRecommended default strategy: {best}")
    print(f"Total compare time: {humanize_time(time.perf_counter() - start)}")

    out_dir = Path(__file__).resolve().parents[1] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / "strategy_comparison.csv"
    pivot_path = out_dir / "strategy_comparison_summary.csv"
    cmp_df.to_csv(cmp_path, index=False)
    pivot.to_csv(pivot_path, index=False)
    print(f"Saved: {cmp_path}")
    print(f"Saved: {pivot_path}")

    return cmp_df, pivot, best


def smoke_all_strategies(n_steps=5, n_configs=1, seeds=(11,)):
    """Quick smoke test: one short world per strategy variant."""
    from arion_strategists.arion_agent import (
        DEFAULT_STRATEGY,
        ArionAgent,
        STRATEGY_VARIANTS,
    )

    print("=== ArionAgent strategy smoke tests ===")
    print(f"Config: {n_steps} steps, {n_configs} config(s), seeds={seeds}\n")

    # Compile check
    import py_compile
    from pathlib import Path as P

    root = P(__file__).resolve().parents[2]
    for rel in ("arion_strategists/arion_agent.py", "arion_strategists/helpers/runner.py"):
        path = root / rel
        py_compile.compile(str(path), doraise=True)
    print("[PASS] py_compile arion_agent.py + runner.py\n")

    rows: list[dict] = []
    all_ok = True

    for name, cls in STRATEGY_VARIANTS.items():
        label = f"{name} ({cls.__name__})"
        print(f"--- {label} ---")
        t0 = time.perf_counter()
        try:
            results_df, summary = benchmark_std(
                competitors=(cls,),
                n_steps=n_steps,
                n_configs=n_configs,
                seeds=seeds,
            )
            arion = summary[summary["agent_type"].str.contains("Arion", na=False)]
            if arion.empty:
                raise RuntimeError("No ArionAgent rows in benchmark summary")
            row = arion.iloc[0]
            elapsed = time.perf_counter() - t0
            score = float(row["score_mean"])
            shortfall = float(row["shortfall_mean"])
            status = "PASS"
            print(f"[{status}] score_mean={score:.4f} shortfall_mean={shortfall:.1f} "
                  f"({humanize_time(elapsed)})\n")
            rows.append(
                dict(
                    strategy=name,
                    class_name=cls.__name__,
                    status=status,
                    score_mean=score,
                    shortfall_mean=shortfall,
                    elapsed_s=round(elapsed, 1),
                )
            )
        except Exception as exc:
            all_ok = False
            elapsed = time.perf_counter() - t0
            status = "FAIL"
            print(f"[{status}] {type(exc).__name__}: {exc} ({humanize_time(elapsed)})\n")
            rows.append(
                dict(
                    strategy=name,
                    class_name=cls.__name__,
                    status=status,
                    score_mean=float("nan"),
                    shortfall_mean=float("nan"),
                    elapsed_s=round(elapsed, 1),
                    error=str(exc),
                )
            )

    # Default ArionAgent (submission class)
    print(f"--- default ArionAgent (strategy={DEFAULT_STRATEGY}) ---")
    t0 = time.perf_counter()
    try:
        _, summary = benchmark_std(
            competitors=(ArionAgent,),
            n_steps=n_steps,
            n_configs=n_configs,
            seeds=seeds,
        )
        arion = summary[summary["agent_type"].str.contains("ArionAgent", na=False)]
        if arion.empty:
            arion = summary[summary["agent_type"] == "ArionAgent"]
        row = arion.iloc[0]
        elapsed = time.perf_counter() - t0
        print(f"[PASS] DEFAULT score_mean={float(row['score_mean']):.4f} "
              f"({humanize_time(elapsed)})\n")
        rows.append(
            dict(
                strategy=f"default({DEFAULT_STRATEGY})",
                class_name="ArionAgent",
                status="PASS",
                score_mean=float(row["score_mean"]),
                shortfall_mean=float(row["shortfall_mean"]),
                elapsed_s=round(elapsed, 1),
            )
        )
    except Exception as exc:
        all_ok = False
        print(f"[FAIL] DEFAULT {type(exc).__name__}: {exc}\n")
        rows.append(
            dict(
                strategy=f"default({DEFAULT_STRATEGY})",
                class_name="ArionAgent",
                status="FAIL",
                error=str(exc),
            )
        )

    out_dir = Path(__file__).resolve().parents[1] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_df = pd.DataFrame(rows)
    smoke_path = out_dir / "smoke_test_results.csv"
    smoke_df.to_csv(smoke_path, index=False)

    print("=== Smoke test summary ===")
    print(tabulate(smoke_df, headers="keys", tablefmt="psql", showindex=False))
    print(f"\nSaved: {smoke_path}")
    print(f"Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    if not all_ok:
        raise SystemExit(1)
    return smoke_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare-strategies":
        n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        n_configs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        compare_strategies(n_steps=n_steps, n_configs=n_configs)
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke-all":
        n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        n_configs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        smoke_all_strategies(n_steps=n_steps, n_configs=n_configs)
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark-std":
        from arion_strategists.arion_agent import ArionAgent

        benchmark_std(competitors=(ArionAgent,))
    else:
        from arion_strategists.arion_agent import ArionAgent

        run([ArionAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
