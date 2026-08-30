#!/usr/bin/env python3
"""Collect local and W&B metrics for the fair LpWM benchmark.

The benchmark launcher registers every deterministic run in ``manifest.json``.
This helper then joins the resolved Hydra config, per-epoch local metrics, optional
W&B summaries, checkpoint metadata, and persistent planning logs into JSON/CSV and
an aggregate Markdown table.  It deliberately keeps local files authoritative so
an interrupted or offline W&B sync cannot erase a completed experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_METRICS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_z_loss",
    "val_z_loss",
    "train_z_visual_err_rollout",
    "val_z_visual_err_rollout",
    "train_z_visual_err_rollout_1framestart",
    "val_z_visual_err_rollout_1framestart",
    "train_generator_active_fraction",
    "val_generator_active_fraction",
    "train_generator_edge_fraction",
    "val_generator_edge_fraction",
    "train_generator_usage_entropy",
    "val_generator_usage_entropy",
    "train_generator_switch_rate",
    "val_generator_switch_rate",
    "num_parameters",
    "num_trainable_parameters",
    "elapsed_time_sec",
    "max_cuda_memory_mb",
    "_runtime",
]

PLAN_METRICS = [
    "final_eval/success_rate",
    "final_eval/mean_visual_dist",
    "final_eval/mean_proprio_dist",
    "final_eval/mean_div_visual_emb",
    "final_eval/mean_div_proprio_emb",
    "final_eval/successes",
]

CSV_COLUMNS = [
    "benchmark_id",
    "profile",
    "model",
    "architecture",
    "state_sparse",
    "law_sparse",
    "run_name",
    "seed",
    "training_complete",
    "planning_complete",
    "wandb_state",
    "wandb_run_id",
    "wandb_url",
    "link",
    "target_p",
    "predictor",
    "predictor_mode",
    "n_rollout",
    "epochs",
    "batch_size",
    "num_projections",
    "checkpoint_size_mb",
    "epoch",
    "train_loss",
    "val_loss",
    "train_z_loss",
    "val_z_loss",
    "train_z_visual_err_rollout",
    "val_z_visual_err_rollout",
    "train_z_visual_err_rollout_1framestart",
    "val_z_visual_err_rollout_1framestart",
    "val_generator_active_fraction",
    "val_generator_edge_fraction",
    "val_generator_usage_entropy",
    "val_generator_switch_rate",
    "num_parameters",
    "num_trainable_parameters",
    "elapsed_time_sec",
    "max_cuda_memory_mb",
    "planning_success_rate",
    "planning_n_evals",
    "planning_mean_visual_dist",
    "planning_mean_proprio_dist",
    "planning_mean_div_visual_emb",
    "run_dir",
    "planning_dir",
]

AGGREGATE_METRICS = [
    "val_z_loss",
    "val_z_visual_err_rollout",
    "val_z_visual_err_rollout_1framestart",
    "planning_success_rate",
    "elapsed_time_sec",
    "max_cuda_memory_mb",
    "num_parameters",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    tmp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warnings.warn(f"Skipping malformed {path}:{line_number}: {exc}")
    return entries


def latest_epoch(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {}
    return max(
        enumerate(entries),
        key=lambda item: (item[1].get("epoch", -1), item[0]),
    )[1]


def latest_final_plan(entries: list[dict[str, Any]]) -> dict[str, Any]:
    finals = [entry for entry in entries if "final_eval/success_rate" in entry]
    return finals[-1] if finals else {}


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - Hydra installs PyYAML in practice.
        raise RuntimeError("PyYAML is required to read hydra.yaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def fetch_wandb_summary(
    run_id: str | None,
    entity: str,
    project: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    if not run_id:
        return {}, None, None
    try:
        import wandb

        run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
        raw_summary = getattr(run.summary, "_json_dict", dict(run.summary))
        summary = {key: jsonable(raw_summary[key]) for key in raw_summary}
        url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        return summary, getattr(run, "state", None), url
    except Exception as exc:  # W&B must never block recovery of persisted local results.
        warnings.warn(f"Could not fetch W&B run {run_id}: {exc}")
        return {}, None, f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def parse_bool(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def register(args: argparse.Namespace) -> None:
    benchmark_dir: Path = args.benchmark_dir
    manifest_path = benchmark_dir / "manifest.json"
    payload = load_json(
        manifest_path,
        {"benchmark_id": args.benchmark_id, "created_at": utc_now(), "runs": []},
    )
    entry = {
        "profile": args.profile,
        "model": args.model,
        "architecture": args.architecture,
        "state_sparse": parse_bool(args.state_sparse),
        "law_sparse": parse_bool(args.law_sparse),
        "run_name": args.run_name,
        "seed": args.seed,
        "planning_dir": str(args.planning_dir),
        "updated_at": utc_now(),
    }
    runs = [run for run in payload.get("runs", []) if run.get("run_name") != args.run_name]
    runs.append(entry)
    payload["benchmark_id"] = args.benchmark_id
    payload["updated_at"] = utc_now()
    payload["runs"] = sorted(runs, key=lambda run: (run["profile"], run["model"], run["seed"]))
    write_json_atomic(manifest_path, payload)
    print(f"Registered {args.run_name} in {manifest_path}")


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 15:
        return table[14]
    if df < 20:
        return table[15]
    if df < 30:
        return table[20]
    return 1.96


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95": None}
    std = statistics.stdev(values)
    ci95 = t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "std": std, "ci95": ci95}


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["profile"], row["model"]), []).append(row)

    aggregates = []
    for (profile, model), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "profile": profile,
            "model": model,
            "architecture": group[0]["architecture"],
            "state_sparse": group[0]["state_sparse"],
            "law_sparse": group[0]["law_sparse"],
            "seeds_registered": len(group),
            "training_complete": sum(bool(row["training_complete"]) for row in group),
            "planning_complete": sum(bool(row["planning_complete"]) for row in group),
        }
        for metric in AGGREGATE_METRICS:
            values = [value for row in group if (value := as_float(row.get(metric))) is not None]
            stats = summarize(values)
            for stat_name, stat_value in stats.items():
                aggregate[f"{metric}_{stat_name}"] = stat_value
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_metric(mean: Any, ci95: Any, percent: bool = False) -> str:
    mean_f = as_float(mean)
    if mean_f is None:
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    ci_f = as_float(ci95)
    if ci_f is None:
        return f"{mean_f * scale:.3f}{suffix}"
    return f"{mean_f * scale:.3f} ± {ci_f * scale:.3f}{suffix}"


def write_markdown(path: Path, aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# LpWM fair benchmark",
        "",
        "Mean ± 95% Student-t confidence interval across completed training seeds. ",
        "A missing interval means only one seed is currently available.",
        "",
        "| Profile | Model | Seeds (train/plan) | val z-loss | rollout error | planning success | Params |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        seeds = f"{row['training_complete']}/{row['planning_complete']}"
        lines.append(
            "| {profile} | {model} | {seeds} | {z} | {rollout} | {success} | {params} |".format(
                profile=row["profile"],
                model=row["model"],
                seeds=seeds,
                z=format_metric(row.get("val_z_loss_mean"), row.get("val_z_loss_ci95")),
                rollout=format_metric(
                    row.get("val_z_visual_err_rollout_mean"),
                    row.get("val_z_visual_err_rollout_ci95"),
                ),
                success=format_metric(
                    row.get("planning_success_rate_mean"),
                    row.get("planning_success_rate_ci95"),
                    percent=True,
                ),
                params=format_metric(row.get("num_parameters_mean"), None),
            )
        )
    lines.extend(
        [
            "",
            "Raw latent errors are comparable only within architecture-matched blocks. ",
            "Use planning success for comparisons between CLS-based LpWM/LeWM and patch-field models.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(args: argparse.Namespace) -> None:
    benchmark_dir: Path = args.benchmark_dir
    manifest_path = benchmark_dir / "manifest.json"
    manifest = load_json(manifest_path, None)
    if not manifest:
        raise FileNotFoundError(f"No benchmark manifest found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for metadata in manifest.get("runs", []):
        run_name = metadata["run_name"]
        run_dir = args.ckpt_base / "outputs" / run_name
        planning_dir = Path(metadata["planning_dir"])
        cfg = load_yaml(run_dir / "hydra.yaml")
        local_train = latest_epoch(read_jsonl(run_dir / "metrics.jsonl"))
        run_id = cfg.get("wandb_run_id")

        wandb_summary: dict[str, Any] = {}
        wandb_state = None
        wandb_url = None
        if args.use_wandb:
            wandb_summary, wandb_state, wandb_url = fetch_wandb_summary(
                run_id, args.wandb_entity, args.wandb_project
            )
        elif run_id:
            wandb_url = f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{run_id}"

        train_summary = {key: wandb_summary[key] for key in TRAIN_METRICS if key in wandb_summary}
        train_summary.update(local_train)
        plan_summary = latest_final_plan(read_jsonl(planning_dir / "logs.json"))

        checkpoint_path = run_dir / "checkpoints" / "model_latest.pth"
        checkpoint_size_mb = (
            checkpoint_path.stat().st_size / (1024**2) if checkpoint_path.exists() else None
        )
        successes = plan_summary.get("final_eval/successes")
        planning_n_evals = len(successes) if isinstance(successes, list) else None

        row = {
            "benchmark_id": manifest["benchmark_id"],
            **{key: metadata.get(key) for key in (
                "profile", "model", "architecture", "state_sparse", "law_sparse", "run_name", "seed"
            )},
            "training_complete": checkpoint_path.exists() and bool(train_summary),
            "planning_complete": "final_eval/success_rate" in plan_summary,
            "wandb_state": wandb_state,
            "wandb_run_id": run_id,
            "wandb_url": wandb_url,
            "link": nested_get(cfg, "link", "kind"),
            "target_p": nested_get(cfg, "regularizer", "target_p", default=cfg.get("target_p")),
            "predictor": nested_get(cfg, "predictor", "_target_"),
            "predictor_mode": nested_get(cfg, "predictor", "mode"),
            "n_rollout": nested_get(cfg, "env", "dataset", "n_rollout"),
            "epochs": nested_get(cfg, "training", "epochs"),
            "batch_size": nested_get(cfg, "training", "batch_size"),
            "num_projections": nested_get(cfg, "regularizer", "num_projections"),
            "checkpoint_size_mb": checkpoint_size_mb,
            **{key: train_summary.get(key) for key in TRAIN_METRICS},
            "planning_success_rate": plan_summary.get("final_eval/success_rate"),
            "planning_n_evals": planning_n_evals,
            "planning_mean_visual_dist": plan_summary.get("final_eval/mean_visual_dist"),
            "planning_mean_proprio_dist": plan_summary.get("final_eval/mean_proprio_dist"),
            "planning_mean_div_visual_emb": plan_summary.get("final_eval/mean_div_visual_emb"),
            "run_dir": str(run_dir),
            "planning_dir": str(planning_dir),
        }
        rows.append(row)
        records.append(
            {
                "metadata": metadata,
                "config": cfg,
                "train_summary": train_summary,
                "planning_summary": plan_summary,
                "wandb_state": wandb_state,
                "wandb_url": wandb_url,
                "checkpoint_size_mb": checkpoint_size_mb,
            }
        )

    rows.sort(key=lambda row: (row["profile"], row["model"], row["seed"]))
    aggregates = aggregate_rows(rows)
    aggregate_columns = sorted({key for row in aggregates for key in row})

    detailed_json = benchmark_dir / "benchmark_results.json"
    detailed_csv = benchmark_dir / "benchmark_results.csv"
    aggregate_json = benchmark_dir / "benchmark_aggregate.json"
    aggregate_csv = benchmark_dir / "benchmark_aggregate.csv"
    markdown_path = benchmark_dir / "benchmark_results.md"

    write_json_atomic(
        detailed_json,
        {
            "benchmark_id": manifest["benchmark_id"],
            "generated_at": utc_now(),
            "runs": records,
        },
    )
    write_csv(detailed_csv, rows, CSV_COLUMNS)
    write_json_atomic(aggregate_json, {"generated_at": utc_now(), "models": aggregates})
    write_csv(aggregate_csv, aggregates, aggregate_columns)
    write_markdown(markdown_path, aggregates)

    print(f"Detailed results: {detailed_json}")
    print(f"Seed-level table: {detailed_csv}")
    print(f"Aggregate table: {aggregate_csv}")
    print(f"Readable summary: {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="register one deterministic run")
    register_parser.add_argument("--benchmark-dir", type=Path, required=True)
    register_parser.add_argument("--benchmark-id", required=True)
    register_parser.add_argument("--profile", choices=("smoke", "screen", "full"), required=True)
    register_parser.add_argument("--model", required=True)
    register_parser.add_argument("--architecture", required=True)
    register_parser.add_argument("--state-sparse", choices=("true", "false", "na"), required=True)
    register_parser.add_argument("--law-sparse", choices=("true", "false", "na"), required=True)
    register_parser.add_argument("--run-name", required=True)
    register_parser.add_argument("--seed", type=int, required=True)
    register_parser.add_argument("--planning-dir", type=Path, required=True)
    register_parser.set_defaults(func=register)

    collect_parser = subparsers.add_parser("collect", help="collect all registered results")
    collect_parser.add_argument("--benchmark-dir", type=Path, required=True)
    collect_parser.add_argument("--ckpt-base", type=Path, required=True)
    collect_parser.add_argument("--wandb-entity", default="twojtys137-tw")
    collect_parser.add_argument("--wandb-project", default="lpwm-sparse-generator")
    collect_parser.add_argument("--use-wandb", action="store_true")
    collect_parser.set_defaults(func=collect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
