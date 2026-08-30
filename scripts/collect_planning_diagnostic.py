#!/usr/bin/env python3
"""Collect the cheap paired PushT planning diagnostic from persistent files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "success_rate",
    "mean_state_dist",
    "mean_visual_dist",
    "mean_proprio_dist",
    "mean_div_visual_emb",
    "mean_div_proprio_emb",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            entries.append(value)
    return entries


def final_entry(entries: list[dict], prefix: str) -> dict:
    key = f"{prefix}/success_rate"
    matches = [entry for entry in entries if key in entry]
    return matches[-1] if matches else {}


def run_row(run_dir: Path) -> dict:
    meta_path = run_dir / "diagnostic_run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = read_jsonl(run_dir / "logs.json")
    prefix = "oracle_eval" if meta["kind"] == "oracle" else "final_eval"
    final = final_entry(entries, prefix)
    mpc = [entry for entry in entries if "mpc/success_rate" in entry]
    row = {
        **meta,
        "run_dir": str(run_dir),
        "modified_time": max(
            (path.stat().st_mtime for path in (run_dir / "logs.json", meta_path) if path.exists()),
            default=0.0,
        ),
        "complete": bool(final),
        "mpc_steps_completed": len(mpc),
        "mpc_first_state_dist": mpc[0].get("mpc/mean_state_dist") if mpc else None,
        "mpc_last_state_dist": mpc[-1].get("mpc/mean_state_dist") if mpc else None,
        "mpc_best_success_rate": max(
            (entry.get("mpc/success_rate", 0.0) for entry in mpc), default=None
        ),
    }
    for metric in METRICS:
        row[metric] = final.get(f"{prefix}/{metric}")
    row["successes"] = final.get(f"{prefix}/successes")
    return row


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_outputs(root: Path, output_dir: Path) -> list[dict]:
    candidates = [run_row(path.parent) for path in root.rglob("diagnostic_run.json")]
    selected = {}
    for row in candidates:
        key = (
            row["kind"],
            row.get("model"),
            row.get("alpha"),
            row.get("train_seed"),
            row.get("plan_seed"),
        )
        previous = selected.get(key)
        rank = (row["complete"], row["modified_time"])
        if previous is None or rank > (previous["complete"], previous["modified_time"]):
            selected[key] = row
    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            0 if row["kind"] == "oracle" else 1,
            row.get("model", ""),
            float(row.get("alpha") or 0),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_results.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )

    fields = [
        "kind",
        "model",
        "alpha",
        "complete",
        "success_rate",
        "mean_state_dist",
        "mean_visual_dist",
        "mean_proprio_dist",
        "mean_div_visual_emb",
        "mean_div_proprio_emb",
        "mpc_steps_completed",
        "mpc_first_state_dist",
        "mpc_last_state_dist",
        "mpc_best_success_rate",
        "run_dir",
    ]
    with (output_dir / "diagnostic_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Planning diagnostic",
        "",
        "All planner rows use the same saved targets and environment seeds.",
        "",
        "| Kind | Model | alpha | Complete | Success | State dist | MPC first → last state dist |",
        "|---|---|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        transition = (
            f"{fmt(row['mpc_first_state_dist'])} → {fmt(row['mpc_last_state_dist'])}"
            if row["kind"] == "plan"
            else "—"
        )
        lines.append(
            "| {kind} | {model} | {alpha} | {complete} | {success} | {state} | {transition} |".format(
                kind=row["kind"],
                model=row.get("model", ""),
                alpha=fmt(row.get("alpha")),
                complete=fmt(row["complete"]),
                success=fmt(row["success_rate"]),
                state=fmt(row["mean_state_dist"]),
                transition=transition,
            )
        )

    oracle = next((row for row in rows if row["kind"] == "oracle"), None)
    lines.extend(["", "## Gate", ""])
    if oracle is None or not oracle["complete"]:
        lines.append("**BLOCKED:** the ground-truth replay is incomplete.")
    elif oracle["success_rate"] is None or oracle["success_rate"] < 0.99:
        lines.append(
            "**BLOCKED:** ground-truth replay success is below 0.99; do not interpret planner comparisons."
        )
    else:
        lines.append(
            "**PASS:** ground-truth replay confirms that the paired goals, simulator and success evaluator agree."
        )
    (output_dir / "diagnostic_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def register(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": args.kind,
        "model": args.model,
        "alpha": args.alpha,
        "train_seed": args.train_seed,
        "plan_seed": args.plan_seed,
        "n_evals": args.n_evals,
        "goal_H": args.goal_h,
        "max_iter": args.max_iter,
    }
    (run_dir / "diagnostic_run.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def check_oracle(args) -> None:
    entry = final_entry(read_jsonl(Path(args.logs)), "oracle_eval")
    if not entry:
        raise SystemExit(f"Ground-truth replay did not write oracle_eval metrics: {args.logs}")
    success = entry.get("oracle_eval/success_rate")
    print(f"Ground-truth replay success_rate={success}")
    if success is None or success < args.min_success:
        raise SystemExit(
            f"Ground-truth replay gate failed: {success} < {args.min_success}. "
            "Do not run MPC comparisons."
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--run-dir", required=True)
    reg.add_argument("--kind", choices=("oracle", "plan"), required=True)
    reg.add_argument("--model", required=True)
    reg.add_argument("--alpha", type=float)
    reg.add_argument("--train-seed", type=int, required=True)
    reg.add_argument("--plan-seed", type=int, required=True)
    reg.add_argument("--n-evals", type=int, required=True)
    reg.add_argument("--goal-h", type=int, required=True)
    reg.add_argument("--max-iter", type=int, required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--root", required=True)
    collect.add_argument("--output-dir", required=True)

    gate = sub.add_parser("check-oracle")
    gate.add_argument("--logs", required=True)
    gate.add_argument("--min-success", type=float, default=0.99)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "register":
        register(args)
    elif args.command == "check-oracle":
        check_oracle(args)
    else:
        rows = write_outputs(Path(args.root), Path(args.output_dir))
        print(f"Collected {len(rows)} diagnostic runs in {args.output_dir}")


if __name__ == "__main__":
    main()
