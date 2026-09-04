"""Single-runtime command budget. CU estimates are NOT Colab billing telemetry.

The rate must be copied from the active Colab runtime. Only wrapped command time
is measured. Interrupted reservations remain charged conservatively. Use one
Colab runtime per ledger; a local lock cannot coordinate two Drive clients.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def charged(ledger):
    return sum(row.get("estimated_cu", row["reserved_cu"]) for row in ledger["runs"])


def run_command(path, total, rate, cap, label, command):
    for name, value in (("total", total), ("rate", rate), ("cap", cap)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not command:
        raise ValueError("A command after -- is required")
    path = Path(path).resolve()
    lock_name = hashlib.sha256(str(path).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"lpwm-budget-{lock_name}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(path.read_text()) if path.exists() else {
            "schema": 1, "total_cu": total, "runs": [],
            "scope": "estimated wrapped-command time; one Colab runtime only",
        }
        if ledger["total_cu"] != total:
            raise ValueError("Existing ledger has a different total; use the same budget")
        previous = [row for row in ledger["runs"] if row["label"] == label]
        if previous:
            last = previous[-1]
            if last["command"] != command:
                raise ValueError("Label already belongs to another command")
            if last.get("returncode") == 0:
                print(f"Already completed: {label}; no command executed", flush=True)
                return 0
            raise ValueError("Attempt exists; inspect its log and use a new label to retry")
        available = total - charged(ledger)
        if cap > available + 1e-10:
            raise ValueError(f"Insufficient estimate budget: cap {cap:g} > remaining {available:g}")
        row = {"label": label, "command": command, "started": stamp(),
               "rate_cu_hour": rate, "reserved_cu": cap, "status": "running"}
        ledger["runs"].append(row)
        save(path, ledger)  # reserve BEFORE launching; a runtime crash cannot erase the spend
        log_path = path.parent / "command_logs" / f"{len(ledger['runs']):03d}.log"
        log_path.parent.mkdir(exist_ok=True)
        row["log"] = str(log_path)
        save(path, ledger)
        seconds = cap / rate * 3600
        grace = min(5.0, seconds * 0.1)
        started = time.monotonic()
        proc = None
        code, status = 1, "failed"
        old_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(*_):
            raise KeyboardInterrupt

        def stop_child():
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    proc.wait(timeout=max(grace, 0.05))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

        signal.signal(signal.SIGTERM, interrupted)
        print(f"{label}: cap {cap:g} estimated CU, {seconds / 3600:.2f} h; log {log_path}", flush=True)
        try:
            with log_path.open("a", buffering=1) as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                while True:
                    remaining = seconds - grace - (time.monotonic() - started)
                    if remaining <= 0:
                        stop_child()
                        code, status = 124, "timeout"
                        break
                    try:
                        code = proc.wait(timeout=min(remaining, 30))
                        status = "completed" if code == 0 else "failed"
                        break
                    except subprocess.TimeoutExpired:
                        print(f"{label}: {(time.monotonic()-started)/60:.1f} min; log {log_path}", flush=True)
        except KeyboardInterrupt:
            stop_child()
            code, status = 130, "interrupted"
        finally:
            stop_child()
            signal.signal(signal.SIGTERM, old_handler)
            elapsed = time.monotonic() - started
            row.update(status=status, returncode=code, finished=stamp(),
                       elapsed_seconds=elapsed, estimated_cu=elapsed * rate / 3600)
            save(path, ledger)
            print(f"{label}: {status}, charged estimate {row['estimated_cu']:.3f} CU; "
                  f"remaining {total-charged(ledger):.2f} CU", flush=True)
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size-128000))
                tail = handle.read().decode(errors="replace")
            print("\n".join(tail.splitlines()[-16:]), flush=True)
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True)
    p.add_argument("--total-cu", type=float, default=1800)
    p.add_argument("--rate-cu-hour", type=float, required=True)
    p.add_argument("--cap-cu", type=float, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    raise SystemExit(run_command(a.ledger, a.total_cu, a.rate_cu_hour, a.cap_cu, a.label, command))


if __name__ == "__main__":
    main()
