"""Pinned upstream PushT baselines. No modified local model implementation is used.

This launcher is intentionally separate from the sparse-generator benchmark.
Installation and download are explicit commands; `run` prints its manifest unless
--execute is supplied. Colab wraps execution with experiments.budget.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import socket
import subprocess
import sys


PINS = {
    "lewm": ("lucas-maes/le-wm", "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"),
    "lpwm": ("YilunKuang/lpworldmodel", "bdd812d9432cccda8c350086006401b436f91982"),
    "dinowm": ("gaoyuezhou/dino_wm", "0a9492fa12044b852ae9e001cc74604b79c8bb0c"),
}
SWM_PIN = "abdced49809d5eae38e24b27dc7b635c502c4812"
HF_MODEL_PIN = "3970e07a65a74097a492f8954b073ec984afb09b"
HF_DATA_PIN = "655cd446b9929369d7d406001da85c15d1457850"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
UV_VERSION = "0.11.33"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def location(args):
    return Path(args.work).resolve() / args.method


def python_for(args):
    return location(args) / ".venv/bin/python"


def checkout(args):
    repo, pin = PINS[args.method]
    target = location(args) / "repo"
    if not target.exists():
        target.mkdir(parents=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin",
                        f"https://github.com/{repo}.git"], check=True)
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain",
                                     "--untracked-files=all"], text=True)
    changes = [line for line in dirty.splitlines() if not (
        line.startswith("?? ") and "__pycache__" in Path(line[3:]).parts
        and line.endswith(".pyc"))]
    if changes:
        raise RuntimeError(f"Keep local changes first: {target}")
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", pin], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", pin], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin:
        raise RuntimeError("Upstream pin mismatch")
    print(f"{repo}@{actual}", flush=True)


def logged_install_command(command, log_path, env=None):
    """Stream installer failures into both notebook output and a durable log."""
    log_path = Path(log_path)
    tail = deque(maxlen=35)
    with log_path.open("a") as log:
        log.write("\nCOMMAND: " + repr(list(map(str, command))) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            tail.append(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"Installation command failed (exit {status}). Full log: {log_path}\n"
                           + "".join(tail))


def isolated_python_ready(env_dir):
    env_dir = Path(env_dir)
    config = env_dir / "pyvenv.cfg"
    python = env_dir / "bin/python"
    if not config.is_file() or not python.exists():
        return False
    fields = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
    fields = {key.strip(): value.strip().lower() for key, value in fields.items()}
    if fields.get("include-system-site-packages") != "false":
        return False
    try:
        result = subprocess.run([str(python), "-I", "-c",
                                 "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                                capture_output=True, text=True)
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == PYTHON_VERSION


def ensure_python_environment(args, log_path):
    env_dir = location(args) / ".venv"
    if isolated_python_ready(env_dir):
        return
    uv = [shutil.which("uv")] if shutil.which("uv") else [sys.executable, "-m", "uv"]
    if len(uv) > 1:
        logged_install_command([sys.executable, "-m", "pip", "install", f"uv=={UV_VERSION}"], log_path)
    if env_dir.exists():
        if not (env_dir / "pyvenv.cfg").is_file():
            raise RuntimeError(f"Expected a virtual environment at {env_dir}; choose a new --work directory")
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup = env_dir.with_name(f".venv.previous-{suffix}")
        env_dir.rename(backup)
        print(f"Preserved incompatible environment: {backup}", flush=True)
    env = os.environ.copy()
    env["UV_PYTHON_INSTALL_DIR"] = str(Path(args.work).resolve() / "python")
    logged_install_command([*uv, "venv", "--python", PYTHON_VERSION, "--seed", str(env_dir)],
                           log_path, env=env)
    if not isolated_python_ready(env_dir):
        raise RuntimeError("Failed to create an isolated Python 3.12 environment")


def install(args):
    # Python 3.13 cannot use the NumPy1 stack. Never inherit Colab site-packages.
    out = Path(args.output) / "environment"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.method}-install.log"
    with log_path.open("a") as log:
        log.write(f"\nINSTALL {datetime.now(timezone.utc).isoformat()} host={sys.version}\n")
    print(f"Installation log: {log_path}", flush=True)
    ensure_python_environment(args, log_path)
    python = str(python_for(args))
    logged_install_command([python, "-m", "pip", "install", "--only-binary=:all:",
                            f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
                            "--index-url", TORCH_INDEX], log_path)
    common = ["hydra-core==1.3.2", "hydra-submitit-launcher==1.2.0", "einops",
              "wandb<1", "scikit-learn", "h5py", "hdf5plugin", "zstandard",
              "huggingface-hub<1", "pygame", "shapely", "matplotlib",
              "imageio[ffmpeg]", "opencv-python-headless<4.12", "numpy==1.26.4",
              f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    if args.method == "lewm":
        packages = common + [
            f"stable-worldmodel @ git+https://github.com/galilai-group/stable-worldmodel.git@{SWM_PIN}",
            "stable-pretraining==0.1.8", "transformers==4.57.6", "pymunk==7.0.1",
            "lightning", "decord", "datasets", "loguru",
        ]
    else:
        packages = common + ["accelerate==0.26.1", "gym==0.26.2", "pymunk==6.11.1",
                             "moviepy<2", "decord", "scikit-image", "tensorboardX",
                             "submitit", "psutil"]
    logged_install_command([python, "-m", "pip", "install", *packages], log_path)
    logged_install_command([python, "-m", "pip", "check"], log_path)
    modules = ["numpy", "torch", "torchvision", "decord", "pymunk", "hydra", "h5py"]
    if args.method == "lewm":
        modules += ["transformers", "stable_pretraining", "stable_worldmodel"]
    smoke = ("import importlib, json, sys; "
             f"[importlib.import_module(name) for name in {modules!r}]; "
             "import torch; print(json.dumps({'python':sys.version,'torch':torch.__version__,"
             "'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available()})); "
             "assert torch.version.cuda is not None, 'CUDA-enabled PyTorch required'")
    logged_install_command([python, "-c", smoke], log_path)
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    (out / f"{args.method}-freeze.txt").write_text(freeze)
    print(f"Dependency snapshot: {out / (args.method+'-freeze.txt')}")


def setup_lewm_paths(args):
    # Symlinks live on Colab's local filesystem: Drive FUSE rejects symlink creation.
    home = location(args) / "runtime-home"
    local_data = Path(args.data).resolve()
    persistent = Path(args.output).resolve() / "lewm/checkpoints"
    home.mkdir(parents=True, exist_ok=True)
    local_data.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(parents=True, exist_ok=True)
    datasets = home / "datasets"
    if datasets.is_symlink() and datasets.resolve() != local_data:
        raise RuntimeError(f"Dataset link points elsewhere: {datasets}")
    if not datasets.exists() and not datasets.is_symlink():
        datasets.symlink_to(local_data, target_is_directory=True)
    if datasets.resolve() != local_data:
        raise RuntimeError(f"Use --data {datasets.resolve()} or a new --output")
    checkpoints = home / "checkpoints"
    if not checkpoints.exists() and not checkpoints.is_symlink():
        checkpoints.symlink_to(persistent, target_is_directory=True)
    if checkpoints.resolve() != persistent:
        raise RuntimeError(f"Checkpoint link points elsewhere: {checkpoints}")
    return home, local_data


def prepare_data(args):
    if args.method != "lewm":
        base = Path(args.data) / "pusht_noise"
        if not (base / "train/seq_lengths.pkl").exists():
            downloader = Path(__file__).with_name("download_lpwmdatasets.py")
            subprocess.run([sys.executable, str(downloader), "--dataset", "pusht_noise",
                            "--output-dir", args.data], check=True)
        audit = {"source": "released OSF pusht_noise; observed counts, no assumed paper size"}
        for split in ("train", "val"):
            directory = base / split
            with (directory / "seq_lengths.pkl").open("rb") as handle:
                lengths = pickle.load(handle)  # original metadata from the official archive
            for required in ("states.pth", "rel_actions.pth", "velocities.pth"):
                if not (directory / required).is_file():
                    raise FileNotFoundError(directory / required)
            videos = [directory / "obses" / f"episode_{i:03d}.mp4" for i in range(len(lengths))]
            missing = [str(path) for path in videos if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete {split} videos: {missing[:5]}")
            audit[split] = {"episodes": len(lengths), "frames": int(sum(lengths)),
                            "verified_video_files": len(videos)}
        write_json(Path(args.output) / args.method / "data_audit.json", audit)
        print(json.dumps(audit, indent=2))
        return
    from huggingface_hub import hf_hub_download
    import h5py
    import hdf5plugin  # noqa: F401
    import zstandard

    _, target = setup_lewm_paths(args)
    h5 = target / "pusht_expert_train.h5"
    if not h5.exists():
        archive = Path(hf_hub_download(
            "quentinll/lewm-pusht", "pusht_expert_train.h5.zst", repo_type="dataset",
            revision=HF_DATA_PIN, cache_dir=str(target / "hf-cache")))
        temporary = h5.with_suffix(".h5.partial")
        # The official download is a zstd-compressed HDF5 file, NOT a tarball.
        with archive.open("rb") as source, temporary.open("wb") as dest:
            zstandard.ZstdDecompressor().copy_stream(source, dest)
        temporary.replace(h5)
    with h5py.File(h5, "r") as handle:
        required = {"pixels", "action", "proprio", "state", "ep_len", "ep_offset"}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"Unexpected official data schema: missing {sorted(missing)}")
        lengths = handle["ep_len"][:]
        audit = {"dataset": "quentinll/lewm-pusht", "revision": HF_DATA_PIN,
                 "file": str(h5), "episodes": int(len(lengths)),
                 "frames": int(lengths.sum()), "mean_episode_length": float(lengths.mean()),
                 "action_shape": list(handle["action"].shape),
                 "pixels_shape": list(handle["pixels"].shape)}
    audit["paper_reported_episodes"] = 20000
    audit["episode_count_matches_paper"] = audit["episodes"] == 20000
    write_json(Path(args.output) / "lewm/data_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["episode_count_matches_paper"]:
        print("The released artifact's episode count differs from the paper's 20,000. "
              "Report the observed count; this run reproduces the released artifact.")


def prepare_checkpoint(args):
    if args.method != "lewm":
        raise ValueError("Published checkpoint importer currently supports native LeWM only")
    sys.dont_write_bytecode = True
    import torch
    import hydra
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    home, _ = setup_lewm_paths(args)
    repo = location(args) / "repo"
    sys.path.insert(0, str(repo))
    cfg_file = hf_hub_download("quentinll/lewm-pusht", "config.json", revision=HF_MODEL_PIN)
    weight_file = hf_hub_download("quentinll/lewm-pusht", "weights.pt", revision=HF_MODEL_PIN)
    cfg = json.loads(Path(cfg_file).read_text())
    targets = {"": "jepa.JEPA", "predictor": "module.ARPredictor",
               "action_encoder": "module.Embedder", "projector": "module.MLP",
               "pred_proj": "module.MLP"}
    for key, value in targets.items():
        (cfg if not key else cfg[key])["_target_"] = value
    model = hydra.utils.instantiate(OmegaConf.create(cfg))
    weights = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.inference_mode():
        batch = {"pixels": torch.zeros(1, 1, 3, 224, 224), "action": torch.zeros(1, 1, 10)}
        emb = model.encode(batch)["emb"]
        assert emb.shape == (1, 1, 192) and torch.isfinite(emb).all()
    dest = home / "checkpoints/published_lewm"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight_file, dest / "weights.pt")
    write_json(dest / "config.json", cfg)
    write_json(dest / "source.json", {"repository": "quentinll/lewm-pusht",
               "revision": HF_MODEL_PIN, "strict_load": True,
               "implementation": f"{PINS['lewm'][0]}@{PINS['lewm'][1]}",
               "changes": "Hydra import targets only; tensor values unchanged"})
    print(f"Published native checkpoint passes strict load and CPU encode: {dest}")


def command_manifest(args):
    repo = location(args) / "repo"
    output = Path(args.output).resolve()
    python = str(python_for(args))
    env = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "MUJOCO_GL": "egl",
           "OMP_NUM_THREADS": str(args.workers), "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
           "PYTHONDONTWRITEBYTECODE": "1", "WANDB_DIR": str(output / "wandb")}
    run_name = args.run_name or f"{args.method}_native_seed{args.seed}_e{args.epochs}"
    if "/" in run_name or "\\" in run_name or run_name in {"", ".", ".."}:
        raise ValueError("run-name must be one directory name")
    if args.method == "lewm":
        home = location(args) / "runtime-home"
        persistent = output / "lewm"
        env["STABLEWM_HOME"] = str(home)
        if args.task == "train":
            cmd = [python, "train.py", "data=pusht", f"trainer.max_epochs={args.epochs}",
                   "loader.batch_size=128", f"num_workers={args.workers}", f"seed={args.seed}",
                   "data.dataset.name=pusht_expert_train.h5", f"output_model_name={run_name}",
                   f"subdir={run_name}", f"hydra.run.dir={persistent / 'hydra' / run_name}",
                   f"+trainer.default_root_dir={persistent / 'lightning' / run_name}"]
            if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_ENTITY"):
                cmd += ["wandb.enabled=true", f"wandb.config.entity={os.environ['WANDB_ENTITY']}",
                        f"wandb.config.project={os.environ.get('WANDB_PROJECT', 'lpwm-native')}"]
            train_dir = persistent / "checkpoints" / run_name
        else:
            policy = "published_lewm/weights.pt" if args.task == "published-eval" else (
                f"{run_name}/weights_epoch_{args.epoch}.pt")
            cmd = [python, "eval.py", "--config-name=pusht", f"policy={policy}",
                   f"eval.num_eval={args.n_evals}", "seed=42", "eval.goal_offset_steps=25",
                   "eval.eval_budget=50",
                   f"hydra.run.dir={persistent / 'hydra' / (run_name+'_'+args.task+'_e'+str(args.epoch))}",
                   f"output.filename={persistent / 'results' / run_name / (args.task+'_e'+str(args.epoch)+'_seed42_n'+str(args.n_evals)+'.txt')}"]
            train_dir = None
    else:
        env.update(DATASET_DIR=str(Path(args.data).resolve()), WORLD_SIZE="1", RANK="0",
                   LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
        # Native legacy checkpoints contain serialized module objects. Scope this
        # compatibility flag to audited official-repository subprocesses only.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        home = output / args.method
        train_dir = home / "outputs" / run_name if args.task == "train" else None
        if args.task == "published-eval":
            raise ValueError("Use LeWM for automated published-checkpoint evaluation first")
        if args.task == "train" and args.method == "lpwm":
            cmd = [python, "train.py", "--config-name=train_rdmreg.yaml", "env=pusht",
                   "frameskip=5", "num_hist=3", "encoder=vit_scratch", "link=reprelu",
                   "target_p=1", "regularizer=rdmreg", "mu=0", "agg=b", "predictor=ar_adaln",
                   "encoder.proj_dim=384", "action_emb_dim=384", "mup=true",
                   "training.mup_lr=1e-4", "reg_weight=0.5", "training.batch_size=64",
                   f"training.epochs={args.epochs}", f"training.seed={args.seed}",
                   "training.save_every_x_epoch=1", "env.dataset.n_rollout=null",
                   f"env.num_workers={args.workers}", f"ckpt_base_path={home}",
                   f"hydra.run.dir={train_dir}", "hydra.job.chdir=true"]
            if os.environ.get("WANDB_PROJECT"):
                cmd.append(f"wandb_project={os.environ['WANDB_PROJECT']}")
        elif args.task == "train":
            raise ValueError("DINO-WM training deferred: use its official checkpoint/protocol; see docs")
        else:
            config = "plan_lewm.yaml" if args.method == "lpwm" else "plan_pusht.yaml"
            cmd = [python, "plan.py", f"--config-name={config}", f"ckpt_base_path={home}",
                   f"model_name={run_name}", f"model_epoch={args.epoch}", f"n_evals={args.n_evals}",
                   "seed=99", "goal_H=5", f"planner.max_iter={args.max_iter}",
                   "planner.sub_planner.num_samples=300", "planner.sub_planner.topk=30",
                   "planner.sub_planner.opt_steps=30",
                   f"hydra.run.dir={home / 'planning' / run_name / ('epoch'+str(args.epoch))}",
                   "hydra.job.chdir=true"]
    return {"method": args.method, "upstream": PINS[args.method][0], "commit": PINS[args.method][1],
            "cwd": str(repo), "command": cmd, "environment": env,
            "train_dir": str(train_dir) if train_dir else None,
            "epoch_note": "LeWM PushT paper:10; LpWM reference:2,10 is duration ablation",
            "protocol_note": "Native LeWM50 raw actions; legacy LpWM max_iter*5*5 raw actions"}


def run(args):
    manifest = command_manifest(args)
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.execute:
        return
    if args.method == "lewm":
        setup_lewm_paths(args)
        if not (Path(args.data) / "pusht_expert_train.h5").is_file():
            raise FileNotFoundError("Run prepare-data first")
        if args.task == "train":
            gate = Path(args.output) / "lewm/published_eval_summary.json"
            if not gate.is_file() or json.loads(gate.read_text())["success_percent"] <= 0:
                raise RuntimeError("Evaluate the published checkpoint first; its native success must exceed zero")
    else:
        if not (Path(args.data) / "pusht_noise/train").is_dir():
            raise FileNotFoundError("Expected full OSF pusht_noise/train under --data")
        audit = Path(args.output) / args.method / "data_audit.json"
        if args.task == "train" and not audit.is_file():
            raise RuntimeError("Run prepare-data to inventory the actual legacy data before training")
    repo = Path(manifest["cwd"])
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if pin != manifest["commit"]:
        raise RuntimeError("Upstream checkout changed; refusing untracked recipe")
    if manifest["train_dir"] and Path(manifest["train_dir"]).exists():
        raise RuntimeError("Fresh runs only: choose a new run-name; native resume is not assumed safe")
    entry = Path(args.output) / "manifests" / f"{args.method}_{args.task}_{args.run_name or 'default'}_e{args.epoch}.json"
    write_json(entry, manifest)
    env = os.environ.copy()
    env.update(manifest["environment"])
    Path(env["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    if args.method != "lewm":
        with socket.socket() as sock:
            sock.bind(("", 0))
            env["MASTER_PORT"] = str(sock.getsockname()[1])
    subprocess.run(manifest["command"], cwd=repo, env=env, check=True)
    if args.method == "lewm" and args.task == "published-eval":
        result_file = Path(next(x.split("=", 1)[1] for x in manifest["command"] if x.startswith("output.filename=")))
        found = re.findall(r"['\"]success_rate['\"]\s*:\s*([0-9.eE+-]+)", result_file.read_text())
        if not found:
            raise RuntimeError(f"Native success metric is missing: {result_file}")
        write_json(Path(args.output) / "lewm/published_eval_summary.json",
                   {"success_percent": float(found[-1]), "n_evals": args.n_evals,
                    "seed": 42, "source": str(result_file), "model_revision": HF_MODEL_PIN,
                    "repo_commit": PINS["lewm"][1], "swm_commit": SWM_PIN})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["checkout", "install", "prepare-data", "prepare-checkpoint", "run"])
    parser.add_argument("--method", choices=PINS, default="lewm")
    parser.add_argument("--work", default="/content/native-worldmodels")
    parser.add_argument("--data", default="/content/native-wm-data")
    parser.add_argument("--output", default="/content/drive/MyDrive/lpwm-native")
    parser.add_argument("--task", choices=["train", "eval", "published-eval"], default="published-eval")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--n-evals", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--run-name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.epoch, args.n_evals, args.workers, args.max_iter) <= 0:
        parser.error("epochs, episode count, workers and max-iter must be positive")
    {"checkout": checkout, "install": install, "prepare-data": prepare_data,
     "prepare-checkpoint": prepare_checkpoint, "run": run}[args.action](args)


if __name__ == "__main__":
    main()
