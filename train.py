import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from metric_logging import append_metrics_jsonl, metric_scalar
from utils import (
    slice_trajdict_with_t,
    cfg_to_dict,
    load_trusted_checkpoint,
    seed,
    sample_tensors,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def training_epoch_range(completed_epochs, epochs, mode="additional"):
    """Legacy epochs are additional; mode='total' sets the final epoch number."""
    if mode not in {"additional", "total"}:
        raise ValueError("training.epochs_mode must be 'additional' or 'total'")
    if epochs < 0:
        raise ValueError("training.epochs must be nonnegative")
    final_epoch = completed_epochs + epochs if mode == "additional" else epochs
    return range(completed_epochs + 1, final_epoch + 1)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class Trainer:
    def __init__(self, cfg):
        self.run_started_at = time.time()
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            _proj = self.cfg.get("wandb_project", None) or f"InfoJEPA_train_{self.cfg.env.name}"
            _entity = self.cfg.get("wandb_entity", None) or os.environ.get("WANDB_ENTITY")
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                _proj = f"{_proj}_debug"
            self.wandb_run = wandb.init(
                entity=_entity,
                project=_proj,
                config=wandb_dict,
                id=wandb_run_id,
                resume="allow",
            )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.link = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        # Keep module objects for the existing planning loader, but optimizer
        # state must be rebound to the parameters used by the resumed model.
        self._keys_to_save = [
            "epoch", "encoder", "predictor", "decoder", "action_encoder",
            "proprio_encoder", "link",
        ]
        self._resume_checkpoint = None
        self.resume_metadata = {"training_state_complete": True, "partial_resume_reasons": []}

        self.init_models()
        self.init_optimizers()
        self.restore_training_state()
        # Do not overwrite the checkpoint's planning configuration when resume
        # validation fails (for example for an incomplete legacy checkpoint).
        if self.accelerator.is_main_process:
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        rng_states = [capture_rng_state()]
        if self.accelerator.num_processes > 1:
            rng_states = [None] * self.accelerator.num_processes
            dist.all_gather_object(rng_states, capture_rng_state())
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if self.__dict__[k] is None:
                    continue
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            ckpt["checkpoint_version"] = 2
            ckpt["optimizer_state_dicts"] = {
                name: getattr(self, name).state_dict() for name in self.optimizer_names()
            }
            ckpt["regularizer_state_dict"] = (
                self.regularizer.state_dict() if self.regularizer is not None else None
            )
            scaler = getattr(self.accelerator, "scaler", None)
            ckpt["grad_scaler_state_dict"] = scaler.state_dict() if scaler is not None else None
            ckpt["rng_states"] = rng_states
            ckpt["resume_metadata"] = dict(self.resume_metadata)
            ckpt["training_epochs_mode"] = self.cfg.training.get("epochs_mode", "additional")
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = load_trusted_checkpoint(filename, map_location="cpu")
        self._resume_checkpoint = ckpt
        for name in self._keys_to_save:
            if name in ckpt:
                setattr(self, name, ckpt[name])

    def optimizer_names(self):
        names = []
        if self.train_encoder:
            names.append("encoder_optimizer")
        if self.cfg.has_predictor and self.train_predictor:
            names.extend(["predictor_optimizer", "action_encoder_optimizer"])
        if self.cfg.has_decoder and self.train_decoder:
            names.append("decoder_optimizer")
        return names

    def restore_training_state(self):
        ckpt = self._resume_checkpoint
        if ckpt is None:
            return
        states = dict(ckpt.get("optimizer_state_dicts", {}))
        # Older trusted checkpoints stored optimizer objects. Extract state only;
        # their serialized parameter references must never become live optimizers.
        for name in self.optimizer_names():
            if name not in states and ckpt.get(name) is not None:
                states[name] = ckpt[name].state_dict()
        missing = [name for name in self.optimizer_names() if name not in states]
        reasons = [f"missing optimizer state: {name}" for name in missing]
        required_modules = ["encoder", "action_encoder", "proprio_encoder"]
        if self.cfg.has_predictor:
            required_modules.append("predictor")
        if self.cfg.has_decoder:
            required_modules.append("decoder")
        reasons.extend(f"missing module: {name}" for name in required_modules
                       if ckpt.get(name) is None)
        rng_states = ckpt.get("rng_states")
        if not rng_states or len(rng_states) != self.accelerator.num_processes:
            reasons.append("missing RNG state or changed number of processes")
            rng_state = None
        else:
            rng_state = rng_states[self.accelerator.process_index]
            if len(rng_state.get("cuda", [])) != torch.cuda.device_count():
                reasons.append("changed number of CUDA devices")
                rng_state = None
        scaler = getattr(self.accelerator, "scaler", None)
        scaler_state = ckpt.get("grad_scaler_state_dict")
        if (scaler is None) != (scaler_state is None):
            reasons.append("missing or incompatible mixed-precision scaler state")
        prior = ckpt.get("resume_metadata", {})
        reasons = list(dict.fromkeys(prior.get("partial_resume_reasons", []) + reasons))
        if reasons and not self.cfg.training.get("allow_partial_resume", False):
            raise RuntimeError(
                "Complete checkpoint-state resume is unavailable: " + "; ".join(reasons) +
                ". Start a fresh run, or explicitly set +training.allow_partial_resume=true "
                "to accept and record a partial resume."
            )
        if reasons:
            log.warning("PARTIAL TRAINING RESUME: %s", "; ".join(reasons))
        for name in self.optimizer_names():
            if name in states:
                getattr(self, name).load_state_dict(states[name])
        if self.regularizer is not None and ckpt.get("regularizer_state_dict") is not None:
            self.regularizer.load_state_dict(ckpt["regularizer_state_dict"])
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        self.resume_metadata = {
            "training_state_complete": not reasons,
            "partial_resume_reasons": reasons,
            "resumed_from_epoch": self.epoch,
            "scope": "Epoch boundary; unchanged model, data, ordering and training configuration required",
        }
        # Initialization, accelerator preparation and optimizer construction can
        # consume randomness. Restore only after all of them have finished.
        if rng_state is not None:
            restore_rng_state(rng_state)
        self._resume_checkpoint = None

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=self.cfg.proprio_emb_dim,
            )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.datasets["train"].action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        action_conditioning = self.cfg.get("action_conditioning", "concat")
        if self.cfg.has_predictor:
            if self.predictor is None:
                if action_conditioning == "adaln":
                    assert action_emb_dim == self.encoder.emb_dim, (
                        f"adaln requires action_emb_dim ({action_emb_dim}) == "
                        f"encoder.emb_dim ({self.encoder.emb_dim})"
                    )
                    self.predictor = hydra.utils.instantiate(
                        self.cfg.predictor,
                        num_frames=self.cfg.num_hist,
                        num_patches=self.encoder.num_patches,
                        input_dim=self.encoder.emb_dim,
                        hidden_dim=self.encoder.emb_dim,
                        output_dim=self.encoder.emb_dim,
                    )
                else:
                    self.predictor = hydra.utils.instantiate(
                        self.cfg.predictor,
                        num_patches=num_patches,
                        num_frames=self.cfg.num_hist,
                        dim=self.encoder.emb_dim
                        + (
                            proprio_emb_dim * self.cfg.num_proprio_repeat
                            + action_emb_dim * self.cfg.num_action_repeat
                        )
                        * (self.cfg.concat_dim),
                    )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = load_trusted_checkpoint(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = ckpt
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )

        self.regularizer = None
        reg_cfg = self.cfg.get("regularizer", None)
        if reg_cfg is not None and reg_cfg.get("_target_", None) is not None:
            self.regularizer = hydra.utils.instantiate(reg_cfg).to(self.device)
            log.info(f"Regularizer: {type(self.regularizer).__name__}")

        link_cfg = self.cfg.get("link", None)
        if self.link is None and link_cfg is not None and link_cfg.get("_target_", None) is not None:
            self.link = hydra.utils.instantiate(link_cfg).to(self.device)
            log.info(f"Link: {getattr(self.link, 'kind', None)}")

        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            action_conditioning=action_conditioning,
            regularizer=self.regularizer,
            reg_weight=self.cfg.get("reg_weight", 0.0),
            detach_target=self.cfg.get("detach_target", True),
            link=self.link,
            lamb_var=self.cfg.get("lamb_var", 0.0),
            lamb_cov=self.cfg.get("lamb_cov", 0.0),
            var_space=self.cfg.get("var_space", "u"),
        )

        if self.cfg.get("mup", False) and not model_ckpt.exists():
            from models.mup import mup_init_
            if self.train_encoder:
                mup_init_(self.encoder, tag="encoder")
            if self.predictor is not None:
                mup_init_(self.predictor, tag="predictor")
            if self.decoder is not None and self.train_decoder:
                mup_init_(self.decoder, tag="decoder")
            log.info("Applied muP init to trained scratch modules "
                     f"(encoder={self.train_encoder}, predictor={self.predictor is not None}, "
                     f"decoder={self.decoder is not None and self.train_decoder}).")

        self.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.num_trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        log.info(
            "Parameters: total=%d trainable=%d",
            self.num_parameters,
            self.num_trainable_parameters,
        )

    def init_optimizers(self):
        mup = self.cfg.get("mup", False)
        if mup:
            from models.mup import mup_param_groups
            mup_lr = self.cfg.training.get("mup_lr", self.cfg.training.predictor_lr)
            mup_wd = self.cfg.training.get("mup_weight_decay", 0.01)
            mup_bw = self.cfg.training.get("mup_base_width", self.cfg.get("embed_dim", 384))

        if self.train_encoder:
            if mup:
                self.encoder_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="encoder"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.encoder_optimizer = torch.optim.Adam(
                    self.encoder.parameters(),
                    lr=self.cfg.training.encoder_lr,
                )
            self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        else:
            self.encoder_optimizer = None
        if self.cfg.has_predictor:
            if mup:
                self.predictor_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.predictor, mup_lr, mup_bw, weight_decay=mup_wd, tag="predictor"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.predictor_optimizer = torch.optim.AdamW(
                    self.predictor.parameters(),
                    lr=self.cfg.training.predictor_lr,
                )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            if mup:
                act_groups = mup_param_groups(self.action_encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="action_encoder") \
                    + mup_param_groups(self.proprio_encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="proprio_encoder")
                self.action_encoder_optimizer = torch.optim.AdamW(
                    act_groups, lr=mup_lr, eps=1e-15,
                )
            else:
                self.action_encoder_optimizer = torch.optim.AdamW(
                    itertools.chain(
                        self.action_encoder.parameters(), self.proprio_encoder.parameters()
                    ),
                    lr=self.cfg.training.action_encoder_lr,
                )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            if mup:
                self.decoder_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.decoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="decoder"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.decoder_optimizer = torch.optim.Adam(
                    self.decoder.parameters(), lr=self.cfg.training.decoder_lr
                )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        for epoch in training_epoch_range(
            self.epoch, self.total_epochs, self.cfg.training.get("epochs_mode", "additional")
        ):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs_linked(obs)  # linked, to match z_obs_out
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs_linked(obs)  # linked, to match z_obs_out
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs_linked(obs_g)  # linked, to match the rollout space
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            scalar_values = [metric_scalar(item) for item in values]
            length = len(scalar_values)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(scalar_values),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        epoch_log["num_parameters"] = self.num_parameters
        epoch_log["num_trainable_parameters"] = self.num_trainable_parameters
        epoch_log["elapsed_time_sec"] = time.time() - self.run_started_at
        epoch_log["max_cuda_memory_mb"] = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            persisted_epoch_log = append_metrics_jsonl("metrics.jsonl", epoch_log)
            self.wandb_run.log(persisted_epoch_log)
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    exit_code = 0
    try:
        trainer = Trainer(cfg)
        trainer.run()
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
