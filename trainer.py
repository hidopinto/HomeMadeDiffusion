from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import cast

import torch
import wandb
from accelerate import Accelerator

from models.models import LatentDiffusion

logger = logging.getLogger(__name__)


def _log_cuda_mem(tag: str) -> None:
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    logger.info("[MEM %s] allocated=%.2f GB  reserved=%.2f GB", tag, alloc, reserved)


def _log_top_tensors(tag: str, top_n: int = 30) -> None:
    entries: list[tuple[int, torch.Size, torch.dtype]] = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                entries.append((obj.element_size() * obj.nelement(), obj.shape, obj.dtype))
        except Exception:
            pass
    entries.sort(key=lambda x: x[0], reverse=True)
    logger.info("[MEM %s] top %d live CUDA tensors:", tag, min(top_n, len(entries)))
    for size, shape, dtype in entries[:top_n]:
        logger.info("  %8.1f MB  %-14s  %s", size / 1e6, str(dtype).replace("torch.", ""), shape)


class DiTTrainer:
    def __init__(
        self,
        config,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        eval_engine=None,
        max_steps: int | None = None,
    ) -> None:
        self.config = config
        self._max_steps = max_steps

        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=getattr(config.training, "gradient_accumulation_steps", 1),
            log_with="wandb",
        )
        self.model = model
        if config.training.gradient_checkpointing:
            self.model.transformer.gradient_checkpointing = True

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        checkpoint_dir = getattr(config.training, "checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.eval_engine = eval_engine

        logger.info("accelerator.prepare starting...")
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )
        logger.info("accelerator.prepare done.")

        # EMA — initialized from live weights after prepare(); 0.0 disables.
        self._ema_decay: float = getattr(config.training, "ema_decay", 0.0)
        self._ema_transformer: dict[str, torch.Tensor] = {}
        self._ema_cmanager: dict[str, torch.Tensor] = {}
        if self._ema_decay > 0:
            unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
            self._ema_transformer = {
                k: v.detach().clone().float().cpu()
                for k, v in unwrapped.transformer.state_dict().items()
            }
            self._ema_cmanager = {
                k: v.detach().clone().float().cpu()
                for k, v in unwrapped.condition_manager.state_dict().items()
            }
            logger.info("EMA initialized (decay=%.4f).", self._ema_decay)

    def _ema_update(self) -> None:
        unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
        with torch.no_grad():
            for k in self._ema_transformer:
                live = unwrapped.transformer.state_dict()[k].detach()
                if live.dtype.is_floating_point:
                    self._ema_transformer[k].mul_(self._ema_decay).add_(
                        live.float().cpu(), alpha=1.0 - self._ema_decay
                    )
                else:
                    self._ema_transformer[k].copy_(live.cpu())
            for k in self._ema_cmanager:
                live = unwrapped.condition_manager.state_dict()[k].detach()
                if live.dtype.is_floating_point:
                    self._ema_cmanager[k].mul_(self._ema_decay).add_(
                        live.float().cpu(), alpha=1.0 - self._ema_decay
                    )
                else:
                    self._ema_cmanager[k].copy_(live.cpu())

    def _ema_save(self, checkpoint_dir: str, full_ckpt_dir: Path, global_step: int) -> None:
        if not (self._ema_decay > 0 and self.accelerator.is_main_process):
            return
        payload = {"transformer": self._ema_transformer, "cmanager": self._ema_cmanager}
        torch.save(payload, str(full_ckpt_dir / "ema.pt"))
        torch.save(
            {"transformer": self._ema_transformer, "cmanager": self._ema_cmanager},
            str(Path(checkpoint_dir) / f"ema_step{global_step:07d}.pt"),
        )

    def _ema_load(self, full_ckpt_dir: Path) -> None:
        ema_path = full_ckpt_dir / "ema.pt"
        if not (self._ema_decay > 0 and ema_path.exists()):
            return
        saved = torch.load(str(ema_path), map_location="cpu", weights_only=True)
        self._ema_transformer = saved["transformer"]
        self._ema_cmanager = saved["cmanager"]
        logger.info("EMA state loaded from %s.", ema_path)

    def _save_checkpoint(self, checkpoint_dir: str, full_ckpt_dir: Path, global_step: int) -> None:
        self.accelerator.wait_for_everyone()
        unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
        path = Path(checkpoint_dir) / f"dit_step{global_step:07d}.pt"
        payload = {
            "transformer": unwrapped.transformer.state_dict(),
            "condition_manager": unwrapped.condition_manager.state_dict(),
        }
        self.accelerator.save(payload, str(path))
        self.accelerator.save_state(str(full_ckpt_dir))
        if self.accelerator.is_main_process:
            (full_ckpt_dir / "step.txt").write_text(str(global_step))
        self._ema_save(checkpoint_dir, full_ckpt_dir, global_step)

    def train_step(self, batch: dict, global_step: int, _diag: bool = False) -> tuple[torch.Tensor, dict]:
        latents = batch["latent"]
        text_embeds = batch["text_embed"]
        grad_log: dict[str, float] = {}
        with self.accelerator.accumulate(self.model):
            loss = self.model(latents, text_embeds)

            if _diag:
                _log_cuda_mem("after-forward")
                _log_top_tensors("after-forward")

            try:
                self.accelerator.backward(loss)
            except torch.cuda.OutOfMemoryError:
                _log_cuda_mem("oom")
                _log_top_tensors("oom")
                logger.error("[MEM oom] memory_summary:\n%s", torch.cuda.memory_summary(abbreviated=False))
                raise

            if self.accelerator.sync_gradients:
                clip_norm = getattr(self.config.training, "gradient_clip_norm", None)
                if clip_norm is not None:
                    total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)
                    grad_log["train/grad_norm"] = total_norm.item()

                grad_norm_interval = getattr(self.config.training, "grad_norm_log_every_steps", 100)
                if global_step % grad_norm_interval == 0:
                    unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
                    transformer = unwrapped.transformer

                    # Collect all norm tensors first, then batch-convert to float with a
                    # single pass of .item() calls. Avoids repeated GPU→CPU sync stalls.
                    norm_tensors: dict[str, torch.Tensor] = {}

                    for i, block in enumerate(transformer.blocks):
                        grads = [p.grad.data.flatten() for p in block.parameters() if p.grad is not None]
                        if grads:
                            norm_tensors[f"grad_norm/block_{i:02d}"] = torch.cat(grads).norm(2)

                    for name, module in [
                        ("patch_embed", transformer.patch_embed),
                        ("t_embedder", transformer.t_embedder),
                        ("final_layer", transformer.final_layer),
                    ]:
                        grads = [p.grad.data.flatten() for p in module.parameters() if p.grad is not None]
                        if grads:
                            norm_tensors[f"grad_norm/{name}"] = torch.cat(grads).norm(2)

                    # condition_manager lives on LatentDiffusion, not DiT
                    condition_manager = unwrapped.condition_manager
                    for proj in condition_manager.projector_modules:
                        key = "adaln_projector" if getattr(proj, "role", "") == "global" else "crossattn_projector"
                        grads = [p.grad.data.flatten() for p in proj.parameters() if p.grad is not None]
                        if grads:
                            norm_tensors[f"grad_norm/{key}"] = torch.cat(grads).norm(2)

                    grads = [p.grad.data.flatten() for p in condition_manager.parameters() if p.grad is not None]
                    if grads:
                        norm_tensors["grad_norm/condition_manager"] = torch.cat(grads).norm(2)

                    grad_log.update({k: v.item() for k, v in norm_tensors.items()})

                    # Block-depth bar chart — shows which layers carry the gradient signal
                    if self.accelerator.is_main_process:
                        block_data = [
                            [f"block_{i:02d}", grad_log.get(f"grad_norm/block_{i:02d}", 0.0)]
                            for i in range(len(transformer.blocks))
                        ]
                        grad_log["grad_norm/block_depth"] = wandb.plot.bar(
                            wandb.Table(columns=["block", "grad_norm"], data=block_data),
                            "block", "grad_norm",
                            title="Gradient Norm by Block Depth",
                        )

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss, grad_log

    def fit(self, epochs: int) -> None:
        self.accelerator.init_trackers(
            self.config.general.wnb_project_name,
            init_kwargs={"wandb": {"entity": self.config.general.wnb_entity}},
        )

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.log({"model/num_params": num_params}, step=0)

        save_every_steps = getattr(self.config.training, "save_every_steps", False)
        infer_every_steps = getattr(self.config.training, "inference_every_steps", False)
        eval_every_steps = getattr(self.config.training, "eval_every_steps", False)
        checkpoint_dir = getattr(self.config.training, "checkpoint_dir", "checkpoints")
        full_ckpt_dir = Path(checkpoint_dir) / "full_ckpt"

        global_step = 0
        resume_from = getattr(self.config.training, "resume_from_checkpoint", False)
        if resume_from and Path(resume_from).exists():
            self.accelerator.load_state(resume_from)
            step_file = Path(resume_from) / "step.txt"
            if step_file.exists():
                global_step = int(step_file.read_text().strip())
            self._ema_load(Path(resume_from))
            self.accelerator.print(f"Resumed training from step {global_step}")

        _first_step_logged = False
        _diag_done = False
        _done = False
        for epoch in range(epochs):
            if _done:
                break
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()
            logger.info("Epoch %d/%d — starting.", epoch + 1, epochs)

            if epoch == 0:
                _log_cuda_mem("before-first-batch")
                _log_top_tensors("before-first-batch")

            for batch in self.dataloader:
                _diag = not _diag_done
                loss, grad_log = self.train_step(batch, global_step, _diag=_diag)
                _diag_done = True
                if not _first_step_logged:
                    logger.info("First batch complete — training loop running.")
                    _first_step_logged = True
                epoch_steps += 1

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if self._ema_decay > 0:
                        self._ema_update()

                    # Defer .item() to here so we only sync GPU→CPU when we actually
                    # need the scalar for logging, not on every micro-step.
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    log_dict: dict = {
                        "train/loss": loss_val,
                        "train/lr": current_lr,
                        **grad_log,
                    }

                    if save_every_steps and global_step % save_every_steps == 0:
                        self._save_checkpoint(checkpoint_dir, full_ckpt_dir, global_step)

                    if infer_every_steps and global_step % infer_every_steps == 0:
                        unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
                        inference_prompt    = getattr(self.config.training, "inference_prompt")
                        sampler_cfg         = self.config.diffusion.samplers[self.config.diffusion.sampler]
                        inference_steps     = getattr(sampler_cfg, "num_steps", 50)
                        inference_eta       = getattr(sampler_cfg, "eta", 0.0)
                        guidance_scale      = getattr(self.config.training, "guidance_scale", 7.5)
                        inference_height    = getattr(self.config.training, "inference_height", 512)
                        inference_width     = getattr(self.config.training, "inference_width", 512)
                        inference_scheduler = getattr(self.config.diffusion, "sampler", "ddim")

                        # Swap in EMA weights for inference if enabled
                        live_xfm_sd: dict | None = None
                        live_cm_sd: dict | None = None
                        if self._ema_decay > 0:
                            _dtype = next(unwrapped.transformer.parameters()).dtype
                            live_xfm_sd = {k: v.clone() for k, v in unwrapped.transformer.state_dict().items()}
                            live_cm_sd = {k: v.clone() for k, v in unwrapped.condition_manager.state_dict().items()}
                            unwrapped.transformer.load_state_dict(
                                {k: v.to(_dtype) for k, v in self._ema_transformer.items()}
                            )
                            unwrapped.condition_manager.load_state_dict(
                                {k: v.to(_dtype) for k, v in self._ema_cmanager.items()}
                            )

                        unwrapped.transformer.eval()
                        images = unwrapped.generate(
                            [inference_prompt],
                            height=inference_height,
                            width=inference_width,
                            num_steps=inference_steps,
                            guidance_scale=guidance_scale,
                            scheduler=inference_scheduler,
                            eta=inference_eta,
                        )
                        img_tensor = images[0].detach().cpu().to(torch.float32)
                        log_dict["inference/images"] = wandb.Image(img_tensor, caption=f"Step {global_step}: {inference_prompt}")
                        log_dict["inference/step"] = global_step

                        # Restore live weights after EMA inference
                        if live_xfm_sd is not None:
                            unwrapped.transformer.load_state_dict(live_xfm_sd)
                        if live_cm_sd is not None:
                            unwrapped.condition_manager.load_state_dict(live_cm_sd)

                        unwrapped.transformer.train()
                        torch.cuda.empty_cache()

                    self.accelerator.log(log_dict, step=global_step)

                    if (
                        eval_every_steps
                        and global_step % eval_every_steps == 0
                        and self.eval_engine is not None
                    ):
                        self.accelerator.wait_for_everyone()
                        gc.collect()
                        torch.cuda.empty_cache()
                        unwrapped = cast(LatentDiffusion, self.accelerator.unwrap_model(self.model))
                        metrics = self.eval_engine.compute(unwrapped, global_step)
                        if metrics and self.accelerator.is_main_process:
                            self.accelerator.log(metrics, step=global_step)
                        torch.cuda.empty_cache()

                    if self._max_steps is not None and global_step >= self._max_steps:
                        logger.info("Reached max_steps=%d — stopping.", self._max_steps)
                        _done = True
                        break

            elapsed = time.time() - t_start
            samples_per_sec = (epoch_steps * self.config.training.batch_size) / elapsed if elapsed > 0 else 0.0
            mean_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0

            self.accelerator.log({
                "train/epoch_loss": mean_loss,
                "train/epoch": epoch + 1,
                "train/samples_per_sec": samples_per_sec,
            }, step=global_step)

        # Always save final state so the last steps are never lost.
        self._save_checkpoint(checkpoint_dir, full_ckpt_dir, global_step)
        self.accelerator.end_training()
