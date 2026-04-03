from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator


class DiTTrainer:
    def __init__(self, config, model, dataloader, optimizer, lr_scheduler) -> None:
        self.config = config

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

        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )

    def train_step(self, batch: dict, global_step: int) -> tuple[torch.Tensor, dict]:
        latents = batch["latent"]
        text_embeds = batch["text_embed"]
        grad_log: dict[str, float] = {}
        with self.accelerator.accumulate(self.model):
            loss = self.model(latents, text_embeds)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                clip_norm = getattr(self.config.training, "gradient_clip_norm", None)
                if clip_norm is not None:
                    total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)
                    grad_log["train/grad_norm"] = total_norm.item()

                unwrapped = self.accelerator.unwrap_model(self.model)
                transformer = unwrapped.transformer

                for i, block in enumerate(transformer.blocks):
                    grads = [p.grad.data.flatten() for p in block.parameters() if p.grad is not None]
                    if grads:
                        grad_log[f"grad_norm/block_{i:02d}"] = torch.cat(grads).norm(2).item()

                for name, module in [
                    ("patch_embed", transformer.patch_embed),
                    ("t_embedder", transformer.t_embedder),
                    ("final_layer", transformer.final_layer),
                ]:
                    grads = [p.grad.data.flatten() for p in module.parameters() if p.grad is not None]
                    if grads:
                        grad_log[f"grad_norm/{name}"] = torch.cat(grads).norm(2).item()

                # condition_manager lives on LatentDiffusion, not DiT
                condition_manager = unwrapped.condition_manager
                for proj in condition_manager.projector_modules:
                    key = "adaln_projector" if getattr(proj, "role", "") == "global" else "crossattn_projector"
                    grads = [p.grad.data.flatten() for p in proj.parameters() if p.grad is not None]
                    if grads:
                        grad_log[f"grad_norm/{key}"] = torch.cat(grads).norm(2).item()

                grads = [p.grad.data.flatten() for p in condition_manager.parameters() if p.grad is not None]
                if grads:
                    grad_log["grad_norm/condition_manager"] = torch.cat(grads).norm(2).item()

                # Block-depth bar chart — shows which layers carry the gradient signal
                if self.accelerator.is_main_process:
                    block_data = [
                        [f"block_{i:02d}", grad_log.get(f"grad_norm/block_{i:02d}", 0.0)]
                        for i in range(len(transformer.blocks))
                    ]
                    wandb.log(
                        {"grad_norm/block_depth": wandb.plot.bar(
                            wandb.Table(columns=["block", "grad_norm"], data=block_data),
                            "block", "grad_norm",
                            title="Gradient Norm by Block Depth",
                        )},
                        step=global_step,
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
        self.accelerator.log({"model/num_params": num_params})

        save_every_steps = getattr(self.config.training, "save_every_steps", False)
        infer_every_steps = getattr(self.config.training, "inference_every_steps", False)
        checkpoint_dir = getattr(self.config.training, "checkpoint_dir", "checkpoints")
        full_ckpt_dir = Path(checkpoint_dir) / "full_ckpt"

        global_step = 0
        resume_from = getattr(self.config.training, "resume_from_checkpoint", False)
        if resume_from and Path(resume_from).exists():
            self.accelerator.load_state(resume_from)
            step_file = Path(resume_from) / "step.txt"
            if step_file.exists():
                global_step = int(step_file.read_text().strip())
            self.accelerator.print(f"Resumed training from step {global_step}")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.dataloader:
                loss, grad_log = self.train_step(batch, global_step)
                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1

                if self.accelerator.sync_gradients:
                    global_step += 1
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.accelerator.log({
                        "train/loss": loss_val,
                        "train/lr": current_lr,
                        **grad_log,
                    }, step=global_step)

                    if save_every_steps and global_step % save_every_steps == 0:
                        self.accelerator.wait_for_everyone()
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        path = Path(checkpoint_dir) / f"dit_step{global_step:07d}.pt"
                        self.accelerator.save(unwrapped.transformer.state_dict(), str(path))
                        self.accelerator.save_state(str(full_ckpt_dir))
                        if self.accelerator.is_main_process:
                            (full_ckpt_dir / "step.txt").write_text(str(global_step))

                    if infer_every_steps and global_step % infer_every_steps == 0:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        unwrapped.transformer.eval()
                        inference_prompt    = getattr(self.config.training, "inference_prompt")
                        sampler_cfg         = self.config.diffusion.samplers[self.config.diffusion.sampler]
                        inference_steps     = getattr(sampler_cfg, "num_steps", 50)
                        inference_eta       = getattr(sampler_cfg, "eta", 0.0)
                        guidance_scale      = getattr(self.config.training, "guidance_scale", 7.5)
                        inference_height    = getattr(self.config.training, "inference_height", 512)
                        inference_width     = getattr(self.config.training, "inference_width", 512)
                        inference_scheduler = getattr(self.config.diffusion, "sampler", "ddim")
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
                        caption = f"Step {global_step}"

                        self.accelerator.log({
                            "inference/images": wandb.Image(img_tensor, caption=caption),
                            "inference/step": global_step,
                        }, step=global_step)
                        unwrapped.transformer.train()

            elapsed = time.time() - t_start
            samples_per_sec = (epoch_steps * self.config.training.batch_size) / elapsed if elapsed > 0 else 0.0
            mean_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0

            self.accelerator.log({
                "train/epoch_loss": mean_loss,
                "train/epoch": epoch + 1,
                "train/samples_per_sec": samples_per_sec,
            }, step=global_step)

        self.accelerator.end_training()
