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

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        latents = batch["latent"]
        text_embeds = batch["text_embed"]
        grad_log: dict[str, float] = {}
        with self.accelerator.accumulate(self.model):
            loss = self.model(latents, text_embeds)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                clip_norm = getattr(self.config.training, "gradient_clip_norm", None)
                if clip_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)

                unwrapped = self.accelerator.unwrap_model(self.model)
                transformer = unwrapped.transformer

                for i, block in enumerate(transformer.blocks):
                    norm_sq = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in block.parameters() if p.grad is not None
                    )
                    grad_log[f"grad_norm/block_{i:02d}"] = norm_sq ** 0.5

                for name, module in [
                    ("patch_embed", transformer.patch_embed),
                    ("t_embedder", transformer.t_embedder),
                    ("text_projector", transformer.text_projector),
                    ("final_layer", transformer.final_layer),
                ]:
                    norm_sq = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in module.parameters() if p.grad is not None
                    )
                    grad_log[f"grad_norm/{name}"] = norm_sq ** 0.5

                grad_norm = sum(v ** 2 for v in grad_log.values()) ** 0.5
                grad_log["train/grad_norm"] = grad_norm

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

        save_every_steps = getattr(self.config.training, "save_every_steps", None)
        infer_every_steps = getattr(self.config.training, "inference_every_steps", None)
        checkpoint_dir = getattr(self.config.training, "checkpoint_dir", "checkpoints")

        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.dataloader:
                loss, grad_log = self.train_step(batch)
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

                    if infer_every_steps and global_step % infer_every_steps == 0:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        unwrapped.transformer.eval()
                        inference_prompt = getattr(self.config.training, "inference_prompt", "a photo")
                        inference_steps = getattr(self.config.training, "inference_steps", 50)
                        images = unwrapped.generate(
                            [inference_prompt],
                            num_steps=inference_steps,
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
