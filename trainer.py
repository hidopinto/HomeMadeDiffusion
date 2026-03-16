import torch
from accelerate import Accelerator


class DiTTrainer:
    def __init__(self, config, model, dataloader, optimizer, lr_scheduler):
        self.config = config

        # BF16 is the standard for 30-series GPUs
        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            log_with="wandb"
        )
        self.model = model
        if config.training.gradient_checkpointing:
            # Reaches into the core DiT model to enable the flag we just added
            self.model.transformer.gradient_checkpointing = True

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Prepare for the RTX 3090
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )

    def train_step(self, batch: dict) -> "torch.Tensor":
        latents = batch["latent"]
        text_embeds = batch["text_embed"]
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            loss = self.model(latents, text_embeds)
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return loss

    def fit(self, epochs):
        self.accelerator.init_trackers(self.config.general.wnb_project_name)
        for epoch in range(epochs):
            self.model.train()
            for batch in self.dataloader:
                loss = self.train_step(batch)
                self.accelerator.log({"train/loss": loss.item()})
        self.accelerator.end_training()
