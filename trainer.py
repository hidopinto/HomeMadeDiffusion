from accelerate import Accelerator


class DiTTrainer:
    def __init__(self, model, dataloader, optimizer, lr_scheduler):
        # Initialize Accelerator with W&B integration
        self.accelerator = Accelerator(log_with="wandb")

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Prepare for the RTX 3090
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )

    def train_step(self, batch):
        pixels, text_input = batch
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            # LatentDiffusion.forward handles the Engine call
            loss = self.model(pixels, text_input)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
        return loss

    def fit(self, epochs, project_name="diffusion-video"):
        self.accelerator.init_trackers(project_name)
        for epoch in range(epochs):
            self.model.train()
            for batch in self.dataloader:
                loss = self.train_step(batch)
                self.accelerator.log({"train/loss": loss.item()})
        self.accelerator.end_training()
