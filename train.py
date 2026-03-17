import huggingface_hub
import torch
import weave
import wandb
from dotenv import load_dotenv
from torch.optim import AdamW

from trainer import DiTTrainer
from model_builder import build_model
from utils import load_config
from data import build_dataloader


def main() -> None:
    load_dotenv()
    huggingface_hub.login()
    wandb.login()

    config = load_config(config_path="config.yaml")

    weave.init(config.general.wnb_project_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build model (frozen giants + DiT + diffusion engine)
    model = build_model(config, device, gradient_checkpointing=config.training.gradient_checkpointing)

    # 2. Optimizer
    optimizer = AdamW(model.transformer.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    # 3. Build cached dataloader (encodes once, reuses on subsequent runs)
    dataloader = build_dataloader(config, model.vae, model.tokenizer, model.text_encoder, device)

    # 4. Execute
    trainer = DiTTrainer(config=config, model=model, dataloader=dataloader, optimizer=optimizer, lr_scheduler=None)
    trainer.fit(epochs=config.training.epochs)


if __name__ == "__main__":
    main()
