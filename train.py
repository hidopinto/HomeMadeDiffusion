import argparse
import math
from functools import partial

import huggingface_hub
import torch
import weave
import wandb
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler

from trainer import DiTTrainer
from model_builder import build_model
from utils import load_config, setup_logging
from data import build_dataloader
from evaluation import EvaluationEngine


def _cosine_lr_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end_ratio: float,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return lr_end_ratio + (1.0 - lr_end_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_peak: float,
    lr_end: float,
) -> object:
    if scheduler_name == "cosine_with_warmup":
        lr_lambda = partial(
            _cosine_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end_ratio=lr_end / lr_peak,
        )
        return LambdaLR(optimizer, lr_lambda)

    return get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Run the VAE caching pass and exit without training (cache_then_train mode only).",
    )
    args = parser.parse_args()

    setup_logging()
    load_dotenv()
    huggingface_hub.login()
    wandb.login()

    config = load_config(config_path="config.yaml")

    weave.init(config.general.wnb_project_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build model (frozen giants + DiT + diffusion engine)
    model = build_model(config, device, gradient_checkpointing=config.training.gradient_checkpointing)
    model.vae.enable_slicing()
    mode = getattr(config.data, "mode", "cache")

    # 2. Optimizer
    optimizer = AdamW(
        list(model.transformer.parameters()) + list(model.condition_manager.parameters()),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        eps=1e-7,
    )

    # 3. Build dataloader — in cache_then_train mode this runs the caching pass first (blocking)
    if mode == "cache_then_train":
        # Offload DiT + condition_manager to CPU before caching: frees ~1.3 GB on GPU so the VAE
        # can encode full batches (disable_slicing) without OOM. Peak VRAM during caching is
        # ~17 GB (batch=64) vs 23.5 GB available. DiT is moved back to GPU after caching.
        model.transformer.cpu()
        model.condition_manager.cpu()
        torch.cuda.empty_cache()
        model.vae.disable_slicing()
    dataloader = build_dataloader(config, model.vae, model.tokenizer, model.text_encoder, device)
    if mode == "cache_then_train":
        model.vae.enable_slicing()
        model.transformer.to(device)
        model.condition_manager.to(device)
        torch.cuda.empty_cache()

    if args.cache_only:
        print("--cache-only: caching complete. Exiting.")
        return

    # Compile text encoder after the caching pass: CUDA graphs no longer occupy VRAM during
    # encoding, and the compiled model is still available for training inference calls.
    # VAE is intentionally not compiled (see original comment above).
    if torch.cuda.is_available():
        model.text_encoder = torch.compile(model.text_encoder, mode="reduce-overhead")

    # 3b. Build val dataloader + evaluation engine (VAE must still be on GPU here)
    torch.cuda.empty_cache()
    eval_engine = None
    if getattr(config.training, "eval_every_steps", False):
        val_split = getattr(config.data, "val_split", None)
        if val_split:
            val_dataloader = build_dataloader(
                config, model.vae, model.tokenizer, model.text_encoder, device,
                split=val_split,
                shuffle=False,
            )
            eval_engine = EvaluationEngine(config, val_dataloader, model, device)

    # 3a. Cache null text embedding before offloading frozen models
    model.cache_null_embed(torch.device(device))
    if mode == "streaming":
        pass  # VAE and text_encoder must stay on GPU — they encode every training batch
    elif mode == "cache_then_train":
        # Caching done: move VAE off GPU. text_encoder stays on GPU for VaeCachedDataset.
        model.vae = model.vae.cpu()
    else:
        model.vae = model.vae.cpu()
        model.text_encoder = model.text_encoder.cpu()
    torch.cuda.empty_cache()

    # 4. Build LR scheduler
    grad_accum_steps: int = getattr(config.training, "gradient_accumulation_steps", 1)
    # streaming DataLoaders wrap an IterableDataset with no __len__; use steps_per_epoch from config
    steps_per_epoch: int = getattr(config.data, "steps_per_epoch", None) or len(dataloader)
    total_training_steps: int = (steps_per_epoch // grad_accum_steps) * config.training.epochs
    num_warmup_steps: int = round(config.training.warmup_ratio * total_training_steps)

    lr_scheduler = build_lr_scheduler(
        scheduler_name=config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
        lr_peak=config.training.lr,
        lr_end=config.training.lr_end,
    )

    # 5. Execute
    trainer = DiTTrainer(
        config=config, model=model, dataloader=dataloader,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        eval_engine=eval_engine,
    )
    trainer.fit(epochs=config.training.epochs)


if __name__ == "__main__":
    main()
