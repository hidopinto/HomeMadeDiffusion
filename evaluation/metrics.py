from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

__all__ = ["EvaluationEngine"]

logger = logging.getLogger(__name__)

_MIN_FID_SAMPLES = 2048


def _fid_stats_path(config, max_real: int) -> Path:
    """Return the path where FID real-image stats are cached for this config."""
    cache_dir = (
        Path(config.data.vae_cache_dir)
        / config.data.dataset_name.replace("/", "--")
        / "train"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"fid_real_stats_n{max_real}.pt"


class EvaluationEngine:
    """Computes FID, Inception Score, and CLIP Score at eval checkpoints.

    All three torchmetrics objects run on the training device. Real FID
    statistics are pre-populated once at construction time (while the VAE is
    still on GPU); a snapshot of those stats is kept so each ``compute()``
    call only needs to generate fake images rather than re-decode the entire
    val set.
    """

    def __init__(
        self,
        config,
        val_dataloader: DataLoader,
        model,
        device: str,
    ) -> None:
        self.config = config
        self.device = device

        self.eval_num_samples: int = getattr(config.training, "eval_num_samples", 2048)
        self.eval_batch_size: int = getattr(config.training, "eval_batch_size", 8)
        prompts_file: str = getattr(config.training, "eval_prompts_file", "eval_prompts.txt")
        with open(prompts_file) as f:
            self.eval_prompts = [line.strip() for line in f if line.strip()]

        sampler_cfg = config.diffusion.samplers[config.diffusion.sampler]
        sampler_default_steps: int = getattr(sampler_cfg, "num_steps", 50)
        self.num_steps: int = getattr(config.training, "eval_num_steps", sampler_default_steps)
        self.eta: float = getattr(sampler_cfg, "eta", 0.0)
        self.guidance_scale: float = getattr(config.training, "guidance_scale", 7.5)
        self.height: int = getattr(config.training, "inference_height", 512)
        self.width: int = getattr(config.training, "inference_width", 512)
        self.scheduler: str = config.diffusion.sampler

        logger.info("Initialising evaluation metrics (FID + IS + CLIPScore)...")
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.isc = InceptionScore(normalize=True).to(device)
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        logger.info("  Metric models on device.")

        max_real: int = max(_MIN_FID_SAMPLES, min(self.eval_num_samples * 4, 10_000))
        fid_stats_path = _fid_stats_path(config, max_real)
        if fid_stats_path.exists():
            logger.info("  Loading cached FID real stats from %s...", fid_stats_path)
            stats = torch.load(fid_stats_path, map_location=device, weights_only=True)
            self._real_sum = stats["real_sum"].to(device)
            self._real_cov_sum = stats["real_cov_sum"].to(device)
            self._real_num = stats["real_num"].to(device)
            logger.info(
                "  FID real stats loaded (%d images). Skipping val encoding.",
                int(self._real_num.item()),
            )
        else:
            logger.info("  Pre-populating FID real statistics (one-time, will be cached)...")
            self._populate_real_stats(val_dataloader, model, max_real)
            torch.save(
                {
                    "real_sum": self._real_sum.cpu(),
                    "real_cov_sum": self._real_cov_sum.cpu(),
                    "real_num": self._real_num.cpu(),
                },
                fid_stats_path,
            )
            logger.info("  FID real stats saved to %s.", fid_stats_path)

        # Offload metric models to CPU — they are only needed during compute().
        # Leaving them on GPU would pin ~1.1 GB of VRAM for the entire training run.
        self.fid = self.fid.cpu()
        self.isc = self.isc.cpu()
        self.clip_score = self.clip_score.cpu()
        torch.cuda.empty_cache()
        logger.info("EvaluationEngine ready (metrics offloaded to CPU until first eval).")

    def _populate_real_stats(self, val_dataloader: DataLoader, model, max_real: int) -> None:
        """Decode val latents and update FID real-image statistics.

        ``self.isc`` and ``self.clip_score`` are unused during this method.
        They are moved to CPU for the duration of the loop to free ~1.5 GB of
        GPU VRAM, then restored to the training device before returning.

        When the underlying dataset exposes ``iter_latents(batch_size)``,
        that path is used instead of iterating the full DataLoader — this
        avoids triggering text encoding (CLIP) which is unnecessary here and
        can cause very long hangs on the first call to a ``torch.compile``d
        encoder.
        """
        scale_factor: float = self.config.dit.vae_scale_factor
        count: int = 0
        logger.info("  Decoding up to %d real images for FID reference stats...", max_real)

        # Offload metrics that are unused during real-stat population.
        self.isc = self.isc.cpu()
        self.clip_score = self.clip_score.cpu()
        torch.cuda.empty_cache()

        dataset = val_dataloader.dataset
        if hasattr(dataset, "iter_latents"):
            latent_source = dataset.iter_latents(self.eval_batch_size)
        else:
            latent_source = (batch["latent"] for batch in val_dataloader)

        model.vae.eval()
        with torch.no_grad():
            for latents in latent_source:
                if count >= max_real:
                    break
                latents = latents.to(self.device)

                # Decode in sub-batches to cap peak VAE decoder memory.
                sub_imgs: list[Tensor] = []
                for sub_latents in _iter_batches_tensor(latents, self.eval_batch_size):
                    sub_imgs.append(
                        _decode_latents_to_unit(model.vae, sub_latents, scale_factor)
                    )
                    torch.cuda.empty_cache()
                imgs: Tensor = torch.cat(sub_imgs, dim=0)

                self.fid.update(imgs.to(self.device), real=True)
                count += imgs.shape[0]
                torch.cuda.empty_cache()

        logger.info("  FID real stats populated (%d images).", count)

        # Restore metrics to GPU for compute() calls.
        self.isc = self.isc.to(self.device)
        self.clip_score = self.clip_score.to(self.device)

        # Snapshot real stats so we can restore them after each reset()
        self._real_sum = self.fid.real_features_sum.clone()
        self._real_cov_sum = self.fid.real_features_cov_sum.clone()
        self._real_num = self.fid.real_features_num_samples.clone()

    def _restore_real_stats(self) -> None:
        self.fid.real_features_sum.copy_(self._real_sum)
        self.fid.real_features_cov_sum.copy_(self._real_cov_sum)
        self.fid.real_features_num_samples.copy_(self._real_num)

    @torch.no_grad()
    def compute(self, model, global_step: int) -> dict[str, float]:
        """Generate ``eval_num_samples`` images, compute FID / IS / CLIP Score.

        Args:
            model: Unwrapped ``LatentDiffusion`` instance.
            global_step: Current training step (unused here; passed for logging symmetry).

        Returns:
            Dict of metric names → scalar values, or empty dict if not enough
            samples for reliable FID.
        """
        logger.info("Running eval (FID / IS / CLIPScore)...")
        self.fid = self.fid.to(self.device)
        self.isc = self.isc.to(self.device)
        self.clip_score = self.clip_score.to(self.device)

        if self.eval_num_samples < _MIN_FID_SAMPLES:
            logger.warning(
                "[EvaluationEngine] eval_num_samples=%d < %d; skipping FID (results would be unreliable).",
                self.eval_num_samples, _MIN_FID_SAMPLES,
            )
            self.fid = self.fid.cpu()
            self.isc = self.isc.cpu()
            self.clip_score = self.clip_score.cpu()
            torch.cuda.empty_cache()
            return {}

        model.transformer.eval()

        fake_imgs: list[Tensor] = []
        batch_prompts: list[str] = []
        generated = 0
        prompt_cycle = 0

        while generated < self.eval_num_samples:
            remaining = self.eval_num_samples - generated
            bs = min(self.eval_batch_size, remaining)
            prompts_batch = [
                self.eval_prompts[(prompt_cycle + i) % len(self.eval_prompts)]
                for i in range(bs)
            ]
            prompt_cycle += bs

            imgs = model.generate(
                prompts_batch,
                height=self.height,
                width=self.width,
                num_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                scheduler=self.scheduler,
                eta=self.eta,
            )  # (bs, 3, H, W) float32 in [0, 1]

            imgs_cpu = imgs.float().cpu()
            fake_imgs.append(imgs_cpu)
            batch_prompts.extend(prompts_batch)
            generated += bs
            torch.cuda.empty_cache()

        model.transformer.train()

        # Update IS and CLIP Score incrementally
        for i, (img_batch, prompt) in enumerate(
            zip(fake_imgs, _iter_batches(batch_prompts, self.eval_batch_size))
        ):
            imgs_gpu = img_batch.to(self.device)
            self.isc.update(imgs_gpu)

            imgs_uint8 = (imgs_gpu * 255).clamp(0, 255).to(torch.uint8)
            self.clip_score.update(imgs_uint8, prompt)

        # FID: update fake stats (real stats already populated)
        all_fake = torch.cat(fake_imgs, dim=0)  # (N, 3, H, W) float32 [0,1]
        self.fid.update(all_fake.to(self.device), real=False)

        fid_val = self.fid.compute().item()
        is_mean, is_std = self.isc.compute()
        clip_val = self.clip_score.compute().item()

        # Reset and restore real stats for next eval call
        self.fid.reset()
        self._restore_real_stats()
        self.isc.reset()
        self.clip_score.reset()

        # Offload back to CPU — metric models are not needed until the next eval pass.
        self.fid = self.fid.cpu()
        self.isc = self.isc.cpu()
        self.clip_score = self.clip_score.cpu()
        torch.cuda.empty_cache()

        logger.info(
            "Eval done — FID=%.2f, IS=%.2f±%.2f, CLIP=%.3f.",
            fid_val, is_mean.item(), is_std.item(), clip_val,
        )
        return {
            "eval/fid": fid_val,
            "eval/is_mean": is_mean.item(),
            "eval/is_std": is_std.item(),
            "eval/clip_score": clip_val,
        }


@torch.no_grad()
def _decode_latents_to_unit(vae, latents: Tensor, scale_factor: float) -> Tensor:
    """Decode VAE latents to float32 images in [0, 1] on CPU."""
    scaled = latents / scale_factor
    imgs = vae.decode(scaled.to(vae.dtype)).sample
    imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0
    return imgs.float().cpu()


def _iter_batches(items: list, batch_size: int):
    """Yield successive batches from a flat list."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _iter_batches_tensor(t: Tensor, batch_size: int):
    """Yield successive sub-batches of a Tensor along dim 0."""
    for i in range(0, t.shape[0], batch_size):
        yield t[i : i + batch_size]
