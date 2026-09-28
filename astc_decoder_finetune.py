"""Opt-in decoder adaptation after a completed Mip0 training run.

Example: python astc_decoder_finetune.py --max_iter 5000 --finetune_steps 1000
All existing train.py options are accepted; this entry leaves train.py unchanged.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def finetune_astc_decoder(trainer, steps: int = 1000, lr_multiplier: float = 0.5):
    """Keep source latents intact while fitting their fixed ASTC-decoded inputs."""
    from Comparison_ASTC import apply_astc_to_feature_grids

    if steps < 1 or not math.isfinite(lr_multiplier) or lr_multiplier <= 0:
        raise ValueError("steps and lr_multiplier must be positive")
    model = trainer.model
    if model.current_iter < 0 or model.current_iter >= trainer.max_iter:
        raise ValueError("Decoder adaptation requires a completed baseline iteration")
    if not trainer.mip0_only or not model.quantize or model.direct_diffuse_enabled:
        raise ValueError("ASTC decoder adaptation requires quantized Mip0 without direct diffuse")
    if model._decoder_adaptation_features is not None:
        raise RuntimeError("Decoder adaptation is already active")
    if trainer.super_resolution_enable and trainer.astc_superres_base_mip0 is None:
        raise ValueError("A deployment ASTC base is required for super-resolution adaptation")

    source = [grid.params.detach().clone() for grid in model.feature_grids]
    decoded = copy.deepcopy(model)
    decoded.simulate_quantize()
    apply_astc_to_feature_grids(decoded, trainer.astc_codec.roundtrip_rgba)
    fixed = [grid.params.detach().clone() for grid in decoded.feature_grids]
    del decoded
    fields = ("trained_iter", "max_iter", "eval_interval", "save_interval", "early_stop",
              "enable_astc_compare", "enable_pbr_compare")
    previous = {name: getattr(trainer, name) for name in fields}
    old_scheduler = model.scheduler
    old_codec = trainer.configs.astc_codec_in_loop_enable
    old_base = trainer.dataset.superres_base_cache[0] if trainer.super_resolution_enable else None
    old_requires_grad = [grid.params.requires_grad for grid in model.feature_grids]
    old_lrs = [group["lr"] for group in model.optimizer.param_groups]
    was_training = model.training
    started = time.perf_counter()
    try:
        model._decoder_adaptation_features = fixed
        for grid in model.feature_grids:
            grid.params.requires_grad_(False)
        if trainer.super_resolution_enable:
            slices = trainer.dataset.canonical_channel_slices
            trainer.dataset.superres_base_cache[0] = torch.cat([
                trainer.astc_superres_base_mip0[..., slices[name][0]:slices[name][1]]
                for name in trainer.dataset.available_textures
            ], dim=-1)
        trainer.configs.astc_codec_in_loop_enable = False
        # Keep Adam moments; fix the smaller learning rate during the extra phase.
        model.optimizer.param_groups[0]["lr"] *= lr_multiplier
        model.scheduler = SimpleNamespace(step=lambda **kwargs: None)
        trainer.trained_iter = model.current_iter + 1
        trainer.max_iter = trainer.trained_iter + steps
        trainer.eval_interval = trainer.save_interval = trainer.max_iter + 1
        trainer.early_stop = False
        trainer.enable_astc_compare = trainer.enable_pbr_compare = False
        model.train()
        trainer.train()
        torch.cuda.synchronize()
        for grid, original in zip(model.feature_grids, source):
            if not torch.equal(grid.params.detach(), original):
                raise RuntimeError("Frozen source latent changed during adaptation")
        for parameter in model.network.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("Nonfinite decoder parameters after adaptation")
        return {"steps": steps, "lr_multiplier": lr_multiplier,
                "seconds": time.perf_counter() - started, "frozen_latent_verified": True}
    finally:
        model._decoder_adaptation_features = None
        model.scheduler = old_scheduler
        model.train(was_training)
        trainer.configs.astc_codec_in_loop_enable = old_codec
        if trainer.super_resolution_enable:
            trainer.dataset.superres_base_cache[0] = old_base
        for grid, requires_grad in zip(model.feature_grids, old_requires_grad):
            grid.params.requires_grad_(requires_grad)
        for group, lr in zip(model.optimizer.param_groups, old_lrs):
            group["lr"] = lr
        for name, value in previous.items():
            setattr(trainer, name, value)
        model._qat_param_cache.clear()
        model._qat_param_cache_iter = -1


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--finetune_steps", type=int, default=1000)
    extra.add_argument("--finetune_lr_multiplier", type=float, default=0.5)
    options, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    from configs import get_args
    from train import Trainer

    params = get_args()
    if params.mode != "train" or params.groups_batching:
        raise ValueError("This entry requires single-material train mode")
    trainer = Trainer(params)
    try:
        # The standalone entry owns the post-training phase explicitly.
        trainer.astc_decoder_finetune_enable = False
        trainer.train()
        baseline_iteration = trainer.model.current_iter
        trainer.model.save(baseline_iteration, trainer.model_path)
        baseline = trainer.run_astc_comparison(curr_iter=baseline_iteration)
        baseline_pbr = trainer.run_pbr_comparison(curr_iter=baseline_iteration)
        stats = finetune_astc_decoder(trainer, options.finetune_steps, options.finetune_lr_multiplier)
        final_iteration = baseline_iteration + options.finetune_steps
        trainer.model.save(final_iteration, trainer.model_path)
        adapted = trainer.run_astc_comparison(curr_iter=final_iteration)
        adapted_pbr = trainer.run_pbr_comparison(curr_iter=final_iteration)
        Path(trainer.save_path, "astc_decoder_finetune.json").write_text(
            json.dumps({"baseline_iteration": baseline_iteration,
                        "adapted_iteration": final_iteration,
                        "baseline": baseline, "baseline_pbr": baseline_pbr,
                        "adapted": adapted, "adapted_pbr": adapted_pbr,
                        "adaptation": stats}, indent=2),
            encoding="utf-8",
        )
    finally:
        trainer.writer.close()


if __name__ == "__main__":
    main()
