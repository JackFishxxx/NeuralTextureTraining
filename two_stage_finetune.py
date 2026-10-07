"""Train a zero-layer feature representation, then a new decoder on fixed ASTC features.

Stage two uses n_neurons, n_hidden_layers and positional encoding from the
resolved config. Both stages honor early stopping. Run with the same options
as train.py, for example: python two_stage_finetune.py --data_dir data/test
"""

import copy
import json
import os
import math
import time
from types import SimpleNamespace

import torch

from configs import Config, get_args
from model import TCNNModel
from train import Trainer


def finetune_astc_decoder(trainer, steps: int = 1000, lr_multiplier: float = 0.5,
                          early_stop: bool = False):
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
              "_early_stop_phase_start_iter", "_early_stop_eval_count",
              "_early_stop_avg_psnr", "_early_stop_prev_psnr",
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
        # Use a fixed learning rate for the newly initialized decoder.
        model.optimizer.param_groups[0]["lr"] *= lr_multiplier
        model.scheduler = SimpleNamespace(step=lambda **kwargs: None)
        trainer.trained_iter = model.current_iter + 1
        trainer.max_iter = trainer.trained_iter + steps
        trainer._early_stop_phase_start_iter = trainer.trained_iter
        if early_stop:
            trainer._early_stop_eval_count = 0
            trainer._early_stop_avg_psnr = 0.0
            trainer._early_stop_prev_psnr = 0.0
            trainer.early_stop = True
            trainer.enable_astc_compare = True
            trainer.enable_pbr_compare = True
        else:
            trainer.early_stop = False
        model.train()
        trainer.train()
        torch.cuda.synchronize()
        for grid, original in zip(model.feature_grids, source):
            if not torch.equal(grid.params.detach(), original):
                raise RuntimeError("Frozen source latent changed during adaptation")
        for parameter in model.network.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("Nonfinite decoder parameters after adaptation")
        completed_steps = model.current_iter - trainer.trained_iter + 1
        return {"steps": completed_steps, "max_steps": steps,
                "early_stopped": completed_steps < steps,
                "lr_multiplier": lr_multiplier,
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


def train_two_stage(params, config=None):
    target = config if config is not None else Config(params)
    if not target.mip0_only or target.direct_diffuse_infer_mode != "disable":
        raise ValueError("Two-stage ASTC training requires Mip0 and direct diffuse disabled")
    if target.n_hidden_layers < 0 or target.two_stage_finetune_max_steps < 1:
        raise ValueError("Stage two requires nonnegative hidden layers and positive fine-tune steps")
    if target.early_stop and (target.early_stop_interval <= 0
                              or target.early_stop_interval % target.eval_interval != 0):
        raise ValueError("early_stop_interval must be positive and a multiple of eval_interval")

    first = copy.deepcopy(target)
    first.n_hidden_layers = 0
    first.n_frequencies = 0
    trainer = Trainer(params, config_override=first)
    try:
        print("[Two Stage] feature stage: 0 hidden layers, no positional encoding", flush=True)
        trainer.train()
        baseline_iteration = int(trainer.model.current_iter)
        trainer.model.save(baseline_iteration, trainer.model_path)
        baseline_astc = trainer.run_astc_comparison(curr_iter=baseline_iteration)
        baseline_pbr = trainer.run_pbr_comparison(curr_iter=baseline_iteration) \
            if trainer.enable_pbr_compare else None

        source = trainer.model
        target.num_mips = first.num_mips
        target.num_channels = first.num_channels
        target.pos_encoding_reference_edge = first.pos_encoding_reference_edge
        second = TCNNModel(target)
        second.astc_proxy = copy.deepcopy(source.astc_proxy)
        for parameter in second.astc_proxy.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            for old_grid, new_grid in zip(source.feature_grids, second.feature_grids):
                new_grid.params.copy_(old_grid.params)
        second.current_iter = baseline_iteration
        trainer.model = second
        trainer.configs = target
        print(
            f"[Two Stage] decoder stage: {target.n_hidden_layers} hidden layers, "
            f"width {target.n_neurons}, {target.n_frequencies} PE frequencies; "
            f"source latent frozen at iter {baseline_iteration}", flush=True,
        )
        stats = finetune_astc_decoder(
            trainer,
            steps=target.two_stage_finetune_max_steps,
            lr_multiplier=target.two_stage_finetune_lr_multiplier,
            early_stop=target.early_stop,
        )
        final_iteration = int(second.current_iter)
        second.save(final_iteration, trainer.model_path)
        adapted_astc = trainer.run_astc_comparison(curr_iter=final_iteration)
        adapted_pbr = trainer.run_pbr_comparison(curr_iter=final_iteration) \
            if trainer.enable_pbr_compare else None
        results = {
            "baseline_iteration": baseline_iteration,
            "adapted_iteration": final_iteration,
            "stage_one": {"hidden_layers": 0, "pe_frequencies": 0},
            "stage_two": {"hidden_layers": target.n_hidden_layers,
                          "neurons": target.n_neurons,
                          "pe_frequencies": target.n_frequencies},
            "baseline": baseline_astc,
            "baseline_pbr": baseline_pbr,
            "adapted": adapted_astc,
            "adapted_pbr": adapted_pbr,
            "adaptation": stats,
        }
        with open(os.path.join(trainer.save_path, "two_stage_finetune.json"), "w", encoding="utf-8") as output:
            json.dump(results, output, indent=2)
        return results
    finally:
        trainer.writer.close()


if __name__ == "__main__":
    train_two_stage(get_args())
