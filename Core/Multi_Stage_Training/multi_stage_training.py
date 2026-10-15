"""Multi-stage training for ASTC: representation learning, decoder fitting and layout refitting.

Run as a module or through train.py with the same CLI/YAML options:
    python -m Core.Multi_Stage_Training.multi_stage_training --data_dir data/test
"""

import copy
import json
import os
import math
import time

import torch

from configs import Config, get_args
from model import TCNNModel
from train import Trainer


def fit_astc_decoder(trainer, steps: int = 1000, lr_multiplier: float = 0.5,
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
              "_early_stop_phase_start_iter",
              "enable_astc_compare", "enable_pbr_compare", "_phase_control")
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
        model.optimizer.param_groups[0]["lr"] *= lr_multiplier
        # Validation-window LR reductions replace the per-batch scheduler.
        model.scheduler = None
        trainer.trained_iter = model.current_iter + 1
        trainer.max_iter = trainer.trained_iter + steps
        trainer._reset_training_phase("decoder", trainer.trained_iter)
        trainer.early_stop = early_stop
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
                "seconds": time.perf_counter() - started, "frozen_latent_verified": True,
                "validation": trainer.last_training_stats}
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


def _record_stage(trainer, stages, name, start_iteration, details):
    """Save one named stage using a common checkpoint and metric schema."""
    model = trainer.model
    iteration = int(model.current_iter)
    model.save(iteration, trainer.model_path)
    astc = trainer.run_astc_comparison(curr_iter=iteration)
    pbr = (trainer.run_pbr_comparison(curr_iter=iteration, return_metrics=True)[1]
           if trainer.enable_pbr_compare else None)
    record = {
        "name": name, "status": "completed",
        "start_iteration": start_iteration, "end_iteration": iteration,
        "checkpoint": os.path.join(trainer.model_path, f"train_result_{iteration}", "model.pth"),
        "model": {"hidden_layers": model.n_hidden_layers, "neurons": model.n_neurons,
                  "pe_frequencies": model.n_frequencies,
                  "feature_gradient_count": model.feature_gradient_count},
        "metrics": {"astc": astc, "pbr": pbr}, "details": details,
    }
    stages.append(record)
    return record


def run_multi_stage_training(params, config=None):
    """Run representation, decoder and optional layout stages without numeric stage names."""
    target = config if config is not None else Config(params)
    if not target.mip0_only or target.direct_diffuse_infer_mode != "disable":
        raise ValueError("Multi-stage ASTC training requires Mip0 and direct diffuse disabled")
    if target.n_hidden_layers < 0 or target.multi_stage_training_decoder_steps < 1:
        raise ValueError("Decoder stage requires nonnegative hidden layers and positive fitting steps")
    if target.early_stop and (target.early_stop_interval <= 0
                              or target.early_stop_interval % target.eval_interval != 0):
        raise ValueError("early_stop_interval must be positive and a multiple of eval_interval")

    representation_config = copy.deepcopy(target)
    representation_config.n_hidden_layers = 0
    representation_config.n_frequencies = 0
    representation_config.feature_gradient_count = 0
    trainer = Trainer(params, config_override=representation_config)
    stages = []
    try:
        print("[Multi Stage Training] representation: affine decoder, no PE or feature differences", flush=True)
        start = trainer.trained_iter
        trainer.train()
        representation = _record_stage(trainer, stages, "representation", start,
                                       {"validation": trainer.last_training_stats})
        os.replace(os.path.join(trainer.save_path, "resolved_config.json"),
                   os.path.join(trainer.save_path, "representation_config.json"))

        source = trainer.model
        target.num_mips = representation_config.num_mips
        target.num_channels = representation_config.num_channels
        target.pos_encoding_reference_edge = representation_config.pos_encoding_reference_edge
        decoder = TCNNModel(target)
        decoder.astc_proxy = copy.deepcopy(source.astc_proxy)
        decoder.astc_proxy.requires_grad_(False)
        with torch.no_grad():
            for old_grid, new_grid in zip(source.feature_grids, decoder.feature_grids):
                new_grid.params.copy_(old_grid.params)
        decoder.current_iter = representation["end_iteration"]
        trainer.model, trainer.configs = decoder, target
        trainer._write_resolved_config(target)
        print(f"[Multi Stage Training] decoder: {target.n_hidden_layers} hidden layers, width {target.n_neurons}, "
              f"{target.n_frequencies} PE frequencies, {target.feature_gradient_count} feature differences", flush=True)
        start = decoder.current_iter + 1
        details = fit_astc_decoder(trainer, steps=target.multi_stage_training_decoder_steps,
                                   lr_multiplier=target.multi_stage_training_decoder_lr_multiplier,
                                   early_stop=target.early_stop)
        fitted = _record_stage(trainer, stages, "decoder", start, details)

        from .layout import refit_layout
        print("[Multi Stage Training] layout_refit", flush=True)
        details = refit_layout(trainer, target.multi_stage_training_layout_refit_steps)
        if details['applied']:
            start = decoder.current_iter + 1
            decoder.current_iter += details['steps']
            final = _record_stage(trainer, stages, "layout_refit", start, details)
        else:
            stages.append({"name": "layout_refit", "status": "skipped", "details": details})
            final = fitted
        results = {"stages": stages, "final_iteration": final["end_iteration"],
                   "final_checkpoint": final["checkpoint"], "final_metrics": final["metrics"]}
        with open(os.path.join(trainer.save_path, "multi_stage_training.json"), "w", encoding="utf-8") as output:
            json.dump(results, output, indent=2)
        return results
    finally:
        trainer.writer.close()


if __name__ == "__main__":
    run_multi_stage_training(get_args())
