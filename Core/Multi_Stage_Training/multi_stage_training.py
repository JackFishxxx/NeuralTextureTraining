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
from .layout import refit_layout


def fit_astc_decoder(trainer, steps: int = 1000, lr_multiplier: float = 0.5,
                     early_stop: bool = False, final_report: bool = True):
    """Fit a decoder on immutable codec inputs; resume follows the same lifecycle."""
    from Core.Feature_Grid.training import prepare_decoder_inputs, release_decoder_inputs

    if steps < 1 or not math.isfinite(lr_multiplier) or lr_multiplier <= 0:
        raise ValueError('steps and lr_multiplier must be positive')
    model = trainer.model
    if not trainer.mip0_only or not model.quantize or model.direct_diffuse_enabled:
        raise ValueError('Decoder fitting requires quantized Mip0 without direct diffuse')
    if trainer.super_resolution_enable and trainer.astc_superres_base_mip0 is None:
        raise ValueError('A deployment ASTC base is required for decoder fitting')
    source = [grid.params.detach().clone() for grid in model.feature_grids]
    was_training = model.training
    started = time.perf_counter()
    if model._decoder_adaptation_features is None:
        if not 0 <= model.current_iter < trainer.max_iter:
            raise ValueError('Decoder fitting requires a completed representation iteration')
        prepare_decoder_inputs(trainer)
        trainer.progress.stage, trainer.progress.status = 'decoder', 'running'
        model.optimizer.param_groups[0]['lr'] *= lr_multiplier
        model.scheduler = None  # Decoder validation windows own this stage's LR.
        trainer.trained_iter = model.current_iter+1
        trainer.max_iter = trainer.trained_iter+steps
        trainer._reset_training_phase('decoder', trainer.trained_iter)
    start_iteration = trainer._phase_control.start
    trainer.early_stop = early_stop
    try:
        model.train()
        trainer.train(final_report=final_report)
        for grid, original in zip(model.feature_grids, source):
            if not torch.equal(grid.params.detach(), original):
                raise RuntimeError('Frozen source latent changed during decoder fitting')
        if not all(torch.isfinite(parameter).all() for parameter in model.network.parameters()):
            raise RuntimeError('Nonfinite decoder parameters after fitting')
        completed = model.current_iter-start_iteration+1
        return dict(steps=completed, max_steps=steps, early_stopped=completed < steps,
                    lr_multiplier=lr_multiplier, seconds=time.perf_counter()-started,
                    frozen_latent_verified=True, validation=trainer.last_training_stats)
    finally:
        release_decoder_inputs(trainer)
        model.train(was_training)


def _record_stage(trainer, name, start_iteration, details, status='completed'):
    """Save one named stage using a common checkpoint and metric schema."""
    model = trainer.model
    iteration = int(model.current_iter)
    trainer._metrics_phase = name
    trainer.progress.stage = name
    deployment = trainer.deployment_metrics()
    astc = trainer.run_astc_comparison(curr_iter=iteration) if trainer.enable_astc_compare else None
    pbr = (trainer.run_pbr_comparison(curr_iter=iteration, return_metrics=True)[1]
           if trainer.enable_pbr_compare else None)
    record = {
        "name": name, "status": status,
        "start_iteration": start_iteration, "end_iteration": iteration,
        "checkpoint": os.path.join(trainer.model_path, f"train_result_{iteration}", "model.pth"),
        "model": {"hidden_layers": model.n_hidden_layers, "neurons": model.n_neurons,
                  "pe_frequencies": model.n_frequencies,
                  "feature_gradient_count": model.feature_gradient_count,
                  "material_cnn_enable": model.material_cnn is not None},
        "metrics": {"deployment": deployment, "astc": astc, "pbr": pbr}, "details": details,
        "metric_reports": {
            "astc": os.path.join(trainer.media_path, f'Compare_ASTC{trainer.astc_block.lower()}',
                                  f'iter_{iteration:08d}', 'metrics.json') if trainer.enable_astc_compare else None,
            "pbr": os.path.join(trainer.media_path, 'Compare_PBRScene', f'iter_{iteration:08d}', 'metrics.json')
                   if trainer.enable_pbr_compare else None,
        },
    }
    trainer.progress.stages.append(record)
    trainer.save_checkpoint(iteration, completed=True)
    _write_stage_report(trainer)
    return record


def _stage_report(progress):
    stages = progress.stages
    latest = next((stage for stage in reversed(stages) if 'checkpoint' in stage), None)
    report = dict(status='completed' if progress.status == 'pipeline_complete' else 'running', stages=stages)
    if latest is not None:
        report.update(final_iteration=latest['end_iteration'], final_checkpoint=latest['checkpoint'],
                      final_metrics=latest['metrics'])
    return report


def _write_stage_report(trainer):
    report = _stage_report(trainer.progress)
    path = os.path.join(trainer.save_path, 'multi_stage_training.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    os.replace(path + '.tmp', path)
    return report


def _finish_pipeline(trainer):
    """Commit completion explicitly, after the final deployment assets exist."""
    from checkpoint import save_training_state
    trainer.progress.status = 'pipeline_complete'
    iteration = trainer.model.current_iter
    path = os.path.join(trainer.model_path, f'train_result_{iteration}', 'model.pth')
    if os.path.isfile(path):
        save_training_state(trainer, iteration)
    else:
        latest = next(s for s in reversed(trainer.progress.stages) if 'checkpoint' in s)
        latest['checkpoint'] = path
        trainer.save_checkpoint(iteration, completed=True)
    return _write_stage_report(trainer)


def _finish_material_cnn(trainer, steps, resume_state=None):
    print('[Multi Stage Training] material_cnn', flush=True)
    trainer.progress.stage, trainer.progress.status = 'material_cnn', 'running'
    start = trainer.model.current_iter + 1
    details = trainer.fit_material_cnn(steps, resume_state=resume_state)
    progress_path = details.pop('progress_checkpoint')
    trainer.model.current_iter += details['steps']
    trainer.configs.material_cnn_enable = trainer.model.material_cnn is not None
    trainer._write_resolved_config(trainer.configs)
    _record_stage(trainer, 'material_cnn', start, details,
                  status='completed' if details['accepted'] else 'rejected')
    trainer.save_compressed_materials(trainer.media_path, trainer.model.current_iter)
    # Completion means all final artifacts exist. Keep recovery data on failure.
    report = _finish_pipeline(trainer)
    os.remove(progress_path)
    return report


def _resume_material_cnn(params, target, path, checkpoint):
    """Resume only the interrupted CNN stage from its frozen source and optimizer."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['version'] != 2:
        raise ValueError('Unsupported material CNN training checkpoint version')
    if not target.material_cnn_enable:
        raise ValueError('Resuming the material CNN stage requires material_cnn_enable')
    if state['source_iteration'] != checkpoint.state['iteration']:
        raise ValueError('Material CNN progress does not match the selected source checkpoint')
    from checkpoint import TrainingProgress, phase_config
    trainer = Trainer(params, config_override=phase_config(target, checkpoint),
                      deployment_config=target, checkpoint=checkpoint)
    try:
        trainer.model.load_state_dict(state['source_model'])
        trainer.progress = TrainingProgress(**state['progress'])
        _write_stage_report(trainer)
        return _finish_material_cnn(trainer, target.material_cnn_steps, resume_state=state)
    finally:
        trainer.writer.close()


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

    from checkpoint import TrainingProgress, phase_config, read_resume
    checkpoint = read_resume(target)
    if checkpoint and checkpoint.state['progress']['status'] == 'pipeline_complete':
        return _stage_report(TrainingProgress(**checkpoint.state['progress']))
    if checkpoint and target.load_iter == -1 and checkpoint.state['progress']['stage'] != 'material_cnn':
        progress = os.path.join(target.load_dir, 'models', 'material_cnn_training.pth')
        if os.path.isfile(progress):
            return _resume_material_cnn(params, target, progress, checkpoint)
    config = phase_config(target, checkpoint) if checkpoint else copy.deepcopy(target)
    if checkpoint is None:
        config.n_hidden_layers = config.n_frequencies = config.feature_gradient_count = 0
        config.material_cnn_enable = False
    trainer = Trainer(params, config_override=config, deployment_config=target, checkpoint=checkpoint)
    if checkpoint is None:
        trainer.progress.stage = 'representation'
    try:
        if trainer.progress.stage == 'representation':
            if not trainer.progress.complete:
                print('[Multi Stage Training] representation', flush=True)
                start = trainer._phase_control.start if trainer._phase_control else trainer.trained_iter
                trainer.train(final_report=False)
                _record_stage(trainer, 'representation', start, {'validation': trainer.last_training_stats})
            os.replace(os.path.join(trainer.save_path, 'resolved_config.json'),
                       os.path.join(trainer.save_path, 'representation_config.json'))
            _create_decoder(trainer, target)
        if trainer.progress.stage == 'decoder':
            if not trainer.progress.complete:
                start = trainer._phase_control.start if trainer._phase_control else trainer.model.current_iter+1
                details = fit_astc_decoder(trainer, steps=target.multi_stage_training_decoder_steps,
                                           lr_multiplier=target.multi_stage_training_decoder_lr_multiplier,
                                           early_stop=target.early_stop, final_report=False)
                _record_stage(trainer, 'decoder', start, details)
            print('[Multi Stage Training] layout_refit', flush=True)
            details = refit_layout(trainer, target.multi_stage_training_layout_refit_steps)
            if details['applied']:
                start = trainer.model.current_iter+1
                trainer.model.current_iter += details['steps']
                _record_stage(trainer, 'layout_refit', start, details)
            else:
                trainer.progress.stages.append({'name': 'layout_refit', 'status': 'skipped', 'details': details})
                _write_stage_report(trainer)
        if trainer.progress.stage == 'material_cnn':
            # A completed CNN checkpoint can precede the completion report.
            trainer.save_compressed_materials(trainer.media_path, trainer.model.current_iter)
            return _finish_pipeline(trainer)
        if target.material_cnn_enable:
            return _finish_material_cnn(trainer, target.material_cnn_steps)
        trainer.save_compressed_materials(trainer.media_path, trainer.model.current_iter)
        return _finish_pipeline(trainer)
    finally:
        trainer.writer.close()


def _create_decoder(trainer, target):
    from Core.Feature_Grid.compression import freeze_proxies

    config = copy.deepcopy(target)
    for key in ('num_mips', 'num_channels', 'pos_encoding_reference_edge',
                'material_cnn_inference_size', 'material_reconstruction_size'):
        setattr(config, key, getattr(trainer.configs, key))
    config.load_dir, config.load_iter, config.material_cnn_enable = None, 0, False
    source, decoder = trainer.model, TCNNModel(config)
    with torch.no_grad():
        for old, new in zip(source.feature_grids, decoder.feature_grids):
            new.params.copy_(old.params)
    decoder.astc_proxy = copy.deepcopy(source.astc_proxy)
    freeze_proxies(decoder)
    decoder.current_iter = source.current_iter
    trainer.model, trainer.configs = decoder, config
    trainer._astc_block_optimizer = trainer._phase_control = None
    trainer.progress.stage, trainer.progress.status = 'decoder', 'running'
    trainer._write_resolved_config(config)


if __name__ == "__main__":
    run_multi_stage_training(get_args())
