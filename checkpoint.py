"""Checkpoint files, training progress and exact resumption of the current format."""
import copy
import hashlib
import os
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


def checkpoint_paths(directory, iteration):
    root = Path(directory) / 'models'
    if iteration != -1:
        return [root / f'train_result_{iteration}' / 'model.pth']
    paths = [p / 'model.pth' for p in root.glob('train_result_*')
             if p.name.removeprefix('train_result_').isdigit() and (p / 'model.pth').is_file()]
    if not paths:
        raise FileNotFoundError(f'No checkpoint found in {root}')
    return sorted(paths, key=lambda p: int(p.parent.name.removeprefix('train_result_')), reverse=True)


def file_digest(path):
    with open(path, 'rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class TrainingProgress:
    stage: str = 'single'
    status: str = 'running'
    stages: list = field(default_factory=list)

    def __post_init__(self):
        if self.stage not in ('single', 'representation', 'decoder', 'layout_refit', 'material_cnn'):
            raise ValueError(f'Unknown training stage: {self.stage}')
        if self.status not in ('running', 'stage_complete', 'pipeline_complete'):
            raise ValueError(f'Unknown training progress status: {self.status}')

    @property
    def complete(self):
        return self.status != 'running'


@dataclass(frozen=True)
class TrainingCheckpoint:
    model_path: Path
    state: dict


# Only the saved stage's architecture and resolved raster settings override
# the caller. Reporting, output paths and other run options stay explicit.
PHASE_CONFIG_KEYS = (
    'n_hidden_layers', 'n_frequencies', 'feature_grid_configs', 'material_cnn_enable',
    'num_mips', 'num_channels', 'pos_encoding_reference_edge',
    'material_cnn_inference_size', 'material_reconstruction_size',
)


def phase_config(target, checkpoint):
    config = copy.deepcopy(target)
    for key, value in checkpoint.state['phase_config'].items():
        setattr(config, key, value)
    return config


def config_values(config):
    return {k: copy.deepcopy(v) for k, v in vars(config).items()
            if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))}


def training_recipe(config):
    ignored = {'load_dir', 'load_iter', 'save_dir', 'data_dir', 'mode', 'device', 'warm_start',
               'num_channels', 'num_mips', 'pos_encoding_reference_edge', 'material_reconstruction_size',
               'material_cnn_inference_size', 'output_loss_weights', 'save_interval',
               'enable_material_compare', 'material_compare_interval', 'eval_inference_tile'}
    result = {k: v for k, v in config_values(config).items() if k not in ignored and not k.startswith('groups_')}
    result['grid_training'] = [asdict(p) for p in config.grid_training]
    return result


def data_contract(directory):
    return {p.name: file_digest(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff')}


def read_resume(config):
    if config.mode != 'train' or not config.load_dir or config.warm_start:
        return None
    paths = checkpoint_paths(config.load_dir, config.load_iter)
    failure = None
    for model_path in paths:
        path = model_path.with_name('training_state.pth')
        if not path.is_file():
            failure = ValueError(f'{model_path} contains weights only; use --warm_start for a new run, '
                                 'or select a checkpoint containing training_state.pth')
        else:
            try:
                state = torch.load(path, map_location='cpu', weights_only=True)
                if file_digest(model_path) != state['model_sha256']:
                    raise ValueError('Model weights do not match the training checkpoint')
            except (EOFError, RuntimeError, KeyError, OSError, pickle.UnpicklingError, ValueError) as exc:
                failure = exc
            else:
                if state.get('version') != 2:
                    raise ValueError('Unsupported training checkpoint version; expected 2')
                expected = training_recipe(config)
                if state['recipe'] != expected:
                    differences = [k for k in state['recipe'].keys() | expected.keys()
                                   if state['recipe'].get(k) != expected.get(k)]
                    raise ValueError('Resume training recipe differs: '+', '.join(sorted(differences))+
                                     '; use --warm_start to start a different experiment')
                if data_contract(config.data_dir) != state['data']:
                    raise ValueError('Resume texture data differs from the saved training input')
                return TrainingCheckpoint(model_path, state)
        if config.load_iter != -1:
            break
    raise ValueError(f'No complete training checkpoint: {failure}') from failure


def save_training_state(trainer, iteration):
    model = trainer.model
    path = Path(trainer.model_path)/f'train_result_{iteration}'/'training_state.pth'
    numpy_rng = np.random.get_state()
    state = dict(
        version=2, model_sha256=file_digest(path.with_name('model.pth')),
        recipe=trainer._training_recipe,
        phase_config={k: getattr(trainer.configs, k) for k in PHASE_CONFIG_KEYS},
        data=trainer._checkpoint_data, iteration=iteration, progress=asdict(trainer.progress),
        max_iter=trainer.max_iter, phase=trainer._phase_control.state_dict() if trainer._phase_control else None,
        last_training_stats=trainer.last_training_stats,
        optimizer=model.optimizer.state_dict(),
        scheduler=model.scheduler.state_dict() if model.scheduler is not None else None,
        codec=trainer._grid_compression_optimizer.state_dict() if trainer._grid_compression_optimizer is not None else None,
        frozen_features=model._decoder_adaptation_features is not None,
        decoder_source_grad=getattr(trainer, '_decoder_source_grad', None),
        requires_grad={name: p.requires_grad for name, p in model.named_parameters()},
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate(), numpy_rng=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
    )
    atomic_torch_save(state, path)


def restore_training_state(trainer, state):
    model = trainer.model
    trainer.trained_iter, trainer.max_iter = state['iteration']+1, state['max_iter']
    trainer.last_training_stats = state['last_training_stats']
    model.current_iter = state['iteration']
    model._codec_active_grids = tuple(i for i, policy in enumerate(trainer.configs.grid_training)
                                      if policy.active(model.current_iter, trainer.configs.max_iter))
    model.optimizer.load_state_dict(state['optimizer'])
    if state['scheduler'] is None:
        model.scheduler = None
    else:
        model.scheduler.load_state_dict(state['scheduler'])
    if state['codec'] is not None:
        trainer._ensure_grid_proxies()
        trainer._grid_compression_optimizer.load_state_dict(state['codec'])
    if state['phase'] is not None:
        trainer._reset_training_phase(state['phase']['phase'], state['phase']['start'])
        trainer._phase_control.load_state_dict(state['phase'])
    if state['frozen_features']:
        from Core.Feature_Grid.training import prepare_decoder_inputs
        prepare_decoder_inputs(trainer)
        trainer._decoder_source_grad = state['decoder_source_grad']
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(state['requires_grad'][name])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    n = state['numpy_rng']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    model._qat_param_cache.clear()
    print(f"[Resume] {trainer.progress.stage}: next iteration {trainer.trained_iter}, status={trainer.progress.status}")
