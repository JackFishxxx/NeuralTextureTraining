"""Current checkpoint contracts, proxy reconstruction and deployment metadata."""

import json
import math
import torch
from .compression import SourceGrid, astc_codec_for, plans_for
from .layout import surface_filename


def inference_spec(model):
    """The numeric interpretation and input/output layout of a deployed model."""
    cnn = model.material_cnn
    from dataset import get_canonical_channel_slices_static
    return dict(
        version=1,
        grids=[dict(**recipe.as_config(), resolution=spec.max_resolution, levels=spec.n_levels,
                    channels=spec.n_features_per_level, quantize_bits=spec.quantize_bits,
                    feature_gradient_count=spec.feature_gradient_count)
               for recipe, spec in zip(model.grid_compressions, model.feature_grid_specs)],
        num_mips=model.num_mips, mip0_only=model.mip0_only,
        normal_encoding=model.normal_encoding, output_channels=model.num_channels,
        output_layout={name: list(bounds) for name, bounds in get_canonical_channel_slices_static(model.normal_encoding).items()},
        output_activation=model.output_activation,
        positional_encoding=dict(type=model.pos_encoding_type, frequencies=model.n_frequencies,
                                 tile_uv=model.pos_encoding_tile_uv),
        decoder=dict(hidden_layers=model.n_hidden_layers, neurons=model.n_neurons),
        super_resolution=dict(enabled=model.super_resolution_enable,
                              base_resolution=model.super_resolution_base_resolution,
                              codec=model.sr_base_codec if model.super_resolution_enable else None,
                              combine='clamp_base_plus_residual_then_bilinear_repeat'),
        direct_diffuse=model.direct_diffuse_infer_mode,
        material_size=list(model.material_reconstruction_size),
        inference_size=list(model.inference_size),
        input_order=['positional_encoding', 'grid_centers_and_differences', 'normalized_mip']
                    + (['canonical_sr_base'] if model.super_resolution_enable else []) + ['constant_one'],
        feature_sampling='bilinear_repeat_texel_center',
        cnn=None if cnn is None else dict(version=cnn.version, mode=cnn.mode, channels=cnn.channels,
                                          hidden=cnn.hidden, rounds=cnn.rounds,
                                          tile_size=int(cnn.tile_size),
                                          inference_size=cnn.inference_size.tolist()),
    )


def _json_tensor(model, value):
    return torch.tensor(list(json.dumps(value, sort_keys=True).encode()),
                        dtype=torch.uint8, device=next(model.parameters()).device)


def _read_metadata(state, key, expected_type):
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(f'Checkpoint requires {key} as a one-dimensional uint8 tensor')
    try:
        result = json.loads(bytes(value.cpu().tolist()).decode())
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid checkpoint metadata: {key}') from exc
    if not isinstance(result, expected_type):
        raise ValueError(f'Checkpoint {key} must contain a JSON {expected_type.__name__}')
    return result


def refresh_inference_contract(model):
    """Refresh only when the model structure changes, never as a side effect of saving."""
    value = _json_tensor(model, inference_spec(model))
    if hasattr(model, '_inference_contract'):
        model._inference_contract = value
    else:
        model.register_buffer('_inference_contract', value)


def validate_contract(model, state):
    if '_inference_contract' not in state:
        raise ValueError('Checkpoint requires the current inference contract')
    saved = _read_metadata(state, '_inference_contract', dict)
    expected = inference_spec(model)
    if saved != expected:
        differences = [key for key in saved.keys() | expected.keys() if saved.get(key) != expected.get(key)]
        raise ValueError('Checkpoint inference configuration does not match: '+', '.join(sorted(differences)))


def proxy_state_kind(proxy):
    return 'astc_packed' if hasattr(proxy, 'surfaces') else proxy.proxy_kind


def proxy_layout_tensor(model):
    return _json_tensor(model, [proxy_state_kind(proxy) for proxy in model.grid_proxies])


def _validate_proxy(model, index, kind, state):
    """Validate the saved representation against its grid before allocating it."""
    if kind == 'source':
        return
    spec, recipe = model.feature_grid_specs[index], model.grid_compressions[index]
    resolution, channels = spec.max_resolution, spec.n_features_per_level
    prefix = f'grid_proxies.{index}.'
    plans = plans_for(model, index)
    shapes = {}
    if kind == 'bc1_rgba':
        if state[prefix+'shape'].tolist() != [resolution, resolution, channels]:
            raise ValueError('BC1 checkpoint dimensions do not match grid')
        if int(state[prefix+'bc1_alpha_version']) != 1:
            raise ValueError('Unsupported BC1 RGBA checkpoint version')
        blocks = math.ceil(resolution/4)**2 * (channels//4)
        shapes = {'endpoints': (blocks, 2, 3), 'weights': (blocks, 16, 4), 'modes': (blocks, 2)}
    elif kind == 'astc_native':
        if recipe.astc_block != '6x6' or channels != 4 or len(plans) != 1:
            raise ValueError('Native ASTC checkpoint requires one four-channel 6x6 surface')
        if state[prefix+'size'].tolist() != [resolution, resolution]:
            raise ValueError('ASTC checkpoint dimensions do not match grid')
        blocks = math.ceil(resolution/6)**2
        shapes = {name: (blocks, *tail) for name, tail in (
            ('endpoints', (32,)), ('weights', (64,)), ('metadata', (16,)),
            ('partitions', (36,)), ('indices', (36, 4)), ('coefficients', (36, 4)))}
        shapes.update(colors=(17, 256), weight_lut=(12, 65))
    else:
        if (kind == 'astc_packed') != (len(plans) > 1):
            raise ValueError('ASTC checkpoint surface layout does not match grid')
        for surface, plan in enumerate(plans):
            name = f'surfaces.{surface}.' if kind == 'astc_packed' else ''
            shape = (resolution, resolution, len(plan.channels))
            shapes[name+'endpoints'] = shape
            if state[prefix+name+'shape'].tolist() != list(shape):
                raise ValueError('ASTC checkpoint dimensions do not match grid')
            if state[prefix+name+'footprint'].tolist() != list(recipe.block):
                raise ValueError('ASTC checkpoint footprint does not match grid')
    for name, shape in shapes.items():
        if tuple(state[prefix+name].shape) != shape:
            raise ValueError(f'Checkpoint {prefix+name} shape does not match grid')


def restore_proxies(model, state):
    from Core.ASTC_Aware.roundtrip_proxy import ASTCRoundtripGrid, PackedASTCRoundtripGrid

    if '_grid_proxy_kinds' not in state:
        raise ValueError('Checkpoint requires explicit grid proxy kinds')
    kinds = _read_metadata(state, '_grid_proxy_kinds', list)
    state.pop('_grid_proxy_kinds')
    if not all(isinstance(kind, str) for kind in kinds):
        raise ValueError('Checkpoint grid proxy kinds must be a list of names')
    count = len(kinds)
    if count not in (0, len(model.grid_compressions)):
        raise ValueError('Checkpoint grid proxy count does not match configuration')
    for index, kind in enumerate(kinds):
        recipe = model.grid_compressions[index]
        allowed = ('source', 'bc1_rgba') if recipe.format == 'bc1' else ('source', 'astc_native', 'astc_roundtrip', 'astc_packed')
        if kind not in allowed:
            raise ValueError(f'Unsupported grid proxy kind: {kind}')
        _validate_proxy(model, index, kind, state)
    if len(model.grid_proxies) == count and all(
        proxy_state_kind(proxy) == kind for proxy, kind in zip(model.grid_proxies, kinds)
    ):
        return
    result = torch.nn.ModuleList()
    for i, kind in enumerate(kinds):
        prefix = f"grid_proxies.{i}."
        if kind == "source":
            proxy = SourceGrid()
        elif kind == "astc_packed":
            plans = plans_for(model, i)
            images = [
                state[f"{prefix}surfaces.{j}.endpoints"].to(model.device) for j in range(len(plans))
            ]
            proxy = PackedASTCRoundtripGrid(
                torch.cat(images, -1), astc_codec_for(model, i), [p.channels for p in plans]
            )
        elif kind == "astc_roundtrip":
            proxy = ASTCRoundtripGrid.from_state(
                state, prefix, model.device, astc_codec_for(model, i)
            )
        else:
            if kind == "astc_native":
                from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid as Proxy
            else:
                from Core.BCn_Aware.alpha_proxy import BC1AlphaProxyGrid as Proxy
            proxy = Proxy.from_state(state, prefix, model.device)
        result.append(proxy)
    model.grid_proxies = result


def deployment_arrays(model):
    manifest = []
    for i, (recipe, spec) in enumerate(zip(model.grid_compressions, model.feature_grid_specs)):
        plans = plans_for(model, i)
        manifest.append(
            dict(
                **recipe.as_config(),
                channels=spec.n_features_per_level,
                binary_channels=(
                    [j for j in range(spec.n_features_per_level) if j % 4 == 3]
                    if recipe.format == "bc1"
                    else []
                ),
                surfaces=[
                    dict(
                        file=surface_filename(i, j, recipe, len(plans)),
                        channels=list(p.channels),
                        block=list(p.block),
                        block_bytes=p.block_bytes,
                    )
                    for j, p in enumerate(plans)
                ],
            )
        )
    import numpy as np

    return dict(
        fntc_tile_size=np.array(model.fntc_tile_size, dtype=np.int32),
        feature_grid_gradient_counts=np.array([spec.feature_gradient_count for spec in model.feature_grid_specs],
                                              dtype=np.int32),
        compression_schema_version=np.array(2, dtype=np.int32),
        feature_grid_compression=np.array(json.dumps(manifest)),
        latent_color_space=np.array("linear_unorm"),
        inference_schema_version=np.array(1, dtype=np.int32),
        inference_spec=np.array(json.dumps(inference_spec(model))),
    )
