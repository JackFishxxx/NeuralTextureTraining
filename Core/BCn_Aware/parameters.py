"""BC1 algorithm-specific binary usage objective and diagnostics."""

import torch


def binary_regularization(trainer):
    from Core.Feature_Grid.compression import proxy_levels
    active = getattr(trainer.model, '_codec_active_grids', None)
    losses = [p.binary_usage_loss() for i, grid in enumerate(trainer.model.grid_proxies)
              for p in proxy_levels(grid)
              if hasattr(p, "binary_usage_loss") and (active is None or i in active)]
    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    trainer._record_training_scalar("BC1/binary_usage_loss", loss)
    if trainer.model.current_iter % 100 == 0:
        for index, grid in enumerate(trainer.model.grid_proxies):
            if active is not None and index not in active:
                continue
            levels = proxy_levels(grid)
            for level, proxy in enumerate(levels):
                if not hasattr(proxy, "usage_stats"):
                    continue
                prefix = f'BC1/grid{index}' + (f'/mip{len(levels)-1-level}' if len(levels) > 1 else '')
                for name, value in proxy.usage_stats().items():
                    trainer._record_training_scalar(f"{prefix}_{name}", value)
    return trainer.configs.bcn_usage_weight * loss
