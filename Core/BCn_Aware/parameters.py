"""BC1 algorithm-specific binary usage objective and diagnostics."""

import torch


def binary_regularization(trainer):
    active = getattr(trainer.model, '_codec_active_grids', None)
    losses = [p.binary_usage_loss() for i, p in enumerate(trainer.model.grid_proxies)
              if hasattr(p, "binary_usage_loss") and (active is None or i in active)]
    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    trainer._record_training_scalar("BC1/binary_usage_loss", loss)
    if trainer.model.current_iter % 100 == 0:
        for index, proxy in enumerate(trainer.model.grid_proxies):
            if not hasattr(proxy, "usage_stats"):
                continue
            for name, value in proxy.usage_stats().items():
                trainer._record_training_scalar(f"BC1/grid{index}_{name}", value)
    return trainer.configs.bcn_usage_weight * loss
