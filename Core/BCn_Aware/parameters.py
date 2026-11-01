"""BC1 algorithm-specific binary usage objective and diagnostics."""

import torch


def binary_regularization(trainer):
    active = getattr(trainer.model, '_codec_active_grids', None)
    losses = [p.binary_usage_loss() for i, p in enumerate(trainer.model.astc_proxy)
              if hasattr(p, "binary_usage_loss") and (active is None or i in active)]
    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    trainer.writer.add_scalar(
        "BC1/binary_usage_loss", loss.detach().item(), trainer.model.current_iter
    )
    if trainer.model.current_iter % 100 == 0:
        for index, proxy in enumerate(trainer.model.astc_proxy):
            if not hasattr(proxy, "usage_stats"):
                continue
            for name, value in proxy.usage_stats().items():
                trainer.writer.add_scalar(
                    f"BC1/grid{index}_{name}", value, trainer.model.current_iter
                )
    return trainer.configs.bcn_usage_weight * loss
