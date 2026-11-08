"""One per-grid training lifecycle for native ASTC, codec STE and BCn."""

import torch
from Core.Feature_Grid.compression import (
    initialize_grid_proxy,
    SourceGrid,
    proxy_levels,
)


def make_scheduler(optimizer, kind, steps, *, on_sample=False):
    if kind == 'none':
        return None
    if kind == 'plateau':
        if on_sample:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=.85, patience=4,
                threshold=.01, threshold_mode='abs')
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.85, patience=2000)
    if kind == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=.9)
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    if kind == 'poly':
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max(1, steps), power=.9)
    raise ValueError(f'Unsupported learning rate scheduler: {kind}')


class GridCompressionOptimizer:
    def __init__(self, trainer):
        self.trainer = trainer
        model = trainer.model
        recipes = model.grid_compressions
        if not len(model.grid_proxies):
            model.grid_proxies = torch.nn.ModuleList([SourceGrid() for _ in recipes])
        self.optimizers = [None for _ in recipes]
        self.updates = [0 for _ in recipes]
        self.iteration = 0
        self.prepare()

    def prepare(self, indices=None):
        trainer = self.trainer
        model = trainer.model
        active = getattr(model, '_codec_active_grids', None) if indices is None else indices
        if active is None:
            active = range(len(self.optimizers))
        progress = min(1., max(0., model.current_iter/max(1, trainer.configs.max_iter-1)))
        for i, recipe in enumerate(model.grid_compressions):
            enabled = trainer.configs.bcn_aware_enable if recipe.format == 'bc1' else trainer.configs.astc_aware_enable
            if not enabled:
                if model.grid_proxies[i].proxy_kind != 'source':
                    model.grid_proxies[i] = SourceGrid()
                self.optimizers[i] = None
                continue
            if i not in active:
                continue
            proxy = model.grid_proxies[i]
            if self.optimizers[i] is None:
                # A codec may activate after warmup validation already reduced
                # the decoder LR. Start it at the same relative schedule age.
                lr_scale = (model.optimizer.param_groups[0]['lr'] /
                            trainer.configs.network_learning_rate
                            if getattr(model, 'inference_mode', 'on_load') == 'on_sample' else 1.)
                if proxy.proxy_kind == 'source':
                    proxy = initialize_grid_proxy(model, i)
                    model.grid_proxies[i] = proxy
                if hasattr(proxy, 'thaw'):
                    proxy.thaw()
                else:
                    proxy.requires_grad_(True)
                levels = proxy_levels(proxy)
                if all(level.proxy_kind == 'astc_native' for level in levels):
                    from Core.ASTC_Aware.differentiable_proxy import ASTCBlockAdam
                    opt = ASTCBlockAdam(torch.nn.ModuleList(levels), trainer.configs.astc_learning_rate * lr_scale,
                                        trainer.configs.astc_learning_rate * lr_scale, fused=True,
                                        quantization_aware=True, packed_pair_search=True)
                else:
                    lr = trainer.configs.bcn_learning_rate if recipe.format == 'bc1' else trainer.configs.astc_learning_rate
                    opt = torch.optim.Adam(proxy.parameters(), lr=lr * lr_scale)
                self.optimizers[i] = opt
            for level in proxy_levels(proxy):
                if level.proxy_kind == 'bc1_rgba':
                    level.temperature.fill_(.25**progress)

    def zero_grad(self):
        for opt in self.optimizers:
            if opt is not None:
                opt.zero_grad()

    def scale_learning_rates(self, factor):
        """Keep codec/decoder adaptation in sync after validation-driven decay."""
        for opt in self.optimizers:
            if isinstance(opt, torch.optim.Optimizer):
                for group in opt.param_groups:
                    group['lr'] *= factor
            elif opt is not None:
                opt.lrs = tuple(lr * factor for lr in opt.lrs)

    def step(self):
        due = getattr(self.trainer, '_codec_due_grids', range(len(self.optimizers)))
        for index, opt in enumerate(self.optimizers):
            if opt is None or index not in due:
                continue
            if isinstance(opt, torch.optim.Optimizer):
                opt.step()
            elif self.updates[index] % 100 == 0 and hasattr(self.trainer, "_codec_batches"):
                touched = {level: torch.where(
                    (proxy.endpoints.grad.abs().sum(1) + proxy.weights.grad.abs().sum(1)) > 0)[0]
                    for level, proxy in enumerate(opt.grids)}
                opt.step(touched, objective=self.trainer._codec_objective)
            else:
                opt.step_full_grid()
            self.updates[index] += 1
        self.trainer.model._qat_param_cache.clear()
        self.iteration += 1
        interval = self.trainer.configs.astc_decoder_projection_interval
        if (interval > 0 and self.iteration % interval == 0
                and any(i in due and p.proxy_kind == 'astc_native' for i, p in enumerate(self.trainer.model.grid_proxies))
                and self.trainer._supports_astc_projection()):
            self.trainer._project_astc_decoder_target()

    def reset_rows(self, index, rows):
        self.optimizers[index].reset_rows(0, rows)

    def state_dict(self):
        return dict(iteration=self.iteration, updates=self.updates,
                    optimizers=[opt.state_dict() if opt is not None else None for opt in self.optimizers])

    def load_state_dict(self, state):
        if len(state['optimizers']) != len(self.optimizers) or len(state['updates']) != len(self.optimizers):
            raise ValueError('Codec optimizer grid count mismatch')
        self.prepare(indices=tuple(i for i, s in enumerate(state['optimizers']) if s is not None))
        for i, (opt, saved) in enumerate(zip(self.optimizers, state['optimizers'])):
            if saved is not None:
                if opt is None:
                    raise ValueError(f'Codec optimizer for disabled grid {i}')
                opt.load_state_dict(saved)
            else:
                self.optimizers[i] = None
        self.iteration, self.updates = state['iteration'], list(state['updates'])
