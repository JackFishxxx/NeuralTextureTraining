"""One per-grid training lifecycle for native ASTC, codec STE and BCn."""

import torch
from Core.Feature_Grid.compression import (
    initialize_grid_proxy,
    grid_recipes,
    proxy_kind,
    SourceGrid,
)


def make_scheduler(optimizer, kind, steps):
    if kind == 'none':
        return None
    if kind == 'plateau':
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
        recipes = grid_recipes(model)
        if not len(model.astc_proxy):
            model.astc_proxy = torch.nn.ModuleList([SourceGrid() for _ in recipes])
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
        for i, recipe in enumerate(grid_recipes(model)):
            enabled = trainer.configs.bcn_aware_enable if recipe.format == 'bc1' else trainer.configs.astc_aware_enable
            if not enabled:
                if proxy_kind(model.astc_proxy[i]) != 'source':
                    model.astc_proxy[i] = SourceGrid()
                self.optimizers[i] = None
                continue
            if i not in active:
                continue
            proxy = model.astc_proxy[i]
            if self.optimizers[i] is None:
                if proxy_kind(proxy) == 'source':
                    proxy = initialize_grid_proxy(model, i)
                    model.astc_proxy[i] = proxy
                if proxy_kind(proxy) == 'astc_roundtrip':
                    proxy.thaw()
                else:
                    proxy.requires_grad_(True)
                if proxy_kind(proxy) == 'astc_native':
                    from Core.ASTC_Aware.differentiable_proxy import ASTCBlockAdam
                    opt = ASTCBlockAdam(torch.nn.ModuleList([proxy]), trainer.configs.astc_learning_rate,
                                        trainer.configs.astc_learning_rate, fused=True,
                                        quantization_aware=True, packed_pair_search=True)
                else:
                    lr = trainer.configs.bcn_learning_rate if recipe.format == 'bc1' else trainer.configs.astc_learning_rate
                    opt = torch.optim.Adam(proxy.parameters(), lr=lr)
                self.optimizers[i] = opt
            if proxy_kind(proxy) == 'bc1_rgba':
                proxy.temperature.fill_(.25**progress)

    def zero_grad(self):
        for opt in self.optimizers:
            if opt is not None:
                opt.zero_grad()

    def step(self):
        due = getattr(self.trainer, '_codec_due_grids', range(len(self.optimizers)))
        for index, opt in enumerate(self.optimizers):
            if opt is None or index not in due:
                continue
            if isinstance(opt, torch.optim.Optimizer):
                opt.step()
            elif self.updates[index] % 100 == 0 and hasattr(self.trainer, "_astc_proxy_batches"):
                proxy = self.trainer.model.astc_proxy[index]
                touched = torch.where(
                    (proxy.endpoints.grad.abs().sum(1) + proxy.weights.grad.abs().sum(1)) > 0
                )[0]
                opt.step({0: touched}, objective=self.trainer._astc_proxy_objective)
            else:
                opt.step_full_grid()
            self.updates[index] += 1
        self.trainer.model._qat_param_cache.clear()
        self.iteration += 1
        self.trainer._astc_codec_pass_count = self.iteration
        interval = self.trainer.configs.astc_decoder_projection_interval
        if (interval > 0 and self.iteration % interval == 0
                and any(i in due and proxy_kind(p) == 'astc_native' for i, p in enumerate(self.trainer.model.astc_proxy))
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
