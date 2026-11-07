"""CNN on decoded PBR material maps, before texture sampling and shading."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Core.Feature_Grid.sampling import tile_regions


class MaterialCNN(nn.Module):
    version = 3
    mode = 'native_material_residual_leaky_relu_0.01_repeat_clamp'

    def __init__(self, channels=10, hidden=16, rounds=2, inference_size=None, tile_size=128):
        super().__init__()
        if channels < 1 or hidden < 1 or rounds not in (1, 2):
            raise ValueError('MaterialCNN requires positive channel counts and 1/2 rounds')
        self.channels, self.hidden, self.rounds = channels, hidden, rounds
        self.halo = rounds
        if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size < 1:
            raise ValueError('Material CNN tile size must be a positive integer')
        self.register_buffer('tile_size', torch.tensor(tile_size, dtype=torch.int64))
        if inference_size is not None and (len(inference_size) != 2 or any(n < 1 for n in inference_size)):
            raise ValueError('Material CNN inference size must contain two positive dimensions')
        self.register_buffer('inference_size', torch.tensor(inference_size or (0, 0), dtype=torch.int64))
        self.layers = nn.ModuleList([nn.ModuleList((nn.Conv2d(channels, hidden, 3),
                                                   nn.Conv2d(hidden, channels, 1))) for _ in range(rounds)])
        for _, last in self.layers:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def valid(self, image):
        """Consume a real halo of width rounds; retain encoded normal channels."""
        value = image
        for spatial, pointwise in self.layers:
            residual = pointwise(F.leaky_relu(spatial(value), negative_slope=.01))
            value = value[..., 1:-1, 1:-1] + residual
        bounded = value.clamp(0, 1)
        return bounded.detach() + (value - value.detach()) if self.training else bounded

    def validate_size(self, size):
        expected = tuple(self.inference_size.tolist())
        if expected != (0, 0) and tuple(size) != expected:
            raise ValueError(f'Material CNN was fitted at {expected}, received {tuple(size)}; '
                             'use the original inference resolution or refit the CNN')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        tile = state_dict.get(prefix+'tile_size')
        if tile is None or tile.ndim != 0 or tile.dtype != torch.int64 or int(tile) != int(self.tile_size):
            raise ValueError('Material CNN checkpoint tile size does not match configuration')
        key = prefix + 'inference_size'
        if key in state_dict:
            self.validate_size(tuple(state_dict[key].tolist()))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _patch(image, y, x, height, width, radius):
        iy = torch.arange(y-radius, y+height+radius, device=image.device) % image.shape[-2]
        ix = torch.arange(x-radius, x+width+radius, device=image.device) % image.shape[-1]
        return image[..., iy[:, None], ix[None, :]]


    def forward(self, image):
        self.validate_size(image.shape[-2:])
        height, width = image.shape[-2:]
        result = torch.empty_like(image)
        for y,x,h,w in tile_regions(height,width,int(self.tile_size)):
            result[..., y:y+h, x:x+w] = self.valid(self._patch(image, y, x, h, w, self.halo))
        return result

    def export_arrays(self):
        arrays = dict(material_cnn_version=np.array(self.version, dtype=np.int32),
                      material_cnn_mode=np.array(self.mode),
                      material_cnn_shape=np.array([self.channels, self.hidden, self.rounds], dtype=np.int32))
        arrays.update({'material_cnn_'+k.replace('.', '_'):v.detach().cpu().numpy()
                       for k,v in self.state_dict().items()})
        return arrays

    @classmethod
    def from_arrays(cls, arrays):
        model = cls(*map(int, arrays['material_cnn_shape']), tile_size=int(arrays['material_cnn_tile_size']))
        if (int(arrays['material_cnn_version']) != cls.version
                or str(arrays['material_cnn_mode']) != cls.mode):
            raise ValueError('Unsupported material CNN export')
        model.load_state_dict({k:torch.from_numpy(np.array(arrays['material_cnn_'+k.replace('.', '_')], copy=True))
                               for k in model.state_dict()}, strict=True)
        return model
