import torch

from vedacore.misc import registry
from .base_meshgrid import BaseMeshGrid


@registry.register_module('meshgrid')
class PointAnchorMeshGrid(BaseMeshGrid):

    def __init__(self, strides):
        super().__init__(strides)

    def gen_anchor_mesh(self,
                        featmap_sizes,
                        img_metas,
                        dtype=torch.float,
                        device='cuda'):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas: Nonsense here
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._gen_anchor_mesh_single(featmap_sizes[i], self.strides[i],
                                             dtype, device))
        return mlvl_points

    def _gen_anchor_mesh_single(self, featmap_size, stride, dtype, device):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
