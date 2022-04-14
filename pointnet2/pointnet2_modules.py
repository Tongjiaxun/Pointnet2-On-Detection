import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from . import model_utils
from typing import List








class PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    

class PointnetSAModuleMSG(PointnetSAModuleBase):

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True,
                 use_xyz: bool=True, pool_method='max_pool', instance_norm=False):

        super().__init__()

        assert len(nsamples) == len(radii) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(model_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method
        



class PointnetFPModule(nn.Module) :

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = model_utils.SharedMLP(mlp, bn=bn)





