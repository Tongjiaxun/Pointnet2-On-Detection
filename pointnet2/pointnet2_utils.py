import torch
import torch.nn as nn
import torch.functional as F
    

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    # def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):




class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz
