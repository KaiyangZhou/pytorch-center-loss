import warnings

import torch
from torch import nn
from torch.nn import functional as F


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 2, use_gpu: bool = None, clamp: int = 1e-12):
        super(CenterLoss, self).__init__()
        if use_gpu is not None:
            warnings.warning(f"Ignoring explicitly set {use_gpu=}. Move the model via .to(device)")
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.clamp = clamp

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        centers = torch.index_select(self.centers, 0, labels.view(-1))  # [Classes, Features] (gather) [Batch]  -> [Batch, Features]
        dist = (x - centers).square().sum(dim=1, keepdim=False)
        return dist.clamp(min=self.clamp, max=1 / self.clamp).mean()
