import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random

from chamfer_distance import ChamferDistance
from utils.projection import projection

class chamfer_distance(nn.Module):
    """Chamfer distance loss. Additionally, the implementation allows the evaluation
    on downsampled point cloud (this is only for comparison to other methods but not recommended,
    because it is a bad approximation of the real Chamfer distance.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target, n_samples):
        batch_size, n_future_steps, H, W = output["rv"].shape
        masked_output = self.projection.get_masked_range_view(output)
        chamfer_distances = {}
        chamfer_distances_tensor = torch.zeros(n_future_steps, batch_size)
        for s in range(n_future_steps):
            chamfer_distances[s] = 0
            for b in range(batch_size):
                output_points = self.projection.get_valid_points_from_range_view(
                    masked_output[b, s, :, :]
                ).view(1, -1, 3)
                target_points = target[b, 1:4, s, :, :].permute(1, 2, 0)
                target_points = target_points[target[b, 0, s, :, :] > 0.0].view(
                    1, -1, 3
                )

                if n_samples != -1:
                    n_output_points = output_points.shape[1]
                    n_target_points = target_points.shape[1]

                    sampled_output_indices = random.sample(
                        range(n_output_points), n_samples
                    )
                    sampled_target_indices = random.sample(
                        range(n_target_points), n_samples
                    )

                    output_points = output_points[:, sampled_output_indices, :]
                    target_points = target_points[:, sampled_target_indices, :]

                dist1, dist2 = self.loss(output_points, target_points)
                dist_combined = torch.mean(dist1) + torch.mean(dist2)
                chamfer_distances[s] += dist_combined
                chamfer_distances_tensor[s, b] = dist_combined
            chamfer_distances[s] = chamfer_distances[s] / batch_size
        return chamfer_distances, chamfer_distances_tensor

class Focal_loss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self,output,target,n_samples):
        pass
