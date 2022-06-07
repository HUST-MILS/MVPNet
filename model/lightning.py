import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
from loss import Loss
from utils.projection import projection
from utils.logger import log_point_clouds, save_range_and_mask, save_point_clouds

class lightning_model(LightningModule):
    def __init__(self,cfg):

        super(lightning_model, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.min_range = self.cfg["DATA_CONFIG"]["MIN_RANGE"]
        self.max_range = self.cfg["DATA_CONFIG"]["MAX_RANGE"]
        self.register_buffer("mean", torch.Tensor(self.cfg["DATA_CONFIG"]["MEAN"]))
        self.register_buffer("std", torch.Tensor(self.cfg["DATA_CONFIG"]["STD"]))

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.use_xyz = self.cfg["MODEL"]["USE"]["XYZ"]
        self.use_intensity = self.cfg["MODEL"]["USE"]["INTENSITY"]

        # Create list of index used in input
        self.inputs = [0]
        if self.use_xyz:
            self.inputs.append(1)
            self.inputs.append(2)
            self.inputs.append(3)
        if self.use_intensity:
            self.inputs.append(4)
        self.n_inputs = len(self.inputs)

        # Init loss
        self.loss = Loss(self.cfg)

        # Init projection class for re-projcecting from range images to 3D point clouds
        self.projection = projection(self.cfg)

        self.chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
    
    def forward(self,x):
        pass
    
    def configure_optimizers(self):
        return super().configure_optimizers()

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def test_epoch_end(self, outputs):
        return super().test_epoch_end(outputs)