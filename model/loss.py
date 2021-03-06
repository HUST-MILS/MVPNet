import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import yaml
import math
import random

import sys
sys.path.append('/data/MVPNet')
from chamfer_distance.chamfer_distance import ChamferDistance
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
class Loss(nn.Module):
    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.loss_weight_cd = self.cfg["TRAIN"]["LOSS_WEIGHT_CHAMFER_DISTANCE"]
        self.loss_weight_cd = 1.0
        self.loss_weight_rv = self.cfg["TRAIN"]["LOSS_WEIGHT_RANGE_VIEW"]
        self.loss_weight_mask = self.cfg["TRAIN"]["LOSS_WEIGHT_MASK"]


        self.chamfer_distance = chamfer_distance(self.cfg)


    def forward(self, output, target, mode):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """

        target_range_image = target[:, 0, :, :, :]

        # Chamfer Distance
        if self.loss_weight_cd > 0.0 or mode == "val" or mode == "test":
            chamfer_distance, chamfer_distances_tensor = self.chamfer_distance(
                output, target, self.cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"]
            )
            loss_chamfer_distance = sum([cd for cd in chamfer_distance.values()]) / len(
                chamfer_distance
            )
            detached_chamfer_distance = {
                step: cd.detach() for step, cd in chamfer_distance.items()
            }
        else:
            chamfer_distance = dict(
                (step, torch.zeros(1).type_as(target_range_image))
                for step in range(self.n_future_steps)
            )
            chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)

            detached_chamfer_distance = chamfer_distance

        loss = (
            self.loss_weight_cd * loss_chamfer_distance
        )

        loss_dict = {
            "loss": loss,
            "chamfer_distance": detached_chamfer_distance,
            "chamfer_distances_tensor": chamfer_distances_tensor.detach(),
            "mean_chamfer_distance": loss_chamfer_distance.detach(),
            "final_chamfer_distance": chamfer_distance[
                self.n_future_steps - 1
            ].detach(),
        }
        return loss_dict


"""
class Focal_Loss(nn.Module):
    r'''
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.
    Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each minibatch.
    Args:
    alpha(1D Tensor, Variable) : the scalar factor for this criterion
    gamma(float, double) : gamma > 0; reduces the relative loss for well-classi???ed examples (p > .5),
    putting more focus on hard, misclassi???ed examples
    size_average(bool): By default, the losses are averaged over observations for each minibatch.
    However, if the field size_average is set to False, the losses are
    instead summed for each minibatch.
    '''

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(Focal_Loss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
                self.gamma = gamma
                self.class_num = class_num
                self.size_average = size_average

    def forward(self, output, targets):
        batch_size,  num_class,H, W = output.shape

        P = F.softmax(output)
        class_mask = output.data.new(batch_size, num_class).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        if output.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]
            probs = (P*class_mask).sum(1).view(-1,1)
            log_p = probs.log()
            #print('probs size= {}'.format(probs.size()))
            #print(probs)
            batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
            #print('-----bacth_loss------')
            #print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


import numpy as np
import random, torch, torchvision
import torch.nn.functional as F
from torch import nn, optim
 
def setup_seed(seed):
    #????????????????????????
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_onehot(label, num_class, high, width):
    label = label.view(-1)
    ones = torch.sparse.torch.eye(num_class)
    ones = ones.index_select(0, label)
    ones = torch.transpose(ones, 0, 1).int()
    ones_high = ones.shape[0]
    label_list = []
    for i in range(ones_high):
        label_list.append(ones[i].reshape(high, width))
    return label_list
 

setup_seed(20)
output = torch.randn((1, 2, 2, 2))
label= torch.randint(2, (1, 2, 2))
N, C, high, width = output.shape
label_list = get_onehot(label, C, high, width)
focalloss  = Focal_Loss(0.25)
label_list = np.array(label_list)
loss = focalloss.forward(output, label_list)
print(loss)
'''
label = label_wwww[:,1,:,:]
N, C, high, width = output.shape
label_list = get_onehot(label, C, high, width)
label0 = label_list[0].unsqueeze(dim=0)
label1 = label_list[1].unsqueeze(dim=0)
print('output:\n', output)
print('label:\n', label)
print('??????label:\n', label0)
print('??????label:\n', label1)
ce = nn.CrossEntropyLoss()
loss = ce(output, label)
print('ce: ', loss)

pt = F.softmax(output, 1)
pt0 = pt[:, 0, :, :]
pt1 = pt[:, 1, :, :]
log0 = torch.log(pt0)
log1 = torch.log(pt1)
print('pt:\n', pt)
print('log0:\n', log0)
print('log1:\n', log1)
loss = -(label0 * log0 + label1 * log1)
print('ce: ', loss.mean())

loss = - 0.25 * label0 * ((1 - pt0) ** 2) * torch.log(pt0) - 0.75 * label1 * ((1 - pt1) ** 2) * log1
print('focal: ', loss.mean())
'''
"""
