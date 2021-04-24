import torch
import torch.nn as nn

class JointBoneLoss(nn.Module):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i+1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

    def forward(self, joint_out, joint_gt):
        J = torch.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        Y = torch.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        loss = torch.abs(J-Y)
        return loss.mean()