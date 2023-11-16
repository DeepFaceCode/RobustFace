import math

import torch
import torch.nn as nn

from torch import distributed
from torch.nn import Parameter


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, logits, label):
        batch_size = logits.size(0)
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max_logits  #logits.sub(max_logits)
        logits = torch.exp(logits)  #logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)

        logits = torch.div(logits, sum_logits_exp) #logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1,device=logits.device)
        loss[index] = logits[index].gather(1, label[index].unsqueeze(1))
        return loss.clamp_min(1e-30).log().mean() * (-1)

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

