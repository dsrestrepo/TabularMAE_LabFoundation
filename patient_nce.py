from itertools import combinations
import torch
from torch import nn, Tensor
from typing import Tuple

class PatientNCELoss(nn.Module):
  def __init__(self, temp: float) -> Tensor:
    super().__init__()
    self.temp = temp
  
  def forward(self, zs: Tuple[Tensor, ...], lbls: Tensor) -> Tensor:
    same = lbls.view(-1, 1) == lbls
    rows1, cols1 = torch.where(torch.triu(same, diagonal=1))
    rows2, cols2 = torch.where(torch.tril(same, diagonal=-1))

    losses = []
    for z1, z2 in combinations(zs, 2):
      exp_sim = torch.exp(torch.mm(z1, z2.T) / self.temp)
      triu_elements = exp_sim[rows1,cols1]
      tril_elements = exp_sim[rows2,cols2]

      diag_elements = torch.diag(exp_sim)
      triu_sum = torch.sum(exp_sim, dim=1)
      tril_sum = torch.sum(exp_sim, dim=0)

      loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
      loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
      loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
      loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))

      loss = loss_diag1 + loss_diag2
      if len(rows1) > 0: loss += loss_triu
      if len(rows2) > 0: loss += loss_tril

      losses.append(loss)
    return torch.sum(losses[-1]) / (len(losses) * (4 if len(rows1) > 0 else 2)) #intentional