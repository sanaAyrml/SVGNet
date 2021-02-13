import torch
import numpy as np
import csv
from itertools import chain
from typing import Iterator, List, Optional
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from src.argoverse.utils.evaluation import get_ade,get_ade_6

MAX_MODES = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
def neg_multi_log_likelihood(
        gt: torch.Tensor, pred: torch.Tensor, confidences: torch.Tensor, avails: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    batch_size, num_modes, future_len, num_coords = pred.shape

    # convert to (batch_size, num_modes, future_len, num_coords)
    if len(gt.shape) != len(pred.shape):
        gt = torch.unsqueeze(gt, 1)  # add modes
    if avails is not None:
        avails = avails[:, None, :, None]  # add modes and cords
        # error (batch_size, num_modes, future_len)
        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    else:
        error = torch.sum((gt - pred) ** 2, dim=-1)  # reduce coords and use availability
    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    return error.reshape(-1)


class PredLoss(nn.Module):
    def __init__(self):
        super(PredLoss, self).__init__()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
        #         self.reg_loss = nn.MSELoss(reduction="sum")
        self.cls_th = 2.0
        self.mgn = 0.2
        self.cls_ignore = 0.2
        self.cls_coef=1.0
        self.reg_coef = 1.0

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:

        cls, reg = out["cls"], out["reg"]


        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0


        num_mods, num_preds = reg.shape[1], reg.shape[2]
        print("here1",reg.shape)

        row_idcs = torch.arange(reg.shape[0]).long().to(reg.device)

        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_idcs, j, -1] - gt_preds[row_idcs, -1])
                            ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)

        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.cls_th).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.cls_ignore
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.mgn
        coef = self.cls_coef
        loss_out["cls_loss"] += coef * (
                self.mgn * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()
        reg = reg[row_idcs, min_idcs]
        print("here2",reg.shape)
        coef = self.reg_coef
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg, gt_preds
        )
        loss_out["num_reg"] += reg.shape[0]*reg.shape[1]
        return loss_out

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.pred_loss = PredLoss()

    def forward(self, out: Dict, data: Dict) -> Dict:
        print(data["gt_preds"])
        loss_out = self.pred_loss(out, data["gt_preds"], data["has_preds"])
        loss_out["loss"] = loss_out["cls_loss"] / (
                    loss_out["num_cls"] + 1e-10
                ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
#         loss_out["loss"] = loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out

