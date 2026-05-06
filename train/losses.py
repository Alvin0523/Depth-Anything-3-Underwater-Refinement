"""
Training losses for underwater depth fine-tuning.

SILog:    penalises depth shape errors while tolerating global scale offset.
Gradient: penalises blurry depth edges via Sobel filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    """Scale-Invariant Log loss (Eigen et al., 2014).

    scale_weight: weight for the mean log-ratio term that anchors absolute
    scale.  Without this the variance-focus term cancels global offset and
    AbsRel can diverge while SILog stays low.
    """

    def __init__(self, variance_focus: float = 0.85, scale_weight: float = 0.1):
        super().__init__()
        self.variance_focus = variance_focus
        self.scale_weight = scale_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0) & torch.isfinite(gt)

        pred_masked = pred[mask].clamp(min=1e-4)
        gt_masked = gt[mask].clamp(min=1e-4)

        g = torch.log(pred_masked) - torch.log(gt_masked)
        # shape term (scale-invariant) + scale anchor (prevents AbsRel drift)
        loss = g.pow(2).mean() - self.variance_focus * g.mean().pow(2) \
               + self.scale_weight * g.mean().abs()
        return loss


class GradientLoss(nn.Module):
    """L1 loss on Sobel depth gradients to preserve sharp edges."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # shape: (1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _grad(self, depth: torch.Tensor):
        # depth: (B, H, W) -> (B, 1, H, W)
        d = depth.unsqueeze(1)
        gx = F.conv2d(d, self.sobel_x, padding=1)
        gy = F.conv2d(d, self.sobel_y, padding=1)
        return gx.squeeze(1), gy.squeeze(1)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0) & torch.isfinite(gt)

        pred_gx, pred_gy = self._grad(pred)
        gt_gx, gt_gy = self._grad(gt)

        loss = (pred_gx - gt_gx).abs()[mask] + (pred_gy - gt_gy).abs()[mask]
        return loss.mean()


class UnderwaterDepthLoss(nn.Module):
    """Combined SILog + gradient loss."""

    def __init__(self, grad_weight: float = 0.5, variance_focus: float = 0.85,
                 scale_weight: float = 0.1):
        super().__init__()
        self.silog = SILogLoss(variance_focus=variance_focus, scale_weight=scale_weight)
        self.grad = GradientLoss()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = (gt > 0) & torch.isfinite(gt) & (pred > 0)
        l_silog = self.silog(pred, gt, mask)
        l_grad = self.grad(pred, gt, mask)
        total = l_silog + self.grad_weight * l_grad
        return {"loss": total, "silog": l_silog, "grad": l_grad}
