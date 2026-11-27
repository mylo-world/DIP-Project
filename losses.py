import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits) if self.from_logits else logits
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """0.75 * DiceLoss + 0.25 * BCEWithLogitsLoss(pos_weight=6).

    pos_weight=6 penalises missing a polyp pixel 6× more than a false positive,
    counteracting the ~10% polyp / 90% background class imbalance.
    Dice weight raised to 0.75 because Dice directly optimises the metric.
    """

    def __init__(self, smooth=1.0, bce_weight=0.25, pos_weight=6.0):
        super().__init__()
        self.dice_weight = 1.0 - bce_weight
        self.bce_weight  = bce_weight
        self.dice = DiceLoss(smooth=smooth, from_logits=True)
        # pos_weight registered as buffer so it moves to GPU automatically
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pw.to(logits.device)
        )
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def dice_coefficient(logits, targets, threshold=0.5, smooth=1.0):
    with torch.no_grad():
        preds   = (torch.sigmoid(logits) > threshold).float()
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (preds * targets).sum(dim=1)
        union        = preds.sum(dim=1) + targets.sum(dim=1)
        dice         = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()