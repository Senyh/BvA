import torch
from torch import nn
from torch.nn.functional import one_hot


class DSCLossH(nn.Module):
    def __init__(self, num_classes=2, inter_weight=1., intra_weights=None, device='cuda', is_3d=False):
        super(DSCLossH, self).__init__()
        if intra_weights is not None:
            intra_weights = torch.tensor(intra_weights).to(device)
        else:
            intra_weights = torch.tensor([1. for _ in range(num_classes)]).to(device) 
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weight = inter_weight
        self.device = device
        self.is_3d = is_3d

    def dice_loss(self, prediction, target, pixel_mask, pixel_weights):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5
        pixel_mask = pixel_mask.unsqueeze(1)
        pixel_weights = pixel_weights.unsqueeze(1)
        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        if self.is_3d:
            dims = (2, 3, 4)
        else:
            dims = (2, 3)
        # Calculate the Dice Similarity Coefficient for each class
        intersection = torch.sum(prediction * target * pixel_mask * pixel_weights, dim=dims)
        union = torch.sum(prediction * pixel_mask + target * pixel_mask, dim=dims)
        dice = (2 * intersection) / (union + smooth)
        one = (torch.sum(1. * pixel_mask * pixel_weights, dim=dims) / (torch.sum(pixel_mask, dim=dims) + smooth)).repeat(1, self.num_classes)
        dice_loss = (one - dice).sum(0) / batchsize
        return dice_loss

    def forward(self, pred, label, pixel_mask=None, pixel_weights=None):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        if pixel_mask is None:
            pixel_mask = torch.ones_like(label).to(self.device)  # B H W or B D H W
        if pixel_weights is None:
            pixel_weights = torch.ones_like(label).to(self.device)  # B H W or B D H W

        cel = (self.ce_loss(pred, label) * pixel_mask * pixel_weights).sum() / (pixel_mask.sum().item() + 1e-5)
        if self.is_3d:
            label_onehot = one_hot(label.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        else:
            label_onehot = one_hot(label.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        dicel = self.dice_loss(pred, label_onehot, pixel_mask, pixel_weights)
        dicel = (dicel * self.intra_weights).mean()
        loss = cel * (1 - self.inter_weight) + dicel * self.inter_weight
        return loss


