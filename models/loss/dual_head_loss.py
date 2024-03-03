import torch
import torch.nn as nn

class DualHeadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        # pred.shape = [batch, seq_len, 4]
        # target.shape = [batch, seq_len, 3]
        # the first 2 channels predict x and y when ground truth p = 0,
        # the last 2 channels predict x and y when ground truth p = 1,
        mask_0 = target[:, :, 2].unsqueeze(2).reshape(-1, 1)  # [batch * seq_len, 1]
        loss_0 = self.mse(pred[:, :, :2], target[:, :, :2]).mean(dim=2, keepdim=True).reshape(-1, 1)  # [batch * seq_len, 1]
        loss_0 = loss_0 * mask_0  # [batch * seq_len, 1]
        mask_1 = 1 - mask_0
        loss_1 = self.mse(pred[:, :, 2:], target[:, :, :2]).mean(dim=2, keepdim=True).reshape(-1, 1) # [batch * seq_len, 1]
        loss_1 = loss_1 * mask_1
        loss = loss_0 + loss_1
        loss = loss.mean()
        return loss
