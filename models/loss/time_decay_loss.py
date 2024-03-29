import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDecayLoss(nn.Module):
    def __init__(self, H, W, temperature=256):
        super().__init__()
        self.H = H
        self.W = W
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        # pred.shape = [batch, seq_len, 2H + 2W]
        # target.shape = [batch, seq_len, 4], 
        # the last dimension of target is (t, x, y, p)
        seq_len = pred.shape[1]
        pred_H_0 = pred[:, :, :self.H]                      # [batch, seq_len, H]
        pred_H_1 = pred[:, :, self.H:2*self.H]              # [batch, seq_len, H]
        pred_W_0 = pred[:, :, 2*self.H:2*self.H+self.W]     # [batch, seq_len, W]
        pred_W_1 = pred[:, :, 2*self.H+self.W:]             # [batch, seq_len, W]
        target_H_0 = torch.zeros_like(pred_H_0)  # [batch, seq_len, H]
        target_H_1 = torch.zeros_like(pred_H_1)  # [batch, seq_len, H]
        target_W_0 = torch.zeros_like(pred_W_0)  # [batch, seq_len, W]
        target_W_1 = torch.zeros_like(pred_W_1)  # [batch, seq_len, W]
        target_p = target[:, :, 3:4]
        target_H_0 = torch.scatter(target_H_0, 2, target[:, :, 1:2].long(), 1 - target_p)  # [batch, seq_len, H]
        target_H_1 = torch.scatter(target_H_1, 2, target[:, :, 1:2].long(), target_p)
        target_W_0 = torch.scatter(target_W_0, 2, target[:, :, 2:3].long(), 1 - target_p)
        target_W_1 = torch.scatter(target_W_1, 2, target[:, :, 2:3].long(), target_p)
        # time decay
        time_diff = target[:, 1:, 0] - target[:, :-1, 0]
        time_decay = torch.exp(-time_diff / self.temperature).unsqueeze(-1)
        # reverse iterate the sequence
        for i in range(seq_len-1, 1, -1):
            target_H_0[:, i-1, :] = target_H_0[:, i-1, :] + target_H_0[:, i, :] * time_decay[:, i-1, :]
            target_H_1[:, i-1, :] = target_H_1[:, i-1, :] + target_H_1[:, i, :] * time_decay[:, i-1, :]
            target_W_0[:, i-1, :] = target_W_0[:, i-1, :] + target_W_0[:, i, :] * time_decay[:, i-1, :]
            target_W_1[:, i-1, :] = target_W_1[:, i-1, :] + target_W_1[:, i, :] * time_decay[:, i-1, :]
        target_0 = 1 - target[:, :, 3]
        target_1 = target[:, :, 3]
        pred_H_0 = pred_H_0.permute(0, 2, 1)  # [batch, H, seq_len]
        pred_H_1 = pred_H_1.permute(0, 2, 1)
        pred_W_0 = pred_W_0.permute(0, 2, 1)
        pred_W_1 = pred_W_1.permute(0, 2, 1)
        _target_H_0 = target_H_0.softmax(dim=2).permute(0, 2, 1)  # [batch, H, seq_len]
        _target_H_1 = target_H_1.softmax(dim=2).permute(0, 2, 1)
        _target_W_0 = target_W_0.softmax(dim=2).permute(0, 2, 1)
        _target_W_1 = target_W_1.softmax(dim=2).permute(0, 2, 1)
        loss_H_0 = self.ce(pred_H_0, _target_H_0)
        loss_H_1 = self.ce(pred_H_1, _target_H_1)
        loss_W_0 = self.ce(pred_W_0, _target_W_0)
        loss_W_1 = self.ce(pred_W_1, _target_W_1)
        loss_0 = (loss_H_0 * target_0 + loss_W_0 * target_0)
        loss_1 = (loss_H_1 * target_1 + loss_W_1 * target_1)
        loss = (loss_0 + loss_1).mean()
        return loss