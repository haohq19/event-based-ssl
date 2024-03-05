import torch
import torch.nn as nn

class TimeDecayLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_0, target):
        # pred.shape = [batch, seq_len, H + W]
        # target.shape = [batch, seq_len, 4]
        target_0 = torch.zeros(target.shape[0], target.shape[1], ).to(target.device)