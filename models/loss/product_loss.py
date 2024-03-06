import torch.nn as nn

class ProductLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        # pred.shape = [batch, seq_len, 4], the first 2 channels are x and y, the last 2 channels are distribution of p
        # target.shape = [batch, seq_len, 4], the 4 channels are t, x, y and p
        # the loss of x and y is MSE, the loss of p is CrossEntropy
        # the loss is the sequence-wise product of the two losses
        target = target[:, :, 1:]  # [batch, seq_len, 3], remove the first column of t
        loss_xy = self.mse(pred[:, :, :2], target[:, :, :2]).mean(dim=2, keepdim=True).reshape(-1, 1) # [batch * seq_len, 1]
        p_pred = pred[:, :, 2:].reshape(-1, 2)
        p_target = target[:, :, 2].reshape(-1).long()
        loss_p = self.ce(p_pred, p_target).unsqueeze(1)  # [batch * seq_len, 1]
        loss = loss_xy * loss_p
        loss = loss.mean()
        return loss
        