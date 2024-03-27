import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # multi-head self-attention
        x = x + self.dropout0(self.attention(x, x, x, attn_mask=mask, is_causal=True)[0])
        x = self.ln0(x)
        # feedforward
        x = x + self.dropout1(self.feedforward(x))
        x = self.ln1(x)
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, d_event, d_model, nhead, num_layers, dim_feedforward, d_out):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.d_out = d_out
        
        # event embedding
        self.embedding = nn.Linear(d_event, d_model)
        # transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        # layer normalization
        self.ln0 = nn.LayerNorm(d_model)
        # output layer
        self.head = nn.Linear(d_model, d_out, bias=False)

        # calculate the model size
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        self.num_params = num_params

    def forward(self, x):
        # auto-regressive transformer decoder
        # x.shape = [batch, time, channel]
        x = self.embedding(x)
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)  # causal mask
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln0(x)
        output = self.head(x)
        return output
        
        