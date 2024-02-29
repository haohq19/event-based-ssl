import torch
import torch.nn as nn
import logging
from ..rwkv4.src.model import TokenMixing, ChannelMixing

logger = logging.getLogger(__name__)


class Layer(nn.Module):
    def __init__(self, d_model, layer_id, num_layers):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id
        self.num_layers = num_layers

        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.attention = TokenMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers)
        self.feedforward = ChannelMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers)

    def forward(self, x):
        x = x + self.attention(self.ln0(x))
        x = x + self.feedforward(self.ln1(x))
        return x


class CausalEventModel(nn.Module):
    def __init__(self, d_event, d_model, num_layers):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.num_layers = num_layers

        # event embedding
        self.embedding = nn.Linear(d_event, d_model)  
        # layer normalization
        self.ln0 = nn.LayerNorm(d_model)
        # rwkv layers
        self.layers = nn.Sequential(*[Layer(d_model=d_model, layer_id=i, num_layers=num_layers) for i in range(num_layers)])
        # layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        # output layer
        self.head = nn.Linear(d_model, d_event, bias=False)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        # x.shape = [batch, seq_len, d_event]

        x = self.embedding(x)
        x = self.ln0(x)
        x = self.layers(x)
        x = self.ln1(x)
        output = self.head(x)

        return output  # output.shape = [batch, seq_len, d_event]