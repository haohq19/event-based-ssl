import torch
import torch.nn as nn
from ..rwkv4.src.model import TokenMixing, ChannelMixing


class RWKVLayer(nn.Module):
    def __init__(self, d_model, layer_id, num_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.attention = TokenMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers)
        self.feedforward = ChannelMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, x):
        # output = self.attention(self.ln0(x))
        output, hidden = self.attention(self.ln0(x))
        self.hidden = hidden
        x = x + output
        x = x + self.feedforward(self.ln1(x))
        return x


class CausalEventModel(nn.Module):
    """
    Causal Event Model class.

    This class represents a causal event model that takes event sequences as input and predicts the corresponding outputs.
    It consists of event embedding, layer normalization, RWKV layers, layer normalization, and an output layer.

    Args:
        d_event (int): The dimension of the event input.
        d_model (int): The dimension of the model.
        num_layers (int): The number of RWKV layers.
        dim_feedforward (int): The dimension of the feedforward layer in each RWKV layer.
        d_out (int): The dimension of the output.

    Attributes:
        d_event (int): The dimension of the event input.
        d_model (int): The dimension of the model.
        num_layers (int): The number of RWKV layers.
        dim_feedforward (int): The dimension of the feedforward layer in each RWKV layer.
        d_out (int): The dimension of the output.
        embedding (nn.Linear): The event embedding layer.
        ln0 (nn.LayerNorm): The first layer normalization layer.
        layers (nn.ModuleList): The list of RWKV layers.
        ln1 (nn.LayerNorm): The second layer normalization layer.
        head (nn.Linear): The output layer.
        num_params (int): The total number of parameters in the model.

    Methods:
        _init_weights: Initializes the weights of the linear and embedding layers.
        forward: Performs the forward pass of the model.

    """

    def __init__(self, d_event, d_model, num_layers, dim_feedforward, d_out):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.d_out = d_out

        # event embedding
        self.embedding = nn.Linear(d_event, d_model)  
        # layer normalization
        self.ln0 = nn.LayerNorm(d_model)
        # rwkv layers
        self.layers = nn.ModuleList([RWKVLayer(d_model=d_model, layer_id=i, num_layers=num_layers, dim_feedforward=dim_feedforward) for i in range(num_layers)])
        # layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        # output layer
        self.head = nn.Linear(d_model, d_out, bias=False)

        # calculate the model size
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        self.num_params = num_params

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, return_hidden=False):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape [batch, seq_len, d_event].
            return_hidden (bool, optional): Whether to return the hidden state. Defaults to False.

        Returns:
            torch.Tensor: The output tensor of shape [batch, seq_len, d_out].
            torch.Tensor: The hidden state of the causal event model, if return_hidden is True.

        """
        x = self.embedding(x)                   # x.shape = [batch, seq_len, d_model]
        x = self.ln0(x)                         # x.shape = [batch, seq_len, d_model]
        for layer in self.layers:
            x = layer(x)                        # x.shape = [batch, seq_len, d_model]
        x = self.ln1(x)                         # x.shape = [batch, seq_len, d_model]
        output = self.head(x)                   # output.shape = [batch, seq_len, d_out]
        if return_hidden:
            hiddens = []
            for layer in self.layers:
                hiddens.append(layer.hidden)            # hidden.shape = [batch, d_model]
                hidden = torch.cat(hiddens, dim=1)    # hidden.shape = [batch, d_model * num_layers]
            return output, hidden
        else:
            return output