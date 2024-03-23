import torch
import torch.nn as nn
from .causal_event_model import CausalEventModel
from ..discrete_event_vae.discrete_event_vae_encoder import DiscreteEventVAEEncoder

class CausalEventModelForDiscreteEventVAEEncoding(nn.Module):
    """
    A module that combines a CausalEventModel and a DiscreteEventVAEEncoder for event-based causal masked self-supervised pretraining.

    Args:
        d_event (int): The dimensionality of the event representation.
        d_model (int): The dimensionality of the model.
        num_layers (int): The number of layers in the model.
        dim_feedforward (int): The dimensionality of the feedforward network.
        nevents_per_token (int): The number of events per token.
        dim_embedding (int): The dimensionality of the event embedding.
        vocab_size (int): The size of the vocabulary.
        pretrained_discrete_event_vae_root (str): The path to the pretrained discrete event VAE model.

    Attributes:
        model (CausalEventModel): The causal event model.
        encoding (DiscreteEventVAEEncoder): The discrete event VAE encoder.
        criterion (nn.CrossEntropyLoss): The cross-entropy loss function.
        num_params (int): The total number of trainable parameters in the module.
    """

    def __init__(self, d_event, d_model, num_layers, dim_feedforward, nevents_per_token, dim_embedding, vocab_size, pretrained_discrete_event_vae_root):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.nevents_per_token = nevents_per_token
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.pretrained_discrete_event_vae_root = pretrained_discrete_event_vae_root

        self.model = CausalEventModel(
            d_event=d_event,
            d_model=d_model,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            d_out=vocab_size,
        )

        self.encoding = DiscreteEventVAEEncoder(
            nevents_per_token=nevents_per_token,
            dim_embedding=dim_embedding,
            vocab_size=vocab_size,
        )

        self.criterion = nn.CrossEntropyLoss()

        checkpoint = torch.load(pretrained_discrete_event_vae_root)
        pretrained_weight = checkpoint['model']
        self.encoding.discrete_event_vae.load_state_dict(pretrained_weight, strict=False)
        for p in self.encoding.discrete_event_vae.parameters():
            p.requires_grad = False

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor of shape [batch, seq_len, d_event].

        Returns:
            torch.Tensor: The computed loss.
        """
        _, seq_len, _ = x.size()
        output = self.model(x)                                                                                        # output.shape = [batch, seq_len, vocab_size]
        target = self.encoding(x, return_hidden=False)                                                                # target.shape = [batch, num_tokens]
        indices = torch.arange(self.nevents_per_token - 1, seq_len - self.nevents_per_token, self.nevents_per_token)  # indices.shape = [num_tokens - 1]
        output = output[:, indices, :].transpose(1, 2)                                                                # output.shape = [batch, vocab_size, num_tokens - 1]
        target = target[:, 1:]                                                                                        # target.shape = [batch, num_tokens - 1]
        loss = self.criterion(output, target)
        return loss