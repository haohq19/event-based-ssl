import torch
import torch.nn as nn
import torch.nn.functional as F
from .discrete_event_vae import DiscreteEventVAE

class DiscreteEventVAEEncoder(nn.Module):
    """
    Encoder module for event sequences with the Discrete Event Variational Autoencoder (dVAE).

    Args:
        nevents_per_token (int): Number of events per token.
        dim_embedding (int): Dimensionality of the embedding.
        vocab_size (int): Size of the vocabulary.

    Attributes:
        nevents_per_token (int): Number of events per token.
        dim_embedding (int): Dimensionality of the embedding.
        vocab_size (int): Size of the vocabulary.
        discrete_event_vae (DiscreteEventVAE): Discrete Event VAE model.

    """

    def __init__(self, nevents_per_token=64, dim_embedding=256, vocab_size=1024):
        super(DiscreteEventVAEEncoder, self).__init__()

        self.nevents_per_token = nevents_per_token
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.discrete_event_vae = DiscreteEventVAE(dim_embedding, vocab_size)

    def forward(self, x, return_hidden=False):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_event). The seq_len must be divisible by nevents_per_token, i.e., seq_len % nevents_per_token == 0.
            return_hidden (bool): Whether to return the hidden representation.

        Returns:
            target (torch.Tensor): The encoded representation of shape (batch_size, num_tokens). num_tokens = seq_len // nevents_per_token.
            if return_hidden is True, also returns hidden (torch.Tensor): The hidden representation of shape (batch_size, vocab_size).

        """
        batch_size, seq_len, _ = x.size()
        assert seq_len % self.nevents_per_token == 0
        num_tokens = seq_len // self.nevents_per_token
        x = x.view(batch_size, num_tokens, self.nevents_per_token, -1)   # x.shape = (batch_size, num_tokens, nevents_per_token, d_event)
        x = x.permute(0, 1, 3, 2).contiguous()                           # x.shape = (batch_size, num_tokens, d_event, nevents_per_token)
        x = x.view(batch_size * num_tokens, -1, self.nevents_per_token)  # x.shape = (batch_size * num_tokens, d_event, nevents_per_token)
        t_start = x[:, 0, :1]
        t_end = x[:, 0, -1:]
        x[:, 0, :] = (x[:, 0, :] - t_start) / (t_end - t_start)
        with torch.no_grad():
            logits = self.discrete_event_vae(x)                                        # logits.shape = (batch_size * num_tokens, vocab_size)
        target = torch.argmax(logits, dim=-1)                            # target.shape = (batch_size * num_tokens)
        target = target.view(batch_size, -1)                             # target.shape = (batch_size, num_tokens)

        if return_hidden:
            hidden = F.one_hot(target, self.vocab_size).float()          # hidden.shape = (batch_size, num_tokens, vocab_size)
            hidden = hidden.sum(dim=1)                                   # hidden.shape = (batch_size, vocab_size)
            return target, hidden
        else:
            return target