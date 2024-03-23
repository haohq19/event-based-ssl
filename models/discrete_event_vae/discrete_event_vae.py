import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding):
        super(PositionalEncoding, self).__init__()

        self.dim_embedding = dim_embedding

        _2i = torch.arange(0, dim_embedding, step=2).float()
        # 'i' means index of dim_embedding
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _phase = 100 / 1000 ** (_2i / dim_embedding)  
        # the phase of sinusoid is different by index of dim_embedding
        # _phase.shape = (dim_embedding / 2)
        self.register_buffer('_phase', _phase)

    def forward(self, x, t):
        # x.shape = (batch_size, dim_embedding, nevents_per_token)
        # t.shape = (batch_size, nevents_per_token)
        batch_size, nevents_per_token = t.size()
        encoding = torch.zeros(batch_size, self.dim_embedding, nevents_per_token, device=x.device, requires_grad=False)
        encoding[:, 0::2, :] = torch.sin(t.unsqueeze(1) * self._phase.unsqueeze(1))  # (b, 1, nevents_per_token) * (dim_embedding/2, 1) -> (b, d_model/2, nevents_per_token)
        encoding[:, 1::2, :] = torch.cos(t.unsqueeze(1) * self._phase.unsqueeze(1)) 
        
        return x + encoding


class DiscreteEventVAE(nn.Module):
    """
    DiscreteEventVAE is a class that represents a discrete Variational Autoencoder (dVAE) for event sequences.

    Args:
        dim_embedding (int): The dimensionality of the embedding layer.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        dim_embedding (int): The dimensionality of the embedding layer.
        vocab_size (int): The size of the vocabulary.
        embedding (nn.Sequential): The embedding layer.
        positional_encoding (PositionalEncoding): The positional encoding layer.
        encoder (nn.Sequential): The encoder layer.

    Methods:
        encode(x, t): Encodes the input sequence into a latent representation.
        forward(x): Performs the forward pass of the dVAE.

    """

    def __init__(
            self,
            dim_embedding,
            vocab_size,
        ):
        super(DiscreteEventVAE, self).__init__()

        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        
        # embedding layer, (batch_size, 3, nevents_per_token) -> (batch_size, dim_embedding, nevents_per_token)
        self.embedding = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, dim_embedding, 1),
            nn.BatchNorm1d(dim_embedding),
            nn.ReLU(),
        )

        # positional encoding layer
        self.positional_encoding = PositionalEncoding(dim_embedding)

        # encoder layer, (batch_size, dim_embedding) -> (batch_size, vocab_size)
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_embedding, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size),
        )
    
    def encode(self, x, t):
        """
        Encodes the input sequence into a latent representation.

        Args:
            x (torch.Tensor): The input sequence of events. Shape: (batch_size, 3, nevents_per_token)
            t (torch.Tensor): The input sequence of timestamps. Shape: (batch_size, nevents_per_token)

        Returns:
            torch.Tensor: The encoded latent representation. Shape: (batch_size, vocab_size)

        """
        x = self.embedding(x)                   # x.shape = (batch_size, 1024, nevents_per_token)
        x = self.positional_encoding(x, t)      # x.shape = (batch_size, 1024, nevents_per_token)
        x, _ = torch.max(x, dim=2)              # x.shape = (batch_size, 1024)
        logit = self.encoder(x)                 # logit.shape = (batch_size, vocab_size)
        return logit

    def forward(self, x):
        """
        Performs the forward pass of the dVAE.

        Args:
            x (torch.Tensor): The input sequence of events and timestamps. Shape: (batch_size, 4, nevents_per_token)

        Returns:
            torch.Tensor: The output logits. Shape: (batch_size, vocab_size)

        """
        events = x[:, 1:4, :]               # events.shape = (batch_size, 3, nevents_per_token)
        t = x[:, 0, :]                      # t.shape = (batch_size, nevents_per_token)
        logit = self.encode(events, t)      # logit.shape = (batch_size, vocab_size)
        return logit