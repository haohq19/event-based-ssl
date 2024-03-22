import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _phase = 100 / 1000 ** (_2i / d_model)  
        # the phase of sinusoid is different by index of d_model
        # _phase.shape = (d_model / 2)
        self.register_buffer('_phase', _phase)


    def forward(self, x, t):
        # self.encoding
        # x.shape = (batch_size, d_model, seq_len)
        # t.shape = (batch_size, seq_len)
        batch_size, seq_len = t.size()
        encoding = torch.zeros(batch_size, self.d_model, seq_len, device=x.device, requires_grad=False)

        encoding[:, 0::2, :] = torch.sin(t.unsqueeze(1) * self._phase.unsqueeze(1))  # (b, 1, seq_len) * (d_model/2, 1) -> (b, d_model/2, seq_len)
        encoding[:, 1::2, :] = torch.cos(t.unsqueeze(1) * self._phase.unsqueeze(1)) 
        
        return x + encoding


class DiscreteEventVAE(nn.Module):
    def __init__(
            self,
            dim_embedding,
            num_tokens,
        ):
        super(DiscreteEventVAE, self).__init__()

        self.dim_embedding = dim_embedding
        self.num_tokens = num_tokens
        
        # embedding layer, (batch_size, 3, seq_len) -> (batch_size, dim_embedding, seq_len)
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

        # encoder layer, (batch_size, dim_embedding) -> (batch_size, num_tokens)
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_embedding, 256),
            nn.ReLU(),
            nn.Linear(256, num_tokens),
        )
    
    def encode(self, x, t):
        # x.shape = (batch_size, 3, seq_len)
        # t.shape = (batch_size, seq_len)
        x = self.embedding(x)                 # x.shape = (batch_size, 1024, seq_len)
        x = self.positional_encoding(x, t)      # x.shape = (batch_size, 1024, seq_len)
        x, _ = torch.max(x, dim=2)              # x.shape = (batch_size, 1024)
        logit = self.encoder(x)                # dist.shape = (batch_size, num_tokens)
        return logit

    def forward(self, x):
        # x.shape = (batch_size, 4, seq_len)
        events = x[:, 1:4, :]  # (batch_size, 3, seq_len)
        t = x[:, 0, :]         # (batch_size, seq_len)
        logits = self.encode(events, t)  # logit.shape = (batch_size, num_tokens)
        return logits
    

class dVAEOutput(nn.Module):
    """
    dVAEOutput is a PyTorch module that combines a DiscreteEventVAE with a CrossEntropyLoss.
    It calculates the loss between the predicted output and the target output.

    Args:
        nevents_per_token (int): Number of events per token.
        dim_embedding (int): Dimension of the embedding.
        vocab_size (int): Size of the vocabulary.

    Attributes:
        nevents_per_token (int): Number of events per token.
        dim_embedding (int): Dimension of the embedding.
        vocab_size (int): Size of the vocabulary.
        dvae (DiscreteEventVAE): Instance of the DiscreteEventVAE class.
        criterion (CrossEntropyLoss): Instance of the CrossEntropyLoss class.

    Methods:
        forward(output, x): Performs the forward pass of the module.

    """

    def __init__(self, nevents_per_token=64, dim_embedding=256, vocab_size=1024):
        super(dVAEOutput, self).__init__()

        self.nevents_per_token = nevents_per_token
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size

        self.dvae = DiscreteEventVAE(dim_embedding, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, x):
        """
        Performs the forward pass of the module.

        Args:
            output (Tensor): The predicted output tensor of shape (batch_size, seq_len, vocab_size).
            x (Tensor): The input tensor of shape (batch_size, seq_len, d_event).

        Returns:
            loss (Tensor): The calculated loss tensor.

        """
        batch_size, seq_len, _ = x.size()
        assert seq_len % self.nevents_per_token == 0
        num_tokens = seq_len // self.nevents_per_token
        x = x.view(batch_size, num_tokens, self.nevents_per_token, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * num_tokens, -1, self.nevents_per_token)
        t_start = x[:, 0, :1]
        t_end = x[:, 0, -1:]
        x[:, 0, :] = (x[:, 0, :] - t_start) / (t_end - t_start)
        with torch.no_grad():
            logits = self.dvae(x)
        target = torch.argmax(logits, dim=-1)
        target = target.view(batch_size, -1)
        indices = torch.arange(0, seq_len, self.nevents_per_token)
        output = output[:, indices, :]
        output = output.permute(0, 2, 1).contiguous()
        loss = self.criterion(output, target)
        return loss
    


if __name__ == '__main__':
    seq_len = 2048
    batch_size = 16
    d_event = 4
    nevents_per_token = 64
    dim_embedding = 256
    vocab_size = 1024
    image_size = 32
    t = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1).float()
    x = torch.randint(0, image_size, (batch_size, seq_len, 2)).float()
    p = torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    x = torch.cat((t, x, p), dim=-1)  # (batch_size, seq_len, d_event)
    output = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    model = dVAEOutput(nevents_per_token, dim_embedding, vocab_size)
    loss = model(output, x)
    print(loss)
    loss.backward()


    
        
        