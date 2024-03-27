# Event-based self-supervised learning

haohq19@gmail.com

## V1.0 

2024/03/27

### Models

#### Causal Event Model 

1. Causal Event Model. 
This model is based on linear RNN to model event sequences auto-regressively.
This is the base model for different tasks with different task-specific heads.

2. Causal Event Model for Discrete Event VAE Encoding.
This model is based on CEM to predict next discrete event token for self-supervised pretraining.

#### Discrete Event VAE:

1. Discrete Event VAE.
This is the base model for event sequence tokenization.

2. Discrete Event VAE Encoder.
This model is the tokenizer for a whole event sequence.
Each slice of length nevents_per_token is tokenized into one token with Discrete Event VAE.

#### Linear Probe

1. Linear Probe.
This model is a linear classifier for transfer pretrained models.

#### Transformer Based

1. Transformer Decoder.
This is the standard transformer decoder based on multi-head attention.

#### Loss

1. Dual Head Loss.
Including L1 and L2 loss.

2. ProductLoss.

3. Time Decay Loss

### Engines

1. Pretrain.
Contain functions for pretraining models.
The forward pass of the model should return loss directly.

2. Transfer.
Contain functions for transferring existed models and cache representations.
The forward pass of the PTM should return its hidden state.

### Utils

1. Distributed 
This provides basic functions for distributed data parallel training.

2. Data
This provides functions to transform, collect data and get dataloader.

