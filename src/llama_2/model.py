import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096  # embedding dimension
    n_layers: int = 32  # N layers
    n_heads: int = 32  # N heads for queries
    n_kv_heads: int | None = None  # N heads for keys and values
    vocab_size: int = -1  # Set when loading the tokenizer
    multiple_of: int = 256  # All intermediate dims will be a multiple of this
    ffn_dim_multiplier: float | None = None  # Multiplier for the FFN dim
    norm_eps: float = 1e-5  # Epsilon for the layer norm numerical stability

    # KV Cache
    max_batch_size: int = 32  # Max batch size
    max_seq_len: int = 2048  # Max sequence length

    device: str | None = None  # Device to run the model on


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str | None = None, theta: float = 10000.0):
    """Precompute the theta and position frequencies.

    Args:
        head_dim (int): The dimension of the head.
        seq_len (int): The length of the sequence.
        device (str | None, optional): The device to run the model on. Defaults to None.
        theta (float, optional): The theta value. Defaults to 10000.0.

    Returns:
        torch.Tensor: The theta and position frequencies.
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    # Build the theta parameters to formula: theta_i = 10000 ^ (-2i / head_dim)
    # Shape: (Head_dim // 2)
    theta_numerator = torch.arange(0, head_dim, 2)
    # Shape: (Head_dim // 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Build the position (m parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Shape: (seq_len) . (Head_dim // 2) -> (seq_len, Head_dim // 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form (seq_len, Head_dim // 2) -> (seq_len, Head_dim)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str | None = None) -> torch.Tensor:
    """Apply the rotary embedding to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        freqs_complex (torch.Tensor): The precomputed frequencies.
        device (str | None, optional): The device to run the model on. Defaults to None.

    Returns:
        torch.Tensor: The input tensor with the rotary embedding applied.
    """
    # This operation takes two consecutive dimensions and fuses them into a single dimension
    # (B, seq_len, n_heads, head_dim) -> (B, seq_len, n_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    freqs_complex = torch.unsqueeze(freqs_complex, 0).unsqueeze(2)
    # (B, seq_len, n_heads, head_dim // 2) * (1, seq_len, 1, head_dim) -> (B, seq_len, n_heads, head_dim // 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, n_heads, head_dim // 2) -> (B, seq_len, n_heads, head_dim // 2, 2)
    x_rotated = torch.view_as_real(x_rotated)
    # (B, seq_len, n_heads, head_dim // 2, 2) -> (B, seq_len, n_heads, head_dim)
    x_out = x_rotated.reshape(*x.shape)
    return x_out.astype(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """RMSNorm layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): The epsilon value. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Dim) * (B, seq_len, Dim) -> (B, seq_len, Dim)
        return self._norm(x.float()).astype(x) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat the keys and values for the number of repetitions.

    Args:
        x (torch.Tensor): (B, seq_len, n_kv_heads, head_dim)
        n_rep (int): The number of repetitions.
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # Essentially expand the last dimension of the tensor then flatten the last two dimensions
        # (B, seq_len, n_kv_heads, 1, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Number of heads for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.args = args
        # Number of heads for queries
        self.n_heads_q = args.n_heads
        # Number of repetitions of the keys and values
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Query, Key, Value projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV Cache
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): (B, seq_len=1, dim)
            start_pos (int): The position of the first token in the sequence.
            freqs_complex (torch.Tensor): (1, seq_len, head_dim)

        Returns:
            torch.Tensor: (B, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        xq = self.wq(x)  # (B, 1, n_heads_q * head_dim)
        xk = self.wk(x)  # (B, 1, n_kv_heads * head_dim)
        xv = self.wv(x)  # (B, 1, n_kv_heads * head_dim)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)  # (B, 1, n_heads_q, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)  # (B, 1, n_kv_heads, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)  # (B, 1, n_kv_heads, head_dim)

        # Apply the rotary embedding
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)

        # Replace the cache with the new keys and values
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Compute the attention weights
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeate the heads of the keys and values for the number of repetitions
        keys = repeat_kv(keys, self.n_rep)  # (B, seq_len, n_heads_q, head_dim)
        values = repeat_kv(values, self.n_rep)  # (B, seq_len, n_heads_q, head_dim)

        # (B, 1, h_q, head_dum) -> (B, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)  # (B, n_heads_q, 1, head_dim)
        keys = keys.transpose(1, 2)  # (B, n_heads_q, seq_len, head_dim)
        values = values.transpose(1, 2)  # (B, n_heads_q, seq_len, head_dim)

        # Compute the attention weights
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # (B, n_heads_q, 1, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (B, n_heads_q, 1, seq_len)

        out = torch.matmul(scores, values)  # (B, n_heads_q, 1, head_dim)
        # (B, n_heads_q, 1, head_dim) -> (B, 1, n_heads_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (B, 1, dim)

        return self.wo(out)  # (B, 1, dim) -> (B, 1, dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # e.g. hidden_size = 7, multiple_of = 5
        # 7 + 5 - 1 // 5 = 11 // 5 = 2
        # 2 * 5 = 10

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))
        x_v = self.w3(swish)
        x = swish * x_v
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Feed-forward normalization
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder block.

        Args:
            x (torch.Tensor): (B, seq_len, dim)
            start_pos (int): The position of the first token in the sequence.
            freqs_complex (torch.Tensor): (1, seq_len, head_dim)

        Returns:
            torch.Tensor: (B, seq_len, dim)
        """
        # (B, Seq_len, dim) + (B, Seq_len, dim) -> (B, Seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        h = h + self.feed_forward.forward(self.ffn_norm(h))
        return h


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # TODO: Add repeating encoder blocks
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # RMSNorm layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute frequencies
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """This model can only be used for inference. So the seq_len size must be 1.
            This is due to KV cache, which stores the keys and values of the previous tokens.

        Args:
            tokens (torch.Tensor): (B, Seq_len)
            start_pos (int): The position of the first token in the sequence.

        Returns:
            torch.Tensor: (B, Seq_len, D)
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token can be processed at a time"

        # Embeddings (B, Seq_len) -> (B, Seq_len, D)
        h = self.tok_embeddings(tokens)

        # Retrieve hte pairs (m, theta), coresponding to the positions [start_pos, start_pos + seq_len]
        # This is precomputed for all positions in the sequence length
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply the encoder blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        # Apply the RMSNorm
        h = self.norm(h)

        # Apply the output layer
        output = self.output(h).float()

        return output
