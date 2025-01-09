import math
from dataclasses import dataclass
from logging import getLogger

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812


logger = getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """This is multi-head attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch (3 * config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, and embedding dimension
        # Calculate query, key, and value for all heads in the batch and move head dimension up
        # nh = number of heads, hs = head size, C (n channels) = nh * hs
        # In GPT-2, nh = 12, hs = 64, C = 768
        qkv = self.c_attn(x)
        # Split the qkv tensor into three separate tensors: q, k, and v
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape the key, query, and value for multi-head attention. Essentially, makes nh as batch dimension parallel
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention (matrtlialzes the large TxT matrix for all queries, keys)
        # Scaled dot product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply the causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # Normalize the attention scores; sum = 1
        att = F.softmax(att, dim=-1)

        # Get output of the attention with the value
        y = att @ v  # (B, nh, T, hs)
        # Reshape the output to be a 3D tensor with the head
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer to project input to 4 times the embedding dimension, allows for more complex representations
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU activation function - approximate tanh, is mostly redudant now but was used in the original GPT-2 paper
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connection around the layer
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # Number of tokens in the dataset: 50k BPE merges + 256 UTF-8 tokens + special tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Get token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos

        for block in self.transformer.h:
            x = block(x)
        # Final Layer Norm
        x = self.transformer.ln_f(x)

        # Get the logits
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            # Compute the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    num_return_sequences = 5
    max_length = 30

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)

    # Prefix tokens
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, how are you?")
    tokens = torch.tensor(tokens, dtype=torch.long)
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    torch.manual_seed(42)
    # Loop and keep adding tokens to the sequence
    while x.size(-1) < max_length:
        # No need to compute gradients
        with torch.no_grad():
            # Get logits from last column
            logits, _ = model(x)
            logits = logits[:, -1, :]
            # Get the probabilities over vocab
            probs = F.softmax(logits, dim=-1)
            # Get top 50 probabilities and indices
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Sample from the top 50 probabilities
            ix = torch.multinomial(topk_probs, num_samples=1)
            # Get the sampled token
            xcol = torch.gather(topk_indices, dim=-1, index=ix)
            # Add the sampled token to the sequence
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
