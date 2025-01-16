import inspect
import logging
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel as DDP


# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


#################################################
# Model Components
#################################################


class CausalSelfAttention(nn.Module):
    """This is multi-head attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch (3 * config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Old - removed with flash attention
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones(config.block_size, config.block_size)).view(
        #         1, 1, config.block_size, config.block_size
        #     ),
        # )

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

        # NOTE: This is the original implementation of the attention mechanism replaced with flash attention
        # # Attention (matrtlialzes the large TxT matrix for all queries, keys)
        # # Scaled dot product attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # Apply the causal mask
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # # Normalize the attention scores; sum = 1
        # att = F.softmax(att, dim=-1)
        # # Get output of the attention with the value
        # y = att @ v  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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


#################################################
# Model
#################################################


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

        # Note that WTE input imbeddings are shared between the input and the output classification
        # This tensor will now be used twice in the forward pass and will accumulate gradients twice
        # This tensor accounts for about 1/3 of the total parameters in the model so
        # its computationally efficient to share the parameters + we expect the parameters to be similar
        # This is a common technique in language models, particularly in the GPT-2 paper and
        # original Attention is All You Need paper
        # Weights sharing scheme!
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_embd) ** -0.5
            # Initialize the weights with a normal distribution STD of 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize the weights with a normal distribution STD of 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            # Compute the loss. We flatten from B, T, C to B*T, C
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

    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4, device_type="cpu"):
        # Step 1: get the parameters to optimize (requires_grad=True)
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        # Step 2: Create optimizer groups. All 2D parameters will be decayed, all others unchanged
        # i.e. all weights tensors in matmuls + embeddings will be decayed, all biases, and layernorms unchanged
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if master_process:
            logger.info(
                f"num decayed parameter tensors: {len(decay_params)}, "
                f"with {sum(p.numel() for p in decay_params):,} parameters"
            )
            logger.info(
                f"num non-decayed parameter tensors: {len(no_decay_params)}, "
                f"with {sum(p.numel() for p in no_decay_params):,} parameters"
            )
        # Fused AdamW optimizer
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            logger.info(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


#################################################
# Data Loader
#################################################


def load_tokens(file_path: str):
    npt = np.load(file_path)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        # Note: OLD
        # with open("data/input.txt", "r", encoding="utf-8") as file:
        #     text = file.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text[:1000])
        # # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # logger.info(f"Loaded {len(self.tokens)} tokens")
        # logger.info(f"1 Epoch: {len(self.tokens) // (self.B * self.T)} batches")

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # Get the shards
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"No shards found for split: {split}"
        if master_process:
            logger.info(f"Loaded {len(self.shards)} shards for split: {split}")

        self.reset()

    def reset(self):
        # This means that each process will start at a different position in the tokens
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T * self.num_processes
        # If we've reached the end of the tokens, move to next shard
        if self.current_pos + (B * T * self.num_processes + 1) >= len(self.tokens):
            # Update the shard if we've reached the end of the current shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y


if __name__ == "__main__":
    # Hyperparameters
    # total_batch_size = 524288  # 2**19 ~ 0.5M used for nice number
    total_batch_size = 262144  # 2**18 ~ 1M used for nice number
    B = 32  # Micro batch size (real is 16/32/64 - depends on GPU size)
    T = 1024  # Sequence length   (real is 1024)

    # Learning rate and optimizer parameters
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 250  # Real is 715
    max_steps = 5000  # Real is 19073

    # Initialize the distributed process group
    # torchrun command sets the env variables RANK, WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP is only supported on CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ.get("RANK"))
        ddp_world_size = int(os.environ.get("WORLD_SIZE"))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0  # For logging and checkpointing (on)
        if master_process:
            logger.info(f"Using device: {device}")
    else:
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        master_process = True
        # Set the device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        logger.info(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    # Gradient Accumulation
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "Total batch size must be divisible by micro batch size * sequence length * number of processes"
    gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        logger.info(f"Total batch size: {total_batch_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Download the tokenized data - this replaces the fineweb.py script
    if master_process:
        logger.info("Downloading tokenized data...")
    repo_id = "jfzhang/edu_fineweb10B_tokens_npy_files"
    local_dir = "./edu_fineweb10B/"
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    if master_process:
        logger.info("Tokenized data downloaded")

    # Get batches for training
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # Set the data type
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")

    # Create the model
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    use_compile = True  # compile can intefere with HellaSwag and text generation
    if device_type == "cuda" and use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    def get_lr(it: int):
        # 1. Linear warmup
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2. If fully decayed, return min_lr
        if it >= max_steps:
            return min_lr
        # 3. Otherwise, return the current learning rate based on the cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{ddp_rank}.txt")
    with open(log_file, "w") as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # Validate
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x = x.to(device)
                    y = y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    logger.info(f"Validation loss: {val_loss_accum:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum:.4f}\n")
                    if step > 0 and (step % 5000 == 0 or last_step):
                        checkpoint_path = os.path.join(log_dir, f"checkpoint_{step:05d}.pth")
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "config": raw_model.config,
                            "step": step,
                            "val_loss": val_loss_accum,
                            "optimizer": optimizer.state_dict(),
                        }
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)  # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            if master_process:
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    logger.info(f"rank {ddp_rank} sample {i}: {decoded}")

        # Train
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_batch in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            if ddp:
                model.require_backward_grad_sync = micro_batch == gradient_accumulation_steps - 1

            # Use autocast to automatically cast the model to the correct precision
            # Note running this on older GPUs may cause problems if not suported (Tesla gpus)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # We need to scale our loss by the number of micro batches to get the correct gradient
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach()
            # By keeping the loss in the loop, we can accumulate the gradients over the micro batches
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Determine the learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # This will wait for all scheduled work to finish
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T * gradient_accumulation_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            logger.info(
                f"Iteration: {step} | Loss: {loss_accum} | lr {lr:.6f} | Norm: {norm:.2f} | Time: {dt:.2f}ms | "
                f"Tokens/s: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum:.6f}\n")

    if ddp:
        destroy_process_group()
