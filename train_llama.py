"""
Distributed Llama 2 training script with data parallelism.

Usage:
$ torchrun --nproc_per_node=8 train_llama.py
"""

import math
import os
import sys
import time
from datetime import timedelta
import glob
import numpy as np
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
from torch.utils.checkpoint import checkpoint


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B, self.T = B, T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files found matching {filename_pattern}"
        self._load_shard(0)

    def _load_shard(self, shard_idx):
        self.current_shard = shard_idx
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_position += self.B * self.T * self.num_processes
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self._load_shard((self.current_shard + 1) % len(self.files))
        return x.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape for broadcasting
    ndim = xq_r.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_r.shape)]
    freqs_cos = freqs_cos.view(shape)
    freqs_sin = freqs_sin.view(shape)

    # Apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert hasattr(self, "mask")
        scores = scores + self.mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        if self.training:
            h = x + checkpoint(self.attention.forward, self.attention_norm(x), freqs_cos, freqs_sin, use_reentrant=False)
            out = h + checkpoint(self.feed_forward.forward, self.ffn_norm(h), use_reentrant=False)
        else:
            h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
            out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        all_params = [p for p in self.parameters() if p.requires_grad]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            all_params,
            lr=learning_rate,
            betas=betas,
            eps=1e-10,  # Smaller epsilon for distributed training stability
            weight_decay=weight_decay,
            **extra_args,
        )

        if master_process:
            num_params = sum(p.numel() for p in all_params)
            print(f"total params: {len(all_params)} tensors, {num_params:,} parameters")
            print(f"{learning_rate=}, {weight_decay=}, {betas=}, eps: 1e-10")
            print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 204e12  # H100 GPU bfloat16 peak flops
        return flops_achieved / flops_promised

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@dataclass
class TrainingConfig:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 100
    eval_only: bool = False

    # Data
    batch_size: int = 4
    max_seq_len: int = 1024
    train_data_path: str = "data/fineweb10B/fineweb_train_*.bin"
    val_data_path: str = "data/fineweb10B/fineweb_val_*.bin"
    vocab_source: str = "custom"
    vocab_size: int = 50257

    # Model (7B configuration)
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    multiple_of: int = 256
    dropout: float = 0.0

    # Optimizer
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    max_iters: int = 100
    weight_decay: float = 0.0
    beta1: float = 0.8
    beta2: float = 0.95

    # Learning rate schedule
    decay_lr: bool = True
    cooldown_frac: float = 0.45
    min_lr_frac: float = 0.1

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True

    def __post_init__(self):
        self.lr_decay_iters = self.max_iters
        self.min_lr = self.learning_rate * self.min_lr_frac

        assert self.vocab_source in ["llama2", "custom"]
        assert self.vocab_source == "custom" or self.vocab_size == 32000


def setup_distributed(config):
    # Assert we have multiple GPUs available
    assert torch.cuda.is_available(), "CUDA must be available for distributed training"
    assert torch.cuda.device_count() > 1, "Multiple GPUs required for distributed training"

    init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
    tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.max_seq_len

    if master_process:
        print(f"tokens per iteration: {tokens_per_iter:,}")
        print(f"breakdown: {config.gradient_accumulation_steps=} * {ddp_world_size=} * {config.batch_size=} * {config.max_seq_len=}")

    torch.manual_seed(1337 + ddp_rank)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"

    device_type = "cuda"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, device_type, ctx


config = TrainingConfig()
ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, device_type, ctx = setup_distributed(config)

train_loader = DistributedDataLoader(config.train_data_path, config.batch_size, config.max_seq_len, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(config.val_data_path, config.batch_size, config.max_seq_len, ddp_rank, ddp_world_size)

iter_num = 0
best_val_loss = 1e9

model_args = ModelArgs(
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,
    vocab_size=config.vocab_size,
    multiple_of=config.multiple_of,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
)
model = Transformer(model_args)
model.to(device)

# FSDP setup - modern approach with mixed precision
fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
}
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)

assert isinstance(model, FSDPModule)

# Create optimizer AFTER FSDP sharding so it sees DTensors
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

if config.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        loader = train_loader if split == "train" else val_loader
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = loader.next_batch()
            _ = model(X, Y)
            loss = model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(step: int):
    x = step / config.max_iters
    assert 0 <= x < 1
    if x < 1 - config.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / config.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


if master_process:
    print(f"Python {sys.version}")
    print(f"PyTorch {torch.version.__version__} (CUDA {torch.version.cuda})")

    import subprocess

    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)

# training loop
X, Y = train_loader.next_batch()
t0 = time.time()
raw_model = model

for iter_num in range(iter_num, config.max_iters):
    lr = get_lr(iter_num) * config.learning_rate if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % config.eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
    if iter_num == 0 and config.eval_only:
        break

    for micro_step in range(config.gradient_accumulation_steps):
        is_last_micro = micro_step == config.gradient_accumulation_steps - 1

        model.set_requires_gradient_sync(is_last_micro, recurse=True)
        model.set_reshard_after_backward(is_last_micro, recurse=True)

        logits = model(X, Y)
        loss = model.last_loss / config.gradient_accumulation_steps
        X, Y = train_loader.next_batch()
        loss.backward()

        if is_last_micro:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        lossf = loss.item() * config.gradient_accumulation_steps
        mfu = model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt * 1000:.2f}ms | mfu {mfu * 100:.2f}%")

destroy_process_group()
