# Implement an MoE based architecture that takes in (B, T, C) of audio signals, give out a good encoding.
# Main trick will be in training: Train to get gaussian distribution of latents
import math

import torch
import torch.nn as nn

from v0_model_try.config import EncoderConfig, get_encoder_defaults


class Router(nn.Module):
    # Simple MLP that takes in token features and returns top-k expert indices.
    def __init__(self, num_experts: int, num_active_experts: int, dim: int, num_layers: int):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if num_active_experts <= 0:
            raise ValueError("num_active_experts must be > 0")
        if num_active_experts > num_experts:
            raise ValueError("num_active_experts must be <= num_experts")

        depth = max(1, num_layers)
        layers = [nn.Linear(dim, dim) for _ in range(depth - 1)]
        layers.append(nn.Linear(dim, num_experts))
        self.layers = nn.Sequential(*layers)
        self.num_active = num_active_experts

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        logits = self.layers(x)
        # Pool token-level logits per sample so a sample selects its active experts.
        pooled = logits.mean(dim=1)
        _, indices = torch.topk(pooled, self.num_active, dim=-1)
        return indices


class Expert(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, scale_up_ffn=4):
        super().__init__()

        layers = []
        mid_dim = scale_up_ffn * input_dim
        layers.append(nn.Linear(input_dim, mid_dim))
        layers.append(nn.SiLU())
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(mid_dim, mid_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(mid_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MoE(nn.Module):
    # just a few FFNs
    def __init__(
        self,
        num_experts: int,
        model_dim: int,
        latent_dim: int,
        scale_up_ffn: int,
        expert_depth: int,
        sparsity: float,
        router_depth: int,
    ):
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if latent_dim <= 0 or model_dim <= 0:
            raise ValueError("latent_dim and model_dim must be > 0")

        self.active_experts = max(1, min(num_experts, math.floor(num_experts * sparsity)))
        self.project_down_layer = nn.Linear(model_dim, latent_dim)
        self.router = Router(
            num_experts=num_experts,
            num_active_experts=self.active_experts,
            dim=latent_dim,
            num_layers=router_depth,
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    expert_depth,
                    input_dim=latent_dim,
                    scale_up_ffn=scale_up_ffn,
                    output_dim=latent_dim,
                )
                for _ in range(num_experts)
            ]
        )
        self.coalesce_layer = nn.Linear(self.active_experts * latent_dim, model_dim)

    def forward(self, x):
        x = self.project_down_layer(x)
        expert_indices = self.router(x)  # (B, active_experts)

        batch_outputs = []
        for sample_idx, sample_indices in enumerate(expert_indices):
            active_outputs = [self.experts[int(i.item())](x[sample_idx : sample_idx + 1]) for i in sample_indices]
            batch_outputs.append(torch.cat(active_outputs, dim=-1))

        stacked = torch.cat(batch_outputs, dim=0)
        return self.coalesce_layer(stacked)


class EncoderBlock(nn.Module):
    # Attention + MoE block
    def __init__(
        self,
        num_experts: int,
        model_dim: int,
        latent_dim: int,
        scale_up_ffn: int,
        expert_depth: int,
        sparsity: float,
        num_attention_heads: int,
        dropout: float,
        router_depth: int,
    ):
        super().__init__()
        self.moe = MoE(
            num_experts=num_experts,
            model_dim=model_dim,
            latent_dim=latent_dim,
            scale_up_ffn=scale_up_ffn,
            expert_depth=expert_depth,
            sparsity=sparsity,
            router_depth=router_depth,
        )
        self.attention = nn.MultiheadAttention(model_dim, num_heads=num_attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x = residual + self.dropout(attn_output)

        residual = x
        moe_output = self.moe(self.norm2(x))
        x = residual + self.dropout(moe_output)
        return x


class Encoder(nn.Module):
    """Simple encoder stack: per-step linear projection, attention and MoE blocks."""

    def __init__(
        self,
        time_dimension=None,
        config: EncoderConfig | None = None,
        **kwargs,
    ):
        super().__init__()

        encoder_cfg = config or get_encoder_defaults()

        self.model_dim = encoder_cfg.model_dim
        self.output_dim = encoder_cfg.output_dim
        self.time_dimension = time_dimension

        self.input_projection = nn.Linear(encoder_cfg.input_dim, encoder_cfg.model_dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    num_experts=encoder_cfg.num_experts,
                    model_dim=encoder_cfg.model_dim,
                    latent_dim=encoder_cfg.latent_dim,
                    scale_up_ffn=encoder_cfg.ffn_scale_up,
                    expert_depth=encoder_cfg.expert_depth,
                    sparsity=encoder_cfg.sparsity,
                    num_attention_heads=encoder_cfg.num_attention_heads,
                    dropout=encoder_cfg.dropout,
                    router_depth=encoder_cfg.router_depth,
                )
                for _ in range(encoder_cfg.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(encoder_cfg.model_dim)
        self.output_projection = nn.Identity() if encoder_cfg.output_dim == encoder_cfg.model_dim else nn.Linear(
            encoder_cfg.model_dim, encoder_cfg.output_dim
        )
        self.time_embedding = (
            nn.Parameter(torch.zeros(1, time_dimension, self.model_dim), requires_grad=True)
            if time_dimension is not None
            else None
        )

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments for Encoder: {', '.join(sorted(kwargs.keys()))}")

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_projection(x)

        if self.time_embedding is not None:
            if x.size(1) > self.time_embedding.size(1):
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds configured max {self.time_embedding.size(1)}. "
                    "Increase time_dimension when constructing encoder."
                )
            x = x + self.time_embedding[:, : x.size(1), :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_projection(x)
        return x


if __name__ == "__main__":
    import torch

    def _run_smoke_tests() -> None:
        cfg = get_encoder_defaults()
        time_dimension = 64
        batch_size = 2

        encoder = Encoder(time_dimension=time_dimension, config=cfg)
        x = torch.randn(batch_size, time_dimension, cfg.input_dim)

        out = encoder(x)
        assert out.shape == (batch_size, time_dimension, cfg.output_dim)

        target = torch.zeros_like(out)
        loss = (out - target).pow(2).mean()
        loss.backward()

        # Ensure gradients flow through the full stack.
        assert encoder.output_projection.weight.grad is not None

        custom_cfg = EncoderConfig(
            num_blocks=1,
            num_attention_heads=2,
            model_dim=128,
            latent_dim=32,
            output_dim=64,
            num_experts=2,
            ffn_scale_up=2,
        )
        encoder_small = Encoder(time_dimension=time_dimension, config=custom_cfg)
        x_small = torch.randn(1, time_dimension, custom_cfg.input_dim)
        out_small = encoder_small(x_small)
        assert out_small.shape == (1, time_dimension, custom_cfg.output_dim)

        # This path should require a longer sequence than configured time embedding.
        long_cfg = EncoderConfig(input_dim=custom_cfg.input_dim, model_dim=custom_cfg.model_dim, output_dim=custom_cfg.output_dim)
        short_encoder = Encoder(time_dimension=4, config=long_cfg)
        too_long_input = torch.randn(1, 8, custom_cfg.input_dim)
        try:
            _ = short_encoder(too_long_input)
            raise AssertionError("Expected ValueError for sequence longer than configured time dimension")
        except ValueError:
            pass

    _run_smoke_tests()
