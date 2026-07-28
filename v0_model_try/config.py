from dataclasses import dataclass


SAMPLE_RATE = 24000
CLIP_LENGTH_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_LENGTH_SECONDS

N_FFT = 1024
HOP_LENGTH = 256
N_BINS = 80
EPS = 1e-6
BATCH_SIZE = 8

EMB_DIM = 512
LATENT_DIM = 128
MODEL_DIM = 256
NUM_EXPERTS = 4
SPARSITY = 0.1

# Router
ROUTER_DEPTH = 2

# Expert
FFN_SCALE_UP = 4


@dataclass(frozen=True)
class EncoderConfig:
    """Centralized encoder hyper-parameter defaults."""

    # Input/output dimensions
    input_dim: int = EMB_DIM
    model_dim: int = MODEL_DIM
    latent_dim: int = LATENT_DIM

    # MoE
    num_experts: int = NUM_EXPERTS
    sparsity: float = SPARSITY
    expert_depth: int = 4
    ffn_scale_up: int = FFN_SCALE_UP

    # Router
    router_depth: int = ROUTER_DEPTH

    # Attention + stack
    num_attention_heads: int = 4
    num_blocks: int = 2

    # Output
    output_dim: int = 128

    # Regularization
    dropout: float = 0.0


def get_encoder_defaults() -> EncoderConfig:
    return EncoderConfig()


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
