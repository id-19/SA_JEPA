import torch
import torch.nn as nn
from model.patch_layer import PatchLayer

from config import CLIP_SAMPLES, HOP_LENGTH, N_BINS, BATCH_SIZE, EMB_DIM, LATENT_DIM
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout, num_ffn_layers):
        super().__init__()
        # PyTorch MultiheadAttention expects (Sequence_Length, Batch, Features) by default, 
        # or (Batch, Sequence_Length, Features) if batch_first=True. Let's use batch_first=True.
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Build the Feedforward Network properly
        ffn_layers = []
        in_dim = d_model
        for i in range(num_ffn_layers):
            # Final sub-layer must project back to d_model to allow the residual connection!
            out_dim = d_model if i == num_ffn_layers - 1 else dim_feedforward
            ffn_layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_ffn_layers - 1:
                ffn_layers.append(nn.GELU())
                ffn_layers.append(nn.Dropout(dropout))
            else:
                # Standard practice: apply normalization at the end of FFN or before projection
                ffn_layers.append(nn.LayerNorm(out_dim))
                
            in_dim = out_dim
            
        self.ffn = nn.Sequential(*ffn_layers)
        
        # Pre-LN structure (matches your forward pass logic)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x):
        # 1. Self-attention + Residual
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_output  
        
        # 2. Feedforward network + Residual (Now shapes match perfectly!)
        norm_x2 = self.norm2(x)
        ff_output = self.ffn(norm_x2)
        x = x + ff_output  
        
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim = LATENT_DIM, num_blocks = 4, d_model = EMB_DIM, n_heads = 4, dim_feedforward = 256, dropout = 0.1, num_ffn_layers = 2):
        super().__init__()
        self.patch_layer = PatchLayer()
        
        # Keep blocks isolated in a ModuleList
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dim_feedforward, dropout, num_ffn_layers) 
            for _ in range(num_blocks)
        ])
        
        # Separate the final projection out of the block loop
        self.final_projection = nn.Linear(d_model, latent_dim)
    
    def forward(self, x):
        x = self.patch_layer(x) # Expected shape: (B, T_patches, d_model)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Project to final latent dimension
        x = self.final_projection(x)
        return x
    
if __name__ == "__main__":
    # Test the Encoder with dummy input
    dummy_input = torch.randn(BATCH_SIZE, N_BINS, CLIP_SAMPLES // HOP_LENGTH + 1)  # (B, F, T)
    encoder = Encoder()
    output = encoder(dummy_input)
    print("Output shape from Encoder:", output.shape)  # Expected: (B, T_patches, latent_dim)