import torch
import torch.nn as nn

from .. import CLIP_SAMPLES, HOP_LENGTH, N_BINS, BATCH_SIZE

def give_nn_conv_layer(in_channels, out_channels, kernel_size, stride):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )

class PatchLayer(nn.Module):
    # Takes in a patch of (freq_dim, time_dim), gives out a (patch_embedding)
    # Input: (B, 1, n_mels, T_frames) — treat as 2D image
    # Output: (B, T_patches, d_model),
    # Conv2d gives (B, d_model, H_out, W_out), which we then cast to (B, T_patches, d_model) by permuting and reshaping

    def __init__(self, freq_dim=N_BINS, time_dim=CLIP_SAMPLES // HOP_LENGTH + 1, n_layers=3, emb_dim=128, time_hop=2, freq_hop=2):
        super().__init__()
        self.freq_dim = freq_dim # Frequency dimension of the patch (number of mel bins)
        self.time_dim = time_dim # Time dimension of the patch (number of time frames)
        self.emb_dim = emb_dim # Desired embedding dimension for each patch(d_model)
        self.time_hop = time_hop
        self.freq_hop = freq_hop
        
        
        self.layers = []
        channel_dims = [(emb_dim // n_layers * (2**i)) for i in range(n_layers)] # Multiply dimensions by 2 every layer, only for the first half of the layers, then keep it constant
        for i in range(n_layers):
            # 
            self.layers.append(
                give_nn_conv_layer(
                    in_channels=1 if i == 0 else channel_dims[i - 1],
                    out_channels=channel_dims[i] ,
                    kernel_size=(freq_dim//n_layers * (2**i), time_dim),
                    stride=(freq_hop, time_hop)
                )
            )
            self.layers.append(nn.ReLU())
        
        self.proj = nn.Sequential(
            [i for i in self.layers],
        )
        
    def forward(self, x):
        B,F,T = x.shape
        # Shape of x: (B, F, T)
        x = x.unsqueeze(1) # (B, 1, F, T)
        # Unsqueeze to (B, 1, F, T)
        x = self.proj(x) # (B, d_model, H_out, W_out)
        x = x.transpose(1, 2) # (B, H_out, d_model, W_out)
        x = x.squeeze(2) # (B, W_out, emb_dim)
        pass
    
if __name__ == "__main__":
    # Local testing
    patch_layer = PatchLayer()
    dummy_input = torch.randn(BATCH_SIZE, N_BINS, CLIP_SAMPLES // HOP_LENGTH + 1) # (B, F, T)
    print("Dummy input shape:", dummy_input.shape)
    output = patch_layer(dummy_input)
    print("Output shape:", output.shape)