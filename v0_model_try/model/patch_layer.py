import torch
import torch.nn as nn
import sys
import math

from ..config import CLIP_SAMPLES, HOP_LENGTH, N_BINS, BATCH_SIZE

def give_nn_conv_layer(in_channels, out_channels, kernel_size, stride):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(0, 1), # No padding, we want to reduce the dimensions
    )

class PatchLayer(nn.Module):
    # Takes in a patch of (freq_dim, time_dim), gives out a (patch_embedding)
    # Input: (B, 1, n_mels, T_frames) — treat as 2D image
    # Output: (B, T_patches, d_model),
    # Conv2d gives (B, d_model, H_out, W_out), which we then cast to (B, T_patches, d_model) by permuting and reshaping

    def __init__(self, 
                 freq_dim=N_BINS, 
                 time_dim=CLIP_SAMPLES // HOP_LENGTH + 1, 
                 n_layers=3,
                 # dimensions
                 emb_dim=128,
                 # Conv param
                 kernel_size=(2, 2),
                time_hop=1
                 ):
        super().__init__()
        self.freq_dim = freq_dim # Frequency dimension of the patch (number of mel bins)
        self.time_dim = time_dim # Time dimension of the patch (number of time frames)
        self.emb_dim = emb_dim # Desired embedding dimension for each patch(d_model)
        self.time_hop = time_hop
        
        
        self.layers = []

        print(f"Time dimension // time hop: {math.log(time_dim // time_hop, 2)}")
        print(f"Initial number of layers requested: {n_layers}")
        n_layers = min(n_layers, math.log(time_dim // time_hop, 2) + 1) # Ensure we don't exceed the dimensions of the input
        n_layers = int(n_layers) # Convert to integer
        print("Adjusted number of layers based on input dimensions:", n_layers)
        
        channel_dims = [(emb_dim//2**i) for i in range(n_layers)][::-1] # Multiply dimensions by 2 every layer, only for the first half of the layers, then keep it constant
        print("Channel dimensions for each layer:", channel_dims)
        
        self.freq_hop = freq_dim // (2 ** (n_layers - 1)) # Calculate the frequency hop based on the number of layers and the input frequency dimension
        for i in range(n_layers):
            # 
            self.layers.append(
                give_nn_conv_layer(
                    in_channels=1 if i == 0 else channel_dims[i - 1],
                    out_channels=channel_dims[i] ,
                    kernel_size=kernel_size,
                    stride=(self.freq_hop, 1)
                )
            )
            self.layers.append(nn.ReLU())
        
        self.proj = nn.Sequential(*self.layers)
        
    def forward(self, x):
        B,F,T = x.shape
        # Shape of x: (B, F, T)
        x = x.unsqueeze(1) # (B, 1, F, T)
        # Unsqueeze to (B, 1, F, T) - (Batch size, 1 channel, W_in, H_in) 
        print("Input shape after unsqueeze:", x.shape)
        for layer in self.layers:
            x = layer(x)
            # print(layer)
            if not isinstance(layer, nn.ReLU):
                print("Shape after layer:", x.shape)
        x = x.squeeze(2) # (B, emb_dim, 1, H_out)
        x = x.transpose(1, 2) # (B, H_out, emb_dim) — treat H_out as T_patches
        return x
    
if __name__ == "__main__":
    # Local testing
    patch_layer = PatchLayer(n_layers=2)
    dummy_input = torch.randn(BATCH_SIZE, N_BINS, CLIP_SAMPLES // HOP_LENGTH + 1) # (B, F, T)
    print("Dummy input shape:", dummy_input.shape)
    output = patch_layer(dummy_input)
    print("Output shape:", output.shape)