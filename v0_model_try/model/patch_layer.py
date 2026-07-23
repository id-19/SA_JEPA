import torch
import torch.nn as nn
# import sys
import math

from v0_model_try.config import CLIP_SAMPLES, HOP_LENGTH, N_BINS, BATCH_SIZE, EMB_DIM

def give_nn_conv_layer(in_channels, out_channels, kernel_size, stride):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(0, 0), # Padding to retain dimensions in time dim
    )

class PatchLayer(nn.Module):
    # Takes in a patch of (freq_dim, time_dim), gives out a (patch_embedding)
    # Input: (B, 1, n_mels, T_frames) — treat as 2D image
    # Output: (B, T_patches, d_model),
    # Conv2d gives (B, d_model, H_out, W_out), which we then cast to (B, T_patches, d_model) by permuting and reshaping

    def __init__(self,
                 freq_dim = N_BINS,
                 time_dim = CLIP_SAMPLES // HOP_LENGTH + 1,
                 # dimensions
                 emb_dim=EMB_DIM, # Desired embedding dimension for each patch (d_model)
                 # Conv param
                 kernel_size=(2, 2),
                 frequency_hop=2,
                 time_hop=1
                 ):
        super().__init__()
        self.freq_dim = freq_dim # Frequency dimension of the patch (number of mel bins)
        self.time_dim = time_dim # Time dimension of the patch (number of time frames)
        self.emb_dim = emb_dim # Desired embedding dimension for each patch(d_model)
        self.time_hop = min(time_hop, kernel_size[1]) # Ensure time_hop does not exceed kernel size in time dimension
        self.frequency_hop = min(frequency_hop, kernel_size[0]) # Ensure frequency_hop does not exceed kernel size in frequency dimension


        self.layers = []

        n_layers = int(math.log(freq_dim, frequency_hop)) # Number of layers needed to reduce freq_dim to 1
        # print("Adjusted number of layers based on input dimensions:", n_layers)

        channel_dims = [(emb_dim//2**i) for i in range(n_layers)][::-1] # Multiply dimensions by 2 every layer, only for the first half of the layers, then keep it constant
        # print("Channel dimensions for each layer:", channel_dims)


        for i in range(n_layers):
            self.layers.append(
                give_nn_conv_layer(
                    in_channels=1 if i == 0 else channel_dims[i - 1],
                    out_channels=channel_dims[i] ,
                    kernel_size=kernel_size,
                    stride=(frequency_hop, time_hop)
                )
            )
            self.layers.append(nn.ReLU())

        self.proj = nn.Sequential(*self.layers)

    def forward(self, x):
        B,F,T = x.shape
        # Shape of x: (B, F, T)
        x = x.unsqueeze(1) # (B, 1, F, T)
        # Unsqueeze to (B, 1, F, T) - (Batch size, 1 channel, W_in, H_in)
        # print("Input shape after unsqueeze:", x.shape)
        for layer in self.layers:
            x = layer(x)
            # print(layer)
            if not isinstance(layer, nn.ReLU):
                print("Shape after layer:", x.shape)
                pass
        x = x.squeeze(2) # (B, emb_dim, 1, H_out)
        x = x.transpose(1, 2) # (B, H_out, emb_dim) — treat H_out as T_patches
        return x

if __name__ == "__main__":
    # Local testing
    patch_layer = PatchLayer()
    dummy_input = torch.randn(BATCH_SIZE, N_BINS, CLIP_SAMPLES // HOP_LENGTH + 1) # (B, F, T)
    print("Dummy input shape:", dummy_input.shape)
    output = patch_layer(dummy_input)
    print("Output shape:", output.shape)
