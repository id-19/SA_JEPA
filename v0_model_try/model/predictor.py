import torch
import torch.nn as nn

from v0_model_try.config import EMB_DIM

class Predictor(nn.Module):
    def __init__(self, inout_dim = EMB_DIM, ):
        super().__init__()
        # Takes in an input embedding, outputs one in the same latent space
        
        
