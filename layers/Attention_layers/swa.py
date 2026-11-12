# Implementing a vanilla attention layer
import torch
from torch.nn import functional as F
from torch import nn

class SWA(nn.Module):
    def __init__(self, d_in, d_out, d_model, n_heads, window, device = "mps"):
        super().__init__()
        self.d_input = d_in
        self.d_output = d_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.window = window
        assert d_model != 0 and d_model % self.n_heads == 0, "number of heads cannot be lesser than d_model"
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_in, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_out)
        self.device = device

    def _build_mask(self, T, device):
        # (query, key). True means that query may attend to that key.
        pos = torch.arange(T, device=device)
        dist = pos.unsqueeze(dim=-1) - pos.unsqueeze(dim=0)
        return (dist >= 0) & (dist <= self.window)

    def forward(self, x, use_flash=True):
        B,T,_ = x.shape
        Q,K,V = self.qkv(x).split([self.d_model, self.d_model, self.d_model], dim=-1) # [B,T, d_model]
        Q = Q.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B,T, self.n_heads, self.head_dim).transpose(1,2) # [B, N, T, d_head]
        mask = self._build_mask(T, x.device)
        if use_flash:
            # Bool mask: True attends. is_causal cannot be combined with a custom mask.
            Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        else:
            scores = torch.einsum('bnth, bnsh->bnts', Q, K) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(~mask.view(1, 1, T, T), float('-inf'))
            weights = F.softmax(scores, dim=-1)
            Y = torch.einsum('bnst,bntv->bnsv', weights, V)
        Y = Y.transpose(1, 2).reshape(B, T, self.d_model)
        return self.W_o(Y)

if __name__ == '__main__':
    import time
    d_in = 1
    m = SWA(d_in, d_out=3, d_model=14, n_heads=7, window=100)
    data = torch.randn((11, 120, d_in))
    m.to('mps')
    data = data.to('mps')
    t1 = time.time()
    outputs_flash = m(data)
    t2 = time.time()
    outputs = m(data, use_flash=False)
    t3 = time.time()
    print(outputs.shape, outputs - outputs_flash)
    print(torch.allclose(outputs, outputs_flash, rtol=1e-3,atol=1e-4))
    print(f"flash: {t2-t1}, regular: {t3-t2}. regular/flash: {(t3-t2)/(t2-t1)}")
