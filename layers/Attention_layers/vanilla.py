# Implementing a vanilla attention layer
import torch
from torch.nn import functional as F
from torch import nn

class VanillaAttention(nn.Module):
    def __init__(self, d_in, d_out, d_model, n_heads):
        super().__init__()
        self.d_input = d_in
        self.d_output = d_out
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model != 0 and d_model % self.n_heads == 0, "number of heads cannot be lesser than d_model"
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_in, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_out)

    def simple_forward(self, x):
        B,T,_ = x.shape
        Q,K,V = self.qkv(x).split([self.d_model, self.d_model, self.d_model], dim=-1) # [B,T, d_model]
        Q = Q.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B,T, self.n_heads, self.head_dim).transpose(1,2) # [B, N, T, d_head]
        M = torch.tril(torch.ones((T, T), device=x.device)).view(1, 1, T, T)
        scores_raw = torch.einsum('bnth, bnsh->bnts', Q, K) / (self.head_dim**0.5)# (B, N, T, T)
        # We don't sum up scores per head, we just concatenate later
        scores = scores_raw.masked_fill(M == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        # Now, multiply with V
        logits = torch.einsum('bnst,bntv->bnsv', scores, V).transpose(1,2)
        print(logits.shape)
        print(B,T,self.d_model )
        logits = logits.reshape(B, T, self.d_model)
        O = self.W_o(logits)
        return O

    def forward(self, x):
        B,T,_ = x.shape
        Q,K,V = self.qkv(x).split([self.d_model, self.d_model, self.d_model], dim=-1) # [B,T, d_model]
        Q = Q.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B,T, self.n_heads, self.head_dim).transpose(1,2) # [B, N, T, d_head]
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        Y = Y.transpose(1, 2).reshape(B, T, self.d_model)
        return self.W_o(Y)

if __name__ == '__main__':
    import time
    d_in = 1
    m = VanillaAttention(d_in, d_out=3, d_model=14, n_heads=7)
    data = torch.randn((11, 13, d_in))
    t1 = time.time()
    outputs_flash = m(data)
    t2 = time.time()
    outputs = m.simple_forward(data)
    t3 = time.time()
    print(outputs.shape, outputs - outputs_flash)
    print(torch.allclose(outputs, outputs_flash, rtol=1e-3,atol=1e-4))
    print(f"flash: {t2-t1}, regular: {t3-t2}. regular/flash: {(t3-t2)/(t2-t1)}")
