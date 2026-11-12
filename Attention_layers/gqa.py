import torch
import torch.nn.functional as F
from torch import nn

class VanillaGQA(nn.Module):
    def __init__(self, d_input, d_output, hdim_qk, hdim_v, n_heads, grouping) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.hdim_qk = hdim_qk
        self.hdim_v = hdim_v
        self.n_heads = n_heads
        self.grouping = grouping
        # dimensions and KV head num
        assert n_heads % grouping == 0, "Heads not diVisible into groups"
        self.KV_heads = n_heads // grouping
        # Weights
        self.W_K = nn.Linear(d_input, self.KV_heads * hdim_qk)
        self.W_V = nn.Linear(d_input, self.KV_heads * hdim_v)
        self.W_Q = nn.Linear(d_input, n_heads * hdim_qk)
        self.W_o = nn.Linear(hdim_v * n_heads, d_output)
        # device
        self.device = 'mps'

    def efficient_simple_forward(self, x):
        # always causal
        B,T,_ = x.shape
        K = self.W_K(x)
        V = self.W_V(x)
        Q = self.W_Q(x)
        Q = Q.view(B, T, self.KV_heads, self.grouping, self.hdim_qk).permute(0, 2, 3, 1, 4)
        K = K.view(B, T, self.KV_heads, self.hdim_qk).permute(0, 2, 1, 3) # KV_heads = n_heads // grouping
        V = V.view(B, T, self.KV_heads, self.hdim_v).permute(0, 2, 1, 3)
        raw_scores = (torch.einsum('bngtk,bnsk->bngts', Q, K)/ (self.hdim_qk ** 0.5)).masked_fill(torch.tril(torch.ones((T, T), device=x.device)) == 0, float('-inf'))
        attn = torch.einsum('bngts, bnsv->bngtv', F.softmax(raw_scores, dim=-1), V).permute(0, 3, 1, 2, 4).reshape(B, T, -1)
        return self.W_o(attn) # (B, T, O)


    def forward(self, x, is_causal=True, use_flash = True):
        B,T,_ = x.shape
        K = self.W_K(x)
        V = self.W_V(x)
        Q = self.W_Q(x)
        Q = Q.view(B,T, self.n_heads, self.hdim_qk)
        K = K.view(B,T, self.KV_heads, self.hdim_qk) # KV_heads = n_heads // grouping
        V = V.view(B,T, self.KV_heads, self.hdim_v)
        # Get attention 
        if use_flash:
            Q = Q.transpose(1,2)
            K = K.transpose(1,2)
            V = V.transpose(1,2)
            attn = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True).transpose(1,2)
        else:
            Q = Q.view(B,T,self.KV_heads, self.grouping, self.hdim_qk)
            raw_scores = torch.einsum('btngd, bsnd->btsng', Q, K) / (self.hdim_qk ** 0.5) # A grouping dim gets added to K
            M = torch.tril(torch.ones((T, T), device=x.device)).view(T, T, 1, 1)
            masked_raw_scores = raw_scores.masked_fill(M == 0, float('-inf'))
            scores = F.softmax(masked_raw_scores, dim=2)
            attn = torch.einsum('bstng,btnv->bsngv', scores, V) # (B, T, n_heads, grouping, hdim_v)
        attn = attn.reshape(B,T, self.hdim_v * self.n_heads)
        return self.W_o(attn)#(B, T, O)

if __name__ == '__main__':
    data = torch.randn((2,3,5))
    model = VanillaGQA(d_input = 5, d_output = 7, hdim_qk = 8, hdim_v = 13, grouping = 4, n_heads = 8)
    flash_outputs = model(data, use_flash=True)
    eff_outs = model.efficient_simple_forward(data)
    assert torch.allclose(flash_outputs, eff_outs, atol=1e-6), f"2!:Outputs are not the same. diff = {flash_outputs - eff_outs}"
