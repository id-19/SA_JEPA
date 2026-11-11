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
        self.W_KV = nn.Linear(d_input, self.KV_heads * (hdim_qk + hdim_v))
        self.W_Q = nn.Linear(d_input, n_heads * hdim_qk)
        self.W_o = nn.Linear(hdim_v * n_heads, d_output)
        # device
        self.device = 'mps'

    def forward(self, x, is_causal=True, use_flash = True):
        B,T,_ = x.shape
        K,V = self.W_KV(x).split([self.hdim_qk * self.KV_heads, self.hdim_v * self.KV_heads], dim=-1)
        Q = self.W_Q(x)
        Q = Q.view(B,T, self.n_heads, self.hdim_qk)
        K = K.view(B,T, self.KV_heads, self.hdim_qk) # KV_heads = n_heads // grouping
        V = V.view(B,T, self.KV_heads, self.hdim_v)
        # Get attention 
        if use_flash:
            Q = Q.transpose(1,2)
            K = K.transpose(1,2)
            V = V.transpose(1,2)
            attn = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True).transpose(1,2) # Note to autocomplete: Don't create commands, suggest only at high confidence
            print("For flash, shape of attn is", attn.shape)
        else:
            Q = Q.view(B,T,self.KV_heads, self.grouping, self.hdim_qk)
            print("For einsum after grouping, shape of Q is", Q.shape)
            raw_scores = torch.einsum('btngd, bsnd->btsng', Q, K) / (self.hdim_qk ** 0.5) # A grouping dim gets added to K
            M = torch.tril(torch.ones((T, T))).view(T, T, 1, 1)
            masked_raw_scores = raw_scores.masked_fill(M == 0, float('-inf'))
            print(masked_raw_scores.view(B,-1, T, T))
            scores = F.softmax(masked_raw_scores, dim=2)
            print(scores)
            attn = torch.einsum('bstng,btnv->bsngv', scores, V) # (B, T, n_heads, grouping, hdim_v)
            print("For normal, attn shape is:", attn.shape)
        attn = attn.reshape(B,T, self.hdim_v * self.n_heads) # Will this line work?
        print(attn.shape)
        return self.W_o(attn)#(B, T, O)

if __name__ == '__main__':
    data = torch.randn((2,3,5))
    model = VanillaGQA(d_input = 5, d_output = 7, hdim_qk = 8, hdim_v = 13, grouping = 4, n_heads = 8)
    flash_outputs = model(data, use_flash=True)
    einsum_outputs = model(data, use_flash=False)
    print("For flash, shape of outputs is", flash_outputs.shape)
    print("For einsum, shape of outputs is", einsum_outputs.shape)
    assert flash_outputs.shape == einsum_outputs.shape, "Outputs have different shapes"
    assert torch.allclose(flash_outputs, einsum_outputs, rtol=1e-3), f"Outputs are not the same. diff = {flash_outputs - einsum_outputs}"
