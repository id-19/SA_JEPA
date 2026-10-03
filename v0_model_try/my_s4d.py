# S4D arm: diagonal of 2x2 rotation*shrink blocks (N/2 complex-conjugate modes)
# Each channel carries a 2-vector state; per-channel (r, omega) rotate+shrink it.
import torch
import torch.nn as nn


class myS4D(nn.Module):
    def __init__(self, d_input=1, d_state=16, d_output=1):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state      # number of channels; each channel is a 2-vector
        self.d_output = d_output

        # transition params, one per channel
        self.log_rate = nn.Parameter(torch.log(torch.arange(1., d_state + 1)))
        self.omega = nn.Parameter(torch.arange(start=-3.14, end=3.14, step = 6.28 / d_state)) # -pi to pi, in appropriate step size

        # input / output
        self.B = nn.Parameter(torch.randn(d_input, d_state, 2)) # (B, 1, d_input) @ (d_input, 2) -> (B, 1, 2)
        self.C = nn.Parameter(torch.randn(d_state, 2, d_output)) # h(B, d_state, 2) @ (d_state, 2, d_out) -> (B, d_out)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.d_input

        h = torch.zeros(B, self.d_state, 2)   # (B, d_state, 2): two numbers per channel
        r = torch.exp(-torch.exp(self.log_rate))   # (d_state,), in (0,1)

        # Bound self.omega so gradients don't get messy

        c = torch.cos(self.omega)             # (d_state,)
        s = torch.sin(self.omega)             # (d_state,)


        step_outs = []
        for t in range(T):
            u = h[:, :, 0]                         # (B, d_state)
            v = h[:, :, 1]                         # (B, d_state)
            # Rotate u and v
            ud, vd = r*((c*u) - (s*v)), r*((s*u) + (c*v))
            # print(ud.shape, vd.shape)
            # Put ud, vd into h
            h = torch.stack((ud,vd), dim=-1)
            # Now we insert the input kick
            # print("h shape before kick", h.shape)
            h += torch.einsum('bc,csk->bsk', x[:,t,:], self.B)
            # print("h shape after kick", h.shape)
            # Get the output
            # print()
            y = torch.einsum("bsk,sko->bo", h, self.C)
            step_outs.append(y)

        #  T *[(B, C)] -> (B, T, C)
        return torch.stack(step_outs, dim=1)

if __name__ == '__main__':
    input = torch.randn((8,16,1))
    model = myS4D(d_input = 1, d_state = 16, d_output = 1)
    outs = model(input)
    print(outs.shape)
