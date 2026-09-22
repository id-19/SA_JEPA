# Simple, discrete SSM(A,B,C)
import torch
import torch.nn as nn

class mySSM(nn.Module):
    def __init__(self, d_input=1, d_state=16, d_output=1):
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # State transition matrices
        self.A = nn.Parameter(torch.randn((d_state, d_state)))
        # self.proj = nn.Linear(d_input, d_input * 4) # Can just skip this step for now
        self.B = nn.Parameter(torch.randn((d_input, d_state)))

        # Output matrix
        self.C = nn.Parameter(torch.randn((d_state, d_output)))

        # re-norm
        self.log_rate = nn.Parameter(torch.log(torch.arange(1., d_state + 1))) # state-wise volume knobs

    def _eff_a(self,):
        # re-norm A
        renorm_a = 0.99 * self.A / torch.sum(torch.abs(self.A), dim=1, keepdim=True)

        # volume sliders
        sliders = torch.exp(-torch.exp(self.log_rate)) # Bound between 0 and 1, these scale every row

        return sliders * renorm_a

    def forward(self, x, debug=False):
        # At every timestep(i.e. every input for del = 1)
        B,T,C = x.shape # C is assumed d_input for now
        assert C == self.d_input
        outputs = []
        eff_a = self._eff_a()
        h = torch.zeros((B, self.d_state))
        step_outs = []
        # print("Batch:", batch)
        for time_step in range(T):
            h = h * eff_a.T + x[:, time_step] @ self.B
            # if time_step % 10 == 0 and debug:
            #     print(f"State mean(.abs()) at time step {time_step} : {torch.abs(h).mean(dim=-1)}")
            out = h @ self.C
            step_outs.append(out)
        # Tensor bana ke, append to the list
        outputs.append(torch.stack(step_outs, dim=0))
        return torch.stack(outputs, dim=0)


if __name__ == '__main__':
    mSSM = mySSM()
    x = torch.randn((5,100,1))
    outputs = mSSM(x)
    print(len(outputs), len(outputs[0]))
    print(outputs.mean())

    # ---- BIBO test: Bounded Input -> Bounded Output, forever ---- AI WRITTEN
    # 1. THE KICK: one bounded input pulse charges the state, then input = 0 forever.
    # 2. SILENCE: h = A @ h only — the system talks only to its own echoes now.
    # 3. THE WATCH: ||h|| each step. Falls geometrically = stable (memory that forgets).
    #    Grows = a mode with |eigenvalue| > 1 amplifying itself = NaN in training.
    with torch.no_grad():
        A_eff = mSSM._eff_a()
        h = torch.zeros(mSSM.d_state)
        h = A_eff @ h + torch.randn(1) @ mSSM.B       # the kick (bounded: a single randn)
        norms = [h.norm().item()]
        for _ in range(30):                           # silence
            h = A_eff @ h
            norms.append(h.norm().item())
    print("BIBO ||h||:", [f"{n:.4f}" for n in norms[:10]], "...", f"{norms[-1]:.2e}")
    print("BIBO verdict:", "PASS (state decays)" if norms[-1] < norms[0] else "FAIL (state grows!)")
