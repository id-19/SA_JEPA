import torch
import time
# from my_ssm import mySSM
# from my_s4d import myS4D
from my_sel_s4d import mySelS4D

t = torch.arange(100).float() / 25 # 4 cycles
f = torch.arange(25).float().unsqueeze(-1) / 5 # Goes from 0 cycles per time step to 5 cycles per timestep
# Unsqueeze because (f,1) with (t) broadcasts into (f, t)
phi = 2 * 3.1415 * torch.arange(25).float().unsqueeze(-1) / 25 # Initial phase goes from 0 to a full cycle

x = torch.sin(2 * 3.1415 * f * t + phi).unsqueeze(-1) # (f,t, 1) so linear can multiply "channel" easily

torch.manual_seed(0)
# model = mySSM(d_input=1, d_state=16)
model = mySelS4D(d_input=1, d_state=16, d_output=1)
optim = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
# print("named params:", [(n, tuple(p.shape)) for n, p in model.named_parameters()])
# print("eff_a before:", model.eff_a)
t0 = time.perf_counter()

for step in range(20000):
    pred = model(x[:, :-1, :]) # Predictions on all time-steps but last
    loss = ((pred - x[:, 1:]) ** 2).mean() # simple MSE

    optim.zero_grad() # Sets all model gradients to zero
    loss.backward() # Populate gradients
    optim.step()
    if step % 50 == 0:
        print(f"loss at step:{step} = {loss.item()}")
print("eff_a after:", model)
print(f"train time: {time.perf_counter() - t0:.3f}s for {step+1} steps")
