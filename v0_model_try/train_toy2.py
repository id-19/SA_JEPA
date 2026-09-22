import torch
from my_ssm import mySSM

t = torch.arange(100).float() / 25 # 4 cycles
f = torch.arange(25).float().unsqueeze(-1) / 5 # Goes from 0 cycles per time step to 5 cycles per timestep
# Unsqueeze because (f,1) with (t) broadcasts into (f, t)
phi = 2 * 3.1415 * torch.arange(25).float().unsqueeze(-1) / 25 # Initial phase goes from 0 to a full cycle

x = torch.sin(2 * 3.1415 * f * t + phi).unsqueeze(-1) # (f,t, 1) so linear can multiply "channel" easily

torch.manual_seed(0)
model = mySSM(d_input=1, d_state=16)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(2000):
    pred = model(x[:, :-1]) # Predictions on all time-steps but last
    loss = ((pred - x[:, 1:]) ** 2).mean() # simple MSE

    optim.zero_grad() # Sets all model gradients to zero
    loss.backward() # Populate gradients
    optim.step()
    if step % 50 == 0:
        print(f"loss at step:{step} = {loss.item()}")
