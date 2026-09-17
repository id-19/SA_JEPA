import torch

## Random sine wave input
# time-steps
t = torch.arange(100).float() / 25 # 4 cycles
f = torch.arange(25).float().unsqueeze(-1) / 5 # Goes from 0 cycles per time step to 5 cycles per timestep
# Unsqueeze because (f,1) with (t) broadcasts into (f, t)
phi = 2 * 3.1415 * torch.arange(25).float().unsqueeze(-1) / 25 # Initial phase goes from 0 to a full cycle

# the waves — (16, 128, 1) = 16 waves, 128 time-steps, 1 value each
x = torch.sin(2 * 3.1416 * f * t + phi).unsqueeze(-1)
# print(x, x.shape)

# boring model on purpose: y = a*x + b, two parameters, no memory
torch.manual_seed(0)
model = torch.nn.Linear(1,1) # Works for now, a single line is as good as any
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(20000):
    pred = model(x[:, :-1])                  # frames 0..126
    loss = ((pred - x[:, 1:]) ** 2).mean()   # vs frames 1..127 — predict the NEXT frame
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(epoch, round(loss.item(), 5))

# print(model.weight)
