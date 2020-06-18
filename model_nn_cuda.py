import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out))
model = model.cuda()
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
y_pred = model(x)
print(y_pred - y)
