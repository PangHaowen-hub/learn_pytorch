import torch
import torch.nn as nn

x = torch.randn(64, 1000)
z = torch.randn(64, 10)


class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.pred = torch.nn.Sequential(nn.Linear(D_in, H, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(H, D_out, bias=False))

    def forward(self, x):
        z_pred = self.pred(x)
        return z_pred


model = Net(1000, 100, 10)
# model = model.cuda()
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(500):
    z_pred = model(x)
    loss = loss_fn(z_pred, z)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
z_pred = model(x)
print(z_pred - z)
