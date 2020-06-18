import torch

x = torch.randn(64, 1000).cuda()
z = torch.randn(64, 10).cuda()

w1 = torch.randn(1000, 100).cuda()
w2 = torch.randn(100, 10).cuda()

learning_rate = 1e-6


for t in range(5000):
    y = x.mm(w1)
    y_relu = y.clamp(min=0)
    z_pred = y_relu.mm(w2)


    loss = (z_pred - z).pow(2).sum()
    print(t, loss.item())
    grad_z_pred = 2.0 * (z_pred - z)
    grad_w2 = y_relu.t().mm(grad_z_pred)
    grad_y_relu = grad_z_pred.mm(w2.t())
    grad_y = grad_y_relu.clone()
    grad_y[y < 0] = 0
    grad_w1 = x.t().mm(grad_y)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
y = x.mm(w1)
y_relu = y.clamp(min=0)
z_pred = y_relu.mm(w2)
print(z_pred - z)
