import torch
# 仅仅将numpy改为torch ， 不涉及其他操作


x = torch.randn(64, 1000)   # 随机生成64*1000数据
y = torch.randn(64, 10)  # 随机生成64*10数据
w1 = torch.randn(1000, 100)
w2 = torch.randn(100, 10)
learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
h = x.mm(w1)
h_relu = h.clamp(min=0)
y_pred = h_relu.mm(w2)
print(y_pred - y)
