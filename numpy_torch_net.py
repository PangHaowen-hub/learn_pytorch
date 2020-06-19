import numpy as np
import torch
import argparse


def numpy_net(input, hide, output, data_size, learning_rate, epoch):
    x = np.random.randn(data_size, input)
    z = np.random.randn(data_size, output)
    w1 = np.random.randn(input, hide)
    w2 = np.random.randn(hide, output)

    for t in range(epoch):
        y = x.dot(w1)
        y_relu = np.maximum(y, 0)
        z_pred = y_relu.dot(w2)

        loss = np.square(z_pred - z).sum()
        print(t, loss)

        grad_z_pred = 2.0 * (z_pred - z)
        grad_w2 = y_relu.T.dot(grad_z_pred)
        grad_y_relu = grad_z_pred.dot(w2.T)
        grad_y = grad_y_relu.copy()
        grad_y[y < 0] = 0
        grad_w1 = x.T.dot(grad_y)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    y = x.dot(w1)
    y_relu = np.maximum(y, 0)
    z_pred = y_relu.dot(w2)
    print(z_pred - z)


def torch_net(input, hide, output, data_size, learning_rate, epoch):
    x = torch.randn(data_size, input)
    z = torch.randn(data_size, output)
    w1 = torch.randn(input, hide)
    w2 = torch.randn(hide, output)

    for t in range(epoch):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default='numpy', help='numpy or torch')
    parser.add_argument('--input', dest='input', type=int, default=1000, help='input dim')
    parser.add_argument('--hide', dest='hide', type=int, default=100, help='hide lawyer dim')
    parser.add_argument('--output', dest='output', type=int, default=10, help='out dim')
    parser.add_argument('--data_size', dest='data_size', type=int, default=64, help='data size')
    parser.add_argument('--epoch', dest='epoch', type=int, default=500, help='epoch')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-6, help='learning rate')
    args = parser.parse_args()

    if args.model == 'numpy':
        numpy_net(args.input, args.hide, args.output, args.data_size, args.learning_rate, args.epoch)
    elif args.model == 'torch':
        torch_net(args.input, args.hide, args.output, args.data_size, args.learning_rate, args.epoch)

    print('model:' + args.model)
    print('input:' + str(args.input))
    print('hide:' + str(args.hide))
    print('output:' + str(args.output))
    print('data size:' + str(args.data_size))
    print('epoch:' + str(args.epoch))
    print('learning rate:' + str(args.learning_rate))