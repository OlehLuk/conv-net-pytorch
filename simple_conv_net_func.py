from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()

# assumes 1-channel input
def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    batch_size, _, input_h, input_w = x_in.size()
    n_filters, _, filter_h, filter_w = conv_weight.size()

    output_h = input_h - filter_h + 1
    output_w = input_w - filter_w + 1

    result = torch.empty((batch_size, n_filters, output_h, output_w)).to(device)
    for n in range(batch_size):
        for f in range(n_filters):
            for i in range(output_h):
                for j in range(output_w):
                    result[n][f][i][j] = (x_in[n][0][i:i+filter_h, j:j+filter_w] * conv_weight[f]).sum() +\
                                         conv_bias[f]

    # Add your code here
    #
    return result


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    batch_size, _, input_h, input_w = x_in.size()
    n_filters, _, filter_h, filter_w = conv_weight.size()

    output_h = input_h - filter_h + 1
    output_w = input_w - filter_w + 1

    w_resh = conv_weight2rows(conv_weight)

    result = torch.empty((batch_size, n_filters, output_h, output_w)).to(device)

    for n in range(batch_size):
        result[n] = (w_resh.matmul(im2col(x_in[n], filter_h, 1, device))
                     + conv_bias.view(-1, 1)).view(n_filters, output_h, output_w)

    return result


# transforms one image from a batch
def im2col(X, kernel_size, stride, device):
    _, input_h, input_w = X.size()

    # difference should be dividable by stride. use // to keep int type
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1

    result = torch.empty((output_h * output_w, kernel_size*kernel_size)).to(device)

    for i in range(output_h):
        for j in range(output_w):
            result[i*output_w+j] = X[0][i: i + kernel_size, j: j + kernel_size].contiguous().view(1, -1)

    return result.t()


def conv_weight2rows(conv_weight):
    return conv_weight.view(conv_weight.size()[0], -1)

# assumes 1-chanel input
def pool2d_scalar(a, device):
    batch_size, _, input_h, input_w = a.size()
    
    output_h = input_h // 2
    output_w = input_w // 2

    result = torch.empty((batch_size, 1, output_h, output_w)).to(device)
    result.requires_grad = True

    # channel_n always equals 1 as 1-channel output is expected thus only 1st channel is allocated
    # tensor is initialized with channel_n though
    for n in range(batch_size):
        for i in range(output_h):
            for j in range(output_w):
                result[n][0][i][j] = a[n][0][2*i:2*i+2, 2*j:2*j+2].max()

    return result


def pool2d_vector(a, device):
    batch_size, channel_n, input_h, input_w = a.size()

    output_h = input_h // 2
    output_w = input_w // 2

    odd_ind = range(0, input_h, 2)
    even_ind = range(1, input_h, 2)

    odd = a[:, :, odd_ind, :].view(2, 1, -1, 2)
    even = a[:, :, even_ind, :].view(2, 1, -1, 2)
    res = torch.cat([odd, even], dim=3).max(dim=3, keepdim=True).values.to(device)

    return res.view([batch_size, channel_n, output_h, output_w]).requires_grad_(True)


def relu_scalar(a, device):
    batch_size, input_size = a.size()

    result = torch.empty((batch_size, input_size)).to(device)
    result.requires_grad = True

    for n in range(batch_size):
        for i in range(input_size):
            if a[n][i] < 0:
                result[n][i] = 0
            else:
                result[n][i] = a[n][i]

    return result


def relu_vector(a, device):
    a[a < 0] = 0
    return a.requires_grad_(True)


def reshape_vector(a, device):
    batch_size = a.size()[0]
    return a.clone().view([batch_size, -1]).requires_grad_(True)


def reshape_scalar(a, device):
    batch_size, channel_n, img_h, img_w = a.size()
    output_size = img_h * img_w * channel_n

    result = torch.empty((batch_size, output_size)).to(device)
    result.requires_grad = True

    for n in range(batch_size):
        for c in range(channel_n):
            for i in range(img_h):
                for j in range(img_w):
                    new_index = c * img_h * img_w + i * img_w + j
                    result[n][new_index] = a[n][c][i][j]

    return result


def fc_layer_scalar(a, weight, bias, device):
    batch_size, input_size = a.size()
    output_size = bias.size()[0]
    result = torch.empty((batch_size, output_size)).to(device)
    result.requires_grad = True

    for n in range(batch_size):
        for i in range(output_size):
            z_i = 0
            for j in range(input_size):
                z_i += a[n][j]*weight[i][j]
            z_i += bias[i]
            result[n][i] = z_i

    return result


def fc_layer_vector(a, weight, bias, device):
    return (weight.matmul(a.t()).t() + bias).clone().requires_grad_(True)

