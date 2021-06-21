import torch

def scal(α, f, batch=False):
    if batch:
        B = α.shape[0]
        return (α.view(B, -1) * f.view(B, -1)).sum(1)
    else:
        return torch.dot(α.view(-1), f.view(-1))


class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)


def squared_distances(x, y):
    D = 1 - torch.matmul(x, y.permute(0, 2, 1))
    return D

def distances(x, y):
    return sqrt_0(squared_distances(x, y))
