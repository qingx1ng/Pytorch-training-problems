'''
    softmax implement
    keepdim参数的含义是求和后是否保持dim而不压缩,一般设置为true,便于广播除法
'''
import torch

def softmax_1(x : torch.Tensor, dim : int = -1):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim, keepdim = True)
    return exp_x / sum_x

def softmax(x : torch.Tensor, dim : int = -1):
    '''
        数值稳定的softmax防止溢,减去最大值,数学理论上保持不变
    '''
    max_x = torch.max(x, dim)
    shift_x = x - max_x
    exp_x = torch.exp(shift_x)
    sum_x = torch.sum(x, dim, keepdim=True)
    return exp_x / sum_x


if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0])
    print("简易版:",softmax_1(x = x, dim = 0))
    print("数值稳定版:",softmax_1(x = x, dim = 0))
    x1 = torch.tensor([[1.0, 2.0, 3.0],[2.0, 4.0, 1.0]])
    print("简易版:",softmax_1(x = x1, dim = 1))
    print("数值稳定版:")

