import torch

class MyRelu(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        # 这里需要存在backward中计算需要利用到的forward的信息，并没有固定格式。本例中backward计算需要利用input所以只需要存input
        ctx.save_for_backward(input)
        # relu的输出
        output = input.clamp(min=0)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 取出 forward 里保存的 input，注意这里是元组，所以需要下面的格式
        (input,) = ctx.saved_tensors
        # grad_output 是链式法则传进来的 dL/d(out)，这是autograd的固定格式要求不要管，只需要借助传过来的梯度算当前这一层的梯度
        grad_input = grad_output.clone()
        # ReLU 在 x <= 0 的地方，导数为 0
        grad_input[input <= 0] = 0
        # 返回这一层的梯度
        return grad_input


# 2. 方便调用的别名，因为autograd的规则是要用apply调用不能直接output = MyRelu(input)，而需要output = MyRelu.apply(input)，这样才会计算图自动求导
my_relu = MyRelu.apply

# 3. 简单测试一下
if __name__ == "__main__":
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
    y = my_relu(x)           # 用我们自定义的 ReLU
    loss = y.sum()           # 随便定义一个标量损失
    loss.backward()          # 反向传播, 自动求导

    print("x:", x)
    print("y = MyReLU(x):", y)
    print("dy/dx (x.grad):", x.grad)   # 期望： [0, 0, 0, 1, 1]