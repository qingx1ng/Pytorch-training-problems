'''
    linear implement
    线性层的简单实现,理解反向传播的过程
'''
import numpy as np

class linear:

    def __init__(self, in_features, out_features, weight_scales=0.01):
        
        self.W = np.random.randn(in_features, out_features) * weight_scales
        self.b = np.zeros(out_features)
        self.dw = np.zeros_like(self.W)
        self.db = np.zeros_like(self.dw)
        self.x_cache = None

    def forward(self, x):

        self.x_cache = x
        out = x @ self.W + self.b
        return out
    
    def backward(self, dout):

        '''
        backward 的 Docstring
        
        :param self: 说明
        :param dout: 上一层得到的梯度
        :out: 输出表示这一层对输入的梯度，然后反向传播给下一层
        '''

        x = self.x_cache
        # 对输入的梯度
        dx = dout @ self.W.T
        # 对权重的梯度
        self.dw = x.T @ dout
        # 对偏置的梯度
        self.db = np.sum(dout, axis=0)
        return dx
    
if __name__ == "__main__":
    np.random.seed(0)
    # 简单定义一个线性层
    layer = linear(3, 2)
    # 输入定义batch为4
    x = np.random.randn(4, 3)
    # 输出,即前向传播
    out = layer.forward(x)
    loss = np.sum(out ** 2)
    # 定义一个简单损失: L = sum(out^2),并计算对输出的梯度，即2 * out
    dout = 2 * out
    # 反向传播
    dx = layer.backward(dout)
    # 打印输出
    print("loss:", loss)
    print("dW:\n", layer.dw)
    print("db:\n", layer.db)
    print("dx:\n", dx)
    # 调参
    # 设置学习率
    lr = 0.001
    layer.W -= lr * layer.dw
    layer.b -= lr * layer.db