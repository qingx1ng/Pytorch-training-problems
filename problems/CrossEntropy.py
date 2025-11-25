'''
    CrossEntropy implement
    以交叉熵损失为例理解pytoch框架的反向传播机制
'''
import torch
# import numpy as np


class CrossEntropy:
    '''
        自定义CrossEntropy实现,自定义实现backward(),而不是用官方的函数
    '''
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None
    
    def softmax(self, x : torch.tensor, dim : int):
        exp_x = torch.exp(x)
        sum_x = torch.sum(exp_x, dim, keepdim=True)
        return exp_x / sum_x
    
    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        self.probs = self.softmax(x = logits, dim = 1)
        # 长度
        N = logits.shape[0]
        loss = -torch.sum(labels * torch.log(self.probs + 1e-12)) / N
        return loss
    
    def backward(self):
        N = self.logits.shape[0]
        dlogits = (self.probs - self.labels) / N
        return dlogits
    
if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    labels = torch.tensor([[1, 0, 0]])
    criterion = CrossEntropy()
    loss = criterion.forward(logits, labels)
    print("前向传播算得loss:", loss)
    dlogits = criterion.backward()
    print("梯度为:", dlogits)
