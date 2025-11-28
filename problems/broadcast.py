'''
    broadcast   implement
    list()变成列表,unsqueeze()是插入一维度,如(2,3)经过unsqueeze(0)就变成了(1,2,3)
    广播机制是从最后一个维度开始比较,如果是相等就pass,不相等的话看是否为1,如果是1也pass
    本代码原理是先把最后两个向量最后一个维度对齐,然后从前往后开始比较,相当于从后开始检查广播机制
'''

import torch

def torch_broadcast(a : torch.Tensor, b : torch.Tensor):
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    
    while(len(a) < len(b)):
        a_shape.insert(0, 1)
        a.unsqueeze(0)
    while(len(b) < len(a)):
        b_shape.insert(0, 1)
        b.unsqueeze(0)
    
    target_shape = []

    for da, db in zip(a_shape, b_shape):
        if da == db or da == 1 or db ==1:
            target_shape.append(max(da, db))
        else:
            raise ValueError(f"维度不匹配：{da} vs {db}")
    
    # 先对齐维度(只是插入1维）在expand扩展到对应形状, *号的作用是提出列表的每一个元素作为参数
    a_exp = a.expand(*target_shape)
    b_exp = b.expand(*target_shape)

    return a_exp, b_exp



