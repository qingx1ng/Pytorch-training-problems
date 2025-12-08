# 🎯 **PyTorch 编程题库（完整版）50题**

------

# ⭐ Part 1：Tensor 基础 / Autograd（1–10）（★）

### **1. 用 PyTorch 实现一个 Softmax（不能用 torch.softmax）**

目标：基础 Tensor 操作

### **2. 手写 CrossEntropy（forward + backward）**

目标：掌握 autograd 机制

### **3. 实现 broadcast：手动实现广播扩展**

目标：了解 broadcast 原理
为什么需要广播举个最常见的例子：给每一列加偏置（bias）

你有一批数据 x，形状是 (batch_size, features)：
```
import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # shape: (2, 3)

bias = torch.tensor([10., 20., 30.])   # shape: (3,)
```

想对 每一行 都加上同一个 bias，也就是想要：

```
[[1+10, 2+20, 3+30],
 [4+10, 5+20, 6+30]]
```
有广播机制时：
```
y = x + bias      # 直接写就行
print(y)
```
PyTorch 看形状 (2,3) 和 (3,)，自动把 bias 看成 (1,3)，再在前面复制成 (2,3) 来做逐元素加法——而且是“虚拟扩展”，不真的复制内存。


### **4. 实现一个 “无 autograd 的” linear 层 + backward**

目标：理解计算图,同时理解反向传播机制传播的内容和训练过程调参的过程

### **5. 自己写一个简单的 ReLU + backward**

目标：自定义 autograd Function

### **6. 用 Hook 打印每一层梯度最大值**

目标：掌握 backward hook

### **7. 用 PyTorch 实现 LayerNorm，不用官方接口**

目标：理解归一化操作

### **8. 写一个参数共享的两层 MLP（共享第一层权重）**

目标：掌握 module parameter tying

### **9. 手写一个 simple RNN（不能使用 nn.RNN）**

目标：掌握循环梯度

### **10. 验证 autograd 是否正确：用 torch.autograd.gradcheck**

目标：科研常用技巧

------

# ⭐⭐ Part 2：nn.Module 深度 + 训练框架（11–20）（★★）

### **11. 自己写一个 ModuleList，用于搭建动态网络**

目标：掌握 module 容器

### **12. 实现一个可视化计算图结构的工具**

目标：了解 forward graph

### **13. 实现一个 tiny Transformer Encoder（不用 nn.Transformer）**

目标：掌握 attention 手写

### **14. 写一个 custom optimizer（例如 RMSProp 或 Adam）**

目标：理解优化器机制

### **15. 写一个 Polynomial Learning Rate Scheduler**

目标：理解 scheduler

### **16. 用 torch.no_grad 实现 “手动梯度下降版本” 的线性回归**

目标：理解 autograd off 的训练

### **17. 实现 Gradient Clipping（手动实现 norm clipping）**

目标：训练稳定性

### **18. 写一个最小化内存占用的训练 loop（使用 gradient checkpointing）**

目标：训练大模型技巧

### **19. 实现一个 Mixup / CutMix**

目标：数据增强机制

### **20. 写一个 mini-logging 系统（loss/time/GPU usage 打印）**

目标：工程能力

------

# ⭐⭐⭐ Part 3：Dataset / DataLoader / 并行训练（21–30）（★★★）

### **21. 实现一个 IterableDataset（模拟数据流输入）**

目标：处理超大数据集

### **22. 写一个自定义 collate_fn（用于可变长度输入）**

目标：数据动态拼接

### **23. 使用 multiprocessing 手写一个 DataLoader（不依赖官方）**

目标：多进程并行

### **24. 实现一个 PrefetchLoader（提前把 batch 搬到 GPU）**

目标：训练加速技巧

### **25. 写一个 Multi-GPU DataParallel 的简化版**

目标：理解数据并行机制

### **26. 写一个最小可运行的 Distributed Data Parallel（DDP）版本**

目标：科研级：理解 AllReduce

### **27. 使用 torch.distributed.launch 训练一个模型**

目标：掌握 DDP 的工程用法

### **28. 实现一个 parameter server 版训练（ps-lite 思路）**

目标：理解传统 ML 并行

### **29. 写一个 Automated Mixed Precision（AMP mini 版）**

目标：训练加速、半精度

### **30. 写一个自动 resume（断点续训）系统**

目标：科研可复现性能力

------

# ⭐⭐⭐⭐ Part 4：进阶 Autograd / 运算图 / 数值稳定性（31–40）（★★★）

### **31. 手写一个 Stable Softmax（log-sum-exp 技巧）**

目标：防止 NaN

### **32. 实现一个 Stable LayerNorm**

目标：掌握数值稳定性

### **33. 用 autograd.grad 实现高阶导（2nd derivative）**

目标：隐式微分/NeRF/SDF 会用到

### **34. 实现一个 implicit layer（比如 Deep Equilibrium Model）**

目标：科研能力要求

### **35. 手写可微 SDF（signed distance function）**

目标：三维视觉基础

### **36. 实现一个可微相机投影矩阵（从 SE3 → 图像坐标）**

目标：SLAM/NeRF 基础工具

### **37. 写一个可微光线采样器（Ray marcher）**

目标：NeRF 关键组件

### **38. 重写 grid_sample（用 PyTorch ops，不用原生函数）**

目标：理解可微插值

### **39. 写一个可微化 KNN（基于 smooth approximation）**

目标：点云论文中常用

### **40. 用 autograd profiler 记录每层算子计算耗时**

目标：深入理解性能瓶颈

------

# ⭐⭐⭐⭐⭐ Part 5：自定义 C++/CUDA 扩展（41–50）（科研顶配）（★★★）

### **41. 写一个最简单的 C++ 扩展（tensor + constant）**

目标：ATen + C++ 前端

### **42. 写一个 add kernel 的 CUDA 版本（forward + backward）**

目标：最小 CUDA 扩展

### **43. 实现一个 CUDA 版本的 ReLU（前向 + 反向）**

目标：掌握 kernel launch

### **44. 写一个 tiled 矩阵乘法（shared memory）**

目标：CUDA 优化

### **45. 写一个 warp reduce（使用 __shfl_down_sync）**

目标：warp-level primitive

### **46. 写一个 nearest neighbor（点云 kNN）的 CUDA 实现**

目标：SLAM/点云加速基础

### **47. 用 CUDA 写一个 bilinear sampling（grid_sample CUDA 版）**

目标：可微插值 + CUDA

### **48. 写一个 tiny NeRF CUDA kernel（光线 marching + accumulating）**

目标：三维视觉科研必备

### **49. 写一个 Gaussian splatting rasterizer CUDA 实现（简化版）**

目标：3DGS pipeline 入门

### **50. 将你的 CUDA算子封装为 PyTorch Autograd Function + 编译为 pip 包**

目标：科研工程闭环

------

### ✅ 1）**GitHub 题库仓库模板**（含模板代码 + 每题 README）


```
pytorch-50-problems/
│── problems/
│    ├── 01_softmax/
│    ├── 02_crossentropy/
│    ├── ...
│── tests/
│── solutions/（可选）
│── README.md
```

### ✅ 2）**逐题讲解 + 参考答案**

------
