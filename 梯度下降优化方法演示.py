'''

公式：
W新 = W旧 - 学习率 * 梯度

存在的问题：
1.遇到平缓区域，梯度下降可能会慢
2.可能会遇到鞍点
3.可能会遇到局部最小值

动量法momentum：
    动量法公式：
        St = 贝塔 * St-1 + （1-贝塔） * Gt
    加入动量法后：
    W新 = W旧 - 学习率 * St

'''

import torch
import torch.nn as nn
import torch.optim as optim

# 1.定义函数，梯度下降优化方法 -> 动量法

def dm01_momentum():
    #1.初始化权重参数
    w = torch.tensor([1.0] , requires_grad=True , dtype=torch.float)
    #2.定义损失函数
    criterion = ((w**2)/2.0)
    #3.创建优化器（函数对象）->基于SGD（随机梯度下降）加入参数momentum
    optimizer = optim.SGD(params=[w], lr=0.01 , momentum=0.9) #momentum = 0 只考虑本次梯度
    #4.计算梯度值：先梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w} , w.grad:{w.grad}')

    #5.重复上述步骤，第二次更新权重参数
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w} , w.grad:{w.grad}')



if __name__ == '__main__':
    dm01_momentum()






