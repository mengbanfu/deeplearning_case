import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def dm01():
    t1 = torch.randint(0, 10, (2,3))
    print(f'{t1.shape[-1]}')
    t1.reshape(3,2)

def dm02():
    t1 = torch.randint(0, 10, (2,3))
    t2 =  t1.unsqueeze(1)   #添加维度
    print(f'{t2.shape}')

def dm03():
    t1 = torch.randint(0, 10, (2,3))
    t2 = t1.transpose(0, 1)   #交换维度
    t3 = torch.randint(0,10,(1,2,3))
    t4 = t3.permute(1,0,2)

def dm04():
    t1 = torch.randint(0, 10, (2,3 , 4))
    print(t1.is_contiguous())
    t2 = t1.view(4,3,2)
    print(t2.is_contiguous())
    t3 = t1.transpose(1,2)
    print(t3.is_contiguous())
    t4 = t3.contiguous()
    print(t4.is_contiguous())

def dm05():
    torch.manual_seed(0)
    t1 = torch.randint(0, 10, (2,3))
    t2 = torch.randint(0, 10, (2,3))
    t3 = torch.cat((t1,t2),1)
    t4 = torch.stack([t1,t2],1)
    print(t1 , t2)
    print(t4)

def dm06():
    #自动微分公式
    #1.定义变量 记录初始权重(初始值 ， 是否自动求导 ， 数据类型)
    w = torch.tensor(10 , requires_grad=True , dtype = torch.float)

    loss = 2* w **2
    #print(loss.sum())

    loss.sum().backward() #自动微分

    w.data = w.data - 0.01 * w.grad

    print(f'更新后的权重{w.data}')

def dm07():
    w = torch.tensor(10 , requires_grad=True , dtype = torch.float)
    loss = w**2 +20
    print(f'开始权重{w.data} , (0.01*w.grad):无 loss:{loss}')

    for i in range(1,101):
        #前向传播
        loss = w**2+20
        #梯度清零
        if w.grad is not None:
            w.grad.zero_()
        #反向传播
        loss.sum().backward()

        #梯度更新
        w.data = w.data - 0.01 * w.grad
    print(f'最终结果{w.data} 梯度：{w.grad} loss:{loss}')

    n1 = w.detach() #通过detach转换numpy 共享一块空间
    print(n1)
    n2 = n1.numpy() #w可以自动微分,n1不可以


if __name__ == '__main__':
    #dm07()
    w = torch.randn(5)
    print(w)