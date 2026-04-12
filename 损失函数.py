'''

x           样本
f(x)            加权求和
S(f(x))         处理后的概率
y               样本x属于某一个类型的真实概率


二分类： loss = -ylog(预测值) - (1-y)log(1-预测值)
'''

import torch
import torch.nn as nn

from 自动微分真实应用场景 import criterion


def dm01():
    #1.手动创建样本的真实值
    #y_true = torch.tensor([[0,1,0] , [1,0,0]] , dtype=torch.float)
    y_true = torch.tensor([1,2])

    #2.手动创建样本的预测值f(x)
    y_pred = torch.tensor([[0.1,0.8,0.1] , [0.7 , 0.2 , 0.1]] , requires_grad=True)

    #3.创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    #计算损失值
    loss = criterion(y_pred, y_true)
    print(f'损失值{loss}')

def dm02():
    y_true = torch.tensor([0,1,0] , dtype=torch.float)

    y_pred = torch.tensor([0.6901,0.5423,0.2639])

    #创建损失函数
    criterion = nn.BCELoss()

    loss = criterion(y_pred, y_true).detach().numpy()
    print(f'损失值{loss}')

def dm03():      #MAE损失函数
    y_true = torch.tensor([2.0 , 2.0 , 2.0] , dtype=torch.float)

    y_pred = torch.tensor([1.0, 1.0, 1.9] , requires_grad=True)

    #实例化损失对象
    criterion = nn.BCELoss()

    criterion2 = nn.MSELoss()

    print(f'损失值{my_loss}')




if __name__ == '__main__':
    dm03()






