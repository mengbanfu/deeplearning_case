
import torch
from tensorboard import summary
from torch.utils.data import TensorDataset #数据集对象
from torch.utils.data import DataLoader     #数据加载器
import torch.nn as nn
import torch.optim as optim     #优化器
from sklearn.model_selection import train_test_split #训练集和测试集
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

'''
实现步骤：
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试

'''

# todo 1.定义函数，构建数据集

def create_dataset():
    #加载csv数据集
    data = pd.read_csv('手机价格预测.csv')
    print(data.head())
    #print(data.shape)
    #获取x特征列 ， y特征列
    x , y = data.iloc[:, :-1] , data.iloc[:, -1]
    #把特征值转换成浮点
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    #切分训练集和测试集
    # 1 . 特征  2. 标签 3 .测试集比例 4. 随机种子 5.样本分布
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0 , stratify=y)

    # 把数据集封装成张量数据库 ， 思路：数据->张量->数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))

    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

# todo 2 搭建神经网络
class PhonePriceModel(nn.Module):
    #1 . 在init中初始化父类成员 ， 及搭建神经网络
    def __init__(self , input_dim , output_dim):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)
    #2.定义前向传播方法
    def forward(self, x):
        # 2.1 隐藏层1: 加权求和 + 激活函数
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2: 加权求和 + 激活函数
        x = torch.relu(self.linear2(x))
        # 2.3 输出层: 加权求和 + 激活函数(softmax)
        #cross_entropy_loss()
        #x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)
        return x

# todo 3 模型训练
def train(train_dataset,test_dataset,input_dim,output_dim):
    torch.manual_seed(0)
    #1.数据集对象
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #2.创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    #移动到 GPU
    model.to(torch.device('cuda'))
    torch.cuda.manual_seed(0)
    #3. 定义损失函数 ， 因为是多分类 ， 用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #4. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #5. 训练模型
    epochs = 50
    for epoch in range(epochs):
        #5.2 训练模型
        total_loss , batch_num = 0.0 , 0
        start = time.time()
        for x,y in train_loader:
            x = x.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))
            model.to(torch.device('cuda'))

            model.train()       #model.eval() 测试模型
            #5.2.1 模型预测
            y_pred = model(x)
            #5.2.2 计算损失
            loss = criterion(y_pred, y)
            #5.2.3 梯度清零 , 反向传播 ， 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #5.2.4 计算损失和
            total_loss += loss.item()
            batch_num += 1
        #5.3 打印损失 , 本次训练的耗时
        print(f"epoch:{epoch} loss:{total_loss / batch_num} time:{time.time() - start}")

    torch.save(model.state_dict(), '手机价格分类模型.pth')


# todo 4 模型测试

# todo 5 模型预测
if __name__ == '__main__':
    train_dataset , test_dataset , input_dim , output_dim = create_dataset()
    #model = PhonePriceModel(input_dim, output_dim)
    #模型对象 ， 输入数据的形状
    #summary(model , input_size = (16 , input_dim))
    train(train_dataset,test_dataset,input_dim,output_dim)



