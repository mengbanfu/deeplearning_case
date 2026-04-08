import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset #数据集对象
from torch.utils.data import DataLoader #数据加载器
from torch import nn #其中有平方损失函数和假设函数
from torch import optim #optim中有优化器函数
from sklearn.datasets import make_regression #创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

def create_dataset():
    x,y,coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        coef=True,
        random_state=3,
    )

    x = torch.tensor(x , dtype=torch.float)
    y = torch.tensor(y , dtype=torch.float)


    return x,y,coef

def train(x,y,coef):

    #数据集
    dataset = TensorDataset(x,y)
    #加载器对象
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

    #创建线性回归模型
    model = nn.Linear(in_features=1,out_features=1)
    #损失函数
    criterion = nn.MSELoss()

    #优化器对象
    optimizer = optim.SGD(model.parameters(),lr=1e-2)

    #具体的的训练过程
    #轮数 ， 每轮损失值， 总损失 ， 平均损失
    epochs , loss_list , total_loss , total_sample = 100 , [] , 0.0 , 0
    for epoch in range(epochs):
        #每轮分批次训练，所以从数据加载器中获取批次数据
        for train_x , train_y in dataloader:
            #模型预测
            y_pred = model(train_x)
            #计算损失值(每批的平均损失)
            loss = criterion(y_pred,train_y.reshape(-1,1)) #-1自动计算
            #记录总损失和样本批次数
            total_loss += loss.item()
            total_sample += 1
            #梯度清零 加 反向传播 加 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss / total_sample)
        print(f"epoch:{epoch} loss:{total_loss / total_sample} ")

    print(f'{epochs}轮的平均损失分别为{loss_list}')
    print(f'模型参数:权重{model.weight} , 偏置{model.bias}')

    #绘制损失曲线
    plt.plot(range(epochs) , loss_list)
    plt.title('损失值曲线变化')
    plt.grid(True)
    plt.show()

    #样本点分布情况
    plt.scatter(x,y)
    #100个样本点的特征
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])

    #计算真实值
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    #绘制预测值和真实值的折线图
    plt.plot(x , y_true , color='red' , label='真实值')
    plt.plot(x , y_pred , color = 'blue' , label = "预测值")
    plt.legend()
    plt.grid()
    plt.show()

    plt.show()


if __name__ == '__main__':

    # dataset = TensorDataset(x,y) #数据集对象
    # dataloader = DataLoader(dataset,batch_size=32,shuffle=True) #构造数据加载器
    # model = nn.Linear(in_features=1,out_features=1) #构造模型
    # criterion = nn.MSELoss()   #构造损失函数
    # optimizer = optim.SGD(model.parameters(),lr=1e-2) #优化函数
    x,y,coef = create_dataset()
    train(x,y,coef)

















