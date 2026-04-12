import torch
import torch.nn as nn
from torchsummary import summary

# todo 搭建神经网络
class ModelDemo(nn.Module):
    def __init__(self):
            #初始化父类
            super().__init__()
            #搭建 ， 隐藏层 + 输出层
            self.linear1 = nn.Linear(3, 3)
            self.linear2 = nn.Linear(3, 2)
            self.output = nn.Linear(2, 2)

            #对隐藏层进行参数初始化
            nn.init.xavier_normal_(self.linear1.weight)
            nn.init.zeros_(self.linear1.bias)

            nn.init.kaiming_normal_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    #前向传播 输入 -> 隐藏 -> 输出
    def forward(self,x):
        #第一层计算
        #分散
        # x = self.linear1(x)
        # x = torch.sigmoid(x)
        #合并版本
        x = torch.sigmoid(self.linear1(x))

        #隐藏层计算
        x = torch.relu(self.linear2(x))

        #输出层计算 dim=-1 表示按行计算 0 表示按列计算
        x = torch.softmax(self.output(x),dim=1)

        #返回预测值
        return x


# todo 2 模型训练
def train():
    #创建模型对象
    my_model = ModelDemo()

    print('===========计算模型参数===========')
    # 参1 神经网络模型对象 参2 输入数据维度
    summary(my_model, input_size=(5, 3) , device='cpu')
    for name, param in my_model.named_parameters():
        print(name)
        print(param)

    #创建数据集对象
    torch.manual_seed(1)
    data = torch.randn(size=(5,3))

    #将数据送到CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_model=my_model.to(device)
    data = data.to(device)

    #调用模型进行模型训练
    output = my_model(data) #自动调用forward前向传播方法
    print(f'{output}')
    print(f'{output.shape}')
    print(f'{output.requires_grad}')

    print('-'*30)





if '__main__' == __name__:
    train()