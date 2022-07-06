import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


""" 
    * 生成器，示例如下    
    list1 = []
    v1 = (1, 2)
    v2 = [3, 4]
    list1.append([0, v1])
    list1.append([*v2])
    list1.append([0, *v1, *v2])
"""

batch_size = 10
data_iter = load_array((features,labels),batch_size)

# Sequential： the list of layers
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

next(iter(data_iter))

if __name__ == '__main__':
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad() # 先把梯度清理
            l.backward() # 再计算梯度
            trainer.step()
        l = loss(net(features),labels)
        print(f'epoch{epoch+1}, loss {l:f}')
