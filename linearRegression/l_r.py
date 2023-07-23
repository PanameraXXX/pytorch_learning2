import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    """生成 y = xw +b +噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]  # 这个可以学习一下
        )
        yield features[batch_indices], labels[batch_indices]
        # yield 生成器


def linreg(X, w, b):
    """线性模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:  # 从W 到 b
            param -= lr * param.grad / batch_size  # .grad表示梯度
            param.grad.zero_()
            # 手动将梯度设置成0 ，在下一次计算梯度的时候就不会和上一次相关了
    """
    with torch.no_grad的作用
    在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
    即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，
    且grad_fn也为None,即不会对w求导。
    """


batch_size = 10

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print("features:", features[0], '\nlabel:', labels[0])

    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(),
                    labels.detach().numpy(), 1)
    d2l.plt.show()

    for X, y in data_iter(batch_size, features, labels):
        print(X, "\n", y)
        break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    """
    requires_grad
    在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
    tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，
    那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）
    当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
    """

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # net->linreg
            l = loss(net(X, w, b), y)  # 计算loss
            l.sum().backward()  # 反向传播 -> 这个过程会自动求导，求导结果保存在grad中，再到sgd中进行更新
            sgd([w, b], lr, batch_size)  # sgd对w,b进行更新
            # 更新的w,b是存在什么地方进行传递的？
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f"epoch {epoch + 1},loss{float(train_l.mean()):f}")
            print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
            print(f'b的估计误差:{true_b - b}')

    pass
