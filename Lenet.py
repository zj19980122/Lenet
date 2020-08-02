import time
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

def load_MNIST(batch_size):
    trans_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST('./data', train=True, transform=trans_img, download=False)
    testset = MNIST('./data', train=False, transform=trans_img, download=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset, testset, trainloader, testloader

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 5, padding=1))
        layer1.add_module('acl', nn.ReLU())
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5, padding=1))
        layer2.add_module('acl', nn.ReLU())
        layer2.add_module('pool2', nn.MaxPool2d(2,2))

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400,120))
        layer3.add_module('acl', nn.ReLU())
        layer3.add_module('fc2', nn.Linear(120,84))
        layer3.add_module('acl', nn.ReLU())
        layer3.add_module('fc3', nn.Linear(84,10))

        self.layer1= layer1
        self.layer2= layer2
        self.layer3= layer3

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0), -1)
        x=self.layer3(x)
        return x


if __name__ == '__main__':
    t_start = time.time()
    learning_rate=0.001
    batch_size = 200
    epoches = 50
    trainset, testset, trainloader, testloader = load_MNIST(batch_size)
    lenet = Lenet()
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)

    f1 = open('result.txt', 'w+')

    for i in range(epoches):
        t_round=time.time()
        loss = 0
        acc = 0
        for ([img, label]) in trainloader:
            optimizer.zero_grad()
            output = lenet(img)
            temp_loss=loss_function(output,label)
            temp_loss.backward()
            optimizer.step()

            loss += temp_loss.item()
            valu, predict = torch.max(output,1)
            correct_num = (predict == label).sum()
            acc += correct_num.item()

        loss /= len(trainset)
        acc /= len(trainset)
        time_spend = time.time()-t_round
        total_time = time.time()-t_start
        print('[%d/%d]: Loss=%.5f, Accuracy=%.5f, time:%.2fs, total time:%.2fs' % (i+1, epoches, loss, acc, time_spend, total_time))
        f1.write('[%d/%d]: Loss=%.5f, Accuracy=%.5f, time:%.2fs, total time:%.2fs\n' % (i+1, epoches, loss, acc, time_spend, total_time))

    f1.close()




