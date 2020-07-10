import argparse
import torch.backends.cudnn as cudnn
from cdata import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from networks import Conv2dBlock
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='midi/spectra/', help='Path to the data.')
opts = parser.parse_args()
cudnn.benchmark = True
batch_size = 64

transform_list = [transforms.ToTensor()]
transform = transforms.Compose(transform_list)
trainset = ImageFolder(os.path.join(opts.dataroot, 'ctrain'), transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
testset = ImageFolder(os.path.join(opts.dataroot, 'test'), transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

class Net(nn.Module):
    def __init__(self, input_dim, dim):
        super(Net, self).__init__()
        cnn_x = []
        cnn_x += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')]
        for _ in range(3):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        self.cnn_x = nn.Sequential(*cnn_x)
        self.fc = nn.Linear(16*16, 3)

    def forward(self, x):
        x = self.cnn_x(x)
        x = x.view(-1, 16*16)
        x = self.fc(x)
        return x

net = Net(1,64).cuda()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(correct, total)
    print('Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))

epoch = 0
iterations = 0
for epoch in range(10):
    epoch += 1
    running_loss = 0.0
    for data in trainloader:
        iterations += 1
        inputs, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if iterations % 200 == 0:
            print('[%d, %5d] loss: %.3f' %
                    (epoch, iterations, running_loss / 200))
            running_loss = 0.0
    if epoch % 5 == 0:
        test()
        PATH = './cnet/classifier_net_' + str(epoch) + '.pth'
        torch.save(net.state_dict(), PATH)

    

