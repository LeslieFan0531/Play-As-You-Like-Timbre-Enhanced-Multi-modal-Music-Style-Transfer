import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from cdata import ImageFolder
from networks import Conv2dBlock
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='test_clf/', help="input image path")
parser.add_argument('--cnet', type=str, default='cnet/classifier_net_30.pth', help="path to classfier checkpoint")
opt = parser.parse_args()

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
net.load_state_dict(torch.load(opt.cnet))

batch_size = 64
transform_list = [transforms.ToTensor()]
transform = transforms.Compose(transform_list)
testset = ImageFolder(opt.dataroot, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

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
print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))