# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.pyplot import figure
from radam import *
import time
from pytorch_four_things_BN import *

import matplotlib.pyplot as plt

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

train_set = dset.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
test_set = dset.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)

batch_size = 128
total_epoch = 50
learning_rate = 1e-3
use_cuda = torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader( 
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

"""## Model class
Simple classifier composed with 2 convolution layer and 1 linear layer
"""

class p1CNN(nn.Module):
    def __init__(self):
        super(p1CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.li = nn.Linear(7*7, 10)
    
    def forward(self, x, batch_norm_type=None, is_training=True):
        x = self.mp(F.relu(self.c1(x)))
        if batch_norm_type == 1:
            # Simple Batch Normalization (batch size =128)
            x = normalization_layer(x, channels_per_group=1, examples_per_group=128, is_training=is_training)
        elif batch_norm_type == 2:
            # Ghost Batch Normalization (ghost batch size = 16)
            x = normalization_layer(x, channels_per_group=1, examples_per_group=16, is_training=is_training)
        elif batch_norm_type == 3:
            # Group Normalization
            x = normalization_layer(x, channel_groups=32, examples_per_group=1, is_training=is_training)
        elif batch_norm_type == 4:
            # Batch/Group Normalization Generalization
            x = normalization_layer(x, channel_groups=32, examples_per_group=2, is_training=is_training)
        x = self.mp(F.relu(self.c2(x)))
        x = x.view(-1, 7*7)
        x = self.li(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    for p in model.parameters():
        p.data.fill_(1)

"""## Model Assign"""

model1 = p1CNN().cuda()
model2 = p1CNN().cuda()
model3 = p1CNN().cuda()
model4 = p1CNN().cuda()
model5 = p1CNN().cuda()
tmp = p1CNN().cuda()
tmp.load_state_dict(model1.state_dict())
model2.load_state_dict(tmp.state_dict())
model3.load_state_dict(tmp.state_dict())
model4.load_state_dict(tmp.state_dict())
model5.load_state_dict(tmp.state_dict())

model_list = [model1, model2, model3, model4, model5]

optimizer1 = RAdam(model1.parameters(), lr=0.01)
optimizer2 = RAdam(model2.parameters(), lr=0.01)
optimizer3 = RAdam(model3.parameters(), lr=0.01)
optimizer4 = RAdam(model4.parameters(), lr=0.01)
optimizer5 = RAdam(model5.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
optim_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]

"""## Optimizer Assign

## Train 5 models
"""

# Train

train_loss_list, test_acc_list = [], []
for i in range(5):
    tmp1, tmp2 = [], []
    for j in range(total_epoch):
        tmp1.append(0)
        tmp2.append(0)
    train_loss_list.append(tmp1)
    test_acc_list.append(tmp2)

for epoch in range(total_epoch):
    # trainning
    for i, (model, optim) in enumerate(zip(model_list, optim_list)):
        model.train()
        if epoch == 0 and i == 0:
            start = time.time()

        total_loss = 0
        total_batch = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            
            
                model.train()
                optim.zero_grad()

                out = model(x, i)
                loss = criterion(out, target)
                total_loss += loss.item()
                loss.backward()
                optim.step()
            total_batch += 1
        train_loss_list[i][epoch] = total_loss / total_batch
        print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, {}'
                .format(epoch, batch_idx+1, total_loss / total_batch, total_batch))

    # testing
        total_loss = 0
        total_batch = 0
        correct_cnt = 0
        total_cnt = 0

        for batch_idx, (x, target) in enumerate(test_loader):
            model.eval()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
                
            out = model(x, i)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum().item()
            
            total_loss += loss.item()
            total_batch += 1
        test_acc_list[i][epoch] = correct_cnt / total_cnt
        print ('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'
                .format(epoch, batch_idx+1, total_loss / total_batch, correct_cnt * 1.0 / total_cnt))
        if epoch == 0 and i == 0:
            end = time.time()
            estimate = end - start
            total = estimate * 9 * total_epoch
            print(f'{total // 60} minute and {total % 60} sec left')

"""## Plot result, using subplot"""

figure(num=None, figsize=(16, 7), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(121)
for line in train_loss_list:
    plt.plot(line)

plt.legend(labels=('Plain', 'BN', 'GBN', 'GN', 'B/GNG'))
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.grid()

plt.subplot(122)
for line in test_acc_list:
    plt.plot(line)

plt.legend(labels=('Plain', 'BN', 'GBN', 'GN', 'B/GNG'))
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.grid()

plt.suptitle('Comparison for No BN, 4 kinds of BNs')
plt.show()


figure(num=None, figsize=(16, 7*4), dpi=80, facecolor='w', edgecolor='k')

for i in range(5):
    plt.subplot2grid((5, 2), (i, 0))
    plt.plot(train_loss_list[i])
    plt.legend(labels=(str(i)))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()

for i in range(5):
    plt.subplot2grid((5, 2), (i, 1))
    plt.plot(test_acc_list[i])
    plt.legend(labels=(str(i)))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()

plt.suptitle('Comparison for No BN, 4 kinds of BNs')
plt.show()
