import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchsummary import summary
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import time
from ptflops import get_model_complexity_info

from mobilenet_model import *

BATCH_SIZE = 1024
NUM_EPOCH = 10
LEARNING_RATE = 1e-3
CRITERION = nn.CrossEntropyLoss()


# %%
# CIFAR100 Dataset
train_dataset = dsets.CIFAR100(root='./data', train=True, 
                              transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]), download=True)
test_dataset = dsets.CIFAR100(root='./data', train=False,
                             transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# Assign model and optimizer

torch.cuda.empty_cache()
res = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).cuda()
ires = MobileNetV2(num_classes=100).cuda()

# model, losses, train_acc = fit(plainnet_model, train_loader)
optimizer1 = torch.optim.Adam(res.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer2 = torch.optim.Adam(ires.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
device = 'cuda:0'

r_loss, r_acc = [], []
i_loss, i_acc = [], []

# %%
# Plot information

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(res, (3, 32, 32), as_strings=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    macs, params = get_model_complexity_info(ires, (3, 32, 32), as_strings=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# %%
# Training Resnet50
 
for epoch in range(NUM_EPOCH):
    start = time.time()
    res.train()
    losses = 0.0
    for i, data in enumerate(train_loader):
        image = data[0].cuda(device)
        label = data[1].cuda(device)
        torch.cuda.synchronize()
        start_f = time.time()
        pred_label = res(image)
        torch.cuda.synchronize()
        estimate_f = time.time() - start_f
        if i == 0 and epoch == 0:
            print(f"forward path : {estimate_f} sec")
        loss = CRITERION(pred_label, label)
        losses += loss.item()

        optimizer1.zero_grad()
        torch.cuda.synchronize()
        start_b = time.time()
        loss.backward()
        torch.cuda.synchronize()
        estimate_b = time.time() - start_b
        if i == 0 and epoch == 0:
            print(f"backward path : {estimate_b} sec")
        optimizer1.step()
    avg_loss = losses/len(train_loader)
    r_loss.append(avg_loss)


    res.eval()
    pred_labels = []
    real_labels = []

    for i, data in enumerate(test_loader):
        image = data[0].cuda(device)
        label = data[1].cuda(device)
        real_labels += list(label.cpu().detach().numpy())
        
        pred_label = res(image)
        pred_label = list(pred_label.cpu().detach().numpy())
        pred_labels += pred_label
        
    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels==pred_labels)/len(real_labels)*100
    r_acc.append(acc)

    if epoch % 5 == 0:
        print(f"[{epoch}/{NUM_EPOCH}] : {r_loss[epoch]}")
    
    if epoch == 0:
        estimate = (time.time() - start) * NUM_EPOCH
        print(f"Estimated total = {estimate // 60} min {estimate % 60} sec")

# %%
# Training MobileNetV2

for epoch in range(NUM_EPOCH):

    ires.train()
    losses = 0.0
    for i, data in enumerate(train_loader):
        image = data[0].cuda(device)
        label = data[1].cuda(device)
        torch.cuda.synchronize()
        start_f2 = time.time()
        pred_label = ires(image)
        torch.cuda.synchronize()
        estimate_f2 = time.time() - start_f2
        if i == 0 and epoch == 0:
            print(f"forward path : {estimate_f2} sec")
        loss = CRITERION(pred_label, label)
        losses += loss.item()

        optimizer2.zero_grad()
        torch.cuda.synchronize()
        start_b = time.time()
        loss.backward()
        torch.cuda.synchronize()
        estimate_b = time.time() - start_b
        if i == 0 and epoch == 0:
            print(f"backward path : {estimate_b} sec")
        optimizer2.step()
    avg_loss = losses/len(train_loader)
    i_loss.append(avg_loss)


    ires.eval()
    pred_labels = []
    real_labels = []

    for i, data in enumerate(test_loader):
        image = data[0].cuda(device)
        label = data[1].cuda(device)
        real_labels += list(label.cpu().detach().numpy())
        
        pred_label = ires(image)
        pred_label = list(pred_label.cpu().detach().numpy())
        pred_labels += pred_label
        
    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels==pred_labels)/len(real_labels)*100
    i_acc.append(acc)
    
    if epoch % 5 == 0:
        print(f"[{epoch}/{NUM_EPOCH}] : {i_loss[epoch]}")

# %%
# Plot graph

figure(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(121)
line1, = plt.plot(r_loss)
line2, = plt.plot(i_loss)
plt.legend(labels=("Residual", "Inverted"))
plt.grid()


plt.subplot(122)
line1, = plt.plot(r_acc)
line2, = plt.plot(i_acc)
plt.legend(labels=("Residual", "Inverted"))
plt.grid()
plt.show()


