import argparse
import torch
import torchvision.transforms as transforms
from fer import FER2013
import os
import torch.optim as optim
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from vgg import VGG
from resnet import ResNet18
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# parser
parser = argparse.ArgumentParser(description='attack to FER on FER2013')
parser.add_argument('--model', type=str, default='Resnet18', choices=['VGG19', 'Resnet18'], help='CNN architecture')
parser.add_argument('--bs', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

# data
cut_size = 44
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
Testset = FER2013(split='PrivateTest', transform=transform_test)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=opt.bs, shuffle=False, num_workers=0)

# model
if opt.model == 'VGG19':
    net = VGG('VGG19')
else:
    net = ResNet18()

# train
use_cuda = torch.cuda.is_available()
best_Test_acc = 0  # best Test accuracy
best_Test_acc_epoch = 0
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 250
path = './model/'

opt.resume = True
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./model/'+opt.model+'.pth')
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = checkpoint['best_Test_acc_epoch'] + 1
    print('best_Test_acc', best_Test_acc)
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()
    print('==> Running on GPU..')
else:
    print('==> Running on CPU..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if epoch > learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
    else:
        current_lr = opt.lr
    utils.set_lr(optimizer, current_lr)  # set the decayed rate
    print('learning_rate: %s' % str(current_lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / total
    return train_acc


def test(epoch):
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(Testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    test_acc = 100.*correct/total

    if test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % test_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,opt.model+'.pth'))
        best_Test_acc = test_acc
        best_Test_acc_epoch = epoch
    return test_acc


def FGSM(eps = 0.001):
    numAdSample = 0
    numFGSMSuccess = 0
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        # get AdSample
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)
        if not predicted.eq(targets.data).cpu().sum():
            continue
        numAdSample += 1
        # attack
        inputs_adv = inputs.detach()
        inputs_adv = Variable(inputs_adv).cuda().requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(net(inputs_adv).view(bs, ncrops, -1).mean(1), targets)
        grad = torch.autograd.grad(loss_ce, [inputs_adv])[0]
        inputs_adv = inputs_adv.detach() + eps * torch.sign(grad.detach())
        inputs_adv = torch.clamp_(inputs_adv, 0, 1)
        # test
        outputs_adv = net(inputs_adv)
        outputs_adv_avg = outputs_adv.view(bs, ncrops, -1).mean(1)  # avg over crops
        _adv, predicted_adv = torch.max(outputs_adv_avg.data, 1)
        if not predicted_adv.eq(targets.data).cpu().sum():
            numFGSMSuccess += 1
        utils.progress_bar(batch_idx, len(Testloader), 'numFGSMSuccess: %d| numAdSample: %d'%(numFGSMSuccess, numAdSample))
    return numFGSMSuccess, numAdSample


def PGD(eps = 0.001, alpha = 1/255, steps = 4):
    numAdSample = 0
    numPGDSuccess = 0
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        # get AdSample
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)
        if not predicted.eq(targets.data).cpu().sum():
            continue
        numAdSample += 1
        # attack
        inputs_adv = inputs.detach() + alpha * torch.randn(inputs.shape).cuda().detach()
        for i in range(steps):
            inputs_adv = Variable(inputs_adv).cuda().requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(net(inputs_adv).view(bs, ncrops, -1).mean(1), targets)
            grad = torch.autograd.grad(loss_ce, [inputs_adv])[0]
            inputs_adv = inputs_adv.detach() + alpha * torch.sign(grad.detach())
            inputs_adv = torch.min(torch.max(inputs - eps, inputs_adv), inputs + eps)
            inputs_adv = torch.clamp_(inputs_adv, 0, 1)
        # test
        outputs_adv = net(inputs_adv)
        outputs_adv_avg = outputs_adv.view(bs, ncrops, -1).mean(1)  # avg over crops
        _adv, predicted_adv = torch.max(outputs_adv_avg.data, 1)
        if not predicted_adv.eq(targets.data).cpu().sum():
            numPGDSuccess += 1
        utils.progress_bar(batch_idx, len(Testloader), 'numPGDSuccess: %d| numAdSample: %d'%(numPGDSuccess, numAdSample))
    return numPGDSuccess, numAdSample


def drawCurve():
    writer = SummaryWriter('./log')
    writer.add_scalar(opt.model + '_train_acc', 17, 0)
    writer.add_scalar(opt.model + '_test_acc', 17, 0)
    for epoch in range(1, total_epoch):
        train_acc = train(epoch)
        test_acc = test(epoch)
        writer.add_scalar(opt.model + '_train_acc', train_acc, epoch)
        writer.add_scalar(opt.model + '_test_acc', test_acc, epoch)


def studyPGDSteps():
    Steps = [1, 2, 3, 4, 5]
    Accuracy = []
    for step in Steps:
        numPGDSuccess, numAdSample = PGD(eps = 0.001, steps = step)
        Accuracy.append((numAdSample - numPGDSuccess)/len(Testloader))
    plt.figure(figsize=(5, 5))
    plt.plot(Steps, Accuracy, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 6, step=1))
    plt.title("Accuracy vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.savefig("./accuracy.jpg")
    plt.show()


def transferPGD(eps = 0.001, alpha = 1/255, steps = 10):
    # load model
    net1 = ResNet18()
    net2 = VGG('VGG19')
    checkpoint1 = torch.load('./model/Resnet18.pth')
    net1.load_state_dict(checkpoint1['net'])
    checkpoint2 = torch.load('./model/VGG19.pth')
    net2.load_state_dict(checkpoint2['net'])
    if use_cuda:
        net1.cuda()
        net2.cuda()
    numAdSample = 0
    numTransPGDSuccess = 0
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        # get AdSample
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net2(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)
        if not predicted.eq(targets.data).cpu().sum():
            continue
        numAdSample += 1
        # attack
        inputs_adv = inputs.detach() + alpha * torch.randn(inputs.shape).cuda().detach()
        for i in range(steps):
            inputs_adv = Variable(inputs_adv).cuda().requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(net1(inputs_adv).view(bs, ncrops, -1).mean(1), targets)
            grad = torch.autograd.grad(loss_ce, [inputs_adv])[0]
            inputs_adv = inputs_adv.detach() + alpha * torch.sign(grad.detach())
            inputs_adv = torch.min(torch.max(inputs - eps, inputs_adv), inputs + eps)
            inputs_adv = torch.clamp_(inputs_adv, 0, 1)
        # test
        outputs_adv = net2(inputs_adv)
        outputs_adv_avg = outputs_adv.view(bs, ncrops, -1).mean(1)  # avg over crops
        _adv, predicted_adv = torch.max(outputs_adv_avg.data, 1)
        if not predicted_adv.eq(targets.data).cpu().sum():
            numTransPGDSuccess += 1
        utils.progress_bar(batch_idx, len(Testloader),
                           'numTransPGDSuccess: %d| numAdSample: %d' % (numTransPGDSuccess, numAdSample))
    return numTransPGDSuccess, numAdSample


def studyTransferPGD():
    eps = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    acc = []
    for ep in eps:
        print('eps:', ep)
        numTransPGDSuccess, numAdSample = transferPGD(eps = ep)
        acc.append((numAdSample - numTransPGDSuccess)/len(Testloader))
    plt.figure(figsize=(5, 5))
    plt.plot(eps, acc, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.06, step=0.01))
    plt.title("Accuracy vs Epsilons")
    plt.xlabel("Epsilons")
    plt.ylabel("Accuracy")
    plt.savefig("./accuracy.jpg")
    plt.show()


if __name__ == '__main__':
    studyTransferPGD()
