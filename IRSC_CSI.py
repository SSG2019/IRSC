import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

from Modules_CSI import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test(model, test_loader_1, test_loader_2, snr):
    model.eval()
    correct_1 = 0
    correct_2 = 0
    total = 0
    with torch.no_grad():

        for test_data_1, test_data_2 in zip(test_loader_1, test_loader_2):
            data_1, target_1 = test_data_1
            data_1 = data_1.to(device)
            target_1 = target_1.to(device)
            data_2, target_2 = test_data_2
            data_2 = data_2.to(device)
            target_2 = target_2.to(device)

            output_1, output_2 = model(data_1, data_2, snr)
            pred_1 = output_1.argmax(dim=1, keepdim=True)
            correct_1 += pred_1.eq(target_1.view_as(pred_1)).sum().item()
            pred_2 = output_2.argmax(dim=1, keepdim=True)
            correct_2 += pred_2.eq(target_2.view_as(pred_2)).sum().item()
            total += target_1.size(0)
    return correct_1 / total, correct_2 / total


def main_train():
    epochs = 200
    snr = 12
    kwargs = {'num_workers': 2, 'pin_memory': True}

    test_mnist = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)
    test_cifar = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    model = Transceiver().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.NLLLoss().to(device)

    t = tqdm(range(len(range(epochs))), desc="epoch")
    for I in t:
        train_mnist = datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        train_mnist_spl = torch.utils.data.Subset(train_mnist, range(50000))
        train_loader_mnist = torch.utils.data.DataLoader(train_mnist_spl, batch_size=128,
                                                         shuffle=True, **kwargs)

        train_loader_cifar = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=128, shuffle=True, **kwargs)

        for train_data_mnist, train_data_cifar in zip(train_loader_mnist, train_loader_cifar):
            optimizer.zero_grad()

            data_mnist, target_mnist = train_data_mnist
            data_mnist = data_mnist.to(device)
            target_mnist = target_mnist.to(device)
            data_cifar, target_cifar = train_data_cifar
            data_cifar = data_cifar.to(device)
            target_cifar = target_cifar.to(device)

            output_mnist, output_cifar = model(data_mnist, data_cifar, snr)

            loss_1 = criterion(output_mnist, target_mnist)
            loss_2 = criterion(output_cifar, target_cifar)
            alpha = loss_1.item() / (loss_1.item() + loss_2.item())
            loss = alpha * loss_1 + (1 - alpha) * loss_2
            loss.backward()
            if I > 10 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.005)
            optimizer.step()
            if I % 2 == 0 or I == 1:
                t.set_description(
                    'iter {:5d}: L1 = {:.5f}, L2 = {:.5f}, alpha = {:.3f}'.format(I, loss_1, loss_2, alpha))
                t.refresh()  # to show immediately the update

        acc1, acc2 = test(model, test_mnist, test_cifar, snr)
        print('1 Test Accuracy:', acc1, '2 Test Accuracy:', acc2)

    if not os.path.exists('checkpoints/interference/csi'):
        os.makedirs('checkpoints/interference/csi')
    with open('checkpoints/interference/csi' + '/model_{}.pth'.format(str(I).zfill(2)),
              'wb') as f:
        torch.save(model.state_dict(), f)


def main_test():
    # SNR = [-6, -3, 0, 3, 6, 9, 12, 15]
    SNR = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    kwargs = {'num_workers': 2, 'pin_memory': True}

    test_mnist = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)
    test_cifar = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    model = Transceiver().to(device)
    model.load_state_dict(torch.load('./checkpoints/interference/csi/model_199.pth'))
    ACC1 = []
    ACC2 = []
    for snr in SNR:
        accuracy1 = 0
        accuracy2 = 0
        t = 10
        for i in range(t):
            acc1, acc2 = test(model, test_mnist, test_cifar, snr)
            accuracy1 += acc1
            accuracy2 += acc2
        print('snr:', snr, '1 Test Accuracy:', accuracy1 / t, '2 Test Accuracy:', accuracy2 / t)
        ACC1.append(accuracy1/t)
        ACC2.append(accuracy2/t)
    print('ACC1:', ACC1, 'ACC2:', ACC2)


if __name__ == '__main__':
    seed_torch(0)
    '''训练'''
    # main_train()
    main_test()

