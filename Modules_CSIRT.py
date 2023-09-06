import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transmitter_MNIST(nn.Module):
    def __init__(self):
        super(Transmitter_MNIST, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 64 * 28 * 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )  # 128 * 14 * 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 256 * 7 * 7
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )  # 512 * 3 * 3
        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )  # 4 * 3 * 3
        self.fc1 = nn.Linear(4 * 3 * 3 + 32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2 * 16)

    def forward(self, x, H11, H21, H12, H22):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.encoder1(x)
        x = torch.reshape(x, (x.size()[0], 4 * 3 * 3))
        x = torch.cat([x, H11, H21, H12, H22], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        tx = x / torch.sqrt(2 * torch.mean(x ** 2))
        return tx


class Transmitter_CIFAR(nn.Module):
    def __init__(self):
        super(Transmitter_CIFAR, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 64 * 28 * 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )  # 128 * 14 * 14
        self.layer1_res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 256 * 7 * 7
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )  # 512 * 4 * 4
        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )  # 4 * 4 * 4
        self.fc1 = nn.Linear(4 * 4 * 4 + 32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2 * 16)

    def forward(self, x, H11, H21, H12, H22):
        x = self.prep(x)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.encoder1(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = torch.cat([x, H11, H21, H12, H22], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        tx = x / torch.sqrt(2 * torch.mean(x ** 2))
        return tx


class Receiver_MNIST(nn.Module):
    def __init__(self):
        super(Receiver_MNIST, self).__init__()
        self.l1 = nn.Linear(2*16+32, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        decoded = F.log_softmax(x, dim=-1)
        return decoded


class Receiver_CIFAR(nn.Module):
    def __init__(self):
        super(Receiver_CIFAR, self).__init__()
        self.l1 = nn.Linear(2*16+32, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        decoded = F.log_softmax(x, dim=-1)
        return decoded


def interference_channel(x1, x2, H11_r, H11_i, H12_r, H12_i, H21_r, H21_i, H22_r, H22_i, sigma):
    x1_r = x1[:, np.arange(0, 16)].reshape(-1, 2, 8).to(device)
    x1_i = x1[:, np.arange(16, 32)].reshape(-1, 2, 8).to(device)
    x2_r = x2[:, np.arange(0, 16)].reshape(-1, 2, 8).to(device)
    x2_i = x2[:, np.arange(16, 32)].reshape(-1, 2, 8).to(device)

    y11_r = torch.matmul(H11_r, x1_r) - torch.matmul(H11_i, x1_i)
    y11_i = torch.matmul(H11_r, x1_i) + torch.matmul(H11_i, x1_r)
    y21_r = torch.matmul(H21_r, x2_r) - torch.matmul(H21_i, x2_i)
    y21_i = torch.matmul(H21_r, x2_i) + torch.matmul(H21_i, x2_r)

    y12_r = torch.matmul(H12_r, x1_r) - torch.matmul(H12_i, x1_i)
    y12_i = torch.matmul(H12_r, x1_i) + torch.matmul(H12_i, x1_r)
    y22_r = torch.matmul(H22_r, x2_r) - torch.matmul(H22_i, x2_i)
    y22_i = torch.matmul(H22_r, x2_i) + torch.matmul(H22_i, x2_r)

    y11 = torch.cat([y11_r.reshape(-1, 2 * 16 // 2), y11_i.reshape(-1, 2 * 16 // 2)], -1)
    y21 = torch.cat([y21_r.reshape(-1, 2 * 16 // 2), y21_i.reshape(-1, 2 * 16 // 2)], -1)
    y12 = torch.cat([y12_r.reshape(-1, 2 * 16 // 2), y12_i.reshape(-1, 2 * 16 // 2)], -1)
    y22 = torch.cat([y22_r.reshape(-1, 2 * 16 // 2), y22_i.reshape(-1, 2 * 16 // 2)], -1)

    y1 = y11 + y21 + (sigma * torch.randn(x1.shape)).to(device)
    y2 = y12 + y22 + (sigma * torch.randn(x2.shape)).to(device)
    return y1, y2


class Transceiver(nn.Module):
    def __init__(self):
        super(Transceiver, self).__init__()
        self.transmitter_mnist = Transmitter_MNIST()
        self.transmitter_cifar = Transmitter_CIFAR()
        self.receiver_mnsit = Receiver_MNIST()
        self.receiver_cifar = Receiver_CIFAR()

    def forward(self, input_mnist, input_cifar, snr):
        H11_r = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)  # (torch.randn([shape1, Nr, Nt])
        H11_i = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H12_r = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H12_i = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H21_r = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H21_i = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H22_r = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        H22_i = (torch.randn([1, 2, 2]) / torch.tensor(np.sqrt(2))).to(device)
        spm = input_mnist.shape[0]
        spc = input_cifar.shape[0]
        e_H11_r = H11_r.expand(spm, -1, -1)
        e_H11_i = H11_i.expand(spm, -1, -1)
        e_H12_r = H12_r.expand(spm, -1, -1)
        e_H12_i = H12_i.expand(spm, -1, -1)
        e_H21_r = H21_r.expand(spc, -1, -1)
        e_H21_i = H21_i.expand(spc, -1, -1)
        e_H22_r = H22_r.expand(spc, -1, -1)
        e_H22_i = H22_i.expand(spc, -1, -1)
        H11 = torch.cat([e_H11_r.reshape(-1, 2 * 2), e_H11_i.reshape(-1, 2 * 2)], -1)
        H12 = torch.cat([e_H12_r.reshape(-1, 2 * 2), e_H12_i.reshape(-1, 2 * 2)], -1)
        H21 = torch.cat([e_H21_r.reshape(-1, 2 * 2), e_H21_i.reshape(-1, 2 * 2)], -1)
        H22 = torch.cat([e_H22_r.reshape(-1, 2 * 2), e_H22_i.reshape(-1, 2 * 2)], -1)

        txm = self.transmitter_mnist(input_mnist, H11, H21, H12, H22)
        txc = self.transmitter_cifar(input_cifar, H11, H21, H12, H22)

        sigma = np.sqrt(0.5 / np.power(10, snr / 10))
        rxm, rxc = interference_channel(txm, txc, e_H11_r, e_H11_i, e_H12_r, e_H12_i, e_H21_r, e_H21_i, e_H22_r,
                                        e_H22_i, sigma)

        zm = self.receiver_mnsit(torch.cat([rxm, H11, H21, H12, H22], -1))
        zc = self.receiver_cifar(torch.cat([rxc, H11, H21, H12, H22], -1))
        return zm, zc
