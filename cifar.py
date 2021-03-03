import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, BatchSampler, Subset
#from augmentation import RandAugment #RandomAugment
from autoaugment import CIFAR10Policy
import random
from utility.cutout import Cutout
import numpy as np

class Cifar:
    def __init__(self, args):
#        self.train_size = args.train_size
        self.batch_size = args.batch_size
        self.threads = args.threads

#        mean, std = self._get_statistics()
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
#        print("4. Cifar.py ", probabilities)

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#            transforms.Normalize(mean, std),
            transforms.Normalize(cifar10_mean, cifar10_std),
            Cutout()
        ])
        if args.add_augment:
            train_transform.transforms.insert(0,CIFAR10Policy())
            train_transform.transforms.insert(0,torchvision.transforms.RandomRotation(30.0))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
#            transforms.Normalize(mean, std)
        ])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        filename = ('data/config/cifar10.%d@%d%s.npy' % (args.seed, args.train_size, args.data_bal) )
        print("Loading data configuration file ", filename)
        train_samples = np.load(filename)
#        print("train_samples ", train_samples)
        self.train_set = Subset(self.train_set, indices=train_samples)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
        sampler = RandomSampler(self.train_set, replacement=False) #, num_samples=self.train_size)
        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=True)

        self.train = torch.utils.data.DataLoader(self.train_set, batch_sampler=batch_sampler, num_workers=self.threads)
#        self.train = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.threads)
        self.test = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

