import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
import os

class Cifar:
#    def __init__(self, seed, prevNumPerClass, num2AddPerClass):
    def __init__(self):
        # Initialization for Cifar10
        self.debug = 0
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        self.cifar10TrainSize = [5000, 10]
        self.numClasses = 10
        self.batch_size = 1000

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        labels = []
        i = 0
        indx = np.zeros([self.numClasses], dtype=int)
        self.samples = np.zeros(self.cifar10TrainSize, dtype=int)
    
        for sample, label in DataLoader(self.train_set, batch_size=self.batch_size ):
            for lab in label:
                l = lab.item()
                labels.append(l)
                self.samples[indx[l],l] = i            
                indx[l] += 1
                i += 1

        for i in range(self.numClasses):
            self.samples[:indx[i],i] = random.sample(self.samples[:indx[i],i].tolist(),k=indx[i])

    def writeFile(self, seed, prevNumPerClass, num2AddPerClass):
        size = self.numClasses*prevNumPerClass + sum(num2AddPerClass)
        print("size ", size)

        indx = np.zeros([self.numClasses], dtype=int)

        filename = ('data/config/cifar10.%d@%d%s.npy' % (seed, self.numClasses*prevNumPerClass, 'equal') )
        source = np.load(filename)
        train_samples = []
        index = 0
        for i in range(self.numClasses):
            for j in range(prevNumPerClass):
                train_samples.append(source[index])
                index += 1
            k = 0
            while (indx[i]<num2AddPerClass[i]):
                if self.samples[k,i] not in train_samples:
                    train_samples.append(self.samples[k,i])
                    indx[i] += 1
                k += 1
        filename = ('data/config/cifar10.%d@%d%s.npy' % (seed, size, 'unequal') )
        print("Writing file ", filename)
        if self.debug == 1:
            # To print clear text versions (for Debug)
            fileOut = open(filename,'w')
            for i in range(self.numClasses):
                fileOut.write(str(num2AddPerClass[i])+", ")
            fileOut.write("\n")
            for i in range(size):
                fileOut.write(str(train_samples[i])+", ")
            fileOut.write("\n")
            fileOut.close()
        else:
            np.save(filename, train_samples)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reconfig_file", default='scripts/reconfig', type=str, help="File name for reconfig file.")
    parser.add_argument("--prevNumPerClass", default=40, type=int, help="Previous size of balanced labeled dataset.")
    args = parser.parse_args()
    print(args)
#    prevNumPerClass = 40

#    fileIn = open(args.reconfig_file,'r')
    if os.path.isfile(args.reconfig_file):
        dataset = Cifar()
        with open(args.reconfig_file,"r") as f:
            for line in f:
                tmp = line.split(" ")
                tmp[10] = tmp[10].strip('\n')
                seed = int(tmp[0])
                num2AddPerClass = []
                for i in range(10):
                    num2AddPerClass.append(int(tmp[i+1]))
                print("For seed ",seed," num2AddPerClass= ", num2AddPerClass)

                dataset.writeFile(seed,args.prevNumPerClass, num2AddPerClass)

    exit(1)
