import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse

class Cifar:
    def __init__(self, numPerClass):
        # Initialization for Cifar10
        debug = 0
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        cifar10TrainSize = [5000, 10]
        numClasses = 10
        batch_size = 1000

        size = numClasses*numPerClass
        print("size ", size)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        labels = []
        i = 0
        indx = np.zeros([numClasses], dtype=int)
        samples = np.zeros(cifar10TrainSize, dtype=int)
    
        for sample, label in DataLoader(self.train_set, batch_size=batch_size ):
            for lab in label:
                l = lab.item()
                samples[indx[l],l] = i            
                indx[l] += 1
                i += 1

        for i in range(numClasses):
            samples[:indx[i],i] = random.sample(samples[:indx[i],i].tolist(),k=indx[i])

        indx *= 0
        for seed in range(args.num_seeds):
            train_samples = []
            for i in range(numClasses):
                for j in range(indx[i],indx[i]+numPerClass):
                    train_samples.append(samples[j,i])
                indx[i] = indx[i] + numPerClass

            filename = ('data/config/cifar10.%d@%d%s.npy' % (seed, size, 'equal') )
            print("Writing file ", filename)
            np.save(filename, train_samples)
            # To print clear text versions (for Debug)
            if debug == 1:
                fileOut = open(filename,'w')
                for i in range(numClasses):
                    fileOut.write(str(numPerClass)+", ")
                fileOut.write("\n")
                for i in range(size):
                    fileOut.write(str(train_samples[i])+", ")
                fileOut.write("\n")
                fileOut.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", default=1, type=int, help="Number of config files to create.")
    parser.add_argument("--num_class", default=40, type=int, help="Number of labeled samples per class.")
    args = parser.parse_args()
    print(args)

#    numPerClass = 40
    
    dataset = Cifar(args.num_class)

    exit(1)
