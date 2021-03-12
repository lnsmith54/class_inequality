import argparse
import torch
import numpy as np
import time
import torch.nn as nn

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.poly_lr import PolyLR
import sys; sys.path.append("..")
from sam import SAM
from pytorch_loss.focal_loss import FocalLossV3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Seed specifies which data configuration to use.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--train_size", default=50000, type=int, help="How many training samples to use.")
    parser.add_argument("--multigpu", default=0, type=int, help="Train using multiple GPUs")
    parser.add_argument("--add_augment", default=1, type=int, help="Train using multiple GPUs")
    parser.add_argument("--loss", default=0, type=int, help="= 0, smooth CE; = 1, focal loss.")
    parser.add_argument("--data_bal", default='equal', type=str, help="Set to 'equal' (default) or 'unequal'.")
    args = parser.parse_args()
    print(args)

    initialize(args, seed=42)

    dataset = Cifar(args)

    log = Log(log_each=10)
    if args.multigpu == 1:
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10)
        model = nn.DataParallel(model).cuda()
    else:
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).cuda()

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    scheduler = PolyLR(optimizer, args.learning_rate, args.epochs)
    criteria = FocalLossV3(alpha=0.25, gamma=2, reduction='none')

    test_class_accuracies = np.zeros((10), dtype=float)
    tic = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.cuda() for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            if args.loss == 0:
                loss = smooth_crossentropy(predictions, targets)
            else:
                loss = criteria(predictions, targets).sum(-1)
            loss.mean().backward()
#            print("loss ", loss.size())
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
#            if args.loss == 0:
#                smooth_crossentropy(model(inputs), targets).mean().backward()
#            else:
#                criteria(model(inputs), targets).sum(-1).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), targets.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda() for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu(), targets.cpu())

    log.flush()
    elapse = (time.perf_counter() - tic) / 3600
    print(f"Total training time {elapse:0.4f} hours")
