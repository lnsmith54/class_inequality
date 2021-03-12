import sys
import os
import numpy as np
import glob
#import math
import statistics
import matplotlib.pyplot as plt

'''
Reads all the files in the directory and extracts a summary of the results.
Expects 4 runs of each scenario
Writes the summary to a file Summary.
The highlights are the Class accuracies for Test and best_test

Typical input at the end of the results files looks like:

┃   Test accuracies by class  ['72.00', '70.50', '70.80', '60.20', '63.20', '58.00', '68.00', '61.10', '74.40', '80.50']

Test set class accuracies  93.15 [96.1, 97.7, 87.6, 76.9, 96.1, 86.1, 98.3, 97.2, 98.3, 97.2]

'''
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


numFiles = int(sys.argv[1])
#File = sys.argv[1]
#skip = int(sys.argv[2])
skip = 0

nEpochs = 20000
nplots = 10

files = os.listdir('.')
listing = glob.glob('./*0')

listing.sort()
#print(listing)
for j in range(len(listing)-skip):

    for i in range(0,numFiles):
        name = listing[j+skip]
        File = name[2:-1] + str(i)
        print(File," exits ",os.path.isfile(File))
        plotAcc  = np.zeros((nEpochs,nplots), dtype=float)
        if os.path.isfile(File):
            cnt = 0
            with open(File,"r") as f:
                for line in f:
                    if (line.find("Test accuracies by class") > 0):
                        line = line.replace("'","")
                        indx1 = line.find("[")
                        indx2 = line.find("]")
                        classAccs = line[indx1+1:indx2]
                        accs = classAccs.split(",")
#                        print(accs)
                        for i in range(nplots):
                            plotAcc[cnt,i] = float(accs[i])
                        cnt += 1

#            print(File)
#            print(classAccs)
#            print(plotAcc[:cnt,0])
            tit = File[:-2]
#            print(tit)
#            exit(1)
            fig, ax = plt.subplots()
            plt.title(tit)
            axes = plt.gca()
#            axes.set_xlim([xmin,xmax])
            axes.set_ylim([20,90])
            for i in range(nplots):
                accs = moving_average(plotAcc[:cnt,i], 200)
                accs = accs[::20]
                ax.plot(range(len(accs)),accs,label=str(i))
            ax.legend()
            fig.savefig('sve/'+File+'.png',format='png')
            plt.close()
#            exit(1)
print("The End")
