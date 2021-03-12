import sys
import os
import numpy as np
import glob
#import math


'''
Reads all the files in the directory and extracts a summary of the results. 
Expects 4 runs of each scenario
Writes the summary to a file Summary.
The highlights are the Class accuracies for Test and best_test

Typical input at the end of the results files looks like: 

┃         199  ┃      0.0201  │     99.44 %  ┃   1.002e-03  │   01:11 min  ┃  0.0704 │    96.94% │    97.10% ┃
┃   Test accuracies by class  [97.4 98.8 96.3 92.7 97.6 94.1 98.6 98.2 98.6 97.1]

'''

#numFiles = int(sys.argv[2])
numFiles = 2
alpha = 0.1
bestAcc = [0]*numFiles

print("=> Writing out files ....")
filename = 'ResultsSummary'
print(filename)
fileOut = open(filename,'w')

files = os.listdir('.')
listing = glob.glob('./*0')

listing.sort()
#print(listing)
for j in range(len(listing)):
#    classAcc = ['', '', '', '']
#    midway   = np.zeros(4, dtype=float)
    bestAcc  = np.zeros(numFiles, dtype=float)
    classAcc  = np.zeros(10, dtype=float)
    avgClassAcc  = np.zeros(10, dtype=float)

#    print(listing[j])
    for i in range(0,numFiles):
        name = listing[j]
        name = name[:-1] + str(i)
#        if name.find("unequal") > 0:
#            continue
        seen = 0
        testclass = ""
#        print(name," exits ",os.path.isfile(name))
        if os.path.isfile(name):
            with open(name,"r") as f:
                for line in f:
                    if (line.find("Test accuracies by class") > 0):
                        testclass = line
                        if testclass.find("]")>0:
                            pref,post = testclass.split("[")
                            post = post[:-2]
                            accs = post.replace("'","").split(",")
                            for k in range(10):
                                avgClassAcc[k] = alpha*float(accs[k]) + (1.0-alpha)*avgClassAcc[k]
                    if (line.find(" min ") > 0):
                        acc = line.split("┃ ")
#                        print("acc ", acc)
                        accs = acc[3].split("│")
                        pre, post = accs[2].split("%") 
                        bestAcc[i] = float(pre)
        classAcc += avgClassAcc
#        if seen == 1:
#            pref,post = testclass.split("[")
#            post = post[:-2]
#            print(post)
#            accs = post.replace("'","").split(",")
#            print(accs)
#            for k in range(10):
#                classAcc[k] += float(accs[k])

    classAcc /= numFiles
    for i in range(10):
        classAcc[i] = "{:.2f}".format(classAcc[i])

    print(classAcc)

#    midAcc = np.mean(midway)
#    print("seen ", seen)
    midway = np.sort(bestAcc)
    midAcc = "{:.2f}".format(np.mean(midway[1:]))
#    acc = np.mean(bestAcc)
    acc = "{:.2f}".format(np.mean(bestAcc))
    accSTD = "{:.2f}".format(np.std(bestAcc))
    print(name," ",midAcc," ",bestAcc," mean, std= ",acc,accSTD)

#    for i in range(0,numFiles):
#        print(classAcc[i])
#        print(numTrainPerClass[i])

#    fileOut.write('{:f}'.format(bestAcc)
    fileOut.write(str(classAcc) + "\n")
    fileOut.write(name+"  "+midAcc+"  [")
    for i in range(0,numFiles):
        fileOut.write(str(bestAcc[i])+" ")
    fileOut.write("] Mean= "+acc)
    fileOut.write(" STD= "+accSTD)
    fileOut.write("\n")
#    for i in range(0,numFiles):
#        fileOut.write(classAcc[i])
#        accs = classAcc[i].split(",")
#        try:
#            for x in accs:
#                fileOut.write('{0:0.2f},  '.format(float(x)))
#        except:
#            pass
#        fileOut.write("\n")
#    for i in range(0,numFiles):
#        fileOut.write(numTrainPerClass[i])
#        fileOut.write("\n")
fileOut.close()
exit(1)
