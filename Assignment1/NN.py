from pybrain.tools.shortcuts import buildNetwork
import os
import numpy as np
import time
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import seaborn as sns
import matplotlib.pyplot as plt

labels = []
input= 0
hidden = 20
out = 1
initialize = True
patients = 0
standardization = 'no'
for i in os.listdir('data'):
    if i.endswith(".csv"):

        stop = False
        with open('data/' + i) as f:
            totals = np.zeros(len(labels))
            counter = 0
            for counter, line in enumerate(f):
                line = line.replace('\n', '')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                if len(line) != 94:
                    stop = True
                    break
                values = list(values[i] for i in [18,39,60,81, 87])

                if counter == 0:
                    if initialize == True:
                        initialize = False
                        labels = values
                        input = len(labels) - out
                        ds = SupervisedDataSet(input, 1)
                    totals = np.zeros(len(values))
                    continue

                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                totals = totals + values
            mean = totals / counter
            if stop == True:
                continue
            patients += 1
        with open('data/' + i) as f:
            next(f)
            for counter, line in enumerate(f):
                line = line.replace('\n', '')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                values = list(values[i] for i in [18,39,60,81, 87])

                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                if standardization == 'yes':
                    ds.addSample(values[0:-1] - mean[0:-1], values[-1] - mean[-1])
                else:
                    ds.addSample(values[0:-1], values[-1])
    break

test, train = ds.splitWithProportion(0.25)
net = buildNetwork(input, hidden, out, bias=True)
trainer = BackpropTrainer(net, train, momentum=0.1, weightdecay=0.01)
start = time.time()
trainer.trainUntilConvergence(maxEpochs=1000, verbose=True, continueEpochs=10, validationProportion=0.25)
print(time.time() - start, 'seconds')


out = net.activateOnDataset(test)
f_out = 0
bench = 0
total = 0
MSE = 0.0
MAE = 0.0
errorVec = []
for i, o in enumerate(out):
    sign = np.sign(out[i] * test['target'][i])
    if sign == -1:
        f_out += 1
    else:
        bench += 1
    total += 1
    error = out[i] - test['target'][i]
    errorVec.append(error[0])
    print(out[i], test['target'][i], error)
    MAE = MAE + abs(out[i] - test['target'][i])
    MSE = MSE + (out[i] - test['target'][i]) ** 2
MSE = MSE / total
MAE = MAE / total

print('total', total)
print('MSE', MSE)
print('MAE', MAE)

# %%
print('NN based on', patients, 'patients')
print('Number of nodes:')
print('   Input:..................', input)
print('   Hidden:.................', hidden)
print('   Output:.................', out)
print('Total training instances:..', len(train))
print('Total test instances:......', len(test))
print('Mean absolute error:.......', np.mean(np.abs(errorVec)))
print('Mean Squared error:........', MSE[0])
print('Standardization:...........', standardization)

sns.set_style('darkgrid')
sns.distplot(errorVec, bins=15)

plt.show()