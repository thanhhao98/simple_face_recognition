import numpy as np
import json
import matplotlib.pyplot as plt

with open('result_10000.json', 'r') as f:
    d = json.load(f)

pos = [i[2] for i in d[1]]
neg = [i[2] for i in d[0]]
TP = 0
TN = 0
FP = 0
FN = 0
thesholds = list(range(10, 100, 1))
thesholds = [i/100 for i in thesholds]
f1s = []
precisions = []
recalls = []
for theshold in thesholds:
    for p in pos:
        if p >= theshold:
            TP += 1
        else:
            FP += 1
    for n in neg:
        if n <= theshold:
            TN += 1
        else:
            FN += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(recall * precision) / (recall + precision)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

# plt.plot(
    # thesholds, precisions, 'r--',
    # thesholds, recalls, 'b--',
    # thesholds, f1s, 'g--'
# )

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(thesholds, precisions, label='precison')  # Plot some data on the axes.
ax.plot(thesholds, recalls, label='recall')  # Plot more data on the axes...
ax.plot(thesholds, f1s, label='f1')  # ... and some more.
ax.set_xlabel('theshold')  # Add an x-label to the axes.
ax.set_ylabel('value')  # Add a y-label to the axes.
ax.set_title("Benmark")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.savefig('a.png')

idx = np.argmax(np.array(f1s), axis=0)
print('Best theshold', thesholds[idx])
print(
    f'precision: {precisions[idx]}, recall: {precisions[idx]}, f1: {f1s[idx]}'
)
