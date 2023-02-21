import matplotlib.pyplot as plt
import numpy as np
import ast
import seaborn as sns

from dataset_loader import *


_max = 0
with open("info.txt","r+") as f:
    for line in f.readlines():
        label,predicted = line.split("\t")
        predicted = ast.literal_eval(predicted)
        if int(label) > _max:
            _max = int(label)
        for key in predicted.keys():
            if key > _max:
                _max = key

l = [[0 for i in range(_max+1)] for i in range(_max+1)]
l = np.array(l,dtype=float)

print(l.shape)

ps = [0 for i in range(_max+1)]
ps2 = [0 for i in range(_max+1)]

with open("info.txt","r+") as f:
    for line in f.readlines():
        label,predicted = line.split("\t")
        predicted = ast.literal_eval(predicted)
        ps[int(label)] += 1 
        tmp_max = 0  
        predicted_label = 0
        for key in predicted.keys():
            if predicted[key] > tmp_max:
                tmp_max = predicted[key]
                predicted_label = key
        l[int(label),predicted_label] += 1
        ps2[predicted_label] += 1


print(ps)
for i in range(_max+1):
    for j in range(_max+1):
        if ps[i] == 0:
            continue
        l[i,j] /= ps[i]
        if l[i,j] > 0:
            print(l[i,j])

# sns.heatmap(l)
# plt.show()

annotations = root_path+"/annotations.txt"

d_training = DMD(annotations,train=True,use_files=True)
d_testing = DMD(annotations,train=False,use_files=True)

training_histogram = {}
ps3 = [0 for i in range(_max+1)]
y_axis = [i for i in range(_max+1)]
c_map = {}
for i in range(len(d_testing.data)):
    label = d_testing.get_item_label(i) 
    numeric = d_testing.class_map[d_testing.get_item_label(i)]
    c_map[numeric] = label
    ps3[numeric] += 1
    if label not in training_histogram.keys():
        training_histogram[label] = 1
    else:
        training_histogram[label] += 1

tmp = list(training_histogram.keys())
x_axis = [i for i in range(len(tmp))]
y_axis = list(training_histogram.values())

for i in range(len(tmp)):
    print(f"X Axis Value: {i} -> {tmp[i]} ")

plt.title("Testing Data Event Histogram")
plt.xlabel("Events")
plt.ylabel("Frequency")
plt.bar(x_axis,y_axis)
plt.show()