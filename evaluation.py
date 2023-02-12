import os
import torch
import gc
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from ml_model import *
from dataset_loader import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.environ["DMD_ROOT"]



if __name__ == "__main__":
    annotations = root_path+"/annotations.txt"

    d_training = DMD(annotations,train=True)
    d_testing = DMD(annotations,train=False,use_files=False)


    model = NeuralNetwork(of=len(d_training.classes)).to(device)
    model.eval()

    labels = []
    counts = {}
    for i in range(d_testing.get_length()):
        tmp = d_testing.get_item_label(i)
        labels.append(tmp)
        if tmp not in counts.keys():
            counts[tmp] = 1
        else:
            counts[tmp] += 1

    events = len(d_testing.classes)
    confusion_matrix = [ [0 for i in range(events)] for i in range(events)]
    confusion_matrix = np.array(confusion_matrix)

    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    f = 0
    correct = 0
    for i, data in enumerate(d_testing):
        # get the inputs; data is a list of [inputs, labels]
        if data == None:
            continue
        inputs, labels = data
        

        print(f"Evaluationg video {i} with {len(inputs)} frames")
        batch_size = 10
        j=0
        while batch_size*j < len(inputs):

            if batch_size*(j+1) > len(inputs):
                frame = inputs[batch_size*j:].to(torch.float32).cuda()
                label = labels[batch_size*j:].cuda()
            else:
                frame = inputs[batch_size*j:batch_size*(j+1)].to(torch.float32).cuda()
                label = labels[batch_size*j:batch_size*(j+1)].cuda()
            


            scores = model(frame)
            _, predictions = scores.max(1)
            label = label[0]
            predicted_labels = {}
            for s in predictions:
                if s == label:
                    correct+=1
                if s.item() not in predicted_labels.keys():
                    predicted_labels[s.item()] = 1
                else:
                    predicted_labels[s.item()] = 1

            confidence = correct/len(predictions)
            if confidence < 0.8:
                _max = 0
                _label = 0
                for i in predicted_labels.keys():
                    v = predicted_labels[i]
                    if v > _max:
                        _max = v
                        _label = i
                
                confusion_matrix[label.item(),_label] += 1
                print(f"Value = {confusion_matrix[label.item(),_label]}\t predicted: {_label}\tActual {label.item()}")
            else:
                confusion_matrix[label.item(),label.item()]+=1
                print(f"Value = {confusion_matrix[label.item(),label.item()]}\t predicted: {label.item()}\tActual {label.item()}")
            gc.collect()
            torch.cuda.empty_cache()
            del frame
        del inputs
        del labels