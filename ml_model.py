import os
import torch
import gc
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import sqlite3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.environ["DMD_ROOT"]

db = sqlite3.connect('dmd.db',check_same_thread=False)
cur = db.cursor()


class NeuralNetwork(nn.Module):
    def __init__(self,of=10):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #This defines the order of execution of the network
        self.linear_relu_stack = nn.Sequential(
            #Applies a linear transformation to input
            #Arg 1 is the size of the input. Meaning, how many dimensions there are.
            #Arg 2 is the output size. 
            nn.Linear(500*500*3, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, of),
            
        )

    def forward(self, x):
        x = self.flatten(x)
        
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    from dataset_loader import *

    annotations = root_path+"/annotations.txt"

    d_training = DMD(annotations,train=True)
    d_testing = DMD(annotations,train=False)

    d_training2 = torchvision.datasets.ImageFolder(root_path,transform=transforms.ToTensor())

    for i in d_training.class_map.keys():
        print(i)
        assert(d_training.class_map[i] == d_testing.class_map[i] )

    transform = transforms.Compose([transforms.PILToTensor()])
    
    d_tr_loader = DataLoader(d_training2,batch_size=10,shuffle=True,num_workers=5)
    # d_te_loader =DataLoader(d_testing,batch_size=10,shuffle=True,num_workers=5)

    
    
    model = torchvision.models.resnet18().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(5):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        f = 0
        print("TRAINING STARTS NOW")
        for i, data in enumerate(d_tr_loader):
            # print(data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data   
            inputs = inputs.cuda()
            labels= labels.cuda()
            # j=0
            # while batch_size*j < len(inputs):

            #     if batch_size*(j+1) > len(inputs):
            #         frame = inputs[batch_size*j:].to(torch.float32).cuda()
            #         label = labels[batch_size*j:].cuda()
            #     else:
            #         frame = inputs[batch_size*j:batch_size*(j+1)].to(torch.float32).cuda()
            #         label = labels[batch_size*j:batch_size*(j+1)].cuda()
                
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print( f"Prediction: {outputs} Label: {label}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i*10)%100 == 0:                
                print(f'[epoch:{epoch + 1}, iter:{i :5d}, frame:{f}] loss: {running_loss/100}')
                running_loss = 0.0
            

            gc.collect()
            torch.cuda.empty_cache()
            # del frame
            del inputs
            del labels
    
            torch.save(model.state_dict(), "./m1")
    


print('Finished Training')