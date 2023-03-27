import os
import torch
import gc
import torchvision
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.environ["DMD_ROOT"]


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
    
def get_accuracy(model, data_loader, criterion):
    total_correct = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            outputs = outputs
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            
    accuracy = 100 * total_correct / total_samples
    loss = criterion(outputs, targets).item()
    
    return accuracy, loss

if __name__ == "__main__":
    # from dataset_loader import *

    annotations = root_path+"/annotations.txt"

    # d_training = DMD(annotations,train=True)
    # d_testing = DMD(annotations,train=False)

    d_training2 = torchvision.datasets.ImageFolder(root_path,transform=transforms.ToTensor())

    # for i in d_training.class_map.keys():
    #     print(i)
    #     assert(d_training.class_map[i] == d_testing.class_map[i] )

    transform = transforms.Compose([transforms.PILToTensor()])
    
    batch_size = 10

    d_tr_loader = DataLoader(d_training2,batch_size=batch_size,shuffle=True,num_workers=5)
    # d_te_loader =DataLoader(d_testing,batch_size=10,shuffle=True,num_workers=5)


    # Define the indices to split the dataset
    dataset_size = len(d_tr_loader)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))  # 80% training data, 20% testing data
    np.random.shuffle(indices)  # shuffle the indices for randomness
    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(d_tr_loader, batch_size=batch_size, sampler=train_sampler, num_workers=5)
    test_loader = DataLoader(d_tr_loader, batch_size=batch_size, sampler=test_sampler,num_workers=5)

    
    model = torchvision.models.resnet18().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 5
    k_folds = 5  # choose the number of folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    print(train_loader.dataset.dataset)
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_loader.dataset.dataset)):
        
        # Create new data loaders for this fold
        train_subset = torch.utils.data.Subset(train_loader.dataset.dataset, train_indices)
   
        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        
        val_subset = torch.utils.data.Subset(train_loader.dataset.dataset, val_indices)
        val_loader_fold = DataLoader(val_subset, batch_size=train_loader.batch_size)



        for epoch in range(epochs):  # loop over the dataset multiple times
            model.train()
            running_loss = 0.0

            f = 0
            print( train_loader_fold)
            print("TRAINING STARTS NOW")
            for i, data in enumerate(train_loader_fold):
                # print(data)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data   
                inputs = inputs.cuda()
                labels= labels.cuda()
                
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
        
            torch.save(model.state_dict(), "./checkpoint")

            accuracy,loss_eval = get_accuracy(model, val_loader_fold, criterion)
            with open("checkpoints.txt", "a+") as f:
                f.write(f" Epoch {epoch}\t Fold: {fold}\t Accuracy: {accuracy}\t Loss: {loss_eval}\n") 
    
    

print('Finished Training')