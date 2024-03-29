import os
import torch
import gc
import torchvision
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.environ["DMD_ROOT"]

# Set the random seed for reproducibility
torch.manual_seed(42)


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

    now = datetime.now() # current date and time
    date = date_time = now.strftime("%m-%d-%Y")

    annotations = root_path+"/annotations.txt"

    # d_training = DMD(annotations,train=True)
    # d_testing = DMD(annotations,train=False)

    d_training2 = torchvision.datasets.ImageFolder(root_path,transform=transforms.ToTensor())


    transform = transforms.Compose([transforms.PILToTensor()])
    
    batch_size = 10


    # Define the indices to split the dataset
    dataset_size = len(d_training2)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(d_training2, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    
    model = torchvision.models.resnet18().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
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
            for i, data in enumerate(tqdm(train_loader_fold)):
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
                if (i*10)%10000 == 0:                
                    print(f'[epoch:{epoch + 1}, iter:{i :5d}, frame:{f}] loss: {running_loss/100}')
                    running_loss = 0.0
                

                gc.collect()
                torch.cuda.empty_cache()
                # del frame
                del inputs
                del labels
        
            torch.save(model.state_dict(), "./checkpoint")

            accuracy,loss_eval = get_accuracy(model, val_loader_fold, criterion)
            with open(f"checkpoints_{date}.txt", "a+") as f:
                f.write(f" Epoch {epoch}\t Fold: {fold}\t Accuracy: {accuracy}\t Loss: {loss_eval}\n") 
    
    

print('Finished Training')
