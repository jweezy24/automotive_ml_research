from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import torch
import torchvision
import os

root_path = os.environ["DMD_ROOT"]


def get_accuracy(model, data_loader, criterion,classes):
    total_correct = 0
    total_samples = 0
    
    model.eval()
    
    confusion_matrix = np.zeros((len(classes),len(classes)))
    with torch.no_grad():
        for inputs, targets in data_loader.dataset:
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            outputs = outputs
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            np_predicted = predicted.cpu().numpy()
            np_targets = targets.cpu().numpy()
            for p in range(np_predicted.shape[0]):
                confusion_matrix[np_predicted[p],np_targets[p]]+=1
            
            
    accuracy = 100 * total_correct / total_samples
    loss = criterion(outputs, targets).item()
    print(confusion_matrix)
    return accuracy, loss


if __name__ == "__main__":
    device = torch.device("cuda")

    batch_size = 10
    
    d_training2 = torchvision.datasets.ImageFolder(root_path,transform=transforms.ToTensor())

    classes = d_training2.classes

    d_tr_loader = DataLoader(d_training2,batch_size=batch_size,shuffle=True,num_workers=5)


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


    checkpoint = torch.load('checkpoint')
    
    print(checkpoint.keys())

    model = torchvision.models.resnet18().cuda()
    model.load_state_dict(checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    get_accuracy(model,test_loader,criterion,classes)