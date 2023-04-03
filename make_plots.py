from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import os

root_path = os.environ["DMD_ROOT"]



def plot_confusion_matrix(cm, classes, normalize=False, title='Face Identification Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.
    :param cm: confusion matrix to be plotted.
    :param classes: array of class names.
    :param normalize: whether to normalize confusion matrix or not.
    :param title: title of the confusion matrix plot.
    :param cmap: color map for the plot.
    :return: None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("./cm.pdf")


def get_accuracy(model, data_loader, criterion,classes):
    total_correct = 0
    total_samples = 0
    
    model.eval()
    
    confusion_matrix = np.zeros((len(classes),len(classes)))
    checkpoint = 1000
    c = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader.dataset):
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
            if c%checkpoint == 0 and c>0:
                plot_confusion_matrix(confusion_matrix,classes,normalize=True)
            c+=1
            
    accuracy = 100 * total_correct / total_samples
    loss = criterion(outputs, targets).item()
    return accuracy, loss, confusion_matrix


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