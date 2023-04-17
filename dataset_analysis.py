from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import os

root_path = os.environ["DMD_ROOT"]

# Set the random seed for reproducibility
torch.manual_seed(42)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def parse_annotations():
    path = "/nobackup.1/jwwest/automotive_ml_research/dataset_dmd/clips/annotations.txt"
    final_dictionary = {}
    with open(path,"r") as f:
        for line in f.readlines():
            
            #Fixing bug with inital path creation
            if "yawning" in line:
                if 'yawning-Yawning without hand' in line:
                    line = line.replace('yawning-Yawning without hand','yawning-Yawning_without_hand')
                if 'yawning-Yawning with hand' in line:
                    line = line.replace('yawning-Yawning with hand','yawning-Yawning_with_hand')
            
            #Splits the meta data and clip information.
            sp1 = line.split("{")
            sp1[0] = sp1[0].replace(" ","\t")
            rebuilt_line = "{".join(sp1)
            sp2 = rebuilt_line.split("\t")
            
            #Matching label with metadata
            label,_,_,meta_data = sp2 
            meta_data = eval(meta_data)
            final_dictionary[label] = meta_data


    return final_dictionary


def match_clip_to_annotation(paths,annotations,predicted,target,incorrect):
    c = 0
    for path in paths:

        #Fixing bug with inital path creation
        if "yawning" in path:
            if 'yawning-Yawning without hand' in path:
                path = path.replace('yawning-Yawning without hand','yawning-Yawning_without_hand')
            if 'yawning-Yawning with hand' in path:
                path = path.replace('yawning-Yawning with hand','yawning-Yawning_with_hand')

        p = path.split("/")
        label = "/".join(p[-3:-1])
        m_data = annotations[label]

        if predicted[c]!= target[c]:
            tmp_age = incorrect["age"]
            tmp_gender = incorrect["gender"]
            tmp_exp = incorrect["exp"]
            tmp_dfreq = incorrect["d_freq"]
            tmp_label = incorrect["label"]

            tmp_age.append(m_data["age"])
            tmp_gender.append(m_data["gender"])
            tmp_exp.append(m_data["exp"])
            tmp_dfreq.append(m_data["d_freq"])
            tmp_label.append(label)
            

            incorrect["age"] = tmp_age 
            incorrect["gender"] = tmp_gender 
            incorrect["exp"] = tmp_exp 
            incorrect["d_freq"] = tmp_dfreq 
            incorrect["label"] = tmp_label 


        c+=1


def make_plot(dictionary,total_samples,title,data_type):
    x_axis = list(dictionary.keys())
    vals = list(dictionary.values())
    y_axis = 100*(np.array(vals)/total_samples)

    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(figsize=(7, 2))
    if "Event" in title or "Freq" in title: 
        ax.barh(range(len(x_axis)), y_axis, tick_label=x_axis)
        ax.tick_params(axis='y', which='major', pad=10)
    else:
        ax.bar(range(len(x_axis)), y_axis, tick_label=x_axis)

    ax.set_xlabel(f"{data_type}")
    ax.set_ylabel(f"Percentage of Failures")
    ax.set_title(f"Misclassifications based on {title}")
    

    plt.savefig(f"figures/{title}_misses.png")
    plt.savefig(f"figures/{title}_misses.pdf")

    plt.clf()

def plot_misses(incorrect,total_samples,classes):
    age_ranges = {20:0,30:0,40:0,50:0}
    genders = {"Male":0,"Female":0,"Non-Binary":0}
    experience = {}
    frequency = {}
    labels = {}

    for key in incorrect.keys():
        if key == "age":
            for age in incorrect[key]:
                for min_age in age_ranges.keys():
                    if age <= min_age:
                        age_ranges[min_age] +=1
                        break
        elif key == "gender":
            for gender in incorrect[key]:
                if gender == "Male" or gender == "Female":
                    genders[gender]+=1
                else:
                    genders["Non-Binary"] += 1
        elif key == "exp":
            for exp in incorrect[key]:
                if exp in experience.keys():
                    experience[exp] += 1
                else:
                    experience[exp] = 1
                
        elif key == "d_freq":
            for freq in incorrect[key]:
                if freq in frequency.keys():
                    frequency[freq] += 1
                else:
                    frequency[freq] = 1
        else:
            for label in incorrect[key]:
                #Getting the event not the video number
                label = label.split("/")[0]
                if label in labels.keys():
                    labels[label] += 1
                else:
                    labels[label] = 1

    make_plot(age_ranges,total_samples,"Age","Ages")
    make_plot(genders,total_samples,"Gender","Genders")
    make_plot(experience,total_samples,"Experience","Experience")
    make_plot(frequency,total_samples,"Driving Frequency","Driving Frequency")
    make_plot(labels,total_samples,"Event","Events")


def get_accuracy(model, data_loader, criterion,classes, annotations):
    total_correct = 0
    total_samples = 0
    
    model.eval()
    
    incorrect = {'age': [], 'gender': [], 'exp': [], 'd_freq': [], "label": []}
 
    with torch.no_grad():
        for inputs, targets, orig_paths in tqdm(data_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            outputs = outputs
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            np_predicted = predicted.cpu().numpy()
            np_targets = targets.cpu().numpy()
            
            match_clip_to_annotation(orig_paths,annotations,np_predicted,np_targets,incorrect)
            
    # plot_confusion_matrix(confusion_matrix,classes,normalize=True)            
    accuracy = 100 * total_correct / total_samples
    loss = criterion(outputs, targets).item()
    
    plot_misses(incorrect,total_samples,classes)

    return accuracy, loss

def main(annotations):
    device = torch.device("cuda")

    batch_size = 10
    
    d_training2 = ImageFolderWithPaths(root_path,transform=transforms.ToTensor())

    classes = d_training2.classes


    d_tr_loader = DataLoader(d_training2,batch_size=batch_size,shuffle=True,num_workers=5)


    # Define the indices to split the dataset
    dataset_size = len(d_training2)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(d_training2, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    print(dir(test_loader))
    
    model = torchvision.models.resnet18().cuda()
    
    checkpoint = torch.load('checkpoint')
    
    model = torchvision.models.resnet18().cuda()
    model.load_state_dict(checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    get_accuracy(model,test_loader,criterion,classes, annotations)





if __name__ == "__main__":
    parsed_info = parse_annotations()
    main(parsed_info)