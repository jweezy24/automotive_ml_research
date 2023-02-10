import os
import torch

from torch.utils.data.dataset import Dataset
from skimage import io
from skimage.transform import resize


root_path = os.environ["DMD_ROOT"]

class DMD(Dataset):
    def __init__(self, annotations,train=True):
        self.annotations = annotations
        self.data = []

        self.class_map = {}

        with open(self.annotations, "r+") as f:
            for line in f.readlines():
                meta_info = line.split(" ")
                clip_path = root_path+"/"+meta_info[0]

              
                frames = 0

                
                label = meta_info[0].split("/")[0]
                numerical_label = int(meta_info[2].strip())
                
                self.data.append((clip_path,label,frames))
                if label not in self.class_map.keys():
                    self.class_map[label] = numerical_label
        
        self.classes = list(self.class_map.keys())
        cut = (3*len(self.data))//4
        
        if train:    
            self.data = self.data[:cut]
        else:
            self.data = self.data[cut:]
        
        self.frame_size = (500,500)

    def get_shortest_clip(self):
        min_ = 100
        m_path = ""
        m_label = ""
        for c,l,frames in self.data:
            if frames < min_:
                min_ = frames
                m_path = c
                m_label = l

        return min_,m_path,m_label

    def __getitem__(self, index):
        if index >= len(self.data):
            return None


        #Get metadata about index
        path,label,frames = self.data[index]
        _, _, files = next(os.walk(path))
        label = self.class_map[label]
        
        #Create frame array
        frames_array = []
        for i in range(len(files)):
            name = "img_{:07d}.jpg".format(i)
            img = io.imread(path+"/"+name)
            img = resize(img,self.frame_size)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img)
            frames_array.append(img)

        tensor_arr = torch.stack(frames_array)
        labels = torch.tensor([label for i in range(len(tensor_arr))])
        
        return (tensor_arr,labels)

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    annotations = root_path+"/annotations.txt"
    d = DMD(annotations)
    for i in d:
        print(i)