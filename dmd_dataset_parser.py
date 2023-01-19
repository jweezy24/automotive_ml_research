import glob
import cv2
import numpy as np
import torch
import cv2
import torchvision

from torch.utils.data import Dataset, DataLoader
from imutils.video import FileVideoStream
from tqdm import tqdm

class DMD(Dataset):
    #Grabs all of the files from parsed DMD dataset generated from dataset_parser.py
    def __init__(self):
        #Grabbing path to parsed datset.
        self.path = "../dataset_dmd/clips/"
        
        #Extracting all files within the folder.
        file_list = glob.glob(self.path + "*")
        
        #initalizing some variables
        self.data = []
        self.classes = []

        print("LOADING DATASET")
        #Adding all MP4s into a data array.
        for class_path in tqdm(file_list):
            class_name = class_path.split("/")[-1].split(";")[1]
            
            if class_name not in self.classes:
                self.classes.append(class_name)

            for vid_path in glob.glob(self.path+"*.mp4"):
                self.data.append([vid_path, class_name])

        #Creating class map
        self.class_map = {}
        i = 0
        for c in self.classes:
            self.class_map[c] = i
            i+=1
        
        #Getting the dimension of the frames
        capture2 = cv2.VideoCapture(self.data[0][0])

        frame_width = int(capture2.get(3))
        frame_height = int(capture2.get(4))
        frame_size = (frame_width,frame_height)

        capture2.release()

        self.img_dim = frame_size    
        
    def __len__(self):
        return len(self.data)    
        
    def __getitem__(self, idx):
        vid_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]

        capture = FileVideoStream(vid_path)

        tmp_arr = []
        capture.start()
        while capture.running():
            frame = capture.read()
            if type(frame) != type(None):
                tmp_arr.append(frame)
        capture.stop()

        out = np.array(tmp_arr)
        tensors = torch.tensor(out)
        class_id = torch.tensor([class_id])
        
        if len(tensors) == 0:
            return None,class_id
        else:
            return  tensors,class_id

if __name__ == "__main__":
    print(torchvision.get_video_backend())
    d = DMD()
    print(torch.cuda.get_device_name(0))
    
    print(d[-1])