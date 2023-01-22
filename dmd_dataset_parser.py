import glob
import cv2
import numpy as np
import torch
import cv2
import torchvision
import os

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from imutils.video import FileVideoStream
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from skimage.transform import resize




class DMD(Dataset):
    #Grabs all of the files from parsed DMD dataset generated from dataset_parser.py
    def __init__(self,amount=0,training=False):
        self.use_bounding = False
        #https://github.com/Itseez/opencv/blob/master/data/haarcascades
        self.face_cascades = []
        #https://github.com/Itseez/opencv/blob/master/data/haarcascades
        self.eye_cascades = []
        
        for root,dirs,files in os.walk("cascades"):
            for file in files:
                if "xml" in file:
                    if "face" in file:
                        self.face_cascades.append(root+"/"+file)
                    elif "eye" in file:
                        self.eye_cascades.append(root+"/"+file)

        #Grabbing path to parsed datset.
        self.path = "../dataset_dmd/clips/"
        file_list = glob.glob(self.path + "*")

        #Retrieve all possible labels
        self.classes = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1].split(";")[1]
            
            if class_name not in self.classes:
                self.classes.append(class_name)

        t_index = (3*len(file_list))//4

        
        if training:
            if amount == 0:
                #Extracting all files within the folder.
                file_list = glob.glob(self.path + "*")[0:t_index]
            else:
                #Grabbing a subset of files
                file_list = glob.glob(self.path + "*")[0:amount]
        else:
            if amount == 0:
                #Extracting all files within the folder.
                file_list = glob.glob(self.path + "*")[t_index:]
            else:
                #Grabbing a subset of files
                if t_index+amount > len(file_list):
                    raise RuntimeError("The amount of testing files exceeds amount of data available.") 
                else:
                    file_list = glob.glob(self.path + "*")[t_index:t_index+amount]
        
        #initalizing some variables
        self.data = []

        print("LOADING DATASET")
        #Adding all MP4s into a data array.
        for class_path in tqdm(file_list):
            class_name = class_path.split("/")[-1].split(";")[1]
            self.data.append([class_path, class_name])

        #Creating class map
        self.class_map = {}
        i = 0
        for c in self.classes:
            self.class_map[c] = i
            i+=1
        
        #Getting the dimension of the frames
        frame_size = (500,500)

       
        self.img_dim = frame_size    
        
    def create_bounding_box(self,frame,indf,inde):
        c=0
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(self.face_cascades[indf]).detectMultiScale(gray,1.3, 5)
        roi_color = frame
        roi_gray = gray
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = cv2.CascadeClassifier(self.eye_cascades[inde]).detectMultiScale(roi_gray)
            if len(eyes) != 2:
                for (ex,ey,ew,eh) in eyes:
                    for (ex2,ey2,ew2,eh2) in eyes:
                        if ex2 == ex and ey2 == ey:
                            continue
                        else:
                            d = euclidean( (ex,ey),(ex2,ey2) )
                            if d > 100 and d < 150:
                                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                                cv2.rectangle(roi_color,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
                                c+=2
            elif len(eyes) == 1:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                c=1
            else:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                c=2
        return c,img,roi_color,roi_gray
    
    def __len__(self):
        return len(self.data)
            
        
    def __getitem__(self, idx):
        #Gets the video path and label
        vid_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        
        #Opens a video stream for directly reading frames
        capture = FileVideoStream(vid_path)

        #Puts frames into an array
        tmp_arr = []
        capture.start()
        
        oldpf = 0
        oldpe = 0
        while capture.running():
            frame = capture.read()
            done = False
            if type(frame) != type(None):
                if self.use_bounding:
                    if oldpe > 0 or oldpf > 0:
                        c,img,rc,rg = self.create_bounding_box(frame,oldpf,oldpe)
                        if c == 2:
                            tmp_arr.append(img)
                            continue
                        
                    
                    for indf in range(len(self.face_cascades)):
                        for inde in range(len(self.eye_cascades)):
                            c,img,rc,rg = self.create_bounding_box(frame,indf,inde)
                    
                            if c == 2 or c == 1:
                                print(f"SUCCESS face cascade = {indf} eye cascade = {inde} Eye Count = {c}")
                                tmp_arr.append(img)
                                done = True
                                # if oldpe != inde or oldpf != indf:
                                #     plt.imshow(img)
                                #     plt.show()
                                oldpf = indf
                                oldpe = inde
                                break
                            
                        
                        if done:
                            break
                else:
                    tmp_arr.append(resize(frame,self.img_dim))
                        

        capture.stop()
        print(len(tmp_arr))
        #Builds tensors out of the array of frames
        out = np.array(tmp_arr)
        tensors = torch.tensor(out)
        class_id = torch.tensor([class_id for i in range(len(out))])
        
        
        return  tensors,class_id

if __name__ == "__main__":
    print(torchvision.get_video_backend())
    d = DMD(amount=10)
    print(torch.cuda.get_device_name(0))
    
    for ele in d:
        print(ele)