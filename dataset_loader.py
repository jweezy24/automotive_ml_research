import os
import torch
import numpy as np
import sqlite3
import cv2

from multiprocessing import Pool
from torch.utils.data.dataset import Dataset
from skimage import io
from skimage.transform import resize

db = sqlite3.connect('/home/jweezy/Drive2/Drive2/Datasets/dmd.db',check_same_thread=False)
cur = db.cursor()

root_path = os.environ["DMD_ROOT"]

ALLOWED_THREADS = 30
def convert_array(text):
    import io
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_converter("ARRAY", convert_array)

def prepare_arr(img,frame):
    img = convert_array(img)
    img = resize(img,(500,500))
    img = torch.from_numpy(img)
    return (img,frame)

class DMD(Dataset):
    def __init__(self, annotations,train=True,use_files=True):
        self.annotations = annotations
        self.data = []

        self.class_map = {}
        self.use_files = use_files

       
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

    def get_length(self):
        return len(self.data)

    def get_item_label(self,index):
        return self.data[index][1]

    def __getitem__(self, index):
        if index >= len(self.data):
            return None

        if self.use_files:
            #Get metadata about index
            path,label,frames = self.data[index]
            _, _, files = next(os.walk(path))
            label = self.class_map[label]
            
            #Create frame array
            frames_array = []
            for i in range(len(files)):
                name = "img_{:07d}.jpg".format(i)
                img = cv2.imread(path+"/"+name)
                img = resize(img,self.frame_size)
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img)
                frames_array.append(img)

            tensor_arr = torch.stack(frames_array)
            labels = torch.tensor([label for i in range(len(tensor_arr))])
            return (tensor_arr,labels)

        else:
            pool = Pool(ALLOWED_THREADS)
            threads = [0 for i in range(ALLOWED_THREADS)]
            path,label,frames = self.data[index]
            print(path)
            vid_num = path.split("/")[-1]
            res = cur.execute(f"SELECT frame, bytes FROM Data WHERE label= '{label}' AND vid_num= {vid_num};")
            print("DONE FETCHING")

            frames_arr = [0 for i in range(len(data))]
            print("LOADING DATA")
            c = 0
            data = res.fetchone()
            if data != None and len(data) > 0:
                while data != None:
                    frame,bts = i
                    t = pool.apply_async(prepare_arr,(bts,frame,))
                    threads[c] = t
                    c+=1

                    if c >= ALLOWED_THREADS:
                        for t in threads:
                            if t != 0:
                                res = t.get()
                                print(t)
                                if res != None:
                                    img,frame = res
                                    frames_arr[frame] = img
                        c=0
                    data = res.fetchone()
                    # img = convert_array(bts)
                    # img = resize(img,self.frame_size)
                    # print(frame)
                    # img = torch.from_numpy(img)
                    # frames.append(img)
                    # frames.append((frame,bts))
                
                for t in threads:
                    if t != 0:
                        res = t.get()
                        print(t)
                        if res != None:
                            img,frame = res
                            frames_arr[frame] = img
            else:
                return None
            
            frames_arr = [i for i in frames_arr if type(i) != type(0)]
            tensor_arr = torch.stack(frames_arr)
            label = self.class_map[label]
            labels = torch.tensor([label for i in range(len(frames_arr))])
            print("DONE")
            return (tensor_arr,labels)

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    import time
    annotations = root_path+"/annotations.txt"
    d = DMD(annotations)
    for i in d:
        print(i)
        time.sleep(0.5)