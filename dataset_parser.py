import os
import sys
import json
import cv2
import gc

from imutils.video import FileVideoStream
from multiprocessing import Pool, Lock

CLIPS_DIRECTORY="../dataset_dmd/clips"
ALLOWED_THREADS = 30
lock = Lock()

def parse_json_file(path):
    l = open(path)
    j = json.load(l)
    valid_mdata = False
    valid_labels = False
    mdata = {}
    #Boolean array to verify the clip is a valid clip
    for k in j.keys():
        age = 0
        gender = ''
        exp = ''
        d_freq = ''

        events = []
        frame_data = []
        c = 0

        print(j[k].keys())
        metadata_str = 'objects'

        #Grabs meta
        if metadata_str in j[k].keys():
            m_data = j[k][metadata_str]
            for d in m_data.keys():
                if 'type' in m_data[d].keys():
                    
                    if m_data[d]['type'] == 'driver':
                        if 'object_data' in m_data[d].keys():
                            driver_info = m_data[d]['object_data']
                            for category in driver_info['num']:
                                if category['name'] == 'age':
                                    age = category['val']  
                                    c+=1
                            for attribute in m_data[d]['object_data']['text']:
                                if attribute['name'] == "gender":
                                    gender = attribute['val']
                                    c+=1
                                elif attribute['name'] == 'experience':
                                    exp = attribute['val']
                                    c+=1
                                elif attribute['name'] == 'drive_freq':
                                    d_freq = attribute['val']
                                    c+=1
                                else:
                                    print(f"DIFFERENT ATTRIBUTE THAN ACCOUNTED FOR {attribute}")
        else:
            print("NO METADATA TO EXTRACT")
        

        framedata_str = 'actions'

        if framedata_str in j[k].keys():
            actions = j[k][framedata_str]
            for t in actions.keys():
                event = actions[t] 
                label = ''
                if 'type' in event.keys():
                    label = event['type']
                    if event['type'] not in events:
                        events.append(label)
                    
                    if "frame_intervals" in event.keys():
                        entries = event["frame_intervals"]
                        for entry in entries:
                            st = entry["frame_start"]
                            en = entry["frame_end"]
                            if label == '':
                                print("bad")
                                break
                            else:
                                valid_labels = True
                                frame_data.append( (st,en,label) )
        #Valid clip
        if c == 4:
            print("CLIP VALID")
            mdata = {'age':age, 'gender':gender, 'exp':exp, 'd_freq':d_freq}
            valid_mdata = True
    
    if valid_mdata and valid_labels:
        return mdata,frame_data,events
    else:
        return None

def return_start_frame_number(entry):
    return entry[0]

def get_clip_object(match_str,l,st,frame_size):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fname = CLIPS_DIRECTORY+"/"+match_str+";"+l.replace("/","-")+f";{st}.mp4"
    if os.path.exists(fname):
        return None
    else:
        lock.acquire()
        w = cv2.VideoWriter(CLIPS_DIRECTORY+"/"+match_str+";"+l.replace("/","-")+f";{st}.mp4", fourcc, 20, frame_size)
        lock.release()
        return w

def write_clip(path,st,en,l,match_str,frame_size):
   
    fname = CLIPS_DIRECTORY+"/"+match_str+";"+l.replace("/","-")+f";{st}.mp4"
    lock.acquire()
    capture = FileVideoStream(path).start()
    lock.release()
    c = 0



    clip = get_clip_object(match_str,l,st,frame_size)
    if clip == None:
        capture.stop()
        print(fname)
        return None

    while True:
        if c == st:

            for i in range(0,en-st):
                frame = capture.read()
                good = type(frame) != type(None)

                if good:
                    print(st,en,l,i)
                    clip.write(frame)
                else:
                    clip.release()
                    capture.stop()
                    print(st,en,l,i)
                    print("STREAM BAD")
                    gc.collect()
                    return None
                c+=1
            break
        elif c < st:
            frame = capture.read()
            good = type(frame) != type(None)
            if not good:
                clip.release()
                capture.stop()
                print(st,en,l,)
                print("STREAM BAD")
                gc.collect()
                return None
            c+=1
    
    lock.acquire()
    clip.release()
    capture.stop()
    lock.release()
    gc.collect()
    return None

def clip_segmenter(path,match_str,segments):
    
   
    pool = Pool(ALLOWED_THREADS)

    segments = sorted(segments,key=return_start_frame_number)
    threads = [0 for i in range(ALLOWED_THREADS)]
    
    capture2 = cv2.VideoCapture(path)

    frame_width = int(capture2.get(3))
    frame_height = int(capture2.get(4))
    frame_size = (frame_width,frame_height)

    capture2.release()


    c = 0
    for st,en,l in segments:
        # write_clip(path,st,en,l,match_str)
        if c < ALLOWED_THREADS:
            p = pool.apply_async(write_clip, (path,st,en,l,match_str,frame_size,))
            threads[c] = (p,st,en,l)
            gc.collect()
            c+=1
        else:
            for i in range(ALLOWED_THREADS):
                t,st1,en1,l1 = threads[i]
                print(t)
                if t != 0:
                    t.get()
                    gc.collect()
                    
                        
                threads[i] = 0
            c = 0
    a = 0
    for i in range(ALLOWED_THREADS):
        if threads[i] != 0:
            t,st1,en1,l1 = threads[i]
            if t != 0:
                t.get()
                gc.collect()
                
        a+=1
    

def iterate_data(pairs):
    total_labels = []
    features = {} 
    for key in pairs.keys():
        json_path,vid_path = pairs[key]

        check = parse_json_file(json_path)
        
        if check != None:
            metadata,segs,labels_set = check
            clip_segmenter(vid_path,key,segs)
            
            for label in labels_set:
                if label in total_labels:
                    continue
                else:
                    total_labels.append(label)
            print(total_labels)

            features[key] = metadata



#Connect json file to mp4 files
def create_file_pairs(path):
    file_pairs = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            if "clips" in root:
                continue

            if ".mp4" in f:
                match_string = f.split(";")[0]
                if match_string not in file_pairs.keys():
                    file_pairs[match_string] = [0,root+"/"+f]
                else:
                    tmp = file_pairs[match_string]
                    tmp[1] = root+"/"+f
                    file_pairs[match_string]= tmp
            if ".json" in f:
                match_string = f.split(";")[0]
                if match_string not in file_pairs.keys():
                    file_pairs[match_string] = [root+"/"+f,0]
                else:
                    tmp = file_pairs[match_string]
                    tmp[0] = root+"/"+f
                    file_pairs[match_string]= tmp
    return file_pairs

if __name__ == "__main__":

    if not os.path.exists(CLIPS_DIRECTORY):
        os.mkdir(CLIPS_DIRECTORY)

    file_pairs = create_file_pairs("../dataset_dmd")
    iterate_data(file_pairs)