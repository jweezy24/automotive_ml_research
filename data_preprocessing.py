import os
import cv2
import gc
from mtcnn import MTCNN
from multiprocessing import Process, Lock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

dataset_path = os.environ["DMD_PATH"] + "/clips"
aligned_dir = os.environ["DMD_PATH"] + "/aligned" 

lock = Lock()
allowed_threads = 32


def process_image(path):
    detector = MTCNN()


    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    new_path = path.replace("clips","aligned")
    label_path = "/".join(new_path.split("/")[:-2])
    iteration_path = "/".join(new_path.split("/")[:-1])
    
    lock.acquire()
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    
    if not os.path.exists(iteration_path):
        os.mkdir(iteration_path)
    lock.release()

    if os.path.exists(new_path):
        del detector
        del image
        return None
    
    result = detector.detect_faces(image)
    
    del detector


    if len(result) > 0:
        x1, y1, w, h = result[0]['box']
        x2, y2 = x1 + w, y1 + h
        aligned = cv2.resize(image[y1:y2, x1:x2], (224, 224))

        cv2.imwrite(new_path,aligned)

        del aligned
        del result
        
    else:
        print(result)
        del result
    
    del image
    
def iterate_instances():
    
    use_threads = True
    threads = [0 for i in range(allowed_threads)]
    c = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for file in files:
            if ".jpg" in file:
                path = root+"/"+file
                if use_threads:
                    if c < allowed_threads:
                        t = Process(target=process_image, args=(path,), daemon=True)
                        gc.collect()
                        threads[c] = t
                        t.start()
                        c+=1
                    else:
                        for i in range(len(threads)):
                            t = threads[i]
                            print(f"finishing thread {i}")
                            if t != 0:
                                t.join()
                                del t
                                gc.collect()
                                
                        c = 0
                        t = Process(target=process_image, args=(path,), daemon=True)
                        gc.collect()
                        threads[c] = t
                        t.start()
                        c+=1
                else:
                    process_image(path)



if __name__ == "__main__":
    
    if not os.path.exists(aligned_dir):
        os.mkdir(aligned_dir)

    iterate_instances()
