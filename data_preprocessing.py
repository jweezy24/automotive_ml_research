import os
import cv2
from mtcnn import MTCNN


dataset_path = './dataset_dmd/clips'
aligned_dir = './dataset_dmd/aligned'
detector = MTCNN()


def process_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    new_path = path.replace("clips_backup","aligned")
    label_path = "/".join(new_path.split("/")[:-2])
    iteration_path = "/".join(new_path.split("/")[:-1])
    
    
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    
    if not os.path.exists(iteration_path):
        os.mkdir(iteration_path)
    
    if os.path.exists(new_path):
        return None
    
    result = detector.detect_faces(image)

    if len(result) > 0:
        x1, y1, w, h = result[0]['box']
        x2, y2 = x1 + w, y1 + h
        aligned = cv2.resize(image[y1:y2, x1:x2], (224, 224))
        cv2.imwrite(new_path,aligned)
    else:
        print(result)
        raise("Bounding Box Not Found")

def iterate_instances():

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for file in files:
            if ".jpg" in file:
                path = root+"/"+file
                process_image(path)


if __name__ == "__main__":
    
    if not os.path.exists(aligned_dir):
        os.mkdir(aligned_dir)

    iterate_instances()
    pass