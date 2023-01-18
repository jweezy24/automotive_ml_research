import os 
import sys

def parse_file(path):
    pass

#Connect json file to mp4 files
def create_file_pairs(path):
    file_pairs = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
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
    file_pairs = create_file_pairs("../dataset_dmd")