import glob
from scipy import io
import os
import numpy as np
from pathlib import Path
import tqdm
import argparse

def mat_to_np(mat_file):   
    mat_file = io.loadmat(mat_file)
    pts = np.array(mat_file['pts'])
    
    return pts

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Face alignment and crop')
    parser.add_argument('--allLmkPontsDisfa', type=str, help='The path to the dataset')
    args = parser.parse_args()
    allLmkPontsDisfa = args.allLmkPontsDisfa + "/Landmark_Points"
    saveLocation = Path("data/datasets/original/DISFA_/Landmark_Points_np")

    allLmkPontsDisfa = list(glob.glob(allLmkPontsDisfa + "/**/**/*.mat"))
    allLmkPontsDisfa.sort()

    print("len(allLmkPontsDisfa):", len(allLmkPontsDisfa))

    for i in tqdm.tqdm(range(0, len(allLmkPontsDisfa))):
        lmk = Path(allLmkPontsDisfa[i])

        frameNumber = lmk.stem.split("_")[0]
        if "l" in frameNumber:
            frameNumber = int(frameNumber[1:]) - 1
        elif "SN" in frameNumber:
            frameNumber = int(frameNumber[2:]) - 1
        else:
            raise Exception("frameNumber not found")

        # save frameNumber to be 4 digits
        frameNumber = str(frameNumber).zfill(4)
        
        fileName = lmk.parent.parent.stem + "_" + str(frameNumber) + "_lm.npy"
        savePath = saveLocation / lmk.parent.parent.name / fileName
        out = mat_to_np(lmk)

        savePath.parent.mkdir(parents=True, exist_ok=True)
        np.save(savePath, out)
