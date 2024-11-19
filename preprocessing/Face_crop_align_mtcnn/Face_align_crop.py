"""
This code was originally written by https://github.com/urbaneman/Face_crop_align_mtcnn
Was modified by Andrew Sumsion for use in the ELEGANT framework
"""

import glob
import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
from align_mtcnn.mtcnn_detector import MtcnnDetector
from pathlib import Path
import glob
import mat73
import numpy as np

def crop_align_face_BP4D(input_dir):
    # output_dir = "./data/datasets/processed/BP4D/"
    # lmk_data_path = Path("./data/datasets/original/BP4D/2DFeatures")
    output_dir = "/home/andreww9/groups/grp_ensembleAU/nobackup/autodelete/data/datasets/processed/BP4D/"
    lmk_data_path = None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    face_size = "256" 

    allActions = list(glob.glob(input_dir + "/**/T*"))

    # randomize:
    np.random.shuffle(allActions)
    
    for actionPath in allActions:
        
        #prepare paths, data and info:
        subjectNumber = Path(actionPath).parent.stem
        action = Path(actionPath).stem
        outputPath_img = Path(output_dir) / "img" / subjectNumber / action
        outputPath_lmks = Path(output_dir) / "lmk" / subjectNumber / action
        
        # in the scenario you have multiple jobs running to do preprocessing, you need to check if the directory exists to send the job to the next action.
        if outputPath_img.exists():
            print("outputPath_img exists")
            continue

        outputPath_img.mkdir(parents=True, exist_ok=True)
        outputPath_lmks.mkdir(parents=True, exist_ok=True)

        if lmk_data_path is not None:
            #load in the landmark data
            thisLMK_data = lmk_data_path / (subjectNumber + "_" + action + ".mat") 
            def mat_to_np(mat_file):   
                f = mat73.loadmat(mat_file)
                np_pts = f['fit']['pred']
                return np_pts
            lmk_np = mat_to_np(thisLMK_data)

        #initialize mtcnn
        ctx = mx.cpu()
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
        mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

        #initialize variables
        allFiles = list(Path(actionPath).rglob("*.jpg"))
        allFiles.sort()
        prevRet = [None]
        previousLMK = [None]
        ret = None
        
        for j in range(0, len(allFiles)):
            file = allFiles[j]

            if lmk_data_path is not None:
                #check if the landmark data is None. If it is, then use the previous landmark data.
                if type(lmk_np[j]) == type(None):
                    if j != 0:#type(previousLMK) != type(None):
                        lmk_np[j] = previousLMK[0]
                    else:
                        #this means that the first value is None. So, get the next value that is not None.
                        
                        for k in range(0, len(lmk_np)):
                            if type(lmk_np[k]) != type(None):
                                lmk_np[j] = lmk_np[k]
                theseLmks = lmk_np[j]
                previousLMK[0] = theseLmks
            
            # find the face: if a face does not exist, then use the previous face
            face_img = cv2.imread(str(file))
            ret = mtcnn.detect_face(face_img)
            faceFound = True
            if ret is None:
                print('%s do not find face'%file)
                faceFound = False
            else:
                bbox, points = ret
                if bbox.shape[0] == 0:
                    print('%s do not find face'%file)
                    faceFound = False
            if not faceFound:
                if j != 0: #prevRet is not None:
                    print("loading in previous ret", prevRet[0])
                    bbox, points = prevRet[0]
                else:
                    # for when the first frame does not have a face, use the next frame that does.
                    done = False
                    for k in range(j, len(allFiles)):
                        if done:
                            continue
                        if j != k:
                            tmp_file = allFiles[k]
                            if tmp_file.is_file():
                                if tmp_file.suffix == ".jpg":
                                    face_img = cv2.imread(str(tmp_file))
                                    ret = mtcnn.detect_face(face_img)
                                    faceFound = True
                                    if ret is None:
                                        faceFound = False
                                    else:
                                        bbox, points = ret
                                        if bbox.shape[0] == 0:
                                            faceFound = False
                                        if faceFound:
                                            prevRet[0] = ret
                                            done = True
            else:
                prevRet[0] = ret
            
            # save the face and landmarks
            for i in range(bbox.shape[0]):
                bbox_ = bbox[i, 0:4]
                points_ = points[i, :].reshape((2, 5)).T
                if lmk_data_path is not None:
                    #inside the mtcnn preprocess we wrote it to do a perspective transform on the lmks.
                    face, new_lmk_points = mtcnn.preprocess(face_img, bbox_, points_, theseLmks, image_size=face_size)
                    file_path_save = outputPath_img / (str(int(file.stem)) + ".jpg")
                    lmk_path_save = outputPath_lmks / (str(int(file.stem)) + ".npy")
                    cv2.imwrite(str(file_path_save), face)
                    np.save(str(lmk_path_save), new_lmk_points)
                else:
                    face, new_lmk_points = mtcnn.preprocess(face_img, bbox_, points_, rotateScalePoints=None, image_size=face_size)
                    file_path_save = outputPath_img / (str(int(file.stem)) + ".jpg")
                    cv2.imwrite(str(file_path_save), face)
                break

def crop_align_face_DISFA(input_dir):
    output_dir = "./data/datasets/processed/DISFA_/"
    lmk_data_path = Path("./data/datasets/original/DISFA_/Landmark_Points_np") #TODO figure out the np file.
    face_size = "256"  

    allSubjects = list(glob.glob(str(input_dir) + "/**"))
    print("len(allSubjects): ", len(allSubjects))

    for vidPath in allSubjects:
        
        #prepare paths, data and info:
        subjectNumber = Path(vidPath).stem.split("Video")[1].split("_")[0]
        thisLMK_data = lmk_data_path / subjectNumber
        imgSaveDir = Path(output_dir) / "img" / subjectNumber
        lmkSaveDir = Path(output_dir) / "lmk" / subjectNumber
        
        # in the scenario you have multiple jobs running to do preprocessing, you need to check if the directory exists
        if imgSaveDir.exists():
            print("imgSaveDir exists")
            continue

        imgSaveDir.mkdir(parents=True, exist_ok=True)
        lmkSaveDir.mkdir(parents=True, exist_ok=True)

        #initialize mtcnn
        ctx = mx.cpu()
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
        mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)       

        #initialize video 
        cam = cv2.VideoCapture(str(vidPath))

        #initialize variables
        prevRet = [None]
        previousLMK = [None]
        ret = None
        j = -1

        while True:
            j += 1
            cam_ret, face_img = cam.read()
            if not cam_ret:
                break

            theseLmks_path = thisLMK_data / (subjectNumber + "_" + str(j).zfill(4) + "_lm.npy")
            exists = True
            if not theseLmks_path.exists():
                exists = False
            theseLmks = np.load(theseLmks_path)

            # find the face:
            ret = mtcnn.detect_face(face_img)
            
            # if a face does not exist, then use the previous face
            faceFound = True
            if ret is None:
                print('%s do not find face'%face_img)
                faceFound = False
            else:
                bbox, points = ret
                if bbox.shape[0] == 0:
                    print('%s do not find face'%face_img)
                    faceFound = False
            if not faceFound:
                if j != 0:
                    print("loading in previous ret", prevRet[0])
                    bbox, points = prevRet[0]
                else:
                    raise Exception("Face not found in first frame. This should not happen.")
            else:
                prevRet[0] = ret

            # save the face and landmarks
            for i in range(bbox.shape[0]):
                bbox_ = bbox[i, 0:4]
                points_ = points[i, :].reshape((2, 5)).T

                #inside the mtcnn preprocess we wrote it to do a perspective transform on the lmks.
                face, new_lmk_points = mtcnn.preprocess(face_img, bbox_, points_, theseLmks, image_size=face_size)
                cv2.imwrite(str(imgSaveDir / (str(int(j)) + ".png")), face)
                np.save(str(lmkSaveDir / (str(int(j)) + ".npy")), new_lmk_points)
                break
    print("Done with DISFA")

if __name__ == '__main__':
    # bring in terminal arguments
    parser = argparse.ArgumentParser(description='Face alignment and crop')
    parser.add_argument('--randomSeed', type=int, default=0, help='The random seed--set to different values for different jobs so each will start on different frames.')
    parser.add_argument('--dataset', type=str, help='The dataset to use: BP4D or DISFA')
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset')
    args = parser.parse_args()

    np.random.seed(args.randomSeed)
    dataset = args.dataset
    dataset_path = args.dataset_path

    if dataset == "BP4D":
        crop_align_face_BP4D(dataset_path)
    elif dataset == "DISFA":
        crop_align_face_DISFA(dataset_path)
    else:
        raise Exception("Dataset not recognized. Please use BP4D or DISFA.")
