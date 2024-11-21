"""
Code originally from https://github.com/CVI-SZU/ME-GraphAU file: model/dataset.py
Modified and adapted to work in the ELEGANT framework by Andrew Sumsion
"""
import glob
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import tarfile
from tqdm import tqdm

def extract_tar_file(tar_file_path, extract_path='.', isMakeHuman=False):
    """
    Extracts a .tar file to a specified directory.
    
    :param tar_file_path: Path to the .tar file.
    :param extract_path: Path to the directory where files will be extracted.
                         Defaults to the current directory.
    """
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted {tar_file_path} to {extract_path}")
    newDir = extract_path / Path(tar_file_path).stem
    if not Path(newDir).exists() and not isMakeHuman:
        # some of the tar files unzip wrong. So, just put them into where they should be.
        newDir.mkdir()
        extract_path_fold1 = extract_path / "fold1"
        extract_path_fold2 = extract_path / "fold2"
        extract_path_fold3 = extract_path / "fold3"
        # move folds to new directory
        os.rename(extract_path_fold1, newDir / "fold1")
        os.rename(extract_path_fold2, newDir / "fold2")
        os.rename(extract_path_fold3, newDir / "fold3")


def make_dataset(image_list, label_list, au_relation=None, landmark_list=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:], landmark_list[i].strip()) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)



class FEC(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, conf=None):

        self._root_path = root_path
        self._train = train
        if train:
            self.img_folder_path = "/root_path//grp_AU_storage/code/FEC_dataset_downloader/train_images"
            # self.img_folder_path = "/tmp/" + str(conf.jobID) + "/train_images"
        else:
            self.img_folder_path = "/root_path//grp_AU_storage/code/FEC_dataset_downloader/test_images"
            # self.img_folder_path = "/tmp/" + str(conf.jobID) + "/test_images"
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        
        if self._train:
            self.allInfo_images = np.load(root_path + "/list/FEC_train_images.npy")
            self.allInfo_annotations = np.load(root_path + "/list/FEC_train_annotations.npy")
        else:
            self.allInfo_images = np.load(root_path + "/list/FEC_test_images.npy")
            self.allInfo_annotations = np.load(root_path + "/list/FEC_test_annotations.npy")
        

    def __getitem__(self, index):
        theImages = self.allInfo_images[index]
        theAnnotations = self.allInfo_annotations[index]
        theImages_out = []
        for i in range(0, 3):
            img = self.loader(os.path.join(self.img_folder_path, theImages[i]))
            # img = self.loader(os.path.join(self.img_folder_path, "0.jpeg"))
            w, h = img.size
            if h > self.crop_size:
                offset_y = random.randint(0, h - self.crop_size)
            else:
                offset_y = 0
            if w > self.crop_size:
                offset_x = random.randint(0, w - self.crop_size)
            else:
                offset_x = 0
            flip = random.randint(0, 1)
            if self._transform is not None:
                if self._train:
                    img = self._transform(img, flip, offset_x, offset_y)
                else:
                    img = self._transform(img)
            theImages_out.append(img)


        return theImages_out[0], theImages_out[1], theImages_out[2], theAnnotations

    def __len__(self):
        return len(self.allInfo_images)

class BP4D(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, trainOnlyOnOne=True, returnPath=False):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        BP4D_Sequence_split = [['F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'],
					   ['M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005'],
					   ['F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014']]
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        # self.lmk_folder_path = os.path.join(root_path,'Landmark_Points_np')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        self.returnPath = returnPath
        if self._train:
            # img
            if trainOnlyOnOne:
                train_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) +'.txt')
            else:
                train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()

            # img labels
            if trainOnlyOnOne:
                train_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            else:
                train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # img landmarks
            if trainOnlyOnOne:
                train_landmark_list_path = os.path.join(root_path, 'list', 'BP4D_test_lmk_path_fold' + str(fold) + '.txt')
            else:
                train_landmark_list_path = os.path.join(root_path, 'list', 'BP4D_train_lmk_path_fold' + str(fold) + '.txt')
            
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            if self._stage == 2:
                if trainOnlyOnOne:
                    au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_test_AU_relation_fold' + str(fold) + '.txt')
                else:
                    au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
            image_list = train_image_list
        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

            image_list = test_image_list

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation, landmark_path = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            # if landmark path exists, load it
            if os.path.exists(os.path.join(self.lmk_folder_path, landmark_path)):
                landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))
            else:
                landmark = np.array([])
            
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation, landmark
        else:
            img_name, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img_name))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            if self.returnPath:
                return img, label, os.path.join(self.img_folder_path, img_name)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, trainOnlyOnOne=True, returnPath=False):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        self.returnPath = returnPath
        if self._train:
            # img
            if trainOnlyOnOne:
                train_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            else:
                train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            if trainOnlyOnOne:
                train_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            else:
                train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # landmarks
            if trainOnlyOnOne:
                train_landmark_list_path = os.path.join(root_path, 'list', 'DISFA_test_landmark_path_fold' + str(fold) + '.txt')
            else:
                train_landmark_list_path = os.path.join(root_path, 'list', 'DISFA_train_landmark_path_fold' + str(fold) + '.txt')
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            if self._stage == 2:
                if trainOnlyOnOne:
                    au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_test_AU_relation_fold' + str(fold) + '.txt')
                else:
                    au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index, returnPath=False):
        if self._stage == 2 and self._train:
            img, label, au_relation, landmark_path = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            # if landmark path exists, load it
            if os.path.exists(os.path.join(self.lmk_folder_path, landmark_path)):
                landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))
            else:
                landmark = np.array([])

            w, h = img.size
            if h > self.crop_size:
                offset_y = random.randint(0, h - self.crop_size)
            else:
                offset_y = 0
            if w > self.crop_size:
                offset_x = random.randint(0, w - self.crop_size)
            else:
                offset_x = 0
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation, landmark
        else:
            img_path, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path,img_path))

            if self._train:
                w, h = img.size
                if h > self.crop_size:
                    offset_y = random.randint(0, h - self.crop_size)
                else:
                    offset_y = 0
                if w > self.crop_size:
                    offset_x = random.randint(0, w - self.crop_size)
                else:
                    offset_x = 0
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            
            if returnPath or self.returnPath:
                return img, label, os.path.join(self.img_folder_path,img_path)
            else:
                return img, label

    def __len__(self):
        return len(self.data_list)


class ensemble_dataset(Dataset):
    def __init__(self, dataset_path, fold, train, train_dataPath=None, allResults=None, jobID=None, returnPath=False, dataset = None, root_path=None):
        super(ensemble_dataset, self).__init__()

        self.dataset_path = Path(dataset_path)
        self.train = train
        self.fold = fold
        self.root_path = root_path
        self.returnPath = returnPath
        self.parents = np.load(self.dataset_path / (allResults + str(fold) + ".npy"))

        if train:
            if Path("/tmp/").exists():
                tmpPath = "/tmp/" + str(jobID) + "/"
            else:
                tmpPath = "tmp_extracting/" + str(jobID) + "/"
            Path(tmpPath).mkdir(exist_ok=True, parents=True)
            print("Extracting:")
            for parent in tqdm(self.parents):
                fromPath = self.dataset_path / (parent + ".tar")
                if not fromPath.exists():
                    print("parent does not exist: ", fromPath)
                    raise Exception("parent does not exist. Please download the correct predictions.")
                else:
                    toPath = Path(tmpPath)
                    if Path(toPath / parent).exists() or Path(toPath / (parent + "_finalLayer")).exists():
                        print("Skipping extraction because it already exists")
                    else:
                        extract_tar_file(fromPath, toPath)
                        print("extracted: ", fromPath, toPath)
                        print("exists: ", toPath.exists())                    
            self.dataset_path = Path(tmpPath)
        else:
            self.dataset_path = train_dataPath

        if dataset == "BP4D":
            if train:
                # img
                train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) + '.txt')
                self.image_list = open(train_image_list_path).readlines()
                self.image_list = [x.replace(".jpg", ".npy").replace(".png", ".npy").replace("\n", "") for x in self.image_list]
                # img labels
                label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
                self.label_list = np.loadtxt(label_list_path)

                if fold == 1:
                    self.modifiedDataset_correspondingFolds = [2,3]
                elif fold == 2:
                    self.modifiedDataset_correspondingFolds = [1,3]
                elif fold == 3:
                    self.modifiedDataset_correspondingFolds = [1,2]
                else:
                    raise Exception("fold must be 1, 2, or 3")
            else:
                # img
                test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
                self.image_list = open(test_image_list_path).readlines()
                self.image_list = [x.replace(".jpg", ".npy").replace(".png", ".npy").replace("\n", "") for x in self.image_list]
                # img labels
                label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
                self.label_list = np.loadtxt(label_list_path)

                if fold == 1:
                    self.modifiedDataset_correspondingFolds = [1]
                elif fold == 2:
                    self.modifiedDataset_correspondingFolds = [2]
                elif fold == 3:
                    self.modifiedDataset_correspondingFolds = [3]
                else:
                    raise Exception("fold must be 1, 2, or 3")
        elif dataset == "DISFA":
            print("DISFA dataset")
            if train:
                # img
                train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
                self.image_list = open(train_image_list_path).readlines()
                self.image_list = ["img/" + x.replace(".jpg", ".npy").replace(".png", ".npy").replace("\n", "") for x in self.image_list]
                # img labels
                label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
                self.label_list = np.loadtxt(label_list_path)

                if fold == 1:
                    self.modifiedDataset_correspondingFolds = [2,3]
                elif fold == 2:
                    self.modifiedDataset_correspondingFolds = [1,3]
                elif fold == 3:
                    self.modifiedDataset_correspondingFolds = [1,2]
                else:
                    raise Exception("fold must be 1, 2, or 3")
            else:
                # img
                test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
                self.image_list = open(test_image_list_path).readlines()
                self.image_list = ["img/" + x.replace(".jpg", ".npy").replace(".png", ".npy").replace("\n", "") for x in self.image_list]
                # img labels
                label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
                self.label_list = np.loadtxt(label_list_path)

                if fold == 1:
                    self.modifiedDataset_correspondingFolds = [1]
                elif fold == 2:
                    self.modifiedDataset_correspondingFolds = [2]
                elif fold == 3:
                    self.modifiedDataset_correspondingFolds = [3]
                else:
                    raise Exception("fold must be 1, 2, or 3")

        self.__getitem__(0)
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        chosenIndex = 0
        allImages = []
        for i in range(0, len(self.parents)):
            if i == 0:
                for j in range(0, len(self.modifiedDataset_correspondingFolds)):
                    exampleImage = self.dataset_path / (self.parents[i]) / "fold{}".format(self.modifiedDataset_correspondingFolds[j]) / self.image_list[index]
                    if exampleImage.exists():
                        chosenIndex = j
                        break
            else:
                exampleImage = self.dataset_path / (self.parents[i]) / "fold{}".format(self.modifiedDataset_correspondingFolds[chosenIndex]) / self.image_list[index]
            exampleImage = np.load(exampleImage)
            if len(allImages) == 0:
                allImages = exampleImage
            else:
                allImages = np.concatenate((allImages, exampleImage), axis=1)
        #squeeze allImages
        allImages = np.squeeze(allImages)

        label = self.label_list[index]

        if self.returnPath:
            return torch.from_numpy(allImages).float(), torch.from_numpy(label).float(), self.image_list[index]

        return torch.from_numpy(allImages).float(), torch.from_numpy(label).float()

class BP4D_makeHuman(Dataset):
    def __init__(self, root_path, makeHumanPath=None, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, trainOnlyOnOne=True, returnPath=False, jobID=0):
        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        BP4D_Sequence_split = [['F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'],
					   ['M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005'],
					   ['F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014']]
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        # self.lmk_folder_path = os.path.join(root_path,'Landmark_Points_np')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        self.returnPath = returnPath
        if Path("/tmp/").exists():
            if train:
                if jobID == 0:
                    while True:
                        try:
                            cntr = 0
                            tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(cntr)
                            while Path(tmpPath).exists():
                                cntr += 1
                                tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(cntr)
                            Path(tmpPath).mkdir(exist_ok=False)
                            break
                        except:
                            print("Attempt to create same directory as other process")
                    print("Extracting:")
                else:
                    for i in range(0, 1000):
                        deletePath = "/tmp/MAKE_HUMAN_dir_" + str(i)
                        if Path(deletePath).exists():
                            os.system("rm -r " + deletePath)

                    #remove all directories with jobID less than 65974600
                    allDirs = list(glob.glob("/tmp/MAKE_HUMAN_dir_*"))
                    # for dir in allDirs:
                    #     if int(dir.split("_")[-1]) < 33987:
                    #         os.system("rm -r " + dir)
                            
                    tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(jobID)
                    Path(tmpPath).mkdir(exist_ok=True)

                
                allParents = list(glob.glob(str(makeHumanPath / "**.tar")))
                if len(allParents) != 200:
                    print("globbed: ", str(makeHumanPath / "**.tar"))
                    print("length of allParents: ", len(allParents))
                    raise Exception("not all tar files are present")
                for parent in tqdm(allParents):
                    fromPath = Path(parent)
                    if not fromPath.exists():
                        print("parent does not exist: ", fromPath)
                        raise Exception("parent does not exist")
                    else:
                        toPath = Path(tmpPath) / fromPath.stem
                        extract_tar_file(fromPath, toPath, isMakeHuman=True)
                        print("extracted: ", fromPath, toPath)
                        print("exists: ", toPath.exists())
                self.makeHumanPath = str(tmpPath)

        if self._train:
            # img
            if trainOnlyOnOne:
                train_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) +'.txt')
            else:
                train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()

            # img labels
            if trainOnlyOnOne:
                train_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            else:
                train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # img landmarks
            if trainOnlyOnOne:
                train_landmark_list_path = os.path.join(root_path, 'list', 'BP4D_test_lmk_path_fold' + str(fold) + '.txt')
            else:
                train_landmark_list_path = os.path.join(root_path, 'list', 'BP4D_train_lmk_path_fold' + str(fold) + '.txt')
            
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            if self._stage == 2:
                if trainOnlyOnOne:
                    au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_test_AU_relation_fold' + str(fold) + '.txt')
                else:
                    au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
            image_list = train_image_list

            allMakeHumanImages = list(glob.glob(self.makeHumanPath + "/**/**/**"))
            allMakeHumanImages.sort()
            data_list_makeHuman = []
            BP4D_use_AUs = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23", "24"]
            cntr = 0
            print("length of available makeHuman images: ", len(allMakeHumanImages))
            for item_og in allMakeHumanImages:
                item = Path(item_og).parent.name
                splitItem = item.split("_")

                if len(splitItem) == 3:
                    secondAU = splitItem[2]
                    firstAU = splitItem[1]
                elif len(splitItem[1]) == 0:
                    firstAU = None
                    secondAU = None
                else:
                    firstAU = splitItem[1]
                    secondAU = None

                if firstAU is None and secondAU is None:
                    thisLabel = np.zeros(12)
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
                elif firstAU in BP4D_use_AUs and secondAU is None:
                    thisLabel = np.zeros(12)
                    thisLabel[BP4D_use_AUs.index(firstAU)] = 1
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
                elif firstAU in BP4D_use_AUs and secondAU in BP4D_use_AUs:
                    thisLabel = np.zeros(12)
                    thisLabel[BP4D_use_AUs.index(firstAU)] = 1
                    thisLabel[BP4D_use_AUs.index(secondAU)] = 1
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
            print("length of data_list_makeHuman: ", len(data_list_makeHuman))
            self.data_list = self.data_list + data_list_makeHuman
        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

            image_list = test_image_list

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation, landmark_path = self.data_list[index]
            if au_relation is not None:
                img = self.loader(os.path.join(self.img_folder_path, img))
                # landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))
            else:
                img = self.loader(img)
                # landmark = torch.tensor([0.0])
                # au_relation = torch.tensor([0.0])
            
            
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            if self.makeHumanPath is None:
                return img, label, au_relation, landmark
            else:
                return img, label
        else:
            img_name, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img_name))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            if self.returnPath:
                return img, label, os.path.join(self.img_folder_path, img_name)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA_makeHuman(Dataset):
    def __init__(self, root_path, makeHumanPath=None, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, trainOnlyOnOne=True, returnPath=False, jobID=0):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        
        if Path("/tmp/").exists():
            if train:
                if jobID == 0:
                    while True:
                        try:
                            cntr = 0
                            tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(cntr)
                            while Path(tmpPath).exists():
                                cntr += 1
                                tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(cntr)
                            Path(tmpPath).mkdir(exist_ok=False)
                            break
                        except:
                            print("Attempt to create same directory as other process")
                    print("Extracting:")
                else:
                    for i in range(0, 1000):
                        deletePath = "/tmp/MAKE_HUMAN_dir_" + str(i)
                        if Path(deletePath).exists():
                            os.system("rm -r " + deletePath)

                    #remove all directories with jobID less than 65974600
                    allDirs = list(glob.glob("/tmp/MAKE_HUMAN_dir_*"))
                    # for dir in allDirs:
                    #     if int(dir.split("_")[-1]) < 33987:
                    #         os.system("rm -r " + dir)
                            
                    tmpPath = "/tmp/MAKE_HUMAN_dir_" + str(jobID)
                    Path(tmpPath).mkdir(exist_ok=True)

                
                allParents = list(glob.glob(str(makeHumanPath / "**.tar")))
                if len(allParents) != 200:
                    print("globbed: ", str(makeHumanPath / "**.tar"))
                    print("length of allParents: ", len(allParents))
                    raise Exception("not all tar files are present")
                for parent in tqdm(allParents):
                    fromPath = Path(parent)
                    if not fromPath.exists():
                        print("parent does not exist: ", fromPath)
                        raise Exception("parent does not exist")
                    else:
                        toPath = Path(tmpPath) / fromPath.stem
                        extract_tar_file(fromPath, toPath, isMakeHuman=True)
                        print("extracted: ", fromPath, toPath)
                        print("exists: ", toPath.exists())
                self.makeHumanPath = str(tmpPath)
        
        if self._train:
            # img
            if trainOnlyOnOne:
                train_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            else:
                train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            if trainOnlyOnOne:
                train_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            else:
                train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # landmarks
            if trainOnlyOnOne:
                train_landmark_list_path = os.path.join(root_path, 'list', 'DISFA_test_landmark_path_fold' + str(fold) + '.txt')
            else:
                train_landmark_list_path = os.path.join(root_path, 'list', 'DISFA_train_landmark_path_fold' + str(fold) + '.txt')
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            if self._stage == 2:
                if trainOnlyOnOne:
                    au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_test_AU_relation_fold' + str(fold) + '.txt')
                else:
                    au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
            allMakeHumanImages = list(glob.glob(self.makeHumanPath + "/**/**/**"))
            allMakeHumanImages.sort()
            data_list_makeHuman = []
            # BP4D_use_AUs = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23", "24"]
            DISAF_use_AUs = ["1", "2", "4", "6", "9", "12", "25", "26"]
            cntr = 0
            print("length of available makeHuman images: ", len(allMakeHumanImages))
            for item_og in allMakeHumanImages:
                item = Path(item_og).parent.name
                splitItem = item.split("_")

                if len(splitItem) == 3:
                    secondAU = splitItem[2]
                    firstAU = splitItem[1]
                elif len(splitItem[1]) == 0:
                    firstAU = None
                    secondAU = None
                else:
                    firstAU = splitItem[1]
                    secondAU = None

                if firstAU is None and secondAU is None:
                    thisLabel = np.zeros(8)
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
                elif firstAU in DISAF_use_AUs and secondAU is None:
                    thisLabel = np.zeros(8)
                    thisLabel[DISAF_use_AUs.index(firstAU)] = 1
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
                elif firstAU in DISAF_use_AUs and secondAU in DISAF_use_AUs:
                    thisLabel = np.zeros(8)
                    thisLabel[DISAF_use_AUs.index(firstAU)] = 1
                    thisLabel[DISAF_use_AUs.index(secondAU)] = 1
                    data_list_makeHuman.append([item_og, thisLabel, None, None])
            print("length of data_list_makeHuman: ", len(data_list_makeHuman))
            self.data_list = self.data_list + data_list_makeHuman

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index, returnPath=False):
        if self._stage == 2 and self._train:
            img, label, au_relation, landmark_path = self.data_list[index]
            if au_relation is not None:
                img = self.loader(os.path.join(self.img_folder_path, img))
            else:
                img = self.loader(img)

            # if landmark path exists, load it
            if landmark_path is not None and os.path.exists(os.path.join(self.lmk_folder_path, landmark_path)):
                landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))
            else:
                landmark = np.array([])
            
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            if self.makeHumanPath is None:
                return img, label, au_relation, landmark
            else:
                return img, label
        else:
            img_path, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path,img_path))

            if self._train:
                w, h = img.size
                if h > self.crop_size:
                    offset_y = random.randint(0, h - self.crop_size)
                else:
                    offset_y = 0
                if w > self.crop_size:
                    offset_x = random.randint(0, w - self.crop_size)
                else:
                    offset_x = 0
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            
            if returnPath:
                return img, label, os.path.join(self.img_folder_path,img_path)
            else:
                return img, label

    def __len__(self):
        return len(self.data_list)
