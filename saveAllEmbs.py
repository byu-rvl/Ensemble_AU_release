from datetime import datetime
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import tarfile
import sys
import random
import glob


from model.encoder_gcn import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    if conf.dataset == 'BP4D':
        valset1 = BP4D(conf.dataset_path, train=False, fold=1, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)
        valset2 = BP4D(conf.dataset_path, train=False, fold=2, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)
        valset3 = BP4D(conf.dataset_path, train=False, fold=3, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)


    elif conf.dataset == 'DISFA':
        valset1 = DISFA(conf.dataset_path, train=False, fold=1, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)
        valset2 = DISFA(conf.dataset_path, train=False, fold=2, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)
        valset3 = DISFA(conf.dataset_path, train=False, fold=3, transform=image_test(crop_size=conf.crop_size), stage = 2, returnPath=True)

    return valset1, valset2, valset3

def saveEmbs(valset, net, conf, fold):
    net.eval()

    dir_name = Path(conf.resume).stem

    saveHere = "/tmp/" + dir_name + "/fold" + str(fold) + "/"
    finalLocation = "data/predictions/" + conf.dataset + "_computeLocal/" + "/" + dir_name
    Path(finalLocation).parent.mkdir(parents=True, exist_ok=True)
        
    
    tar_path = str(Path(saveHere).parent / f"{Path(saveHere).name}.tar")
    final_tar_path = str(Path(finalLocation).parent / f"{Path(finalLocation).name}.tar")
    if Path(final_tar_path).exists():
        print(f"tar file already exists: {tar_path}")
        return
    Path(saveHere).mkdir(parents=True)

    for img, label, path in tqdm(valset):
        with torch.no_grad():
            img = img.unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()
            outputs, _, _, _ = net(img)
            path = Path(path)
            person = path.parent.parent.stem
            movement = path.parent.stem
            frame = path.stem
            savePath = saveHere + "/" + person + "/" + movement + "/" + frame + ".npy"
            Path(savePath).parent.mkdir(parents=True, exist_ok=True)
            np.save(savePath, outputs.cpu().detach().numpy())

    return saveHere, final_tar_path
    

def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
        numberLmks=49
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
        numberLmks=66
    elif conf.dataset == "FEC":
        dataset_info = FEC_infolist
        # numberLmks=49
        numberLmks=66
    
    valset1, valset2, valset3 = get_dataloader(conf)
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, numEncoderLayers=conf.numEncoderLayers, numLandmarks=numberLmks)

    # resume
    if conf.resume != '':
        logging.info("Resume from | {} ]".format(conf.resume))
        weight_paths = "data/weights/baseLearners/" + conf.dataset + "/" + conf.resume
        net = load_state_dict(net, weight_paths)
    else:
        raise Exception("No resume file found")

    if torch.cuda.is_available():
        num_gpus = int(os.environ['SLURM_GPUS'])
        gpu_ids = list(range(num_gpus))
        if len(gpu_ids) > 1:
          net = nn.DataParallel(net, device_ids=gpu_ids).cuda()
        else:
          net = net.cuda()
        print(f"Number of available GPUs: {num_gpus}")

    saveHere, final_tar_path = saveEmbs(valset1, net, conf, fold=1)
    saveHere, final_tar_path = saveEmbs(valset2, net, conf, fold=2)
    saveHere, final_tar_path = saveEmbs(valset3, net, conf, fold=3)
    saveHere = Path(saveHere).parent

    with tarfile.open(final_tar_path, "w") as tar:
        tar.add(saveHere, arcname=os.path.basename(saveHere))
    print(f"Created tar archive: {saveHere}")

    # delete the original directory and all its contents rm -rf type
    os.system(f"rm -rf {saveHere}")

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
