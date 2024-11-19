"""
Code originally from https://github.com/CVI-SZU/ME-GraphAU file: model/test.py
Modified and adapted to work in the ELEGANT framework by Andrew Sumsion
Modified and adapted to work in the EnsembleAU framework by Andrew Sumsion
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model.encoder_gcn import MEFARG
from dataset import *
from utils import *
from model.ensemble import *
from conf import get_config,set_logger,set_outdir,set_env

from torchmetrics.classification import BinaryCalibrationError


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        datasetPath = "data/predictions/BP4D_computeLocal"
        allResults = "whichPredictions_fold"
        training_data = ensemble_dataset(datasetPath, conf.fold, train=True, allResults=allResults, jobID=conf.jobID, dataset="BP4D", root_path=conf.dataset_path)
        valset = ensemble_dataset(datasetPath, conf.fold, train=False, train_dataPath=training_data.dataset_path, allResults=allResults, jobID=conf.jobID, returnPath=True, dataset="BP4D", root_path=conf.dataset_path)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        datasetPath = "data/predictions/DISFA_computeLocal"
        allResults = "whichPredictions_fold"
        includeLambda = False
        training_data = ensemble_dataset(datasetPath, conf.fold, train=True, allResults=allResults, jobID=conf.jobID, dataset="DISFA", root_path=conf.dataset_path)
        valset = ensemble_dataset(datasetPath, conf.fold, train=False, train_dataPath=training_data.dataset_path, allResults=allResults, jobID=conf.jobID, returnPath=True, dataset="DISFA", root_path=conf.dataset_path)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'FEC':
        valset = FEC(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return val_loader, len(valset), valset


# Val
def val(net, val_loader):
    net.eval()
    statistics_list = None
    
    allOutputs = []
    allTargets = []
    for batch_idx, (inputs, targets, img_path) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
            
            #put outputs on cpu and targets on cpu
            outputs = outputs.cpu()
            targets = targets.cpu()
            

            if type(allOutputs) == list:
                allOutputs = outputs
                allTargets = targets
            else:
                allOutputs = torch.cat((allOutputs, outputs), 0)
                allTargets = torch.cat((allTargets, targets), 0)

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    return mean_f1_score, f1_score_list, mean_acc, acc_list

def update_statistics_list_FEC(statistics_list, outputs, label):
    diff_first_second = torch.norm(outputs[:,0,:] - outputs[:,1,:], p=2, dim=1)
    diff_first_third = torch.norm(outputs[:,0,:] - outputs[:,2,:], p=2, dim=1)
    diff_second_third = torch.norm(outputs[:,1,:] - outputs[:,2,:], p=2, dim=1)

    combined = torch.cat((torch.unsqueeze(diff_second_third,1), torch.unsqueeze(diff_first_third,1), torch.unsqueeze(diff_second_third,1)), 1)
    predictions = torch.argmin(combined, 1)

    # add 1 to all prediction values so indexing is the same as the labels
    predictions += 1

    correct = predictions == label
    incorrect = ~correct
    num_correct = torch.sum(correct, dim=0)
    num_incorrect = torch.sum(incorrect, dim=0)
    total = num_correct + num_incorrect

    if statistics_list is None:
        statistics_list = [num_correct, num_incorrect, total]
    else:
        statistics_list = [statistics_list[0] + num_correct, statistics_list[1] + num_incorrect, statistics_list[2] + total]
    return statistics_list

def val_FEC(net, val_loader, criterion):
    print("==> Starting validation")
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    losses = AverageMeter()
    for batch_idx, (imgs0, imgs1, imgs2, label) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            if torch.cuda.is_available():
                imgs0, imgs1, imgs2, label = imgs0.cuda(), imgs1.cuda(), imgs2.cuda(), label.cuda()

            _, _, emb_out0, _ = net(imgs0)
            _, _, emb_out1, _ = net(imgs1)
            _, _, emb_out2, _ = net(imgs2)
            emb_out0 = emb_out0.unsqueeze(1)
            emb_out1 = emb_out1.unsqueeze(1)
            emb_out2 = emb_out2.unsqueeze(1)
            outputs = torch.cat((emb_out0, emb_out1, emb_out2), 1)
            loss = criterion[0](outputs, label)
            statistics_list = update_statistics_list_FEC(statistics_list, outputs, label)
            losses.update(loss.data.item(), outputs.size(0))
    num_correct = statistics_list[0]
    num_incorrect = statistics_list[1]
    total = statistics_list[2]
    mean_acc = num_correct / total
    return losses.avg, mean_acc

def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
        num_classes = 12
        numberLmks=49
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
        num_classes = 8
        numberLmks=66
    elif conf.dataset == 'FEC':
        dataset_info = FEC_infolist
        # num_classes = 12
        # numberLmks=49
        num_classes = 8
        numberLmks=66

    # data
    val_loader, val_data_num, valset = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    
    if conf.dataset == 'FEC':
        net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, numEncoderLayers=conf.numEncoderLayers, numLandmarks=numberLmks)
    else:
        tmp = valset.__getitem__(0)
        tmp = tmp[0]
        net = MixtureOfExperts(input_size=tmp.shape[0], output_size=num_classes, num_experts=int(num_classes*conf.stacking_times), hidden_size=2500)

    # resume
    if conf.resume != '':
        logging.info("Resume from | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    #test
    if conf.dataset == 'FEC':
        margin = 0.2
        criterion = [TripleContrasitiveLoss(margin=margin)]
        val_loss, val_mean_acc = val_FEC(net, val_loader, criterion)
    else:
        val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader)

    # log
    if conf.dataset == 'FEC':
        infostr = {'val_loss {:.2f}' .format(val_loss)}
        logging.info(infostr)
        infostr = {'val_mean_acc {:.2f}' .format(100.* val_mean_acc)}
        logging.info(infostr)
    else:
        infostr = {'val_mean_f1_score {:.2f} val_mean_acc {:.2f}' .format(100.* val_mean_f1_score, 100.* val_mean_acc)}
        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

