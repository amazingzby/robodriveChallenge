import os, sys
import cv2, imageio
import numpy as np
import torch
import pickle
import glob

#valInfoPath = "/home/zby/work/project/robodrive-24/toolkit/track-3/SurroundOcc/data/nuscenes_infos_val.pkl"
valInfoPath = "/home/zby/work/project/robodrive-24/toolkit/track-3/SurroundOcc/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
with open(valInfoPath,"rb") as f:
    valInfo = pickle.load(f)
    infos = valInfo['infos']
    #print(len(infos)) # 6091
    num_iter = 0
    prev_token = ''
    for elem in infos:
        num_iter += 1
        print(num_iter)
        print(elem['sweeps'])
        print(elem.keys())
        exit(0)
        cur_token = elem['token']
        occ_path1 = elem['occ_path']
        occ_path2 = glob.glob("/home/zby/data/nuscenes/gts/*/"+elem['token']+"/labels.npz")
        #print(cur_token)
        gt_semantics = np.full((200, 200, 16), 17, np.uint8)
        gt = np.load(occ_path1)
        gt_semantics[gt[:, 1].astype(np.int), gt[:, 0].astype(np.int), gt[:, 2].astype(np.int)] = gt[:, 3]
        gt_semantics = np.flip(gt_semantics,[1])
        unique_elements1, unique_indices1 = np.unique(gt_semantics, return_inverse=True)
        counts1 = np.bincount(unique_indices1)
        print(dict(zip(unique_elements1, counts1)))

        gts = np.load(occ_path2[0])['semantics']
        unique_elements2, unique_indices2 = np.unique(gts, return_inverse=True)
        counts2 = np.bincount(unique_indices2)
        print(dict(zip(unique_elements2, counts2)))




