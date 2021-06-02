import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state

from val import *
from tqdm import tqdm
import glob
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-id', type=str, default='detections',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='path to the checkpoints directory')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    args = parser.parse_args()

    #net = PoseEstimationWithMobileNet()
    net = PoseEstimationWithMobileNet(num_heatmaps=(19+2), num_pafs=(38+4))
    modeldir = '{}/*.pth'.format(args.checkpoint_dir)

    sorted_dirs = sorted( glob.glob(modeldir) )
    for checkpoint_path in sorted_dirs:
        checkpoint = torch.load(checkpoint_path)
        load_state(net, checkpoint)

        chkpt_no = os.path.basename(checkpoint_path).split('.')[0].split('_')[-1]
        output_dir = '{}/{}'.format(args.checkpoint_dir, args.output_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_name = '{}/{}/{}.json'.format(args.checkpoint_dir, args.output_id, chkpt_no)
        print('{}:{}'.format(checkpoint_path, output_name))
        evaluate(args.labels, output_name, args.images_folder, net, args.multiscale, False)
