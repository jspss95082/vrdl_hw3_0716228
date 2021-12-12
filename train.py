from __future__ import print_function
import numpy as np
import torchvision
import random
import argparse
from torchvision.models.detection import FasterRCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import cv2
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import warnings
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection import MaskRCNN
warnings.filterwarnings("ignore")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', transforms=None):
        imgs_fold = os.listdir('dataset/' + mode + '/')
        self.transforms = transforms
        self.imgs_fold = imgs_fold

    def __getitem__(self, idx):
        fold_name = self.imgs_fold[idx]
        while True:
            h_f = torch.rand(1) > 0.5
            v_f = torch.rand(1) > 0.5
            r_f = torch.rand(1) > 0.5
            x = random.randint(0, 750)
            y = random.randint(0, 750)
            cr = (x, y, x + 250, y + 250)
            img = Image.open('dataset/' + 'train/' + fold_name +
                             '/images/' + fold_name + '.png').convert("RGB")
            img = img.crop(cr)
            if r_f:
                img = img.rotate(90, Image.BILINEAR)
            if v_f:
                img = F.vflip(img)
            if h_f:
                img = F.hflip(img)
            img = np.array(img)
            image_id = torch.tensor([idx])
            boxes = []
            labels = []
            area = []
            masks = []
            iscrowd = []
            mask = os.listdir('dataset/' + 'train/' + fold_name + '/masks/')
            for it in range(len(mask)):
                mask_img = Image.open('dataset/' + 'train/' +
                                      fold_name + '/masks/' + mask[it])
                mask_img = mask_img.crop(cr)
                if r_f:
                    mask_img = mask_img.rotate(90, Image.BILINEAR)
                if v_f:
                    mask_img = F.vflip(mask_img)
                if h_f:
                    mask_img = F.hflip(mask_img)
                if np.all(np.array(mask_img) == 0):
                    continue
                left, top, right, down = mask_img.getbbox()
                if left == 0 or top == 0 or right >= 249 or down >= 249:
                    continue
                mask_img = np.array(mask_img)
                masks.append(mask_img)
                boxes.append([left, top, right, down])
                labels.append(1)
                area.append((right - left)*(down - top))
                iscrowd.append(0)
                masks.append(mask_img)
                boxes.append([left, top, right, down])
                labels.append(1)
                area.append((right - left)*(down - top))
                iscrowd.append(0)

            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks/255
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            img = self.transforms(img)
            if len(boxes) > 0:
                return img, target

    def __len__(self):
        return len(self.imgs_fold)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained_backbone=False,
        pretrained=False,
        box_detections_per_img=1200,
        min_size=750,
        max_size=1300,
    )

    # create an anchor_generator for the FPN which by default has 5 outputs
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0, 4) for _ in range(5)])
    )
    model.rpn.anchor_generator = anchor_generator
    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(
        256, anchor_generator.num_anchors_per_location()[0])
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def main():
    warnings.filterwarnings("ignore")
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 100)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )

    start_epoch = 0
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    load = input('Do you want to load model y/n ? : ')

    if load[0] == 'y' or load[0] == 'Y':
        model_name = input('Model name? : ')
        model = torch.load(model_name)
        start_epoch = int(input('Start epoch ?: '))
    else:
        model = get_instance_segmentation_model(2)
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = Dataset(
        transforms=transform)

    print(np.array(train_data[0][1]['masks']).max())
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        collate_fn=utils.collate_fn
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300, 1000,1500], gamma=0.1)

    # train
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_loader,
                        device, epoch, print_freq=200)
        scheduler.step()
        if epoch % 100 == 0:
            torch.save(model, 'model.pt')
    print("That's it!")


if __name__ == "__main__":
    main()
