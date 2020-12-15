"""
Evaluates meta-test-set performance of the FSS-1000 baseline model.

References:
    [1] Wei et al. 2020. FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation. https://github.com/HKUSTCV/FSS-1000. https://openaccess.thecvf.com/content_CVPR_2020/html/Li_FSS-1000_A_1000-Class_Dataset_for_Few-Shot_Segmentation_CVPR_2020_paper.html
    [2] Hendryx et al. 2020. https://meta-learn.github.io/2020/papers/44_paper.pdf
"""
import argparse
import random
import time
import warnings
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FSS-1000 baseline model.")
    parser.add_argument('--input-data', default='/xdisk/claytonm/projects/arete-realsim/fss-1000/data/')
    parser.add_argument("--output-root", help="path to results directory (hpc on default)",
                        default='/xdisk/claytonm/projects/arete-realsim/results')
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-modelf", "--feature_encoder_model", type=str, default='pretrained_model/feature_encoder.pkl')
    parser.add_argument("-modelr", "--relation_network_model", type=str,
                        default='pretrained_model/relation_network.pkl')
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use CUDA if available."
    )
    parser.add_argument("--n-support-per-class", type=int, default=5)
    parser.add_argument("--k", default=5, type=int, help="Number of training shots per task")
    args = parser.parse_args()
    return args


def validate_args(args):
    assert os.path.exists(args.input_data)
    assert os.path.exists(args.feature_encoder_model)
    assert os.path.exists(args.relation_network_model)


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn(pretrained=False).features)
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1)
        )
        self.features = nn.ModuleList(features)[1:]  # .eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        return x, results


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU()
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU()
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )  # 112 x 112
        self.double_conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )  # 256 x 256

    def forward(self, x, concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out)  # block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out)  # block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out)  # block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out)  # block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out)  # block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)

        out = F.sigmoid(out)
        return out


def build_model(args):
    print("Building neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    if args.use_cuda:
        feature_encoder.cuda(args.gpu)
        relation_network.cuda(args.gpu)

    if os.path.exists(args.feature_encoder_model):
        if args.use_cuda:
            feature_encoder.load_state_dict(torch.load(args.feature_encoder_model))
            print("loaded CUDA-enabled feature encoder")
        else:
            feature_encoder.load_state_dict(torch.load(args.feature_encoder_model, map_location=torch.device('cpu')))
            print("loaded CPU feature encoder")
    else:
        raise RuntimeError('Can not load feature encoder: %s' % args.feature_encoder_model)
    if os.path.exists(args.relation_network_model):
        if args.use_cuda:
            relation_network.load_state_dict(torch.load(args.relation_network_model))
            print("loaded GPU relation network")
        else:
            relation_network.load_state_dict(torch.load(args.relation_network_model, map_location=torch.device('cpu')))
            print("loaded CPU relation network")
    else:
        raise RuntimeError('Can not load relation network: %s' % args.relation_network_model)

    print("Neural networks successfully built.")
    return feature_encoder, relation_network


def get_fss_test_set():
    dirname = os.path.dirname(__file__)
    path = "fss_test_set.txt"  # File containing the test tasks
    filename = os.path.join(dirname, path)
    with open(filename, "r") as file:
        tasks = [line.rstrip("\n") for line in file]
    return tasks


def get_one_test_shot_batch(data_dir: str, k: int = 5, class_num: int = 1):
    test_shots: int = 1

    support_images = np.zeros((class_num * k, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((class_num * k, class_num, 224, 224), dtype=np.float32)
    query_images = np.zeros((class_num * test_shots, 3, 224, 224), dtype=np.float32)
    query_labels = np.zeros((class_num * test_shots, class_num, 224, 224), dtype=np.float32)
    zeros = np.zeros((class_num * test_shots, 1, 224, 224), dtype=np.float32)

    # Load tuples of images and masks in data_dir
    labels = [x for x in os.listdir(data_dir) if x.endswith(".png")]
    images = [x.replace(".png", ".jpg") for x in labels]
    labels = [os.path.join(data_dir, x) for x in labels]
    images = [os.path.join(data_dir, x) for x in images]
    images_labels = list(zip(images, labels))

    # Randomly sample k support tuples
    random.shuffle(images_labels)
    images_labels = [(np.transpose(cv2.imread(i)[:, :, ::-1] / 255., (2, 0, 1)), cv2.imread(l)[:, :, 0]) for i, l in images_labels]
    images_labels = [(i, l) for i, l in images_labels if i.shape[1] == 224 and i.shape[2] == 224 and l.shape[0] == 22 and l.shape[1] == 224]
    assert k <= len(images_labels) - test_shots

    support_image_labels = images_labels[:k]
    test_image_labels = images_labels[k:k + test_shots]

    for i, (image, label) in enumerate(support_image_labels):
        support_images[i] = image
        support_labels[i][0] = label
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor, support_labels_tensor), dim=1)

    for i, (image, label) in enumerate(test_image_labels):
        query_images[i] = image
        query_labels[i][0] = label
    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor


def forward(feature_encoder, relation_network, support_images_tensor, query_images_tensor, args):
    class_num = 1
    if args.use_cuda:
        var = Variable(support_images_tensor).cuda(args.gpu)
    else:
        var = Variable(support_images_tensor)
    sample_features, _ = feature_encoder(var)
    sample_features = sample_features.view(class_num, args.k, 512, 7, 7)
    sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
    if args.use_cuda:
        batch_features, ft_list = feature_encoder(Variable(query_images_tensor).cuda(args.gpu))
    else:
        batch_features, ft_list = feature_encoder(Variable(query_images_tensor))
    sample_features_ext = sample_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
    output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)
    return output


def evaluate_predictions(predictions, labels):
    pred = predictions.data.cpu().numpy()
    ious = []
    # for i in n_images:
    i = 0
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    testlabel = labels.numpy()[i][0].astype(bool)
    pred = pred.astype(bool)
    # compute IOU
    overlap = testlabel * pred
    union = testlabel + pred
    iou = overlap.sum() / float(union.sum())
    print('iou=%0.4f' % iou)
    ious.append(iou)
    return ious


def ci95(a: Union[List[float], np.ndarray]):
    """Computes the 95% confidence interval of the array `a`."""
    sigma = np.std(a)
    return 1.96 * sigma / np.sqrt(len(a))


def main():
    args = parse_args()
    validate_args(args)
    n_eval_samples_per_task = 2  # Following Hendryx et al. 2020

    # Get the 240 test tasks:
    test_tasks = get_fss_test_set()
    test_tasks = [os.path.join(args.input_data, x) for x in test_tasks]

    feature_encoder, relation_network = build_model(args)
    
    # Loop through each task, evaluating each task n_eval_samples_per_task times
    ious = []
    for _ in range(n_eval_samples_per_task):
        for task in test_tasks:
            # Load examples for the task:
            support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor = get_one_test_shot_batch(task, k=args.k)
            predictions = forward(feature_encoder, relation_network, support_images_tensor, query_images_tensor, args)

            # Evaluate the IoU on queries:
            ious.extend(evaluate_predictions(predictions, query_labels_tensor))
    
    # Report mean and 95% CI:
    print(f"Mean IoU +- 95% CI: {np.nanmean(ious)} +- {ci95(ious)}")


if __name__ == "__main__":
    main()
