import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
# import base_model
# import base_model_block as base_model
# import base_model_ban as base_model
import base_model_ab as base_model
# import base_model_v_only as base_model
# import base_model_sfce as base_model

from train_ab import train
from train_ab import evaluate
from ClassAwareSampler import ClassAwareSampler
# from train_GGE import train
import utils
import click

from vqa_debias_loss_functions import *


# ckpt = torch.load('/content/drive/MyDrive/vqa-cp/logs/[1]/model-8.pth')
# # states_ = ckpt
# # model.load_state_dict(states_)
# for k, v in ckpt.items():
#     print(k)


# tmp = [{'answer': {'question_type': 1, 'label_counts': {1: 7, 3: 3}}},
#        {'answer': {'question_type': 1, 'label_counts': {1: 6, 3: 4}}},
#        {'answer': {'question_type': 1, 'label_counts': {2: 7, 3: 3}}},
#        {'answer': {'question_type': 1, 'label_counts': {4: 7, 3: 3}}},
#        {'answer': {'question_type': 2, 'label_counts': {6: 7, 3: 3}}},
#        {'answer': {'question_type': 2, 'label_counts': {6: 7, 3: 3}}},
#        {'answer': {'question_type': 2, 'label_counts': {7: 7, 3: 3}}},
#        {'answer': {'question_type': 2, 'label_counts': {7: 7, 3: 3}}}]
# now = ClassAwareSampler(tmp)
# for i in now:
#     print(i)

# with open('util/cpv2_notype_mask.json', 'r') as f:
#     p1 = json.load(f)
# with open('util/v2_type_mask.json', 'r') as f:
#     p2 = json.load(f)
# print(len(p1))



# print(438183-329019)
# with open('sample-list.json', 'r') as f:
#     now = json.load(f)

# tmp = torch.zeros(438183)
# for i in now:
#     tmp[i] += 1
# num = 0
# for a, b in enumerate(tmp):
#     if b == 0:
#        num += 1
# print(num)


# import numpy as np
# import matplotlib.pyplot as plt
#
# N = 1
#
# boys = [20]
# girls = (25)
# boyStd = (2)
# girlStd = (3)
# ind = np.arange(N)
# width = 0.35
#
# fig = plt.figure(figsize=(2,7))
# ax = fig.add_subplot(111)
#
#
# p1 = plt.bar(ind, boys, width)
# p2 = plt.bar(ind, girls, width,
#              bottom=boys)
# ax.text(1,10, "%d%%" % (200), ha='center')
# plt.axis('off')

# plt.ylabel('Contribution')
# plt.title('Contribution by the teams')
# plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0]), ('boys', 'girls'))

# plt.savefig('visual/VQA/ bar-test.svg', bbox_inches='tight')
# plt.show()
# tmp = np.zeros((3, 10))
# print(np.amax(tmp))
print(53.43-46.79)