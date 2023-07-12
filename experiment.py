"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random
import json
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb


##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:

    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(data, n, num_samples_cls=1):
    tmp = data
    cls_iter, data_iter_list = tmp['class_iter'], tmp['data_iter_list']
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    def __init__(self, entries, num_samples_cls=1):
        self.data = {}
        self.datalen = 0
        empty = []
        self.data[-1] = []
        for i, entry in enumerate(entries):
            ans = entry['answer']
            if ans['label_counts'] is None or len(ans['label_counts']) == 0:
                self.data[-1].append(i)
            else:
                tmp = max(ans['label_counts'], key=ans['label_counts'].get)
                if tmp not in self.data:
                    self.data[tmp] = []
                self.data[tmp].append(i)

        self.data = self.sol(self.data)
        self.datalen = 438183
        self.num_samples_cls = num_samples_cls
        # for _, tmp in self.data.items():
        #     print(tmp)


    def sol(self, tmp):
        label_list = np.unique([x for x in tmp])
        num_classes = len(label_list)
        now = {}
        mapping = {}
        for i, j in enumerate(label_list):
            mapping[j] = i

        now['class_iter'] = RandomCycleIter(range(num_classes))
        now['cls_data_list'] = [list() for _ in range(num_classes)]
        for a, b in tmp.items():
            now['cls_data_list'][mapping[a]] = b
        now['data_iter_list'] = [RandomCycleIter(x) for x in now['cls_data_list']]
        # now['num_samples'] = max([len(x) for x in now['cls_data_list']]) * len(now['cls_data_list'])
        now['num_samples'] = len(tmp)
        return now

    def __iter__(self):
        return class_aware_sample_generator(self.data, self.datalen, self.num_samples_cls)
            # yield class_aware_sample_generator(tmp['class_iter'], tmp['data_iter_list'],
            #                                 tmp['num_samples'], self.num_samples_cls)

    def __len__(self):
        return self.datalen


def get_sampler():
    return ClassAwareSampler

##################################