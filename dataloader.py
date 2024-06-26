#   Copyright (c) 2024 orangerec Authors. All Rights Reserved.
from collections import defaultdict
from torch.utils.data import IterableDataset
import torch 
import numpy as np

class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.config = config
        self.file_list = file_list
        self.init()

    def init(self):
        all_field_id = [
            '101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
            '125', '126', '127', '128', '129', '205', '206', '207', '210',
            '216', '508', '509', '702', '853', '301'
        ]
        self.all_field_id_dict = defaultdict(int)
        # self.max_len = self.config.get("hyper_parameters.max_len", 3)
        self.max_len = self.config.max_len
        for i, field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False, i]
        self.padding = 0

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    features = l.strip().split(',')
                    ctr = int(features[1])
                    ctcvr = int(features[2])

                    output = [(field_id, [])
                              for field_id in self.all_field_id_dict]
                    output_list = []
                    for elem in features[4:]:
                        field_id, feat_id = elem.strip().split(':')
                        if field_id not in self.all_field_id_dict:
                            continue
                        self.all_field_id_dict[field_id][0] = True
                        index = self.all_field_id_dict[field_id][1]
                        output[index][1].append(int(feat_id))

                    for field_id in self.all_field_id_dict:
                        visited, index = self.all_field_id_dict[field_id]
                        self.all_field_id_dict[field_id][0] = False
                        if len(output[index][1]) > self.max_len:
                            output_list.append(
                                np.array(output[index][1][:self.max_len])
                                .astype('int64'))
                        else:
                            for ii in range(self.max_len - len(output[index][
                                    1])):
                                output[index][1].append(self.padding)
                            output_list.append(
                                np.array(output[index][1]).astype('int64'))
                    output_list.append(np.array([ctr]).astype('int64'))
                    output_list.append(np.array([ctcvr]).astype('int64'))
                    #print("----------------------output_list-------------\n:", output_list)
                    '''
                        ----------------------output_list-------------
                         [array([28874,     0,     0]), array([ 534, 6000, 3724]), array([28859, 28860, 28861]), array([28851, 28852, 28853]), array([12099, 28830, 28831]), array([28844,     0,     0]), array([12096,     0,     0]), array([2670,    0,    0]), array([11209,     0,     0]), array([8438,    0,    0]), 
                         array([15,  0,  0]), array([16,  0,  0]), array([10537,     0,     0]), array([28892,     0,     0]), array([18497,     0,     0]), array([0, 0, 0]), array([20565, 20558, 28887]), array([28891,     0,     0]), array([0, 0, 0]), array([0, 0, 0]), array([0, 0, 0]), array([0, 0, 0]), array([9281,    0,    0]), 
                         array([0]), array([0])]
                    
                    '''
             
                    yield output_list
