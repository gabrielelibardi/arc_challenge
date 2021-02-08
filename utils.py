import os
import json
from pathlib import Path
import torch
import numpy as np

from torch.utils.data import Dataset    

def inp2img(inp):
    inp = np.array(inp)
    img = np.full((11, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(11):
        img[i] = (inp==i)
    return img


def check_max_rows_cols(data):
    max_cols = 0
    max_rows = 0
    
    for task in data:
        for sample in task['train']:

            x = torch.Tensor(sample['input'])
            y = torch.Tensor(sample['output'])

            max_cols = max(max_cols, x.shape[0], y.shape[0])
            max_rows = max(max_rows, x.shape[1], y.shape[1])
     
    return max_cols, max_rows

def add_padding(tensor, max_cols, max_rows):
    
    # either is odd or even cols_n or rows_
    padding = np.ones([max_cols, max_rows])*10
    point1_cols = max_cols//2 -  tensor.shape[0]//2
    point2_cols = max_cols//2 +  tensor.shape[0]//2
    if tensor.shape[0] % 2 != 0:
        point2_cols += 1
        
    point1_rows = max_cols//2 -  tensor.shape[1]//2
    point2_rows = max_cols//2 +  tensor.shape[1]//2
    if tensor.shape[1] % 2 != 0:
        point2_rows += 1
    
    padding[point1_cols:point2_cols, point1_rows: point2_rows] = tensor
    
    return padding
    
        
def pad_all_tensors(data, max_cols, max_rows):
     
    for idx1, task in enumerate(data):
        for idx2, sample in enumerate(data[idx1]['train']):
            
            x = np.array(sample['input'])
            y = np.array(sample['output'])
            x_padded = add_padding(x, max_cols, max_rows)
            y_padded = add_padding(y, max_cols, max_rows)
            data[idx1]['train'][idx2]['input'] = x_padded
            data[idx1]['train'][idx2]['output'] = y_padded
            
            