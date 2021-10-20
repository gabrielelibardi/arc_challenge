import os
import json
from pathlib import Path
import torch

from torch.utils.data import Dataset    
from utils import *
import copy

class DatasetARC(Dataset):
    """
    Creates a dataset for the ARC dataset.
    """

    def __init__(self, dir_path, number_permutations):
        
        self.number_permutations = number_permutations
        data_path = Path(dir_path)
        train_path = data_path / 'training'
        self.train_tasks = [json.load(task.open()) for task in train_path.iterdir() ]
        # add filter grid size
        self.train_tasks= filter_size(self.train_tasks, max_size=10)
        print('Total Number Training Tasks:', len(self.train_tasks))
        #self.train_tasks += permute_colors(self.train_tasks)
        #print('added permutations')
        augmented_tasks = augment(self.train_tasks)
        print('added augmentation')
        self.train_tasks += augmented_tasks
        print('Total Number Training Tasks with augmentation:', len(self.train_tasks))
        self.max_cols, self.max_rows = check_max_rows_cols(self.train_tasks)
        pad_all_tensors(self.train_tasks, self.max_cols, self.max_rows)

        
        l = [0,1,2,3,4,5,6,7,8,9]
        self.colors_permutations = [{old_col:new_col for  old_col, new_col in enumerate(new_colors)} for new_colors in random_permutations(l, self.number_permutations)]

        for ii in range(len(self.colors_permutations)):
            self.colors_permutations[ii][10] =10 
        
    def __len__(self):
        return len(self.train_tasks*self.number_permutations)
        
    def __getitem__(self, idx):
        true_idx = idx // self.number_permutations
        color_idx = idx % self.number_permutations
        task = self.train_tasks[true_idx].copy()

        new_task = {'train':[], 'test':[]}
        for sample in task['train']:
            new_task['train'].append(change_colors(sample.copy(), self.colors_permutations[color_idx]))
        for sample in task['test']:
            new_task['test'].append(change_colors(sample.copy(), self.colors_permutations[color_idx]))
            
        return new_task
    
class DatasetARC_Test(Dataset):
    """
    Creates a dataset for the ARC dataset.
    """

    def __init__(self, dir_path):

        data_path = Path(dir_path)

        
        test_path = data_path / 'evaluation'
        self.test_tasks = [json.load(task.open()) for task in test_path.iterdir() ]
        print(len(self.test_tasks))
        self.test_tasks = filter_size(self.test_tasks , max_size=10)
        print('Total Number Validation Tasks:', len(self.test_tasks))
        self.max_cols, self.max_rows = check_max_rows_cols(self.test_tasks)
        print(self.max_cols, self.max_rows)
        pad_all_tensors(self.test_tasks, self.max_cols, self.max_rows)


    def __len__(self):
        return len(self.test_tasks)
        
    def __getitem__(self, idx):
        return self.test_tasks[idx]

if __name__ == '__main__':
    
    dir_path = 'data/abstraction-and-reasoning-challenge'
    data = DatasetARC(dir_path)

    