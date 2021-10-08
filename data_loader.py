import os
import json
from pathlib import Path
import torch

from torch.utils.data import Dataset    
from utils import *

class DatasetARC(Dataset):
    """
    Creates a dataset for the ARC dataset.
    """

    def __init__(self, dir_path):
        
        data_path = Path(dir_path)
        train_path = data_path / 'training'
        self.train_tasks = [json.load(task.open()) for task in train_path.iterdir() ]
        augmented_tasks = augment(self.train_tasks)
        self.train_tasks += augmented_tasks
        self.max_cols, self.max_rows = check_max_rows_cols(self.train_tasks)
        pad_all_tensors(self.train_tasks, self.max_cols, self.max_rows)
        

    def __len__(self):
        return len(self.train_tasks)
        
    def __getitem__(self, idx):
        return self.train_tasks[idx]
    
class DatasetARC_Test(Dataset):
    """
    Creates a dataset for the ARC dataset.
    """

    def __init__(self, dir_path):

        data_path = Path(dir_path)

        
        test_path = data_path / 'evaluation'
        self.test_tasks = [json.load(task.open()) for task in test_path.iterdir() ]
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

    