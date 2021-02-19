import os
import json
from pathlib import Path
import torch
import numpy as np
import csv
import copy
from shutil import copy2
from matplotlib import colors
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



def augment(tasks):
    # bottom up flipping
    augmented_tasks = []
    for task in tasks:
        augmended_task = {'train':[], 'test':[]}
        for sample in task['train']:
            some_dict = {}
            some_dict['input'] = sample['input'][::-1]
            some_dict['output'] = sample['output'][::-1]
            augmended_task['train'].append(some_dict)
            
        for sample in task['test']:    
            some_dict = {}
            some_dict['input'] = sample['input'][::-1]
            some_dict['output'] = sample['output'][::-1]
            augmended_task['test'].append(some_dict)
        
        augmented_tasks.append(augmended_task)
    
    return augmented_tasks
        

def save_model(model, fname, iter):
    #avoid overwrite last model for safety
    torch.save(model.state_dict(), fname + ".tmp")  
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))
    

def clean_borders(arr):
    return arr[~np.all(arr == 10,axis=0), :][:,~np.all(arr == 10,axis=1)]


class img_logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def save_imgs(self, num_img, samples, pred, loss):
        samples = copy.deepcopy(samples)
        pred = pred.clone()
        imgs_dict = {}  
        for idx1,sample in enumerate(samples['train']):
            samples['train'][idx1]['input'] = clean_borders(sample['input']).tolist()
            samples['train'][idx1]['output'] = clean_borders(sample['output']).tolist()
            
        for idx2,sample in enumerate(samples['test']):
            samples['test'][idx2]['input'] = clean_borders(sample['input']).tolist()
            samples['test'][idx2]['output'] = clean_borders(sample['output']).tolist()
        pred = pred.argmax(1).squeeze().cpu().numpy().tolist()
        imgs_dict = {'sample': samples, 'pred':pred, 'loss': loss}
        log_dir = self.log_dir + '/imgs/'
        json.dump(imgs_dict, open(os.path.join(log_dir, 'imgs_' + str(num_img)+ ".json"), "w"), indent=4)

class LossWriter(object):
    def __init__(self, log_dir, fieldnames = ('l'), header=''):
        
        assert log_dir is not None
        
        filename = '{}/loss_monitor.csv'.format(log_dir)
        self.f = open(filename, "wt")
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, training_info):
        if self.logger:
            self.logger.writerow(training_info)
            self.f.flush()
            
            
class ImgWriter(object):
    def __init__(self, log_dir, fieldnames = ('l'), header=''):
        
        assert log_dir is not None
        
        filename = '{}/loss_monitor.csv'.format(log_dir)
        self.f = open(filename, "wt")
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, training_info):
        if self.logger:
            self.logger.writerow(training_info)
            self.f.flush()


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((11, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(11):
        img[i] = (inp==i)
    return img

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
    
def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()
    
def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])
            

def check_max_rows_cols(data):
    max_cols = 0
    max_rows = 0
    
    for task in data:
        for sample in task['train']:

            x = torch.Tensor(sample['input'])
            y = torch.Tensor(sample['output'])

            max_cols = max(max_cols, x.shape[0], y.shape[0])
            max_rows = max(max_rows, x.shape[1], y.shape[1])
            
        for sample in task['test']:

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
            
        for idx3, sample in enumerate(data[idx1]['test']):

            x = np.array(sample['input'])
            y = np.array(sample['output'])
            x_padded = add_padding(x, max_cols, max_rows)
            y_padded = add_padding(y, max_cols, max_rows)
            data[idx1]['test'][idx3]['input'] = x_padded
            data[idx1]['test'][idx3]['output'] = y_padded
            
            