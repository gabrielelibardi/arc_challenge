import torch
import numpy as np
from data_loader import DatasetARC, DatasetARC_Test
from models import CAModel, CAModelConvLSTM, MetaConvLSTM_CA
from nar_autoencoder import NARAutoencoder
import torch.nn as nn
import torch.nn.functional as F
from utils import inp2img, LossWriter, img_logger, save_model
import os
import sys
import argparse
import json
import random
from copy import copy

""" This script is to train the VQ-VAE and the Neural Abstract Reasoner autoencoders with Spectral Regularization
    TO DO:
    - Add the regularization
    - Find a way to turn off/on the padding
    - Add permutations of the colors
"""

def solve_tasks(tasks):
    for task in tasks:
        solve_task(task)


def solve_task(task, max_steps=40, recurrent=True):
    
    if recurrent:
        model = CAModelConvLSTM(11).to(device)
    else:
        model = CAModel(11).to(device)


    
    model = model.train()
    num_epochs = 500
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1))
        print((0.1 / (num_steps * 10)))
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for sample in task['train']:
                # predict output from input
                x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
                y = torch.from_numpy(sample["output"]).unsqueeze(0).long().to(device)
                
                y_pred = model(x, num_steps)
                
                loss += criterion(y_pred, y)
                
                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1) 
                loss += criterion(y_pred, y)
                

            loss.backward()
            optimizer.step()
            
            losses[(num_steps - 1) * num_epochs + e] = loss.item()

            if e % 100 == 0:
                print('Loss:',loss.item())
            
            if loss < 0.0001:
                break
                
    print('------------------FINAL LOSS:', loss, '-------------------------------')      
    return model, num_steps, losses


def tensorize_concatenate(sample, device):
    x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
    y = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
    #cat = torch.cat([x,y], dim= 1)
    return x, y


def permute_colors(task):

    # last color 10 is the alpha channel deciding what should be considered as task, that one should not be permuted

    new_colors = np.arange(10)
    np.random.shuffle(new_colors)
    new_task = task.copy()
    for i in range(len(new_task['train'])):
        new_task['train'][i] = change_colors(new_task['train'][i], new_colors)

    for j in range(len(new_task['test'])):
        new_task['test'][j] = change_colors(new_task['test'][j], new_colors)

    return new_task


def change_colors(sample, new_colors):
    for new_color, old_color in enumerate(new_colors):
        sample['input'][sample['input'] == old_color] = new_color
        sample['output'][sample['input'] == old_color] = new_color

    return sample




def meta_solve_task(tasks, val_tasks, max_steps=20, recurrent=True, log_dir = '', load_model=None):
    
    # if recurrent:
    #     model = MetaConvLSTM_CA(11).to(device)
    # else:
    #     model = CAModel(11).to(device)
        
    # if load_model:
    #     model.load_state_dict(torch.load(load_model, map_location=device))

    model = NARAutoencoder(11).to(device)
    model = model.train()
    num_epochs = 1000000
    batch_size = 20
    criterion = nn.CrossEntropyLoss()
    loss_writer = LossWriter(log_dir, fieldnames = ('epoch','train_loss', 'val_loss'))
    img_writer = img_logger(log_dir)
    losses = np.zeros((max_steps - 1) * num_epochs)
    
    for e in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.0001))
        optimizer.zero_grad()
        loss = 0.0
        epoch_loss = 0.0
        rand_idxs = random.sample(range(0, len(tasks)), len(tasks)) 
        for ii, idx_task in enumerate(rand_idxs):
            task = tasks[idx_task]
            task = permute_colors(task)
            X = torch.cat([object_in_tuple for sample in task['train']  for object_in_tuple in tensorize_concatenate(sample, device) ], dim=0)
            test_x = torch.from_numpy(inp2img(task['test'][0]['input'])).unsqueeze(0).float().to(device)
            test_y = torch.from_numpy(task['test'][0]['output']).unsqueeze(0).long().to(device)
            # for now we don't include test_x in the autoencoder training
            y_pred = model(X, max_steps)
            
            loss += criterion(y_pred, test_y)
            
            if e % 100 == 0:
                img_writer.save_imgs(idx_task, task, y_pred, criterion(y_pred, test_y).detach().cpu().item())

            
                
            if ii % batch_size == 0:
                loss.backward()
                epoch_loss += loss
                loss = 0.0
                
            if ii % batch_size*4 == 0:
                optimizer.step()
                optimizer.zero_grad()

      
        if e % 100 == 0:
            save_model(model, os.path.join(log_dir + "/models/","arc.state_dict"), e)
            
        if e % 10 == 0:
            ## EVALUATE
            model.eval()
            val_loss = 0.0
            rand_idxs = random.sample(range(0, len(val_tasks)), len(val_tasks))
            with torch.no_grad():
                for ii, idx_task in enumerate(rand_idxs):
                    val_task = val_tasks[idx_task]
                    X = torch.cat([tensorize_concatenate(sample, device) for sample in val_task['train']], dim=0)
                    test_x = torch.from_numpy(inp2img(val_task['test'][0]['input'])).unsqueeze(0).float().to(device)
                    test_y = torch.from_numpy(val_task['test'][0]['output']).unsqueeze(0).long().to(device)
                    y_pred = model(X, test_x, max_steps)
                    if e % 100 == 0:
                        img_writer.save_imgs(len(tasks) + idx_task, val_task, y_pred,  criterion(y_pred, test_y).detach().cpu().item())

                    val_loss += criterion(y_pred, test_y)
            val_loss /= batch_size
            print('VAL LOSS', val_loss)
            
        if e % 1 == 0:
            epoch_loss /= (400/batch_size)
            print('Loss:',epoch_loss.item())
            training_info = {'epoch': e, 'train_loss': epoch_loss.item(), 'val_loss':val_loss.item()}
            loss_writer.write_row(training_info)
            
            model.train()
            
        

    print('------------------FINAL LOSS:', loss, '-------------------------------')      
    return model, num_steps, losses

def get_args():
    parser = argparse.ArgumentParser(description='ARC')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--log-dir',default='',help='Log dir')
    parser.add_argument(
        '--data-dir',default='data/abstraction-and-reasoning-challenge',help='Log dir') 
    parser.add_argument(
        '--load-model',default=None,help='directory with the dir to load the model') 
    
    args = parser.parse_args() 
    
    args.log_dir = os.path.expanduser(args.log_dir)
    
    return args





if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)

    os.system("mkdir "+ args.log_dir)
    json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)
    #os.system("cp "+__file__+ " " + args.log_dir + "/"+__file__ )
    os.system("mkdir "+ args.log_dir + "/models")
    os.system("mkdir "+ args.log_dir + "/imgs")

    train_tasks = DatasetARC(args.data_dir)
    test_tasks = DatasetARC_Test(args.data_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    meta_solve_task(train_tasks, test_tasks, log_dir = args.log_dir,load_model = args.load_model)
    