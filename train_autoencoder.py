import torch
import numpy as np
from data_loader import DatasetARC, DatasetARC_Test
from models import CAModel, CAModelConvLSTM, MetaConvLSTM_CA
from autoencoders import NARAutoencoder, VQVAE
import torch.nn as nn
import torch.nn.functional as F
from utils import inp2img, LossWriter, img_logger, save_model
import os
import sys
import argparse
import json
import random
from copy import copy
import wandb
from utils import plot_sample, clean_borders
import matplotlib.pyplot as plt
from matplotlib import colors


""" This script is to train the VQ-VAE and the Neural Abstract Reasoner autoencoders with Spectral Regularization
    TO DO:
    - Add the regularization
    - Find a way to turn off/on the padding
    - Add permutations of the colors
"""
cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#3d9970' ])

norm = colors.Normalize(vmin=0, vmax=10)


def save_imgs_wandb(pred, x, caption_str):

    for index in range(x.shape[0]):

        original_im = plt.imshow(x[index].argmax(0).cpu().numpy(), cmap=cmap, norm=norm)
        plt.close('all')

        prediction_im = plt.imshow(pred[index].argmax(0).cpu().numpy(), cmap=cmap, norm=norm)
        plt.close('all')

        wandb.log({caption_str + ' log images': [wandb.Image(original_im, caption="Original Image"), 
            wandb.Image(prediction_im, caption="Prediction")]})
            

def solve_tasks(tasks):
    for task in tasks:
        solve_task(task)


def tensorize_concatenate(sample, device):
    x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
    y = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
    #cat = torch.cat([x,y], dim= 1)
    return x, y


def random_permutations(l, n):    
    pset = set()
    while len(pset) < n:
        random.shuffle(l)
        pset.add(tuple(l))
    return pset

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




def meta_solve_task(tasks, val_tasks, max_steps=20, recurrent=True, log_dir = '', load_model=None, num_epochs=1000, batch_size=16, lr=0.0001, lamb=0.01, nar=False):

    if nar:
        power_iterations=10 
        model = NARAutoencoder(11, lamb, power_iterations).to(device)
    else:
        num_hiddens = 128
        num_residual_hiddens = 32
        num_residual_layers = 2

        embedding_dim = 64
        num_embeddings = 512

        commitment_cost = 0.25
        decay = 0.99
        power_iterations=10 


        learning_rate = 1e-3
        model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay,  power_iterations, lamb)

    model = model.train()

    criterion = nn.BCELoss()

    wandb.init(entity='gabrielelibardi', project ='arc')
    
    wandb.config.epochs = num_epochs
    wandb.config.lr = lr
    wandb.config.bs = batch_size
    wandb.config.nar = nar

    wandb.config.log_dir = log_dir
    
    wandb.run.name = log_dir
    wandb.run.save()
    
    for e in range(num_epochs):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=(lr))
        optimizer.zero_grad()
        loss = 0.0
        epoch_loss = 0.0
        rand_idxs = random.sample(range(0, len(tasks)), len(tasks)) 
        perfect_scores = 0.0
        for ii, idx_task in enumerate(rand_idxs):
            task = tasks[idx_task]
            X = torch.cat([object_in_tuple for sample in task['train']  for object_in_tuple in tensorize_concatenate(sample, device) ], dim=0)
            test_x = torch.from_numpy(inp2img(task['test'][0]['input'])).unsqueeze(0).float().to(device)
            test_y = torch.from_numpy(task['test'][0]['output']).unsqueeze(0).long().to(device)
            # for now we don't include test_x in the autoencoder training
            if nar:
                y_pred = model(X)
                y_pred = torch.clamp(y_pred, min=0.0, max = 1.0)
                loss = criterion(y_pred, X)
                if loss == 0.0:
                    perfect_scores += X.shape[0]
            else:
                vqloss, y_pred, preplexity = model(X)
                y_pred = torch.sigmoid(y_pred)
                loss = criterion(y_pred, X)
                loss += vqloss

            loss.backward()

            if ii % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({'training loss': loss, 'epoch': e})


            if ii % batch_size*100 == 0:
                save_imgs_wandb(torch.clone(y_pred), torch.clone(X), 'training')
                
        if e % 5 == 0:
            save_model(model, os.path.join(log_dir + "/models/","arc.state_dict"), e)

        if e %1 == 0:
            wandb.log({'training perfect scores': perfect_scores, 'epoch': e})
            perfect_scores = 0.0
            
        if e % 1 == 0:
            ## EVALUATE
            val_perfect_scores = 0.0
            model.eval()
            val_loss = 0.0
            rand_idxs = random.sample(range(0, len(val_tasks)), len(val_tasks))
            with torch.no_grad():
                for ii, idx_task in enumerate(rand_idxs):
                    task = val_tasks[idx_task]

                    X = torch.cat([object_in_tuple for sample in task['train']  for object_in_tuple in tensorize_concatenate(sample, device) ], dim=0)
                    test_x = torch.from_numpy(inp2img(task['test'][0]['input'])).unsqueeze(0).float().to(device)
                    test_y = torch.from_numpy(task['test'][0]['output']).unsqueeze(0).long().to(device)
                    # for now we don't include test_x in the autoencoder training
                    if nar:
                        y_pred = model(X)
                        y_pred = torch.clamp(y_pred, min=0.0, max = 1.0)
                        val_loss += criterion(y_pred, X)
                        if val_loss == 0.0:
                            val_perfect_scores += X.shape[0]

                    else:
                        vqloss, y_pred, preplexity = model(X)
                        y_pred = torch.sigmoid(y_pred)

                        val_loss += criterion(y_pred, X) + vqloss

            wandb.log({'val loss': val_loss/len(val_tasks), 'epoch': e})
            save_imgs_wandb(y_pred, X, 'val')
            wandb.log({'val perfect scores': val_perfect_scores, 'epoch': e})
            
            
        # if e % 1 == 0:
        #     epoch_loss /= (400/batch_size)
        #     print('Loss:',epoch_loss.item())
        #     training_info = {'epoch': e, 'train_loss': epoch_loss.item(), 'val_loss':val_loss.item()}
        #     loss_writer.write_row(training_info)
            
        #     model.train()
            
    print('------------------FINAL LOSS:', loss, '-------------------------------')      
    return model

def get_args():
    parser = argparse.ArgumentParser(description='ARC')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='number of epochs')
    parser.add_argument(
        '--nar', default=False, action='store_true', help='Use the NAR autoencoder or the VQ-VAE autoencoder')
    parser.add_argument(
        '--lamb', type=float, default=0.001, help='Lambda parameter for the Spectral Regularization')
    parser.add_argument(
        '--log-dir',default='',help='Log dir')
    parser.add_argument(
        '--data-dir',default='data/abstraction-and-reasoning-challenge',help='Log dir') 
    parser.add_argument(
        '--load-model',default=None,help='directory with the dir to load the model') 
    parser.add_argument(
        '--permute-colors',default=False, action='store_true', help='sample radnom color permutations when learning') 
    parser.add_argument(
        '--max-gird',default=10, type=int, help='max size of the grid used in training') 
    
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

    train_tasks = DatasetARC(args.data_dir, number_permutations=10)
    test_tasks = DatasetARC_Test(args.data_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    meta_solve_task(train_tasks, test_tasks, log_dir = args.log_dir, load_model = args.load_model, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lamb=args.lamb, nar=args.nar)
    