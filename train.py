import torch
import numpy as np
from data_loader import DatasetARC
from models import CAModel, CAModelConvLSTM, MetaConvLSTM_CA
import torch.nn as nn
import torch.nn.functional as F
from utils import inp2img


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
                print('Loss:',loss)
            
            if loss < 0.0001:
                break
                
    print('------------------FINAL LOSS:', loss, '-------------------------------')      
    return model, num_steps, losses


def tensorize_concatenate(sample):
    x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
    y = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
    cat = torch.cat([x,y], dim= 1)
    return cat


def meta_solve_task(tasks, max_steps=20, recurrent=True):
    
    if recurrent:
        model = MetaConvLSTM_CA(11).to(device)
    else:
        model = CAModel(11).to(device)
        
    model = model.train()
    num_epochs = 1000000
    batch_size = 20
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)
    
    for e in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.0001))
        optimizer.zero_grad()
        loss = 0.0
        n_tasks = 0
        
        for idx_task, task in enumerate(tasks):

            X = torch.cat([tensorize_concatenate(sample) for sample in task['train']], dim=0)
            n_tasks += 1
            test_x = torch.from_numpy(inp2img(task['test'][0]['input'])).unsqueeze(0).float().to(device)
            test_y = torch.from_numpy(task['test'][0]['output']).unsqueeze(0).long().to(device)
            y_pred = model(X, test_x, max_steps)

            loss += criterion(y_pred, test_y)
            #print('Task loss:',criterion(y_pred, test_y))

            # predit output from output
            # enforces stability after solution is reached
#                 y_in = torch.from_numpy(inp2img(task['test'][0]["output"])).unsqueeze(0).float().to(device)
#                 y_pred = model(X, y_in, 1) 
#                 loss += criterion(y_pred, test_y)
                
            if idx_task % batch_size == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0.0
                n_tasks = 0
                

        #losses[(num_steps - 1) * num_epochs + e] = loss.item()

        if e % 1 == 0:
            print('Loss:',loss)

    print('------------------FINAL LOSS:', loss, '-------------------------------')      
    return model, num_steps, losses



if __name__ == '__main__':
    
    dir_path = 'data/abstraction-and-reasoning-challenge'
    tasks = DatasetARC(dir_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #solve_tasks(tasks)
    meta_solve_task(tasks)
    