import torch
import numpy as np
from data_loader import DatasetARC
from models import CAModel, CAModelConvLSTM
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
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)))
        
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



if __name__ == '__main__':
    
    dir_path = 'data/abstraction-and-reasoning-challenge'
    tasks = DatasetARC(dir_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    solve_tasks(tasks)
    