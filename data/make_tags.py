import csv
import json
import os

raw_file = csv.DictReader(open("training_tasks_tagged.csv"))
list_file = [row for row in raw_file]

root = '/home/gabbo/Documents/arc_challenge/data/'

for task in list_file:
    task_name = task['task_name']
    tags = [tag for tag in task.keys() if task[tag] == '1']
    print(task_name) 
    if os.path.exists(os.path.join(root, 'abstraction-and-reasoning-challenge/training/' + task_name)):
        with open(root + 'abstraction-and-reasoning-challenge/training/' + task_name) as f:
            task_dict = json.load(f)

        task_dict['tags'] = tags
        with open(root + 'tagged_abstraction-and-reasoning-challenge/training/'+ task_name, 'w+') as f:
            json.dump(task_dict, f)
   
    if os.path.isdir(os.path.join(root, 'abstraction-and-reasoning-challenge/test/', task_name)):
        with open(root + 'abstraction-and-reasoning-challenge/test/' + task_name) as f:
            task_dict = json.load(f)

        task_dict['tags'] = tags
        with open(root + 'tagged_abstraction-and-reasoning-challenge/test/'+ task_name, 'w+') as f:
            json.dump(task_dict, f)

    if os.path.isdir(os.path.join(root, 'abstraction-and-reasoning-challenge/evaluation/', task_name)):
        with open(root + 'abstraction-and-reasoning-challenge/evaluation/' + task_name) as f:
            task_dict = json.load(f)

        task_dict['tags'] = tags
        with open(root + 'tagged_abstraction-and-reasoning-challenge/evaluation/'+ task_name, 'w+') as f:
            json.dump(task_dict, f)


