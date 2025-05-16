
# Standard Python imports
import sys
import os
sys.path.append(f'{os.getcwd()}/')

import argparse
import json
from functools import partial

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Other imports
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# Inner-project imports
from networks import construct_network
from datasets import get_dataset
from datasets import PseudoLabeledDataset
from utils import parameter_groups
from utils import make_deterministic
from utils import infer_input
from utils import get_transforms
from utils import Mixup
from utils import KLDivergence
from utils import SoftCrossEntropy
from utils import Entropy
from utils import entropy

# BN & FC INIT
from initialization import StandardFC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def arguments():
    parser = argparse.ArgumentParser(description='Training arguments')

    # Normal parameters
    parser.add_argument('--config', default='./configs/progadv/train.json', type=str, help='path to config file')
    parser.add_argument('--seed', default='134895', type=str, help='rng seed for reproducability')

    # Pull out the command-line arguments that were unknown
    _, unknown_args = parser.parse_known_args()

    # Add the unknown arguments to the argparser
    for arg in unknown_args:
        if arg.startswith(('--')):
            parser.add_argument(arg.split('=')[0], type=infer_input)

    # Parse out all of the arguments now that the new ones were added
    args = vars(parser.parse_args())

    # Load the config file
    with open(args['config'], 'r') as f:
        config = json.load(f)

    # Parse the unknown arguments into the config file hierarchy
    for arg, value in args.items():
        if 'config' not in arg:
            # A single level arg in the config
            if '.' not in arg:
                config[arg] = value

            # An arg in the config that goes into nested dictionaries
            else:
                levels = arg.split('.')
                if len(levels) == 2:
                    config[levels[0]][levels[1]] = value
                elif len(levels) == 3:
                    config[levels[0]][levels[1]][levels[2]] = value
                elif len(levels) == 4:
                    config[levels[0]][levels[1]][levels[2]][levels[3]] = value
                elif len(levels) == 5:
                    config[levels[0]][levels[1]][levels[2]][levels[3]][levels[4]] = value

    # Now just rename the config to use the args variable name
    args = config

    # Adjust the experiment name to also include the seed value
    # args['name'] = f'{args["name"]}

    # Append the necessary info to the 'path' argument to access the required directories
    # Images
    args['teacher_dataset']['root'] = f'{args["path"]}/{args["teacher_dataset"]["name"]}/images/{args["teacher_dataset"]["subname"]}/'
    args['student_dataset']['root'] = f'{args["path"]}/{args["student_dataset"]["name"]}/images/{args["student_dataset"]["subname"]}/'

    # Networks
    args['teacher']['network_dir'] = f'{args["path"]}/mnist/networks/'
    args['student']['network_dir'] = f'{args["path"]}/mnist/networks/'

    # Results
    args['results_dir'] = f'{args["path"]}/mnist/results/'       

    # Create those directories if they don't already exist
    if not os.path.isdir(os.path.abspath(args['teacher_dataset']['root'])): os.makedirs(os.path.abspath(args['teacher_dataset']['root']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['student_dataset']['root'])): os.makedirs(os.path.abspath(args['student_dataset']['root']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['teacher']['network_dir'])): os.makedirs(os.path.abspath(args['teacher']['network_dir']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['student']['network_dir'])): os.makedirs(os.path.abspath(args['student']['network_dir']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['results_dir'])): os.makedirs(os.path.abspath(args['results_dir']), exist_ok=True)
    
    if 'cuda' in args['device']:
        assert torch.cuda.is_available(), 'Device set to GPU but CUDA not available'

    return args

if __name__ == '__main__':
    # Get command line arguments
    args = arguments()

    # Set the rng seed for reproducibility
    make_deterministic(args['seed'])

    # Get the transforms for the data
    teacher_transform = get_transforms(normalization=args['teacher']['normalization'], **args['teacher_augmentations'])
    student_transform = get_transforms(normalization=args['student']['normalization'], **args['student_augmentations'])
    

    # Create the dataset object
    teacher_dataset = get_dataset(transform=teacher_transform, num_classes=10, **args['teacher_dataset'])
    student_dataset = get_dataset(transform=student_transform, num_classes=10, **args['student_dataset'])
    num_classes = len(teacher_dataset.classes)

    # Construct the networks
    teacher_network = construct_network(num_classes=num_classes, **args['teacher'])
    student_network = construct_network(num_classes=num_classes, **args['student'])

    teacher_network[1].fc = StandardFC(teacher_network[1].fc)
    student_network[1].fc = StandardFC(student_network[1].fc)
    
    # Load the weights into the network
    teacher_network_path = f'{args["teacher"]["network_dir"]}/{args["teacher"]["weights"]}'
    teacher_network_dict = torch.load(teacher_network_path, map_location='cpu', weights_only=True)
    teacher_network.load_state_dict(teacher_network_dict, strict=True)

    student_network_path = f'{args["student"]["network_dir"]}/{args["student"]["weights"]}'
    student_network_dict = torch.load(student_network_path, map_location='cpu', weights_only=True)
    student_network.load_state_dict(student_network_dict, strict=True)

    # Make sure no gradients for teacher (or any other updating)
    teacher_network.eval()
    teacher_network.requires_grad_(False)

    student_network.eval()
    student_network.requires_grad_(False)

    # Disable computing gradients
    torch.set_grad_enabled(False)

    # Move networks
    teacher_network = teacher_network.to(args['device'])
    student_network = student_network.to(args['device'])

    # Optionally take a pseudolabeled subset (corresponding to the teacher) -- TODO probably a better way to do this
    # TODO provide option to get a Long Tail
    if args['teacher_dataset']['pseudolabel']:
        teacher_dataset = PseudoLabeledDataset(teacher_dataset, teacher_network, selection=args['teacher_dataset']['pseudolabel'], device=args['device'])
    if args['student_dataset']['pseudolabel']: # TODO: doesn't seem useful to pseudolabel with student? maybe pseudolabel with teacher?
        student_dataset = PseudoLabeledDataset(student_dataset, student_network, selection=args['student_dataset']['pseudolabel'], device=args['device'])

    # Create the data loaders
    teacher_dataloader = data.DataLoader(teacher_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    student_dataloader = data.DataLoader(student_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    # Identify the number of training, validation, and test examples
    num_teacher_instances = len(teacher_dataset)
    num_student_instances = len(student_dataset)

    # Identify the number of training, validation, and test batches
    num_teacher_batches = len(teacher_dataloader)
    num_student_batches = len(student_dataloader)

    # Lists for keeping track of things
    teacher_labels = []
    teacher_outputs = []
    teacher_features = []

    student_labels = []
    student_outputs = []
    student_features = []

    # Hook function
    def hook_fn(module, inp, out, collection):
        collection.append(inp[0].detach().cpu().numpy().astype(float))

    # Teacher GAP hook
    teacher_hook_fn = partial(hook_fn, collection=teacher_features)
    teacher_network[1].fc.fc.register_forward_hook(teacher_hook_fn)

    # Student GAP hook
    student_hook_fn = partial(hook_fn, collection=student_features)
    student_network[1].fc.fc.register_forward_hook(student_hook_fn)
    
    # Iterate over the TEACHER batches
    with torch.no_grad():
        for _ in range(args['num_cycles']):
            for batch_num, (images, labels) in enumerate(teacher_dataloader):
                # Perform mixup if desired
                
                # Send images and labels to compute device
                images = images.to(args['device'])
                labels = labels.to(args['device'])
                
                # Forward propagation
                teacher_logits = teacher_network(images)
                
                # Threshold for flat prediction
                _, teacher_predictions = torch.max(teacher_logits, 1)
                
                # Record the actual and predicted labels for the instance
                teacher_labels.append(labels.detach().cpu().numpy().astype(int))
                teacher_outputs.append(teacher_logits.detach().cpu().numpy().astype(float))

                # Give epoch status update
                print(' ' * 100, end='\r', flush=True) 
                print(f'Teacher: {100. * (batch_num + 1) / num_teacher_batches : 0.1f}% ({batch_num + 1}/{num_teacher_batches})', end='\r', flush=True)
            
            # Clear the status update message
            print(' ' * 100, end='\r', flush=True) 

    # Concatenate
    teacher_labels = np.concatenate(teacher_labels)
    teacher_outputs = np.concatenate(teacher_outputs)
    teacher_features = np.concatenate(teacher_features)

    # Iterate over the STUDENT batches
    with torch.no_grad():
        for _ in range(args['num_cycles']):
            for batch_num, (images, labels) in enumerate(student_dataloader):
                
                # Send images and labels to compute device
                images = images.to(args['device'])
                labels = labels.to(args['device'])
                
                # Forward propagation
                student_logits = student_network(images)
                
                # Record the actual and predicted labels for the instance
                student_labels.append(labels.detach().cpu().numpy().astype(int))
                student_outputs.append(student_logits.detach().cpu().numpy().astype(float))

                # Give epoch status update
                print(' ' * 100, end='\r', flush=True) 
                print(f'Student: {100. * (batch_num + 1) / num_student_batches : 0.1f}% ({batch_num + 1}/{num_student_batches})', end='\r', flush=True)
            
            # Clear the status update message
            print(' ' * 100, end='\r', flush=True) 

    # Concatenate
    student_labels = np.concatenate(student_labels)
    student_outputs = np.concatenate(student_outputs)
    student_features = np.concatenate(student_features)

    # Get the predictions of teacher and student
    teacher_predictions = np.argmax(teacher_outputs, axis=1)
    student_predictions = np.argmax(student_outputs, axis=1)
    
    # Accuracy
    teacher_accuracy = metrics.accuracy_score(teacher_labels, teacher_predictions)
    student_accuracy = metrics.accuracy_score(student_labels, student_predictions)

    teacher_set = 'Train' if args['teacher_dataset']['train'] else 'Test'
    student_set = 'Train' if args['student_dataset']['train'] else 'Test'

    print(f'Teacher: {args["teacher_dataset"]["name"].upper()} {teacher_set} Accuracy: {teacher_accuracy * 100.0 : 0.3f}')
    print(f'Student: {args["student_dataset"]["name"].upper()} {student_set} Accuracy: {student_accuracy * 100.0 : 0.3f}')

    # Prediction Histograms
    teacher_prediction_counts = np.histogram(teacher_predictions, bins=num_classes)[0]
    teacher_prediction_entropy = entropy(teacher_prediction_counts / teacher_prediction_counts.sum())
    student_prediction_counts = np.histogram(student_predictions, bins=num_classes)[0]
    student_prediction_entropy = entropy(student_prediction_counts / student_prediction_counts.sum())

    print(f'Teacher Predictions: {teacher_prediction_counts}')
    print(f'\tEntropy: {teacher_prediction_entropy : 0.5f}')
    print(f'Student Predictions: {student_prediction_counts}')
    print(f'\tEntropy: {student_prediction_entropy : 0.5f}')

    # Visualize the GAP features
    teacher_X = teacher_features[:, 0]
    teacher_Y = teacher_features[:, 1]
    student_X = student_features[:, 0]
    student_Y = student_features[:, 1]

    # Determine the min and max
    min_X = -10
    max_X = 10
    min_Y = -10
    max_Y = 10

    # Get the decision boundary
    xx, yy = torch.meshgrid(torch.linspace(min_X - 1, max_X + 1, 500), torch.linspace(min_Y - 1, max_Y + 1, 500), indexing='ij')

    # Predict class probabilities on the grid
    grid = torch.column_stack([xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)]).to(args['device'])
    with torch.no_grad():
        teacher_decision_space = teacher_network[1].fc.fc(grid).detach().cpu().numpy()
        student_decision_space = student_network[1].fc.fc(grid).detach().cpu().numpy()

    teacher_decision_space = np.argmax(teacher_decision_space, axis=1)# .reshape(xx.shape)
    student_decision_space = np.argmax(student_decision_space, axis=1)# .reshape(xx.shape)


    # TEACHER
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    # colors = plt.get_cmap('Paired', num_classes)
    c = 'gist_rainbow'
    colors = plt.get_cmap(c, num_classes)

    # Plot each of the classes
    for i in range(num_classes):
        tx = teacher_X[teacher_predictions == i]
        ty = teacher_Y[teacher_predictions == i]

        tc = teacher_predictions[teacher_predictions == i]

        l = f'Class {i}'

        axes.scatter(tx, ty, color=colors(i), s=0.5, label=l)

    axes.scatter(xx, yy, c=teacher_decision_space, cmap=c, alpha=0.06, s=0.5)

    # axes.legend(markerscale=6)

    axes.set_title('CIFAR10',fontsize=64)
    axes.set_ylabel('   ',fontsize=64)

    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticklabels([])
    axes.set_yticks([])


    axes.set_ylim([-10, 10])
    axes.set_xlim([-10, 10])

    fig.tight_layout()

    fig.savefig(f'./good_dataset/results/cifar10_teacher.png', dpi=300)

    #########################################

    # STUDENT
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    # colors = plt.get_cmap('Paired', num_classes)
    c = 'gist_rainbow'
    colors = plt.get_cmap(c, num_classes)

    # Plot each of the classes
    for i in range(num_classes):
        sx = student_X[student_predictions == i]
        sy = student_Y[student_predictions == i]

        sc = student_predictions[student_predictions == i]

        l = f'Class {i}'

        axes.scatter(sx, sy, color=colors(i), s=0.5, label=l)

    axes.scatter(xx, yy, c=student_decision_space, cmap=c, alpha=0.06, s=0.5)

    # axes.legend(markerscale=6)

    axes.set_ylabel('   ',fontsize=64)
    axes.set_title('   ',fontsize=64)

    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticklabels([])
    axes.set_yticks([])

    axes.set_ylim([-10, 10])
    axes.set_xlim([-10, 10])

    fig.tight_layout()

    fig.savefig(f'./good_dataset/results/cifar10_student.png', dpi=300)

# EOF      