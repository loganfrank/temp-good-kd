
# Standard Python imports
import sys
import os
sys.path.append(f'{os.getcwd()}/')
print(sys.path)
print(os.getcwd())

import argparse
import copy
import json

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torchvision.transforms import v2

# Other imports
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics

# Inner-project imports
from networks import construct_network

from utils import make_deterministic
from utils import make_complex
from utils import get_transforms
from utils import infer_input
from utils import entropy
from utils import Mixup

from datasets import get_dataset
from datasets import PseudoLabeledDataset

# BN & FC INIT
from initialization import StandardFC

from good_dataset import BIM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

def arguments():
    parser = argparse.ArgumentParser(description='Training arguments')

    # Normal parameters
    parser.add_argument('--config', default='./configs/teacher_predictions_good_dataset.json', type=str, help='path to config file')
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
    # args['name'] = f'{args["name"]}_{args["teacher_dataset"]["name"]}_{args["teacher"]["name"]}_teacher_to_{args["student"]["name"]}_student_using_{args["distillation_dataset"]["name"]}_seed{args["seed"]}'

    # Append the necessary info to the 'path' argument to access the required directories
    # Images
    args['teacher_dataset']['root'] = f'{args["path"]}/{args["teacher_dataset"]["name"]}/images/'
    args['student_dataset']['root'] = f'{args["path"]}/{args["student_dataset"]["name"]}/images/{args["student_dataset"]["subname"]}/'

    # Networks
    args['teacher']['network_dir'] = f'{args["path"]}/{args["teacher_dataset"]["name"]}/networks/'

    # Results
    args['results_dir'] = f'{args["path"]}/{args["teacher_dataset"]["name"]}/results/'       

    # Create those directories if they don't already exist
    if not os.path.isdir(os.path.abspath(args['teacher_dataset']['root'])): os.makedirs(os.path.abspath(args['teacher_dataset']['root']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['student_dataset']['root'])): os.makedirs(os.path.abspath(args['student_dataset']['root']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['teacher']['network_dir'])): os.makedirs(os.path.abspath(args['teacher']['network_dir']), exist_ok=True)
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
    transform = get_transforms(normalization=args['teacher']['normalization'], **args['augmentations'])

    # Create the dataset for passing through the network
    if args['teacher_dataset']['name'] in ('cifar10', 'eurosat'):
        num_classes = 10
    elif args['teacher_dataset']['name'] == 'cifar100':
        num_classes = 100
    elif args['teacher_dataset']['name'] == 'tiny':
        num_classes = 200
    elif args['teacher_dataset']['name'] == 'fgvca':
        num_classes = 70
    elif args['teacher_dataset']['name'] == 'pets':
        num_classes = 37

    dataset = get_dataset(transform=transform, num_classes=num_classes, **args['student_dataset'])

    # Set up the rest of training
    network = construct_network(name=args['teacher']['name'], num_classes=num_classes, normalization=args['teacher']['normalization'], image_size=args['teacher']['image_size'])

    # Load the weights into the network
    network_path = f'{args["teacher"]["network_dir"]}/{args["teacher"]["weights"]}'
    network_dict = torch.load(network_path, map_location='cpu', weights_only=True)
        
    fc_fc_check = sum(['fc.fc' in k for k in network_dict.keys()]) > 0
    if fc_fc_check:
        network[1].fc = StandardFC(network[1].fc)

    # Check if utilizing the pretrained weights from previous DFKD papers
    if args['teacher']['name'] == 'resnet34':
        # TODO will check later
        temp_dict = network_dict['state_dict']

        network_dict = {}
        for k,v in temp_dict.items():
            if 'shortcut' in k:
                network_dict[k.replace('shortcut', 'downsample')] = v
            elif 'linear' in k:
                network_dict[k.replace('linear', 'fc')] = v
            else:
                network_dict[k] = v
        
        network[1].load_state_dict(network_dict, strict=True)

    # Big Transfer
    elif args['teacher']['name'] == 'resnet152x2':
        network[1].load_from(np.load(network_path))

    else:
        # This assumes my weights are being employed
        network.load_state_dict(network_dict, strict=True)

    # Adjust teacher fully connected layer (mainly in case of ResNet34s)
    # if not fc_fc_check and (args['teacher']['name'] != 'resnet152x2'):
    #     network[1].fc = StandardFC(network[1].fc)

    # Make sure no gradients for teacher (or any other updating)
    network.eval()
    network.requires_grad_(False)
    
    # Display the teacher and student networks
    print(network[1])

    # Move teacher network to device
    network = network.to(args['device'])
    
    print()
    print('*' * 50)
    print()

    if args['student_dataset']['pseudolabel']:
        dataset = PseudoLabeledDataset(dataset, network, image_size=args['student_dataset']['image_size'], selection=args['student_dataset']['pseudolabel'], longtail=args['student_dataset']['longtail'], device=args['device'])
    
    # Create the data loaders
    dataloader = data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True, num_workers=4)

    # Create the first attack if requested
    if args['border_attack']['name'] == 'bim':
        attack1 = BIM(network, **args['border_attack'])
    else:
        attack1 = None

    # Create the second attack if requested -- only creates the attack if you created the first attack
    if args['deeper_attack']['name'] != 'none':
        attack2 = BIM(network, **args['deeper_attack'])
    else:
        attack2 = None

    # Handle the mixup
    if args['augmentations']['mixup'] is not None:
        mixup_transform = Mixup(alpha1=args['augmentations']['mixup'], alpha2=args['augmentations']['mixup'], num_classes=num_classes)
    else:
        mixup_transform = None

    # Identify the number of training, validation, and test examples
    num_instances = len(dataset)

    # Identify the number of training, validation, and test batches
    num_batches = len(dataloader)

    # Prepare for training by disabling gradients
    torch.set_grad_enabled(False)
    
    # Instantiate two arrays for keeping track of ground truth labels and predicted labels
    all_softmaxes = []
    all_labels = []
    all_predictions = []

    all_features = []
    

    # Set the output print for torch tensors
    torch.set_printoptions(precision=1, threshold=10)

    # Iterate over the synthetic data
    for batch_num, (images, labels) in enumerate(dataloader):

        # Perform mixup if desired
        if mixup_transform is not None:
            if args['teacher_dataset']['name'] == args['student_dataset']['name']:
                mixed_images, mixed_labels = mixup_transform(images, labels)
            else:
                mixed_images, mixed_labels = mixup_transform(images, torch.ones(len(labels), dtype=torch.int64))

            images = mixed_images

        if attack1 is not None:
            ####################
            ##### ATTACK 1 #####
            ####################
            
            # TODO if OpenGL vs other
            if args['student_dataset']['name'] == 'opengl':
                targets = labels
            else:
                images = images.to(args['device'])
                with torch.no_grad():
                    l = network(images)
                    _, p = torch.max(l, 1)

                target_options = [[i for i in range(num_classes) if i != j] for j in p]
                targets = [np.random.choice(options, size=1)[0] for options in target_options]
                targets = torch.tensor(targets, dtype=torch.int64).squeeze()

            # Add the mixup synthetic images to the list of attack images
            attack_images = images
            attack_labels = targets

            # Send images and labels to compute device
            attack_images = attack_images.to(args['device'])
            attack_labels = attack_labels.to(args['device'])

            # Perform that actual adversarial attack
            border_images, border_pair_images, initial_predictions, final_predictions, final_predictions_pair, _ = attack1(attack_images, attack_labels)
            post_attack_label = final_predictions

            # Create a selection condition
            condition = torch.ones_like(attack_labels, dtype=bool)

            # Apply precondition that initial prediction was not in the desired target label
            condition = condition & (initial_predictions.to(attack_labels.device) != attack_labels)
            
            # Apply the postcondition that the final prediction was the desired target label
            condition = condition & (final_predictions.to(attack_labels.device) == attack_labels)

            # Filter the final border images
            border_images = border_images[condition]
            border_labels = attack_labels[condition]
            initial_predictions = initial_predictions[condition.cpu()]
            final_predictions = final_predictions[condition.cpu()]

            # Optionally filter the pair images
            border_pair_images = border_pair_images[condition]
            final_predictions_pair = final_predictions_pair[condition.cpu()]

            border_images = torch.cat([border_images, border_pair_images], dim=0)
            border_labels = torch.cat([final_predictions, final_predictions_pair], dim=0)

            ####################
            ##### ATTACK 2 #####
            ####################

            # Move the previous attack's example to the appropriate device
            border_images = border_images.to(args['device'])
            border_labels = border_labels.to(args['device'])

            # Perform that actual adversarial attack, this attack should never include an adversarial pair
            deeper_images, _, deeper_predictions, _ = attack2(border_images, border_labels)

            # Create the initial condition mask (to include all examples)
            condition = torch.ones_like(border_labels, dtype=bool)

            # Filter the deeper examples
            deeper_images = deeper_images[condition]
            deeper_labels = deeper_predictions.to(args['device'])[condition]

            #####################################################
            ##### DETERMINE WHICH EXAMPLES FOR DISTILLATION #####
            #####################################################

            images = torch.cat([border_images, deeper_images], dim=0)
            labels = torch.cat([border_labels, deeper_labels], dim=0)
        
        # Send images and labels to compute device
        images = images.to(args['device'])
        labels = labels.to(args['device'])

        def hook_fn(m, inp, out):
            all_features.append(inp[0].detach().cpu())
        hook = network[1].fc.fc.register_forward_hook(hook_fn)

        # Forward propagation
        with torch.no_grad():
            logits = network(images)

        hook.remove()
        
        # Threshold for flat prediction
        _, predictions = torch.max(logits, 1)

        # Compute the softmax scores
        softmax = F.softmax(logits, dim=1)

        # Record the values
        all_softmaxes.append(softmax.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_predictions.append(predictions.detach().cpu())
        
        # Give epoch status update
        print(' ' * 100, end='\r', flush=True) 
        print(f'Parsing Predictions: {100. * (batch_num + 1) / num_batches : 0.1f}% ({batch_num + 1}/{num_batches})', end='\r', flush=True)

    # Clear the status update message
    print(' ' * 100, end='\r', flush=True) 

    all_softmaxes = torch.cat(all_softmaxes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    all_features = torch.cat(all_features, dim=0)

    # Set the output print for torch tensors
    # torch.set_printoptions(prec)

    # Calculate how many examples were predicted in each of the network's possible outputs
    prediction_counts = torch.histogram(all_predictions.to(torch.float32), bins=num_classes)[0]
    print(prediction_counts)

    torch.set_printoptions(precision=4)

    # List of all argmax maximums
    argmax_maximums = []
    unnormalized_feature_spread = []
    normalized_feature_spread = []

    for i in range(num_classes):
        
        selector = (all_predictions == i)

        if sum(selector) == 0:
            print(f'Skipping {i}: No Predictions')
            unnormalized_feature_spread.append(0)
            normalized_feature_spread.append(0)
            continue

        softmaxes = all_softmaxes[selector]
        labels = all_labels[selector]
        predictions = all_predictions[selector]
        features = all_features[selector]
        normalized_features = features / torch.linalg.vector_norm(features, dim=1).reshape(-1, 1)

        unnormalized_feature_spread.append(features.std(dim=1).mean())
        normalized_feature_spread.append(normalized_features.std(dim=1).mean())

        maximums = softmaxes.max(dim=1)[0]
        argmax_maximums.append(maximums)

    argmax_maximums = torch.cat(argmax_maximums)
    unnormalized_feature_spread = torch.tensor(unnormalized_feature_spread)
    normalized_feature_spread = torch.tensor(normalized_feature_spread)

    prediction_entropy = entropy(prediction_counts / prediction_counts.sum())
    print(f'Entropy: {prediction_entropy : 0.5f}')

    # print(f'UnMixed Softmax Maximums: {softmax_maximums.mean()}')
    # print(f'UnMixed Softmax T Maximums: {sofmaxT_maximums.mean()}')
    print(f'Softmax Maximums: {argmax_maximums.mean() : 0.5f}')

    # Set the output print for torch tensors
    torch.set_printoptions(precision=3)
    print(f'Normalized Feature Spread: {normalized_feature_spread}')
    print(f'UNnormalized Feature Spread: {unnormalized_feature_spread}')

    print(f'')
    print('stop')
# EOF      