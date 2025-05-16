# Standard Python imports
import sys
import os
sys.path.append(f'{os.getcwd()}/')
from functools import partial

# PyTorch imports
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as v2F

# Huggingface
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
from transformers import AutoModelForImageClassification
from transformers import AutoFeatureExtractor
from transformers import AutoImageProcessor

from datasets import load_dataset

import evaluate

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

# Other imports
import numpy as np
import wandb

# Inner-project imports
from utils import get_config
from utils import make_deterministic
from utils import init_network
from utils import KLDivergence
from utils import Mixup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class Distiller(Trainer):
    def __init__(self, teacher_network=None, student_network=None, output_loss_function=None, mixup_transform=None, *args, **kwargs):
        super().__init__(model=student_network, *args, **kwargs)
        self.teacher_network = teacher_network
        self.student_network = student_network
        self.output_loss_function = output_loss_function
        self.mixup_transform = mixup_transform

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.teacher_network = self.teacher_network.to(device)
        self.teacher_network.requires_grad_(False)
        self.teacher_network = self.teacher_network.eval()

    def __repr__(self):
        return 'Knowledge Distillation Trainer'

    def create_scheduler(self, num_training_steps, optimizer=None):
        ...

    def compute_loss(self, network, inputs, return_outputs=False, num_items_in_batch=None):
        # Optionally apply mixup
        if self.mixup_transform is not None:
            inputs['pixel_values'], _ = self.mixup_transform(inputs['pixel_values'], inputs['labels'])

        # Obtain outputs from the teacher network
        with torch.no_grad():
          teacher_output = self.teacher_network(**inputs)

        # Obtain outputs from the student network
        student_output = network(**inputs)

        # Compute the loss (in most cases KL-divergence)
        loss = self.output_loss_function(teacher_output.logits, student_output.logits)

        return (loss, student_output) if return_outputs else loss
    
    def prediction_step(self, network, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            student_outputs = network(**inputs)
        return (student_outputs.loss, None, None) if prediction_loss_only else (student_outputs.loss, student_outputs.logits, inputs['labels'])

##################################
##### COMMAND LINE ARGUMENTS #####
##################################

def arguments():
    # Now just rename the config to use the args variable name
    config = get_config()
    args = config

    # Images
    args['distillation_image_dir'] = f'{args["path"]}/{args["distillation_dataset"]}/images/' 
    args['test_image_dir'] = f'{args["path"]}/{args["test_dataset"]}/images/' 

    # Networks
    args['network_dir'] = f'{args["path"]}/{args["test_dataset"]}/networks/'

    # Results
    args['results_dir'] = f'{args["path"]}/{args["test_dataset"]}/results/'      

    # Create those directories if they don't already exist
    if not os.path.isdir(os.path.abspath(args['network_dir'])): os.makedirs(os.path.abspath(args['network_dir']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['results_dir'])): os.makedirs(os.path.abspath(args['results_dir']), exist_ok=True)

    # Set the device
    torch.cuda.set_device(args['device'])

    ###########################################
    ##### Handle Weights & Biases Logging #####
    ###########################################

    wandb.init(
        entity=args['wandb_entity'],
        project=args['wandb_project'],
        name=args['name'],
        tags=args['wandb_tags'],
        mode=args['wandb_mode'],
        config=config
    )

    return args

if __name__ == '__main__':
    # Get command line arguments
    args = arguments()

    # Set the rng seed for reproducibility
    make_deterministic(args['seed'])

    #########################
    ##### NETWORK SETUP #####
    #########################

    # Teacher
    teacher_image_processor = AutoImageProcessor.from_pretrained('nateraw/vit-base-patch16-224-cifar10', use_fast=True)
    teacher_network = AutoModelForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')

    # Student
    student_config = AutoConfig.from_pretrained('timm/resnet18.tv_in1k', num_labels=10)
    student_network = AutoModelForImageClassification.from_config(student_config)
    init_network(student_network, name='resnet18')

    # Some information extracted from hf
    num_classes = teacher_network.config.num_labels
    image_size = teacher_network.config.image_size
    normalization_statistics = {'mean': teacher_image_processor.image_mean, 'std': teacher_image_processor.image_std}
    normalization_statistics_int = {n : [round(x * 255) for x in v] for n, v in normalization_statistics.items()}

    ###########################
    ##### TRANSFORMATIONS #####
    ###########################

    # Preprocessing Transformations
    distillation_preprocess = v2.Compose([v2.Resize(image_size)])
    test_preprocess = v2.Compose([v2.Resize(image_size), v2.CenterCrop(image_size)])

    # Per-Batch Transformations
    distillation_transform = v2.Compose([
        v2.PILToTensor(),
        v2.RandomCrop(image_size, padding=None),
        v2.RandAugment(num_ops=(2 if args['distillation_dataset'] != 'opengl' else 4), magnitude=14, fill=normalization_statistics_int['mean']),
        nn.Identity() if args['distillation_dataset'] != 'opengl' else v2.RandomApply(nn.ModuleList([v2.ElasticTransform(alpha=50, sigma=5.0, fill=normalization_statistics_int['mean'])]), p=0.5),
        nn.Identity() if args['distillation_dataset'] != 'opengl' else v2.RandomInvert(p=0.1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(image_size, padding=12, fill=normalization_statistics_int['mean']),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**normalization_statistics)
    ])
    test_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**normalization_statistics)
    ])

    # Create mixup
    mixup_transform = Mixup(args['augmentations']['mixup'], device=args['device'])

    ####################
    ##### DATASETS #####
    ####################

    dataset_map = {
        'cifar10' : 'uoft-cs/cifar10',
        'cifar100' : 'uoft-cs/cifar100',
        'tiny_imgnet' : 'zh-plus/tiny-imagenet',
        'fgvca' : 'Voxel51/FGVC-Aircraft',
        'pets' : 'timm/oxford-iiit-pet',
        'eurosat' : 'timm/eurosat-rgb',
        'food' : 'ethz/food101',
        'opengl' : 'imagefolder',
    }

    # Load the distillation and test datasets
    distillation_dataset = load_dataset(dataset_map[args['distillation_dataset']], data_dir=(None if args['distillation_dataset'] != 'opengl' else args['distillation_image_dir']))
    test_dataset = load_dataset(dataset_map[args['test_dataset']], data_dir=(None if args['test_dataset'] != 'opengl' else args['test_image_dir']))

    #########################
    ##### PREPROCESSING #####
    #########################

    # Preprocess the dataset (if applicable)
    def process(examples, processor):    
        examples['img'] = [processor(img.convert('RGB')) for img in examples['img']]
        return examples
    
    # Determine the original columns in each (that will eventually be removed)
    distillation_columns = [cn for cn in distillation_dataset['train'].column_names if cn not in ['img', 'label']]
    test_columns = [cn for cn in test_dataset['test'].column_names if cn not in ['img', 'label']]

    # Apply preprocessing to the datasets
    distillation_preprocessor = partial(process, processor=distillation_preprocess)
    test_preprocessor = partial(process, processor=test_preprocess)
    distillation_dataset = distillation_dataset['train'].map(distillation_preprocessor, batched=True, remove_columns=distillation_columns)
    test_dataset = test_dataset['test'].map(test_preprocessor, batched=True, remove_columns=test_columns)

    ##############################
    ##### DATA AUGMENTATIONS #####
    ##############################

    def augment(examples, augmentations):
        examples['pixel_values'] = [augmentations(img) for img in examples['img']]
        del examples['img']
        return examples

    # Add the data augmentation
    distillation_augmentations = partial(augment, augmentations=distillation_transform)
    test_augmentations = partial(augment, augmentations=test_transform)
    distillation_dataset = distillation_dataset.with_transform(distillation_augmentations)
    test_dataset = test_dataset.with_transform(test_augmentations)

    #########################
    ##### LOSS FUNCTION #####
    #########################

    loss_function = KLDivergence(temperature=args['training']['loss']['temperature'])

    ####################################
    ##### OPTIMIZER + LR SCHEDULER #####
    ####################################

    optimizer = create_optimizer_v2(student_network, args['training']['optimizer']['name'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['training']['num_epochs'], eta_min=1e-6)

    ##########################
    ##### TRAINING STUFF #####
    ##########################

    distillation_args = TrainingArguments(
        output_dir=f'{args["network_dir"]}/{args["name"]}/',
        eval_strategy='epoch',
        per_device_train_batch_size=args['training']['batch_size'],
        per_device_eval_batch_size=args['training']['batch_size'],
        num_train_epochs=args['training']['num_epochs'],
        save_strategy='no', # TODO idk, was 'epoch'
        report_to=['wandb'],
        run_name=args['name'],
        logging_dir=f'{args["results_dir"]}/',
        logging_strategy='epoch',
        remove_unused_columns=False
    )

    ############################
    ##### EVALUATION STUFF #####
    ############################

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {"accuracy": acc["accuracy"]}

    #####################################################
    ##### CREATE THE DISTILLER, TRAIN, AND EVALUATE #####
    #####################################################

    distiller = Distiller(teacher_network=teacher_network,
              student_network=student_network,
              output_loss_function=loss_function,
              mixup_transform=mixup_transform,
              args=distillation_args,
              train_dataset=distillation_dataset,
              eval_dataset=test_dataset,
              compute_metrics=compute_metrics,
              optimizers=(optimizer, scheduler))

    t = distiller.evaluate(test_dataset)

    distiller.train()

    t = distiller.evaluate(test_dataset)

    wandb.finish()

# EOF      