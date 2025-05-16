# Standard Python imports
import sys
import os
sys.path.append(f'{os.getcwd()}/')
from functools import partial

# PyTorch imports
import torch
import torchvision

# Huggingface
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
from transformers import TimmWrapperConfig
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import BaseImageProcessor
from transformers import DefaultDataCollator
from transformers import TrainerCallback

from datasets import load_dataset

import evaluate

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

# Other imports
import numpy as np
import wandb

# Inner-project imports
from utils import get_config
from utils import parameter_groups
from utils import make_deterministic
from utils import get_transforms
from utils import Mixup

from utils import KLDivergence
from utils import SoftCrossEntropy

# Initialization work
from networks import init_network
from approaches.initialization import get_normalization_type
from approaches.initialization import init_normalization_layers
from approaches.initialization import init_classifier

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

    # Networks
    args['network_dir'] = f'{args["path"]}/{args["dataset"]}/networks/'

    # Results
    args['results_dir'] = f'{args["path"]}/{args["dataset"]}/results/'      

    # Create those directories if they don't already exist
    if not os.path.isdir(os.path.abspath(args['network_dir'])): os.makedirs(os.path.abspath(args['network_dir']), exist_ok=True)
    if not os.path.isdir(os.path.abspath(args['results_dir'])): os.makedirs(os.path.abspath(args['results_dir']), exist_ok=True)

    # Set the device
    torch.cuda.set_device(args['device'])

    ###########################################
    ##### Handle Weights & Biases Logging #####
    ###########################################

    wandb.init(
        entity='loganfrank-dissertation',
        project='student-initialization',
        name=args['name'],
        tags=args['tags'],
        dir=f'{args["results_dir"]}/',
        mode=args['wandb_mode'],
        config=config
    )

    return args

if __name__ == '__main__':
    # Get command line arguments
    args = arguments()

    # Set the rng seed for reproducibility
    make_deterministic(args['seed'])

    ###########################
    ##### TRANSFORMATIONS #####
    ###########################

    # Get the transforms for the data
    distillation_preprocess, distillation_transform, distillation_normalization = get_transforms(args['augmentations']['train'], args['augmentations']['normalization'], args['image_size'])
    test_preprocess, test_transform, test_normalization = get_transforms(args['augmentations']['test'], args['augmentations']['normalization'], args['image_size'])

    # Combine augmentations with normalization
    distillation_transform = torchvision.transforms.v2.Compose(distillation_transform.transforms + distillation_normalization.transforms)
    test_transform = torchvision.transforms.v2.Compose(test_transform.transforms + test_normalization.transforms)

    # Create mixup -- TODO may need to change device
    if args['augmentations']['mixup'] is not None:
        mixup_transform = Mixup(args['augmentations']['mixup'], device=args['device'])
    else:
        mixup_transform = None

    ####################
    ##### DATASETS #####
    ####################

    dataset_map = {
        'cifar100' : 'uoft-cs/cifar100',
        'pets' : 'timm/oxford-iiit-pet',
    }
    dataset = load_dataset(dataset_map[args['dataset']])
    num_classes = dataset['train'].features['label'].num_classes

    # Use the  default collator
    data_collator = DefaultDataCollator()

    #########################
    ##### PREPROCESSING #####
    #########################

    # Preprocess the dataset (if applicable)
    def process(examples, processor):    
        examples['image'] = [processor(img.convert('RGB')) for img in examples['image']]
        return examples
    
    # Determine the original columns in each
    distillation_columns = [cn for cn in dataset['train'].column_names if cn not in ['image', 'label']]
    test_columns = [cn for cn in dataset['test'].column_names if cn not in ['image', 'label']]

    distillation_preprocessor = partial(process, processor=distillation_preprocess)
    test_preprocessor = partial(process, processor=test_preprocess)

    # Apply preprocessing to the datasets
    distillation_dataset = dataset['train'].map(distillation_preprocessor, batched=True, remove_columns=distillation_columns)
    test_dataset = dataset['test'].map(test_preprocessor, batched=True, remove_columns=test_columns)

    ##############################
    ##### DATA AUGMENTATIONS #####
    ##############################

    def augment(examples, augmentations):
        examples['pixel_values'] = [augmentations(img) for img in examples['image']]
        del examples['image']
        return examples

    distillation_augmentations = partial(augment, augmentations=distillation_transform)
    test_augmentations = partial(augment, augmentations=test_transform)

    # Add the data augmentation
    distillation_dataset = distillation_dataset.with_transform(distillation_augmentations)
    test_dataset = test_dataset.with_transform(test_augmentations)

    #################################
    ##### TEACHER NETWORK SETUP #####
    #################################

    teacher_network_map = {
        'resnet50': partial(AutoConfig.from_pretrained, 'timm/resnet50.tv2_in1k', num_labels=num_classes),
        'vit_sm': partial(TimmWrapperConfig, architecture='vit_small_patch16_224_ls', num_labels=num_classes),
        'convnext_tiny': partial(AutoConfig.from_pretrained, 'timm/convnext_tiny.fb_in1k', num_labels=num_classes),
    }

    student_network_map = {
        'resnet18': partial(AutoConfig.from_pretrained, 'timm/resnet18.tv_in1k', num_labels=num_classes),
        'resnet_milli': partial(TimmWrapperConfig, architecture='resnet_milli', num_labels=num_classes),
        'efficientvit_m4': partial(AutoConfig.from_pretrained, 'timm/efficientvit_m4.r224_in1k', num_labels=num_classes),
        'mobilenetv4': partial(AutoConfig.from_pretrained, 'timm/mobilenetv4_conv_medium.e500_r224_in1k', num_labels=num_classes),
    }

    teacher_network = AutoModelForImageClassification.from_config(teacher_network_map[args['teacher']['name']]())
    student_network = AutoModelForImageClassification.from_config(student_network_map[args['student']['name']]())

    # Output some info about the networks
    teacher_params = sum(p.numel() for p in teacher_network.parameters() if p.requires_grad)
    print(f'Teacher #Params: {teacher_params}')
    student_params = sum(p.numel() for p in student_network.parameters() if p.requires_grad)
    print(f'Student #Params: {student_params}')

    ##########################
    ##### INITIALIZATION #####
    ##########################

    # Pretrained weights for the teacher
    network_dict = torch.load(f'{args["network_dir"]}/{args["teacher"]["pretrained"]}', map_location='cpu', weights_only=True)
    teacher_network.load_state_dict(network_dict, strict=True)

    # First, random initialization to the student
    init_network(student_network, name=args['student']['name'], **args['weights'])

    ###########################################
    ##### MAKE ADJUSTMENTS TO THE STUDENT #####
    ###########################################

    ##############################################
    ### NORMALIZATION ############################
    ##############################################

    # Determine the normalization layer type in the student network
    norm_type = get_normalization_type(student_network)

    # Adjust (via scaling) the normalization layer affine transformation weights in the student network
    init_normalization_layers(student_network, norm_type=norm_type, **args['normalization'])

    ###########################################
    ### FULLY CONNECTED #######################
    ###########################################

    # Get the teacher's FC layer
    teacher_classifier = teacher_network.timm_model.get_classifier()

    # Get the student's FC layer
    student_classifier = student_network.timm_model.get_classifier()

    # Do the student FC initialization
    student_classifier = init_classifier(student_classifier, teacher_classifier, **args['classifier'])    

    ########################################
    ##### CHECK FOR PRETRAINED WEIGHTS #####
    ########################################

    # Optionally load some pretrained weights for the student to begin
    if args['student']['pretrained'] != -1:
        network_dict = torch.load(f'{args["network_dir"]}/{args["student"]["pretrained"]}', map_location='cpu', weights_only=True)
        student_network.load_state_dict(network_dict, strict=False)

    ####################################
    ##### SAVE THE STARTING POINTS #####
    ####################################

    # Save the initial weights of the student for later analysis
    student_network.save_pretrained(f'{args["network_dir"]}/{args["name"]}/{args["name"]}_0.pt')

    #########################
    ##### LOSS FUNCTION #####
    #########################

    if args['training']['loss']['function'] == 'kl':
        output_loss_function = KLDivergence(temperature=args['training']['loss']['temperature'], logit_normalization=args['training']['loss']['logit_normalization'])
    elif args['training']['loss']['function'] == 'ce':
        output_loss_function = SoftCrossEntropy(temperature=args['training']['loss']['temperature'], logit_normalization=args['training']['loss']['logit_normalization'])

    # TODO intermediate loss functions - gap, other layers, etc.

    ####################################
    ##### OPTIMIZER + LR SCHEDULER #####
    ####################################

    parameter_group_fn = partial(parameter_groups, **args['training']['optimizer'])
    optimizer = create_optimizer_v2(student_network, args['training']['optimizer']['name'], param_group_fn=parameter_group_fn)

    lr_max = max([h['lr'] for _, h in args['training']['optimizer'].items() if isinstance(h, dict)])
    scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args['training']['num_epochs'], step_on_epochs=True, min_lr=(lr_max / args['training']['optimizer']['min_lr_reduction']), warmup_lr=(lr_max / args['training']['optimizer']['warmup_lr_reduction']), warmup_epochs=args['training']['warmup_epochs'], warmup_prefix=False)

    # if args['training']['warmup_epochs'] > 0:
    #     warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=(1 / args['training']['warmup_epochs']), total_iters=args['training']['warmup_epochs'])
    #     primary = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args['training']['num_epochs'] - args['training']['warmup_epochs']), eta_min=(lr_max / args['training']['optimizer']['min_lr_reduction']))
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, primary], milestones=[args['training']['warmup_epochs']])
    # else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['training']['num_epochs'], eta_min=(lr_max / args['training']['optimizer']['min_lr_reduction']))

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

    checkpoint_saver = CheckpointSaver(args['name'], save_frequency=args['save_frequency'], save_thresholds=args['save_thresholds'])

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
              output_loss_function=output_loss_function,
              mixup_transform=mixup_transform,
              args=distillation_args,
              train_dataset=distillation_dataset,
              eval_dataset=test_dataset,
              data_collator=data_collator,
              compute_metrics=compute_metrics,
              optimizers=(optimizer, scheduler))

    t = distiller.evaluate(test_dataset)

    distiller.train()

    t = distiller.evaluate(test_dataset)

    wandb.finish()

    """
    # Create an output file in addition to the wandb logging
    output_log = []

    # Identify the number of training, validation, and test examples
    num_train_instances = len(distillation_dataset)
    num_test_instances = len(test_dataset)

    # Identify the number of training, validation, and test batches
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # Checkpoint counter
    checkpoint = 0

    # Training
    for epoch in range(args['training']['num_epochs']):
        
        # Print out the epoch number
        print(f'Epoch {epoch}:')

        # Get the start time
        start_time = time.time()
        
        # Prepare for training by enabling gradients
        student_network.train()
        teacher_network.eval()
        torch.set_grad_enabled(True)
        
        # Create the lists for keeping track of desired values
        training_loss = []

        teacher_training_softmaxes = []
        student_training_softmaxes = []
        student_testing_softmaxes = []

        training_labels = []
        testing_labels = []
        
        # Iterate over the TRAINING batches
        for batch_num, (images, labels) in enumerate(train_dataloader):

            # If transforms should be done on CUDA
            if 'cuda' in args['transform_device']:
                images = images.to(args['device'])
                labels = labels.to(args['device'])

            # Perform batch-wise transforms -- PyTorch sucks and has the same exact transformation applied to all images in the batch (i.e., SLOW)
            images = torch.stack([collate_distillation_transform(img) for img in images])

            # Perform mixup if desired
            if mixup_transform is not None:
                images, _ = mixup_transform(images, labels)
            
            # Move the images and labels to the device
            if not images.is_cuda:
                images = images.to(args['device'])
                labels = labels.to(args['device'])
            
            # Get teacher logits
            with torch.no_grad():
                teacher_logits = teacher_network(images)

            # Zero the previous gradients
            optimizer.zero_grad(set_to_none=True)
                    
            # Forward propagation
            student_logits = student_network(images)
            
            # Compute loss
            loss = loss_func(teacher_logits, student_logits)
            
            # Backward propagation
            loss.backward()

            # Optionally do gradient clipping
            if args['training']['gradient_clip']: 
                nn.utils.clip_grad_norm_(student_network.parameters(), args['training']['gradient_clip'], norm_type=2.0)
            
            # Adjust weights
            optimizer.step()
  
            # Compute softmax on the logits
            teacher_softmax = F.softmax(teacher_logits, dim=1)
            student_softmax = F.softmax(student_logits, dim=1)

            # Accumulate values from training
            training_loss.append(loss.item())
            teacher_training_softmaxes.append(teacher_softmax.detach())
            student_training_softmaxes.append(student_softmax.detach())
            training_labels.append(labels.detach())
            
            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Epoch {epoch}: {batch_num + 1}/{num_train_batches}', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 

        # Take a LR scheduler step
        scheduler.step()

        # # Take a mixup scheduler step
        # if mixup_scheduler is not None:
        #     mixup_scheduler.step()

        # Disable computing gradients
        student_network.eval()
        torch.set_grad_enabled(False)
        
        # Iterate over the TEST batches
        with torch.no_grad():
            for batch_num, (images, labels) in enumerate(test_dataloader):

                # If transforms should be done on CUDA
                if 'cuda' in args['transform_device']:
                    images = images.to(args['device'])
                    labels = labels.to(args['device'])

                # Perform batch-wise transforms -- PyTorch sucks and has the same exact transformation applied to all images in the batch (i.e., SLOW)
                images = torch.stack([collate_test_transform(img) for img in images])
                
                # Send images and labels to compute device
                if not images.is_cuda:
                    images = images.to(args['device'])
                    labels = labels.to(args['device'])
                
                # Forward propagation
                logits = student_network(images)

                # Compute the entropy through the teacher model
                softmax = F.softmax(logits, dim=1)

                # Record the actual and predicted labels for the instance
                student_testing_softmaxes.append(softmax.detach())
                testing_labels.append(labels.detach())

                # Give epoch status update
                print(' ' * 100, end='\r', flush=True) 
                print(f'Testing: {batch_num + 1}/{num_test_batches}', end='\r', flush=True)
            
            # Clear the status update message
            print(' ' * 100, end='\r', flush=True) 

        # Aggregate all lists
        training_loss = torch.tensor(training_loss).cpu()
        teacher_training_softmaxes = torch.cat(teacher_training_softmaxes).cpu()
        student_training_softmaxes = torch.cat(student_training_softmaxes).cpu()
        student_testing_softmaxes = torch.cat(student_testing_softmaxes).cpu()
        training_labels = torch.cat(training_labels).cpu()
        testing_labels = torch.cat(testing_labels).cpu()

        # Determine average train loss
        training_loss = (sum(training_loss) / num_train_batches).item()

        # Determine predictions
        _, teacher_training_predictions = torch.max(teacher_training_softmaxes, dim=1)
        _, student_training_predictions = torch.max(student_training_softmaxes, dim=1)
        _, student_testing_predictions = torch.max(student_testing_softmaxes, dim=1)

        # Calculate entropy per example
        teacher_training_entropy = entropy(teacher_training_softmaxes)
        student_training_entropy = entropy(student_training_softmaxes)
        student_testing_entropy = entropy(student_testing_softmaxes)

        # Calculate the global average and standard deviation of entropy, and the max entropy (of an individual instance)
        teacher_training_entropy_mean = torch.mean(teacher_training_entropy)
        teacher_training_entropy_std = torch.std(teacher_training_entropy)
        teacher_training_entropy_max = torch.max(teacher_training_entropy)
        teacher_training_entropy_min = torch.min(teacher_training_entropy)

        student_training_entropy_mean = torch.mean(student_training_entropy)
        student_training_entropy_std = torch.std(student_training_entropy)
        student_training_entropy_max = torch.max(student_training_entropy)
        student_training_entropy_min = torch.min(student_training_entropy)

        student_testing_entropy_mean = torch.mean(student_testing_entropy)
        student_testing_entropy_std = torch.std(student_testing_entropy)
        student_testing_entropy_max = torch.max(student_testing_entropy)
        student_testing_entropy_min = torch.min(student_testing_entropy)

        # Calculate train accuracy (did the student predict the same as the teacher)
        train_accuracy = (sum(student_training_predictions == teacher_training_predictions) / len(student_training_predictions)).item()

        # Calculate the test accuracy (did the student get the right prediction)
        test_accuracy = metrics.accuracy_score(testing_labels, student_testing_predictions)

        # Print metrics
        print(f'Training Loss: {training_loss : 0.6f}')

        print(f'Train Accuracy: {train_accuracy * 100.0 : 0.3f}')
        print(f'Test Accuracy: {test_accuracy * 100.0 : 0.3f}')

        print(f'Teacher Training Entropy: {teacher_training_entropy_mean : 0.4f} +/- {teacher_training_entropy_std : 0.4f} ({teacher_training_entropy_max : 0.4f}, {teacher_training_entropy_min : 0.4f})')
        print(f'Student Training Entropy: {student_training_entropy_mean : 0.4f} +/- {student_training_entropy_std : 0.4f} ({student_training_entropy_max : 0.4f}, {student_training_entropy_min : 0.4f})')
        print(f'Student Testing Entropy: {student_testing_entropy_mean : 0.4f} +/- {student_testing_entropy_std : 0.4f} ({student_testing_entropy_max : 0.4f}, {student_testing_entropy_min : 0.4f})')

        # PyTorch synchronize if we care about precise timing
        # torch.cuda.synchronize()

        # Get the end time
        end_time = time.time()
        print(f'Epoch {epoch} Time: {round(end_time - start_time, 6)}')

        # Add to table
        epoch_results = {
            'train_loss' : round(training_loss, 6),
            'student_train_accuracy' : round(train_accuracy * 100, 6),
            'student_test_accuracy' : round(test_accuracy  * 100, 6),
            'teacher_train_entropy_mean' : round(teacher_training_entropy_mean.item(), 6),
            'teacher_train_entropy_std' : round(teacher_training_entropy_std.item(), 6),
            'teacher_train_entropy_max' : round(teacher_training_entropy_max.item(), 6),
            'teacher_train_entropy_min' : round(teacher_training_entropy_min.item(), 6),
            'student_train_entropy_mean' : round(student_training_entropy_mean.item(), 6),
            'student_train_entropy_std' : round(student_training_entropy_std.item(), 6),
            'student_train_entropy_max' : round(student_training_entropy_max.item(), 6),
            'student_train_entropy_min' : round(student_training_entropy_min.item(), 6),
            'student_test_entropy_mean' : round(student_testing_entropy_mean.item(), 6),
            'student_test_entropy_std' : round(student_testing_entropy_std.item(), 6),
            'student_test_entropy_max' : round(student_testing_entropy_max.item(), 6),
            'student_test_entropy_min' : round(student_testing_entropy_min.item(), 6),
            'epoch_time': round(end_time - start_time, 6)
        }

        # Add per-class entropy?
        # Add per-class accuracy of student?

        # Weights & Biases logging
        wandb.log(epoch_results)
        output_log.append(epoch_results)

        # Save model weights by some frequency, also save them to wandb
        if (epoch + 1) % args['save_frequency'][checkpoint] == 0:
            torch.save(student_network.state_dict(), f'{args["student"]["network_dir"]}/{args["base_name"]}/{args["name"]}_{epoch}.pt')

            # Move the checkpoint counter if we reach certain points
            if epoch == 500:
                checkpoint += 1
            elif epoch == 1000:
                checkpoint += 1

        # Separate epoch outputs
        print()
    
    # Save the final model weights
    torch.save(student_network.state_dict(), f'{args["student"]["network_dir"]}/{args["name"]}_final.pt')
    wandb.save(f'{args["student"]["network_dir"]}/{args["name"]}_final.pt')

    print('-' * 100)
    print('-' * 100)
    print('-' * 100)

    # Arrange the custom output log to something usable
    output_log = {key: [values[key] for values in output_log] for key in output_log[0]}
    output_keys = list(output_log.keys())
    output_values = [list(values)[0] for values in zip(output_log.values())]

    # Save custom output log to file
    with open(f'{args["results_dir"]}/{args["name"]}.txt', 'w') as f:
        json.dump(args, f, indent=4)
        f.write('\n\n')
        f.write(','.join(output_keys))
        f.write('\n')
        for values in zip(*output_values):
            values = [str(num) for num in values]
            f.write(','.join(values))
            f.write('\n')

    # Finish wandb run
    wandb.finish()"
    """

# EOF      