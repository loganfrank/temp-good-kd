from .determinism import (
    make_complex, 
    make_deterministic
)

from .transforms import (
    get_transforms, 
    Mixup
)

from .initializations import (
    init_network
)

from .kl_divergence import (
    KLDivergence
)

import json
import argparse
import yaml
import re

def get_config():
    parser = argparse.ArgumentParser(description='Arguments for script')

    # Normal parameters
    parser.add_argument('--config', default='', type=str, help='path to config file')
    parser.add_argument('--seed', default='134895', type=str, help='rng seed for reproducability')

    # Pull out the command-line arguments that were unknown
    _, unknown_args = parser.parse_known_args()

    # Add the unknown arguments to the argparser
    for arg in unknown_args:
        if arg.startswith(('--')):
            parser.add_argument(arg.split('=')[0], type=infer_input)

    # Parse out all of the arguments now that the new ones were added
    args = vars(parser.parse_args())

    # Check if user provided a config file
    assert args['config'] != '', 'no config file provided'

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

    # Output the args to console
    print(json.dumps(config, indent=4))

    return config
        
def infer_input(x):
    """
    Will determine the appropriate type for input x.

    :param x: the input for which we will determine the type / parse it appropriately
    """


    if x in ('none', 'None', 'NONE'):
        x = None
    elif x == '[]':
        x = []
    elif (len(x.split(',')) > 1):
        x = [yaml.safe_load(s) for s in x.split(',')]
    else:
        x = yaml.safe_load(x)
    
    return x

########################################################################
##### PARAMETER GROUP FUNCTIONS FOR DECIDING WHO GETS WEIGHT DECAY #####
########################################################################

# TODO needs some work -- what to do with cls_token, pos_embed, etc. -- timm seems to add them into no_decay group, so I could create a new group for them
# But how to differeniate them from normalization parameters
# Furthermore how to specifically differentiate between normalization and transformation biases
# Could iterate over module then further iterate over parameters in each??? TODO
# TODO incorporate skip list too
def lodi_param_group_fn(network, transformation_weight_hyperparams=None, transformation_bias_hyperparams=None, normalization_weight_hyperparams=None, normalization_bias_hyperparams=None, classifier_weight_hyperparams=None, classifier_bias_hyperparams=None, **kwargs):    

    # Set of parameters that should not have weight decay
    use_no_weight_decay = kwargs.get('use_no_weight_decay', True)
    no_weight_decay = getattr(network, 'no_weight_decay', lambda: set())() if use_no_weight_decay else set()

    # Default for nestrov momentum if not specified
    nesterov = kwargs.get('nesterov', False)

    # Overall default hyperparameters
    default_hyperparams = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0001
    }

    # Set up some defaults
    transformation_weight_hyperparams = transformation_weight_hyperparams if transformation_weight_hyperparams is not None else default_hyperparams
    transformation_bias_hyperparams = transformation_bias_hyperparams if transformation_bias_hyperparams is not None else default_hyperparams
    normalization_weight_hyperparams = normalization_weight_hyperparams if normalization_weight_hyperparams is not None else default_hyperparams
    normalization_bias_hyperparams = normalization_bias_hyperparams if normalization_bias_hyperparams is not None else default_hyperparams
    classifier_weight_hyperparams = classifier_weight_hyperparams if classifier_weight_hyperparams is not None else default_hyperparams
    classifier_bias_hyperparams = classifier_bias_hyperparams if classifier_bias_hyperparams is not None else default_hyperparams

    # Initialize lists for each group
    transformation_weight = []
    transformation_bias = []
    normalization_weight = []
    normalization_bias = []
    classifier_weight = []
    classifier_bias = []

    # Get the classifier module.
    classifier_module = network.get_classifier()
    classifier_param_ids = set(id(p) for p in classifier_module.parameters())

    # Regular expressions to identify normalization and classifier layers
    normalization_pattern = re.compile(r'(bn|norm|ln|gn)', re.IGNORECASE)

    for name, param in network.named_parameters():
        # Skip parameters that are not being optimized.
        if not param.requires_grad:
            print(f'{name} SKIPPED')
            continue

        # Classifier layer
        if id(param) in classifier_param_ids:
            # Parameter belongs to the classifier module.
            if 'bias' in name:
                print(f'{name} added to classifier bias')
                classifier_bias.append(param)
            else:
                print(f'{name} added to classifier weight')
                classifier_weight.append(param)

        # Normalization layers -- TODO could catch transformation biases
        elif normalization_pattern.search(name) or len(param.shape) == 1:
            if 'bias' in name:
                print(f'{name} added to normalization bias')
                normalization_bias.append(param)
            else:
                print(f'{name} added to normalization weight')
                normalization_weight.append(param)

        # Other / Transformation layers
        else:
            # Assume remaining parameters belong to transformation layers.
            if 'bias' in name:
                print(f'{name} added to transformation bias')
                transformation_bias.append(param)
            else:
                print(f'{name} added to transformation weight')
                transformation_weight.append(param)

    # Assemble parameter groups
    param_groups = [
        {"params": transformation_weight, 'nesterov': nesterov, **transformation_weight_hyperparams},
        {"params": transformation_bias, 'nesterov': nesterov, **transformation_bias_hyperparams},
        {"params": normalization_weight, 'nesterov': nesterov, **normalization_weight_hyperparams},
        {"params": normalization_bias, 'nesterov': nesterov, **normalization_bias_hyperparams},
        {"params": classifier_weight, 'nesterov': nesterov, **classifier_weight_hyperparams},
        {"params": classifier_bias, 'nesterov': nesterov, **classifier_bias_hyperparams},
    ]
    
    return param_groups

# Adapted from timm library
def timm_param_group_fn(network, lr=None, momentum=None, weight_decay=None, **kwargs):

    # Set of parameters that should not have weight decay
    use_no_weight_decay = kwargs.get('use_no_weight_decay', True)
    no_weight_decay = getattr(network, 'no_weight_decay', lambda: set())() if use_no_weight_decay else set()

    # Default for nestrov momentum if not specified
    nesterov = kwargs.get('nesterov', False)

    # Overall default hyperparameters
    default_hyperparams = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0001
    }

    # Set up the hyperparameters
    lr = lr if lr is not None else default_hyperparams['lr']
    momentum = momentum if momentum is not None else default_hyperparams['momentum']
    weight_decay = weight_decay if weight_decay is not None else default_hyperparams['weight_decay']

    # Some defaults for the classifier if not specified
    classifier_lr = kwargs.get('classifier_lr', lr)
    classifier_weight_decay = kwargs.get('classifier_weight_decay', weight_decay)

    # Initialize lists for each group
    decay = []
    no_decay = []
    classifier_weight = []
    classifier_bias = []

    # Get the classifier module.
    classifier_module = network.get_classifier()
    classifier_param_ids = set(id(p) for p in classifier_module.parameters())

    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in classifier_param_ids:
            # Parameter belongs to the classifier module.
            if 'bias' in name:
                print(f'{name} added to classifier bias')
                classifier_bias.append(param)
            else:
                print(f'{name} added to classifier weight')
                classifier_weight.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
            print(f'{name} added to NO_decay')
            no_decay.append(param)
        else:
            print(f'{name} added to DECAY')
            decay.append(param)

    # Assemble parameter groups
    param_groups = [
        {"params": decay, 'nesterov': nesterov, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
        {"params": no_decay, 'nesterov': nesterov, 'lr': lr, 'momentum': momentum, 'weight_decay': 0.},
        {"params": classifier_weight, 'nesterov': nesterov, 'lr': classifier_lr, 'momentum': momentum, 'weight_decay': classifier_weight_decay},
        {"params": classifier_bias, 'nesterov': nesterov, 'lr': classifier_lr, 'momentum': momentum, 'weight_decay': 0.},
    ]
    
    return param_groups

# Puts all parameters into a single group
def all_param_group_fn(network, lr=None, momentum=None, weight_decay=None, **kwargs):

    # Default for nestrov momentum if not specified
    nesterov = kwargs.get('nesterov', False)

    # Overall default hyperparameters
    default_hyperparams = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0001
    }

    # Set up the hyperparameters
    lr = lr if lr is not None else default_hyperparams['lr']
    momentum = momentum if momentum is not None else default_hyperparams['momentum']
    weight_decay = weight_decay if weight_decay is not None else default_hyperparams['weight_decay']

    # Assemble parameter groups
    param_groups = [
        {"params": network.parameters(), 'nesterov': nesterov, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
    ]
    
    return param_groups

