import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BIM:
    """
    L-inf Basic Iterative Method that has a constant epsilon and will iterate until the desired label is reached.
    :param network: 
    :param epsilon: 
    :param random_init:
    :param reduction: 
    :return: an adversarial example
    """
    def __init__(self, network, epsilon, num_iterations, alpha=1, reduction='none', targeted=False, include_pair=False, bold_driver=False, softmax_threshold=0.9, random_init=None, include_all=False, **kwargs):
        self.name = 'bim'
        
        # The network we are attacking
        self.network = network
        
        # Attack epsilon budget, adjusted for [0, 1] space
        self.epsilon = (epsilon / 255)

        # Number of maximum attack iterations
        self.num_iterations = num_iterations

        # Attack step size, adjusted for [0, 1] space
        self.alpha = (alpha / 255)

        # How to reduce the cross entropy loss function
        self.reduction = reduction 
        self.loss_function = nn.CrossEntropyLoss(reduction=reduction)
        
        # Whether to include the "pair" example that comes right before the final example
        self.include_pair = include_pair
        
        # Flag for if we are doing a targeted attack
        self.targeted = targeted
        
        # Flag for if we are doing this bold driver style
        self.bold_driver = bold_driver
        
        # Upper limit softmax threshold for stopping condition
        # Only matters when doing bold driver attack
        self.softmax_threshold = softmax_threshold

        # Flag for if we are going to include all steps of the attack
        self.include_all = include_all
        
        # Flag for if we want to do a small random perturbation at the start
        # True: PGD
        # False: BIM
        self.random_init = random_init is not None
        self.random_init_range = (random_init / 255) if self.random_init else None

        # Print a warning message if bold_driver=True and include_all=True
        if self.bold_driver and self.include_all:
            print('bold_driver=True and include_all=True, you can only select one therefore this attack will default to the bold driver attack')

    def __repr__(self):
        return f'BIM: epsilon={self.epsilon * 255 : 0.2f}, num_iterations={self.num_iterations}, alpha={self.alpha * 255 : 0.2f}, targeted={self.targeted}'

    def bare_minimum_attack(self, images, labels):
        # TODO need to update this with names and other stuff

        # Ensure the network is in eval mode
        self.network.eval()

        # The tensor of the final adversarial images
        final_adversarial_images = torch.zeros(images.shape)
        
        # If we want to include the "adversarial" images just before the successful ones
        if self.include_pair:
            # Always stores the adversarial images from the previous iteration
            previous_adversarial_images = None
            
            # The tensor of the "adversarial" images just before the successful ones
            final_adversarial_images_pair = torch.zeros(images.shape)

            # Store the final predicted label for each successful attack TODO TODO
            final_predictions_pair = torch.zeros(len(final_adversarial_images_pair), dtype=torch.int16)
            
        # Store the number of iterations required for each successful attack
        final_iteration = torch.zeros(len(final_adversarial_images), dtype=torch.int16)
        
        # Store the final predicted label for each successful attack
        initial_predictions = None
        
        # Store the final predicted label for each successful attack
        final_predictions = torch.zeros(len(final_adversarial_images), dtype=torch.int16)
        
        # Stores whether an image (in the batch) has been successfully attacked or not
        successful = torch.zeros(len(final_adversarial_images), dtype=bool)

        # Zero the gradients in the model
        self.network.zero_grad(set_to_none=True)

        # Allocate the adversarial images and other information
        adversarial_images = images.clone().requires_grad_(True)

        # Compute L_inf epsilon ball
        lower_bound = images - self.epsilon
        upper_bound = images + self.epsilon

        # Determine the upper and lower bound of the adversarial perturbation
        lower_bound = torch.clamp(lower_bound, max=1, min=0)
        upper_bound = torch.clamp(upper_bound, max=1, min=0)
            
        # Run through pgd iterations
        for iteration in range(self.num_iterations):

            # Clone the existing adversarial example, for some reason this is needed
            _adversarial_images = adversarial_images.clone().requires_grad_(True)

            # Feed current state of adversarial image into the network and compute loss
            logits = self.network(_adversarial_images)
            loss = self.loss_function(logits, labels)

            # Get the initial prediction of the natural example
            if initial_predictions is None:
                _, initial_predictions = torch.max(logits, 1)
                initial_predictions = initial_predictions.detach().cpu()

            # Remember the previous adversarial examples
            if self.include_pair:
                # Get the prediction of the current adversarial example
                _, previous_predictions = torch.max(logits, 1)

                previous_adversarial_images = _adversarial_images.detach().cpu()
                previous_adversarial_predictions = previous_predictions.detach().cpu()

            # Get the gradient
            if self.reduction == 'none':
                gradients = torch.autograd.grad(loss, _adversarial_images, grad_outputs=torch.ones_like(loss), only_inputs=True)[0]
            else:
                gradients = torch.autograd.grad(loss, _adversarial_images)[0]

            # Adjust the adversarial example and project into L-infinity
            with torch.no_grad():
                # If we are doing a targeted attack, minimize w.r.t. the label
                # else maximize with respect to the ground truth label
                if self.targeted:
                    adversarial_images -= self.alpha * gradients.sign()
                else:
                    adversarial_images += self.alpha * gradients.sign()

                # Adjust image according to bounds
                adversarial_images = torch.where(adversarial_images < lower_bound, lower_bound, adversarial_images).requires_grad_(True)
                adversarial_images = torch.where(adversarial_images > upper_bound, upper_bound, adversarial_images).requires_grad_(True)

            # Attack iteration has completed, do one more pass through the network and determine if we keep or save the adversarial image and whether or not we are done
            logits = self.network(adversarial_images)

            # Get the prediction of the current adversarial example
            _, predictions = torch.max(logits, 1)
            
            # Determine if the current adversarial example is successful or not
            if self.targeted:
                s = (predictions == labels).detach().cpu()
            else:
                s = (predictions != labels).detach().cpu()
                
            # Detach the predictions and move them to cpu
            predictions = predictions.detach().cpu()

            # Detach and cpu the adversarial image
            current_adversarial_images = adversarial_images.detach().cpu()

            # Determine the condition for adding the adversarial example into the final ones
            cond = (s & (torch.logical_not(successful))).bool()

            # Keep track of the official adversarial images and where they were successful at
            final_adversarial_images = torch.where(cond[:, None, None, None], current_adversarial_images, final_adversarial_images)
            
            # If we want to include the "adversarial" images just before the successful ones
            if self.include_pair:
                final_adversarial_images_pair = torch.where(cond[:, None, None, None], previous_adversarial_images, final_adversarial_images_pair)
                final_predictions_pair = torch.where(cond, previous_adversarial_predictions, final_predictions_pair)
            
            # Keep track of the final epsilon values and their predictions
            final_iteration = torch.where(cond, iteration + 1, final_iteration)
            final_predictions = torch.where(cond, predictions, final_predictions)

            # Signify if certain examples have already reached success
            successful = successful | s

            # Determine if we are done (all adversarial examples are successful)
            if successful.all():
                break
            
        # If there are any remaining images that aren't successful
        final_adversarial_images = torch.where(successful[:, np.newaxis, np.newaxis, np.newaxis], final_adversarial_images, current_adversarial_images)
        
        # Also include their pairs (optionally)
        if self.include_pair:
            final_adversarial_images_pair = torch.where(successful[:, np.newaxis, np.newaxis, np.newaxis], final_adversarial_images_pair, previous_adversarial_images)
            final_predictions_pair = torch.where(successful, previous_adversarial_predictions, final_predictions_pair)
        
        # The final epsilons / label predictions for the adversarial examples
        final_iteration = torch.where(successful, final_iteration, 0)
        final_predictions = torch.where(successful, final_predictions, predictions)

        # Construct the return tensor depending on if we are returning the pair or not
        if self.include_pair:
            return (final_adversarial_images.float().to(images.device),
                    final_adversarial_images_pair.float().to(images.device),
                    initial_predictions, final_predictions, final_predictions_pair, final_iteration)
        else:
            return (final_adversarial_images.float().to(images.device),
                    initial_predictions, final_predictions, final_iteration)
 
    def bold_driver_attack(self, images, labels):
        # Condition 1: image has not crossed the boundary
        # -> Increase the step size
        
        # Condition 2: image has crossed, but softmax is too high
        # -> Decrease the step size
        
        # Condition 3? softmax is too low

        # Condition 3? image has crossed the boundary, but the pair softmax is too high
        # -> Decrease the step size
    
        # Determines how large of an increase to take on "success" and decrease on "failure"
        step_increase = 1.1
        step_decrease = 0.5
        
        # Create a step size for each image in the batch
        step = torch.tensor([self.alpha]).repeat(len(images)).to(images.device)

        # The tensor of the final adversarial images
        final_adversarial_images = torch.zeros(images.shape)
        
        # If we want to include the "adversarial" images just before the successful ones
        if self.include_pair:
            # Always stores the adversarial images from the previous iteration
            previous_adversarial_images = None
            
            # The tensor of the "adversarial" images just before the successful ones
            final_adversarial_images_pair = torch.zeros(images.shape)

            # Store the final predicted label for each successful attack
            final_predictions_pair = torch.zeros(len(final_adversarial_images_pair), dtype=torch.uint8)
        
        # Store the final predicted label for each successful attack
        initial_predictions = None
        
        # Store the final predicted label for each successful attack
        final_predictions = torch.zeros(len(final_adversarial_images), dtype=torch.uint8)
        
        # Stores whether an image (in the batch) has been successfully attacked or not
        successful = torch.zeros(len(final_adversarial_images), dtype=bool)

        # Stores the number of iterations it took for an attack to be successful
        num_iterations = torch.ones(len(final_adversarial_images), dtype=torch.uint8) * self.num_iterations

        # Allocate the adversarial images and other information
        adversarial_images = images.clone().requires_grad_(True)

        # Compute L_inf epsilon ball
        lower_bound = images - self.epsilon
        upper_bound = images + self.epsilon

        # Determine the upper and lower bound of the adversarial perturbation
        lower_bound = torch.clamp(lower_bound, max=1, min=0)
        upper_bound = torch.clamp(upper_bound, max=1, min=0)
        
        # Add some random noise if we have a random initialization of the perturbation mask, this is a PGD attack
        if self.random_init:
            # Apply the initial random mask
            initial_perturbation = torch.empty_like(adversarial_images).uniform_(-self.random_init_range, self.random_init_range).requires_grad_(True)
            adversarial_images = adversarial_images + initial_perturbation

            # Clip at the bounds
            adversarial_images = torch.where(adversarial_images < lower_bound, lower_bound, adversarial_images)
            adversarial_images = torch.where(adversarial_images > upper_bound, upper_bound, adversarial_images)

        # Make sure the environment is set up appropriately for the attack
        self.network.eval()
        torch.set_grad_enabled(True)
            
        # Run through iterations
        for iteration in range(self.num_iterations):

            # Zero the gradients in the model
            self.network.zero_grad(set_to_none=True)

            # Clone the existing adversarial example, for some reason this is needed
            _adversarial_images = adversarial_images.clone().requires_grad_(True)

            # Feed current state of adversarial image into the network and compute loss
            logits = self.network(_adversarial_images)
            loss = self.loss_function(logits, labels)

            # Get the initial prediction of the natural example
            if initial_predictions is None:
                _, initial_predictions = torch.max(logits, 1)
                initial_predictions = initial_predictions.detach().cpu()

            # Remember the previous adversarial examples
            # Get the prediction of the current adversarial example
            _, previous_predictions = torch.max(logits, dim=1)
            previous_softmaxes = F.softmax(logits, dim=1).detach()
            previous_softmaxes, _ = torch.max(previous_softmaxes, dim=1)

            previous_adversarial_images = _adversarial_images.clone()
            previous_adversarial_predictions = previous_predictions.detach().cpu()

            # Get the gradient
            if self.reduction == 'none':
                gradients = torch.autograd.grad(loss, _adversarial_images, grad_outputs=torch.ones_like(loss), only_inputs=True)[0]
            else:
                gradients = torch.autograd.grad(loss, _adversarial_images)[0]

            # Adjust the adversarial example and project into L-infinity
            with torch.no_grad():
                # If we are doing a targeted attack, minimize w.r.t. the label
                # else maximize with respect to the ground truth label
                if self.targeted:
                    adversarial_images -= step[:, None, None, None] * gradients.sign()
                else:
                    adversarial_images += step[:, None, None, None] * gradients.sign()

                # Adjust image according to bounds
                adversarial_images = torch.where(adversarial_images < lower_bound, lower_bound, adversarial_images).requires_grad_(True)
                adversarial_images = torch.where(adversarial_images > upper_bound, upper_bound, adversarial_images).requires_grad_(True)

            # One attack iteration completed
            # Do another pass through to determine whether to keep the current example based on conditions
            # and how to adjust the alpha (bold driver) if necessary
            logits = self.network(adversarial_images)
            
            # Detach and cpu the adversarial image
            current_adversarial_images = adversarial_images.detach().cpu()

            # Get the prediction of the current adversarial example
            softmaxes = F.softmax(logits, dim=1)
            argmax_softmax, _ = torch.max(softmaxes, dim=1)
            _, predictions = torch.max(logits, dim=1)
            
            # Determine if the current adversarial example has crossed the boundary -- CONDITION 1
            if self.targeted:
                crossed_boundary_condition = (predictions == labels).detach().cpu()
            else:
                crossed_boundary_condition = (predictions != labels).detach().cpu()
                
            # Determine if the argmax softmax is below (or potentially above) the desired threshold -- CONDITION 2
            softmax_condition = (argmax_softmax < self.softmax_threshold).detach().cpu()

            # Remember the previous adversarial examples
            if self.include_pair:
                # Determine if the argmax softmax of the previous example is below (or potentially above) the desired threshold -- CONDITION 3
                softmax_pair_condition = (previous_softmaxes < self.softmax_threshold).detach().cpu()
            else:
                softmax_pair_condition = torch.ones_like(softmax_condition, dtype=bool)
                
            # Detach the predictions and move them to cpu
            predictions = predictions.detach().cpu()
            
            #######################################################################
            # At this point we have the previous and current adversarial examples #
            #######################################################################
            
            ### Saving the final adversarial examples where the stopping condition was satisfied
            
            # Determine if the adversarial example has satisified all stopping conditions
            stopping_conditions_satisfied = (crossed_boundary_condition & softmax_condition & softmax_pair_condition)

            # Determine if this example has just satisfied all conditions or it has done so previously
            add_final_condition = (stopping_conditions_satisfied & torch.logical_not(successful)).bool()

            # Keep track of the official adversarial images and where they were successful at
            final_adversarial_images = torch.where(add_final_condition[:, None, None, None], current_adversarial_images, final_adversarial_images)
            
            # If we want to include the "adversarial" images just before the successful ones
            if self.include_pair:
                final_adversarial_images_pair = torch.where(add_final_condition[:, None, None, None], previous_adversarial_images.detach().cpu(), final_adversarial_images_pair)
                final_predictions_pair = torch.where(add_final_condition, previous_adversarial_predictions, final_predictions_pair)
            
            # Keep track of the final predictions
            final_predictions = torch.where(add_final_condition, predictions, final_predictions)

            # Signify if certain examples have already reached success
            successful = successful | stopping_conditions_satisfied
            
            ### If the image didn't cross the boundary, increase the step size
            step = torch.where(torch.logical_not(crossed_boundary_condition).to(step.device), step * step_increase, step)
            
            ### If the image did cross the boundary but the softmax was too high, decrease the step size
            too_high_softmax_condition = (crossed_boundary_condition & torch.logical_not(softmax_condition))
            step = torch.where(too_high_softmax_condition.to(step.device), step * step_decrease, step)

            ### If the image did cross the boundary, the softmax of the result was in the right range, but the pair's softmax was too high
            if self.include_pair:
                too_high_softmax_pair_condition = (crossed_boundary_condition & softmax_condition & torch.logical_not(softmax_pair_condition))
                step = torch.where(too_high_softmax_pair_condition.to(step.device), step * step_decrease, step)

            # Remember the number of iterations it took for an example to be successful
            num_iterations = torch.where(add_final_condition, iteration, num_iterations)
            
            # If the image crossed the boundary but the softmax was too high, revert the image to the previous step
            if self.include_pair:
                revert_images_condition = (too_high_softmax_condition | too_high_softmax_pair_condition).to(step.device)[:, None, None, None]
            else:
                revert_images_condition = too_high_softmax_condition.to(step.device)[:, None, None, None]
            adversarial_images = torch.where(revert_images_condition, previous_adversarial_images, adversarial_images)

            # Determine if we are done (all adversarial examples are successful)
            if successful.all():
                break
            
        ### Clean up after all iterations have been performed, these examples may not be taken anyways
        
        # If there are any remaining images that aren't successful
        final_adversarial_images = torch.where(successful[:, None, None, None], final_adversarial_images, current_adversarial_images)
        
        # Also include their pairs (optionally)
        if self.include_pair:
            final_adversarial_images_pair = torch.where(successful[:, None, None, None], final_adversarial_images_pair, previous_adversarial_images.detach().cpu())
            final_predictions_pair = torch.where(successful, final_predictions_pair, previous_adversarial_predictions)
        
        # The final epsilons / label predictions for the adversarial examples
        final_predictions = torch.where(successful, final_predictions, predictions)

        # Construct the return tensor depending on if we are returning the pair or not
        if self.include_pair:
            return (final_adversarial_images.float().to(images.device),
                    final_adversarial_images_pair.float().to(images.device),
                    initial_predictions, final_predictions, final_predictions_pair, num_iterations)
        else:
            return (final_adversarial_images.float().to(images.device),
                    initial_predictions, final_predictions, num_iterations)
 
    def __call__(self, images, labels):   
        # An attack to accomplish certain goals using a bold driver based step size     
        if self.bold_driver:
            return self.bold_driver_attack(images, labels)
        else:
            return self.bare_minimum_attack(images, labels)