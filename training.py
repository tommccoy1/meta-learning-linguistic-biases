import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random

from models import *
from utils import *
from phonology_task_creation import *
from load_data import *

# Compute the loss and accuracy for a single batch
def get_loss(model, batch, criterion):
    correct = 0
    total = 0

    inp, outp = batch
    output, logits = model(inp)
    seq_loss = 0

    # Count how many predicted outputs are correct
    for index, output_guess in enumerate(output):
        if process_output(output_guess) == outp[index]:
            correct += 1
        total += 1


    # Create a tensor for the correct output
    all_seqs = []
    for sequence in outp:
        this_seq = []
        for elt in sequence:
            ind = model.char2ind[elt]
            this_seq.append(ind)
        this_seq.append(model.char2ind["EOS"])
        all_seqs.append(torch.LongTensor(this_seq))

    all_seqs = torch.nn.utils.rnn.pad_sequence(all_seqs)

    # Compute the loss based on the predictions and the correct output
    for index, logit in enumerate(logits):
        if index >= len(all_seqs):
            break
        seq_loss += criterion(logit[0], all_seqs[index])

    # Average the loss over the sequence
    loss = seq_loss / len(logits)

    # Return the loss over the batch, the number of correct predictions,
    # and the total number of predictions
    return loss, correct, total
    

def train_model(model, task, max_epochs=10, lr=0.001, batch_size=100, print_every=1000, patience=10, threshold=None):
    optimizer = torch.optim.Adam(model.params(), lr=lr)
    done = False
    count_since_improved = 0
    best_dev_acc = 0.0

    training_set = batchify_list(task[0], batch_size=batch_size)
    dev_set = batchify_list(task[1], batch_size=batch_size)
    test_set = batchify_list(task[1], batch_size=batch_size)
    vocab = task[2]

    model.set_dicts(vocab)
    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(max_epochs):
        if done:
            epoch -= 1
            break

        for i, batch in enumerate(training_set):
            batch_loss, batch_correct, batch_total = get_loss(model, batch, criterion) 
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % print_every == 0:
                dev_correct = 0
                dev_total = 0

                # Compute the dev loss and dev accuracy
                for batch in dev_set:
                    batch_loss, batch_correct, batch_total = get_loss(model, batch, criterion)
                    dev_correct += batch_correct
                    dev_total += batch_total

                dev_acc = dev_correct * 1.0 / dev_total
                print("Dev accuracy at iteration " + str(i) + ":", dev_acc)
               
                # Determine whether to early stop
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    count_since_improved = 0

                else:
                    count_since_improved += 1

                if count_since_improved >= patience:
                    done = True
                    break

                if threshold is not None:
                    if dev_acc >= threshold:
                        done = True
                        break
 
    return epoch * len(training_set) * batch_size + i * batch_size, best_dev_acc

# Fit a model to a task (i.e., perform one inner loop)
# meta: True if this is being called as part of metatraining; False otherwise
# lr_inner: learning rate
# batch_size: the batch size
def fit_task(model, task, meta=False, train=True, lr_inner=0.01, batch_size=100):
    # This task's training set, test set, and vocab
    training_set = batchify_list(task[0], batch_size=batch_size)
    test_set = batchify_list(task[1], batch_size=batch_size)
    vocab = task[2]

    # Copy the model
    model_copy = model.create_copy(same_var=meta)
    model_copy.set_dicts(vocab)

    # Loss function; ignore_index is to handle padding (so you
    # don't compute loss for the padded parts)
    criterion = nn.NLLLoss(ignore_index=0)


    # Training
    if train:
        for batch in training_set:
            
            # Compute the loss on this batch
            batch_loss, batch_correct, batch_total = get_loss(model_copy, batch, criterion)

            # Backprop the loss; setting create_graph as True enables 
            # double gradients for MAML
            batch_loss.backward(create_graph=meta, retain_graph=True)

            # Update the model's parameters
            for name, param in model_copy.named_params():
                grad = param.grad
                
                # This is to deal with cases where '' is the input
                if grad is None:
                    continue
                
                if meta:
                    model_copy.set_param(name, param - lr_inner * grad)
                else:
                    model_copy.set_param(name, (param - lr_inner * grad).data.requires_grad_())
                
                
    # Testing
    test_correct = 0
    test_total = 0
    test_loss = 0

    # Compute the test loss and test accuracy
    for batch in test_set:
        batch_loss, batch_correct, batch_total = get_loss(model_copy, batch, criterion)
        test_loss += batch_loss
        test_correct += batch_correct
        test_total += batch_total

    # Average the test loss over the number of test batches
    test_loss /= len(test_set)

    # Compute the test accuracy
    test_acc = test_correct * 1.0 / test_total

    # Return the test loss and test accuracy
    return test_loss, test_acc, model_copy



# Perform transfer learning
# model = the model being trained
# train_set =  training set
# dev_set = development set
# max_epochs = maximum number of epochs to run before stopping
# lr_inner = learning rate for the inner loop
# inner_batch_size = batch size for the inner loop (i.e. the number of training examples within a given task to view before updating the temporary copied model's weights
# print_every = number of outer loop iterations to go through (specifically, the number of tasks, not the number of batches) before printing the dev set accuracy
# patience = number of dev set evaluations to allow without any improvement before early stopping
# save_prefix = prefix of the filename where the weights will be saved
def transfer(model, train_set, dev_set, max_epochs=10, lr_inner=0.01, inner_batch_size=100, print_every=100, patience=10, save_prefix="model"):
    done = False
    count_since_improved = 0
    best_dev_acc = 0.0
    
    for _ in range(max_epochs):
        test_loss = 0
        if done:
            break

        for i, t in enumerate(train_set):
            
            # Get the loss for one training task
            _, _, model = fit_task(model, t, meta=False, lr_inner=lr_inner, batch_size=inner_batch_size)

            # Evaluate on the dev set
            if i % print_every == 0:
                dev_acc = average_acc(model, dev_set, lr_inner=lr_inner, batch_size=inner_batch_size)
                print("Dev accuracy at iteration " + str(i) + ":", dev_acc)

                # Determine whether to early stop
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    count_since_improved = 0

                    # Save the model if its performance has improved
                    torch.save(model.state_dict(), save_prefix + ".weights")

                else:
                    count_since_improved += 1

                if count_since_improved >= patience:
                    done = True
                    break
 

# Perform MAML
# model = the model being trained
# train_set =  training set
# dev_set = development set
# max_epochs = maximum number of epochs to run before stopping
# lr_inner = learning rate for the inner loop
# lr_outer = learning rate for the outer loop
# outer_batch_size = batch size for the outer loop (i.e., the number of tasks to train on before updating the model's weights)
# inner_batch_size = batch size for the inner loop (i.e. the number of training examples within a given task to view before updating the temporary copied model's weights
# print_every = number of outer loop iterations to go through (specifically, the number of tasks, not the number of batches) before printing the dev set accuracy
# patience = number of dev set evaluations to allow without any improvement before early stopping
# save_prefix = prefix of the filename where the weights will be saved
def maml(model, train_set, dev_set, max_epochs=10, lr_inner=0.01, lr_outer=0.001, outer_batch_size=1, inner_batch_size=100, print_every=100, patience=10, save_prefix="model"):
    optimizer = torch.optim.Adam(model.params(), lr=lr_outer)
    done = False
    count_since_improved = 0
    best_dev_acc = 0.0
    
    for _ in range(max_epochs):
        test_loss = 0
        if done:
            break

        for i, t in enumerate(train_set):
            
            # Get the loss for one training task
            task_loss, task_acc, _ = fit_task(model, t, meta=True, lr_inner=lr_inner, batch_size=inner_batch_size)
            test_loss += task_loss
            
            # Compute the gradient on the test loss, backpropagate it, and update the model's weights
            if (i + 1) % outer_batch_size == 0:
                # Average the test loss over the batch
                test_loss /= outer_batch_size

                # Backpropagate the test loss
                test_loss.backward(create_graph=False, retain_graph=True)
                
                # Update the weights of the model
                optimizer.step()
                optimizer.zero_grad()
                test_loss = 0
                
            # Evaluate on the dev set
            if i % print_every == 0:
                dev_acc = average_acc(model, dev_set, lr_inner=lr_inner, batch_size=inner_batch_size)
                print("Dev accuracy at iteration " + str(i) + ":", dev_acc)

                # Determine whether to early stop
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    count_since_improved = 0

                    # Save the model if its performance has improved
                    torch.save(model.state_dict(), save_prefix + ".weights")

                else:
                    count_since_improved += 1

                if count_since_improved >= patience:
                    done = True
                    break
               
# Compute the average accuracy across all tasks in a dataset (e.g. the dev set)
def average_acc(model, dataset, lr_inner=0.01, batch_size=100, train=True):
    total_acc = 0

    for task in dataset:
        loss, acc, _ = fit_task(model, task, lr_inner=lr_inner, meta=False, batch_size=batch_size, train=train)
        total_acc += acc

    average_acc = total_acc * 1.0 / len(dataset)
    
    return average_acc




