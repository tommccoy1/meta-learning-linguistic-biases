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


def fit_task(model, task, train=True, create_graph=False, lr_inner=0.01, same_var=True):
    training_set = batchify_list(task[0])
    test_set = batchify_list(task[1])
    vocab = task[2]

    model_copy = model.create_copy(same_var=same_var)
    model_copy.set_dicts(vocab)
    
    criterion = nn.NLLLoss(ignore_index=0)

    def get_loss(dataset):
        loss = 0
        correct = 0
        total = 0

        for pair in dataset:
            inp, outp = pair
            output, logits = model_copy(inp)
            seq_loss = 0

            for index, output_guess in enumerate(output):
                if process_output(output_guess) == outp[index]:
                    correct += 1
                total += 1

            acc = correct * 1.0 / total

            all_seqs = []
            for sequence in outp:
                this_seq = []
                for elt in sequence:
                    ind = model_copy.char2ind[elt]
                    this_seq.append(ind)
                this_seq.append(model_copy.char2ind["EOS"])
                all_seqs.append(torch.LongTensor(this_seq))

            all_seqs = torch.nn.utils.rnn.pad_sequence(all_seqs)

            for index, logit in enumerate(logits):
                if index >= len(all_seqs):
                    break
                seq_loss += criterion(logit[0], all_seqs[index])

            loss += seq_loss / len(logits)

        loss /= len(dataset)

        return loss, acc

    train_loss, train_acc = get_loss(training_set)
    train_loss.backward(create_graph=create_graph, retain_graph=True)
 
    for name, param in model_copy.named_params():
        grad = param.grad
        model_copy.set_param(name, param - lr_inner * grad)
           
    # Test
    test_loss, test_acc = get_loss(test_set) 
            
    return test_loss, test_acc



def maml(model, epochs, train_set, dev_set, lr_inner=0.01, lr_outer=0.001, batch_size=1):
    optimizer = torch.optim.Adam(model.params(), lr=lr_outer)
    
    for _ in range(epochs):
        test_loss = 0


        # Copy the model
        # same_var means that its parameters are the same variables
        # as the original model's
        for i, t in enumerate(train_set):
            #new_model = model.create_copy(same_var=True) #EncoderDecoder(3,6,10,128)
            
            # Get the loss for one training task
            # create_graph must be true to allow for higher-order derivatives
            task_loss, task_acc = fit_task(model, t, create_graph=True, lr_inner=lr_inner)
            test_loss += task_loss

            if (i + 1) % batch_size == 0:
                test_loss /= batch_size
                test_loss.backward(create_graph=False, retain_graph=True)
                
                optimizer.step()
                optimizer.zero_grad()
                test_loss = 0
                
            if i % 100 == 0:
                print(i)
                total_acc = 0
                for task in dev_set:
                    loss, acc = fit_task(model, task, lr_inner=lr_inner, same_var=False)

                    total_acc += acc
                print("AVERAGE:", total_acc / len(dev_set))
                print("")
                







