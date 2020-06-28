
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random
import argparse

from load_data import *
from utils import *
from training import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", help="prefix for the datasets", type=str, default="phonology")
parser.add_argument("--vocab_size", help="vocab size for the model", type=int, default=34)
parser.add_argument("--emb_size", help="embedding size for the model", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size for the model", type=int, default=256)
parser.add_argument("--method", help="training method", type=str, default="maml")
parser.add_argument("--lr_inner", help="inner loop learning rate", type=float, default=1.0)
parser.add_argument("--inner_batch_size", help="inner loop batch size", type=int, default=100)
parser.add_argument("--lr_outer", help="outer loop learning rate", type=float, default=0.001)
parser.add_argument("--outer_batch_size", help="outer loop batch size", type=int, default=1)
parser.add_argument("--print_every", help="how many iterations to pass before printing dev accuracy", type=int, default=1000)
parser.add_argument("--patience", help="how many prints to pass before early stopping", type=int, default=5)
parser.add_argument("--save_prefix", help="prefix for saving the weights file", type=str, default="phonology")
args = parser.parse_args()

# Training a model

# The sets of languages to train/evaluate the model on
# (each of these is a set of languages, where each language
# in turn has its own training set, dev set, and test set)
train_set = load_dataset(args.data_prefix + ".train")

dev_set = load_dataset(args.data_prefix + ".dev")
test_set = load_dataset(args.data_prefix + ".test")

# Initialize the model
model = EncoderDecoder(args.vocab_size,args.emb_size,args.hidden_size)

# If you need a randomly-initialized model, save the model as-is, with no further training
if args.method == "random":
    torch.save(model.state_dict(), "../models/" + args.save_prefix + ".weights")

# If meta-training, meta-train the model
# The weights are saved inside the maml() function
if args.method == "maml":
    maml(model, train_set, dev_set, lr_inner=args.lr_inner, lr_outer=args.lr_outer, outer_batch_size=args.outer_batch_size, 
            inner_batch_size=args.inner_batch_size, print_every=args.print_every, patience=args.patience, save_prefix=args.save_prefix)


