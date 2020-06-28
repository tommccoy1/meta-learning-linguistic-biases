import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random

from load_data import *
from utils import *
from training import *
from models import *

# Load the data
train_set = load_dataset_cv("cv.train")
dev_set = load_dataset_cv("cv.dev")
test_set = load_dataset_cv("cv.test")

# Create the model
cv_model = EncoderDecoder(6,6,10,128)

# Meta-train the model
maml(cv_model, train_set, dev_set, outer_batch_size=1, lr_inner=1.0,
     print_every=1000, patience=2, save_prefix="cv_model")

# Load the saved model
loaded_model = EncoderDecoder(6,6,10,128)
loaded_model.load_state_dict(torch.load("../models/cv_model.weights"))

# Evaluate it on the test set
test_acc = average_acc(loaded_model, test_set, lr_inner=0.01, batch_size=100)
print("Test accuracy:", test_acc)




