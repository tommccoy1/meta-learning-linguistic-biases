# Create CV tasks from their keys
import sys
import argparse
import random

from phonology_task_creation import *
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="number of training examples to generate per language", type=int, default=20)
parser.add_argument("--ranking_prefix", help="prefix for the output files", type=str, default="cv")
args = parser.parse_args()

train_lang_list = load_languages("cv.train_keys")
dev_lang_list = load_languages("cv.dev_keys")
test_lang_list = load_languages("cv.test_keys")

all_input_outputs = load_io("yo_nc_io_correspondences.txt")

train_set = []
dev_set = []
test_set = []

fo_train = open("../data/" + args.ranking_prefix + ".train", "w")
fo_dev = open("../data/" + args.ranking_prefix + ".dev", "w")
fo_test = open("../data/" + args.ranking_prefix + ".test", "w")

for elt in train_lang_list:
    ranking = elt[0]

    task = make_task_cv(ranking, all_input_outputs, n=args.n_train)

    train_set.append(task)

    if len(train_set) % 1000 == 0:
        print(len(train_set))

for elt in train_set:
    train_examples = " ".join([",".join(x[:2]) for x in elt[0]])
    dev_examples = ""
    test_examples = " ".join([",".join(x[:2]) for x in elt[1]])
    vocab = " ".join(elt[2])

    fo_train.write("\t".join([train_examples, dev_examples, test_examples, vocab]) + "\n")


for elt in dev_lang_list:
    ranking = elt[0]

    task = make_task_cv(ranking, all_input_outputs, n=args.n_train)

    dev_set.append(task)

    if len(dev_set) % 1000 == 0:
        print(len(dev_set))

for elt in dev_set:
    train_examples = " ".join([",".join(x[:2]) for x in elt[0]])
    dev_examples = ""
    test_examples = " ".join([",".join(x[:2]) for x in elt[1]])
    vocab = " ".join(elt[2])

    fo_dev.write("\t".join([train_examples, dev_examples, test_examples, vocab]) + "\n")


for elt in test_lang_list:
    ranking = elt[0]

    task = make_task_cv(ranking, all_input_outputs, n=args.n_train)

    test_set.append(task)

    if len(test_set) % 1000 == 0:
        print(len(test_set))

for elt in test_set:
    train_examples = " ".join([",".join(x[:2]) for x in elt[0]])
    dev_examples = ""
    test_examples = " ".join([",".join(x[:2]) for x in elt[1]])
    vocab = " ".join(elt[2])

    fo_test.write("\t".join([train_examples, dev_examples, test_examples, vocab]) + "\n")





