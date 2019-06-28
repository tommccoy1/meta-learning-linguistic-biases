# Create full tasks from their keys
import sys
import argparse
import random

from phonology_task_creation import *
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="number of training examples to generate per language", type=int, default=100)
parser.add_argument("--n_test", help="number of test examples to generate per language", type=int, default=100)
parser.add_argument("--ranking_prefix", help="prefix for the output files", type=str, default="phonology")
args = parser.parse_args()

alphabet = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
        '.']

train_lang_list = load_languages(args.ranking_prefix + ".train_keys")
dev_lang_list = load_languages(args.ranking_prefix + ".dev_keys")
test_lang_list = load_languages(args.ranking_prefix + ".test_keys")

all_input_outputs = load_io("input_output_correspondences.txt")

train_set = []
dev_set = []
test_set = []

fo_train = open(args.ranking_prefix + ".train", "w")
fo_dev = open(args.ranking_prefix + ".dev", "w")
fo_test = open(args.ranking_prefix + ".test", "w")

for elt in train_lang_list:
    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_test=args.n_test, v_list=v_list, c_list=c_list)

    train_set.append(task)

    if len(train_set) % 1000 == 0:
        print(len(train_set))

for elt in train_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    test_examples = " ".join([",".join(x) for x in elt[1]])
    vocab = " ".join(alphabet)

    fo_train.write("\t".join([train_examples, test_examples, vocab]) + "\n")


for elt in dev_lang_list:
    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_test=args.n_test, v_list=v_list, c_list=c_list)

    dev_set.append(task)

    if len(dev_set) % 1000 == 0:
        print(len(dev_set))

for elt in dev_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    test_examples = " ".join([",".join(x) for x in elt[1]])
    vocab = " ".join(alphabet)

    fo_dev.write("\t".join([train_examples, test_examples, vocab]) + "\n")


for elt in test_lang_list:
    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_test=args.n_test, v_list=v_list, c_list=c_list)

    test_set.append(task)

    if len(test_set) % 1000 == 0:
        print(len(test_set))

for elt in test_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    test_examples = " ".join([",".join(x) for x in elt[1]])
    vocab = " ".join(alphabet)

    fo_test.write("\t".join([train_examples, test_examples, vocab]) + "\n")





