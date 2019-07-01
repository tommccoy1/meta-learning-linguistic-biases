# Create full tasks from their keys
import sys
import argparse
import random

from phonology_task_creation import *
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="number of training examples to generate per language", type=int, default=100)
parser.add_argument("--n_dev", help="number of dev examples to generate per language", type=int, default=0)
parser.add_argument("--n_test", help="number of test examples to generate per language", type=int, default=100)
parser.add_argument("--n_train_tasks", help="number of training tasks to generate", type=int, default=None)
parser.add_argument("--n_dev_tasks", help="number of dev tasks to generate", type=int, default=None)
parser.add_argument("--n_test_tasks", help="number of test tasks to generate", type=int, default=None)
parser.add_argument("--n_train_tasks_per_ranking", help="number of training tasks to generate per ranking", type=int, default=None)
parser.add_argument("--n_dev_tasks_per_ranking", help="number of dev tasks to generate per ranking", type=int, default=None)
parser.add_argument("--n_test_tasks_per_ranking", help="number of test tasks to generate ranking", type=int, default=None)
parser.add_argument("--ranking_prefix", help="prefix for the output files", type=str, default="phonology")
parser.add_argument("--train_rankings", help="rankings to include in the training set", type=str, default="abcdefgh")
parser.add_argument("--dev_rankings", help="rankings to include in the dev set", type=str, default="abcdefgh")
parser.add_argument("--test_rankings", help="rankings to include in the test set", type=str, default="abcdefgh")
args = parser.parse_args()

random.seed(12345)

alphabet = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
        '.']

ranking_dict = {}
ranking_dict["a"] = [0,1,2,3]
ranking_dict["b"] = [0,1,3,2]
ranking_dict["c"] = [0,2,3,1]
ranking_dict["d"] = [0,3,2,1]
ranking_dict["e"] = [2,3,0,1]
ranking_dict["f"] = [3,2,0,1]
ranking_dict["g"] = [1,2,3,0]
ranking_dict["h"] = [1,3,2,0]

inv_ranking_dict[(0,1,2,3)] = "a"
inv_ranking_dict[(0,1,3,2)] = "b"
inv_ranking_dict[(0,2,3,1)] = "c"
inv_ranking_dict[(0,3,2,1)] = "d"
inv_ranking_dict[(2,3,0,1)] = "e"
inv_ranking_dict[(3,2,0,1)] = "f"
inv_ranking_dict[(1,2,3,0)] = "g"
inv_ranking_dict[(1,3,2,0)] = "h"

ranking_count_dict_train = {}
ranking_count_dict_dev = {}
ranking_count_dict_test = {}

for letter in ["a", "b", "c", "d", "e", "f", "g", "h"]:
    ranking_count_dict_train[letter] = 0
    ranking_count_dict_dev[letter] = 0
    ranking_count_dict_test[letter] = 0


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

for index, elt in enumerate(train_lang_list):
    if args.n_train_tasks is not None:
        if index >= args.n_train_tasks:
            break

    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    if args.n_train_tasks_per_ranking is not None:
        if ranking_count_dict_train[inv_ranking_dict[tuple(ranking)]] == args.n_train_tasks_per_ranking:
            continue

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list)

    train_set.append(task)
    ranking_count_dict_train[inv_ranking_dict[tuple(ranking)]] += 1

    if len(train_set) % 1000 == 0:
        print(len(train_set))

for elt in train_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    dev_examples = " ".join([",".join(x) for x in elt[1]])
    test_examples = " ".join([",".join(x) for x in elt[2]])
    vocab = " ".join(alphabet)
    v_list = " ".join(elt[3])
    c_list = " ".join(elt[4])
    ranking = " ".join([str(x) for x in elt[5]])
    key = ",".join([v_list, c_list, ranking])

    fo_train.write("\t".join([train_examples, dev_examples, test_examples, vocab, key]) + "\n")


for index, elt in enumerate(dev_lang_list):
    if args.n_dev_tasks is not None:
        if index >= args.n_dev_tasks:
            break

    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    if args.n_dev_tasks_per_ranking is not None:
        if ranking_count_dict_dev[inv_ranking_dict[tuple(ranking)]] == args.n_dev_tasks_per_ranking:
            continue

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list)

    dev_set.append(task)
    ranking_count_dict_dev[inv_ranking_dict[tuple(ranking)]] += 1

    if len(dev_set) % 1000 == 0:
        print(len(dev_set))

for elt in dev_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    dev_examples = " ".join([",".join(x) for x in elt[1]])
    test_examples = " ".join([",".join(x) for x in elt[2]])
    vocab = " ".join(alphabet)
    v_list = " ".join(elt[3])
    c_list = " ".join(elt[4])
    ranking = " ".join([str(x) for x in elt[5]])
    key = ",".join([v_list, c_list, ranking])

    fo_dev.write("\t".join([train_examples, dev_examples, test_examples, vocab, key]) + "\n")


for index, elt in enumerate(test_lang_list):
    if args.n_test_tasks is not None:
        if index >= args.n_test_tasks:
            break

    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    if args.n_test_tasks_per_ranking is not None:
        if ranking_count_dict_test[inv_ranking_dict[tuple(ranking)]] == args.n_test_tasks_per_ranking:
            continue
    

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list)

    test_set.append(task)
    ranking_count_dict_test[inv_ranking_dict[tuple(ranking)]] += 1

    if len(test_set) % 1000 == 0:
        print(len(test_set))

for elt in test_set:
    train_examples = " ".join([",".join(x) for x in elt[0]])
    dev_examples = " ".join([",".join(x) for x in elt[1]])
    test_examples = " ".join([",".join(x) for x in elt[2]])
    vocab = " ".join(alphabet)
    v_list = " ".join(elt[3])
    c_list = " ".join(elt[4])
    ranking = " ".join([str(x) for x in elt[5]])
    key = ",".join([v_list, c_list, ranking])

    fo_test.write("\t".join([train_examples, dev_examples, test_examples, vocab, key]) + "\n")





