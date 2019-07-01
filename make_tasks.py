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
parser.add_argument("--periods", help="whether to include periods in the surface representations", type=str, default="True")
parser.add_argument("--test_part_new", help="whether to make every element of the test set include 1 new segment", type=str, default="False")
parser.add_argument("--test_all_new", help="whether to make every element of the test set include all new segments", type=str, default="False")

parser.add_argument("--input_allowed_lengths", help="lengths the input string may be", type=str, default=None)
parser.add_argument("--input_must_be_first", help="character that must be first in the input", type=str, default=None)
parser.add_argument("--input_cant_be_first", help="character that can't be first in the input", type=str, default=None)
parser.add_argument("--input_must_be_last", help="character that must be last in the input", type=str, default=None)
parser.add_argument("--input_cant_be_last", help="character that can't be last in the input", type=str, default=None)
parser.add_argument("--input_allowed_strings", help="strings that can be the input", type=str, default=None)
parser.add_argument("--input_disallowed_strings", help="strings that can't be the input", type=str, default=None)

parser.add_argument("--output_allowed_lengths", help="lengths the output string may be", type=str, default=None)
parser.add_argument("--output_must_be_first", help="character that must be first in the output", type=str, default=None)
parser.add_argument("--output_cant_be_first", help="character that can't be first in the output", type=str, default=None)
parser.add_argument("--output_must_be_last", help="character that must be last in the output", type=str, default=None)
parser.add_argument("--output_cant_be_last", help="character that can't be last in the output", type=str, default=None)
parser.add_argument("--output_allowed_strings", help="strings that can be the output", type=str, default=None)
parser.add_argument("--output_disallowed_strings", help="strings that can't be the output", type=str, default=None)

parser.add_argument("--abstract_input_allowed_lengths", help="lengths the abstract input string may be", type=str, default=None)
parser.add_argument("--abstract_input_must_be_first", help="character that must be first in the abstract input", type=str, default=None)
parser.add_argument("--abstract_input_cant_be_first", help="character that can't be first in the abstract input", type=str, default=None)
parser.add_argument("--abstract_input_must_be_last", help="character that must be last in the abstract input", type=str, default=None)
parser.add_argument("--abstract_input_cant_be_last", help="character that can't be last in the abstract input", type=str, default=None)
parser.add_argument("--abstract_input_allowed_strings", help="strings that can be the abstract input", type=str, default=None)
parser.add_argument("--abstract_input_disallowed_strings", help="strings that can't be the abstract input", type=str, default=None)

parser.add_argument("--abstract_output_allowed_lengths", help="lengths the abstract output string may be", type=str, default=None)
parser.add_argument("--abstract_output_must_be_first", help="character that must be first in the abstract output", type=str, default=None)
parser.add_argument("--abstract_output_cant_be_first", help="character that can't be first in the abstract output", type=str, default=None)
parser.add_argument("--abstract_output_must_be_last", help="character that must be last in the abstract output", type=str, default=None)
parser.add_argument("--abstract_output_cant_be_last", help="character that can't be last in the abstract output", type=str, default=None)
parser.add_argument("--abstract_output_allowed_strings", help="strings that can be the abstract output", type=str, default=None)
parser.add_argument("--abstract_output_disallowed_strings", help="strings that can't be the abstract output", type=str, default=None)

args = parser.parse_args()

replace_one_small = False
replace_one_med = False
replace_one_large = False
replace_all_small = False
replace_all_med = False
replace_all_large = False

if args.n_train >= args.n_test and args.n_test >= args.n_dev:
    replace_one_med = args.test_part_new == "True"
    replace_all_med = args.test_all_new == "True"
elif args.n_train >= args.n_dev and args.n_dev >= args.n_test:
    replace_one_small = args.test_part_new == "True"
    replace_all_small = args.test_all_new == "True"
elif args.n_test >= args.n_train and args.n_train >= args.n_dev:
    replace_one_large = args.test_part_new == "True"
    replace_all_large = args.test_all_new == "True"
elif args.n_test >= args.n_dev and args.n_dev >= args.n_train:
    replace_one_large = args.test_part_new == "True"
    replace_all_large = args.test_all_new == "True"
elif args.n_dev >= args.n_train and args.n_train >= args.n_test:
    replace_one_small = args.test_part_new == "True"
    replace_all_small = args.test_all_new == "True"
elif args.n_dev >= args.n_test and args.n_test >= args.n_train:
    replace_one_med = args.test_part_new == "True"
    replace_all_med = args.test_all_new == "True"


if args.input_allowed_lengths is not None:
    input_allowed_lengths = [int(x) for x in args.input_allowed_lengths.split(",")]
else:
    input_allowed_lengths = None
input_must_be_first = args.input_must_be_first
input_cant_be_first = args.input_cant_be_first
input_must_be_last = args.input_must_be_last
input_cant_be_last = args.input_cant_be_last
if args.input_allowed_strings is not None:
    input_allowed_strings = args.input_allowed_strings.split(",")
else:
    input_allowed_strings = None
if args.input_disallowed_strings is not None:
    input_disallowed_strings = args.input_disallowed_strings.split(",")
else:
    input_disallowed_strings = None
input_filter_function = filtering_function(allowed_lengths=input_allowed_lengths, must_be_first=input_must_be_first, cant_be_first=input_cant_be_first, must_be_last=input_must_be_last, cant_be_last=input_cant_be_last, allowed_strings=input_allowed_strings, disallowed_strings=input_disallowed_strings)

if args.output_allowed_lengths is not None:
    output_allowed_lengths = [int(x) for x in args.output_allowed_lengths.split(",")]
else:
    output_allowed_lengths = None
output_must_be_first = args.output_must_be_first
output_cant_be_first = args.output_cant_be_first
output_must_be_last = args.output_must_be_last
output_cant_be_last = args.output_cant_be_last
if args.output_allowed_strings is not None:
    output_allowed_strings = args.output_allowed_strings.split(",")
else:
    output_allowed_strings = None
if args.output_disallowed_strings is not None:
    output_disallowed_strings = args.output_disallowed_strings.split(",")
else:
    output_disallowed_strings = None
output_filter_function = filtering_function(allowed_lengths=output_allowed_lengths, must_be_first=output_must_be_first, cant_be_first=output_cant_be_first, must_be_last=output_must_be_last, cant_be_last=output_cant_be_last, allowed_strings=output_allowed_strings, disallowed_strings=output_disallowed_strings)



if args.abstract_input_allowed_lengths is not None:
    abstract_input_allowed_lengths = [int(x) for x in args.abstract_input_allowed_lengths.split(",")]
else:
    abstract_input_allowed_lengths = None
abstract_input_must_be_first = args.abstract_input_must_be_first
abstract_input_cant_be_first = args.abstract_input_cant_be_first
abstract_input_must_be_last = args.abstract_input_must_be_last
abstract_input_cant_be_last = args.abstract_input_cant_be_last
if args.abstract_input_allowed_strings is not None:
    abstract_input_allowed_strings = args.abstract_input_allowed_strings.split(",")
else:
    abstract_input_allowed_strings = None
if args.abstract_input_disallowed_strings is not None:
    abstract_input_disallowed_strings = args.abstract_input_disallowed_strings.split(",")
else:
    abstract_input_disallowed_strings = None
abstract_input_filter_function = filtering_function(allowed_lengths=abstract_input_allowed_lengths, must_be_first=abstract_input_must_be_first, cant_be_first=abstract_input_cant_be_first, must_be_last=abstract_input_must_be_last, cant_be_last=abstract_input_cant_be_last, allowed_strings=abstract_input_allowed_strings, disallowed_strings=abstract_input_disallowed_strings)


if args.abstract_output_allowed_lengths is not None:
    abstract_output_allowed_lengths = [int(x) for x in args.abstract_output_allowed_lengths.split(",")]
else:
    abstract_output_allowed_lengths = None
abstract_output_must_be_first = args.abstract_output_must_be_first
abstract_output_cant_be_first = args.abstract_output_cant_be_first
abstract_output_must_be_last = args.abstract_output_must_be_last
abstract_output_cant_be_last = args.abstract_output_cant_be_last
if args.abstract_output_allowed_strings is not None:
    abstract_output_allowed_strings = args.abstract_output_allowed_strings.split(",")
else:
    abstract_output_allowed_strings = None
if args.abstract_output_disallowed_strings is not None:
    abstract_output_disallowed_strings = args.abstract_output_disallowed_strings.split(",")
else:
    abstract_output_disallowed_strings = None
abstract_output_filter_function = filtering_function(allowed_lengths=abstract_output_allowed_lengths, must_be_first=abstract_output_must_be_first, cant_be_first=abstract_output_cant_be_first, must_be_last=abstract_output_must_be_last, cant_be_last=abstract_output_cant_be_last, allowed_strings=abstract_output_allowed_strings, disallowed_strings=abstract_output_disallowed_strings)





random.seed(12345)

periods = args.periods == "True"

if periods:
    alphabet = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
        '.']
else:
    alphabet = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']

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



def filtering_function(allowed_lengths=None, must_be_first=None, cant_be_first=None, must_be_last=None, cant_be_last=None, allowed_strings=None, disallowed_strings=None):
    f_list = []
    if allowed_lengths is not None:
        f1 = lambda x: len(x) in allowed_lengths
        f_list.append(f1)
    if must_be_first is not None:
        f2 = lambda x: x[0] == must_be_first
        f_list.append(f2)
    if cant_be_first is not None:
        f3 = lambda x: x[0] != cant_be_first
        f_list.append(f3)
    if must_be_last is not None:
        f4 = lambda x: x[-1] == must_be_last
        f_list.append(f4]
    if cant_be_last is not None:
        f5 = lambda x: x[-1] != cant_be_last
        f_list.append(f5)
    if allowed_strings is not None:
        f6 = lambda x: x in allowed_strings
        f_list.append(f6)
    if disallowed_strings is not None:
        f7 = lambda x: x not in disallowed_strings
        f_list.append(f7)

    def combined_functions(x):
        for function in f_list:
            if not function(x):
                return False

        return True

    return lambda x: combined_functions(x)




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

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter=input_filter_function, output_filter=output_filter_function, abstract_input_filter=abstract_input_filter_function, abstract_output_filter=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large)

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

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter=input_filter_function, output_filter=output_filter_function, abstract_input_filter=abstract_input_filter_function, abstract_output_filter=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large)

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
    

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter=input_filter_function, output_filter=output_filter_function, abstract_input_filter=abstract_input_filter_function, abstract_output_filter=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large)

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





