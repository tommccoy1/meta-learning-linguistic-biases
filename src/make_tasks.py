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
parser.add_argument("--ranking_prefix", help="prefix for the input files", type=str, default="phonology")
parser.add_argument("--output_prefix", help="prefix for the output files", type=str, default=None)
parser.add_argument("--periods", help="whether to include periods in the surface representations", type=str, default="True")
parser.add_argument("--test_part_new", help="whether to make every element of the test set include 1 new segment", type=str, default="False")
parser.add_argument("--test_all_new", help="whether to make every element of the test set include all new segments", type=str, default="False")
parser.add_argument("--constraints", help="which constraint set to use", type=str, default="yonc")
parser.add_argument("--correspondences", help="file of input/output correspondences to use (if not default)", default=None)
parser.add_argument("--aio_shuffle", help="list of input/output correspondence files to shuffle over", default=None)
parser.add_argument("--n_v", help="number of vowels", type=int, default=None)
parser.add_argument("--n_c", help="number of consonants", type=int, default=None)

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
        f_list.append(f4)
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


rankings_no_nc = [
    [0,1,2,3],
    [0,2,1,3],
    [0,3,2,1],
    [1,2,0,3],
    [1,3,2,0],
    [2,3,1,0],
    [2,3,0,1],
    [3,2,1,0],
    [3,2,0,1],
]

rankings_yo_nc = [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,3,1],
    [0,3,2,1],
    [1,2,3,0],
    [1,3,2,0],
    [2,3,1,0],
    [3,2,1,0]
]

rankings_no_yc = [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,3,1],
    [0,3,2,1],
    [1,2,3,0],
    [1,3,2,0],
    [2,3,1,0],
    [3,2,1,0]
]

rankings_yo_yc = [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,3,1],
    [0,3,2,1],
    [1,2,3,0],
    [1,3,2,0],
    [2,3,1,0],
    [2,3,0,1],
    [3,2,1,0],
    [3,2,0,1]
]


if args.constraints == "nonc":
    rankings = rankings_no_nc
elif args.constraints == "yonc":
    rankings = rankings_yo_nc
elif args.constraints == "noyc":
    rankings = rankings_no_yc
elif args.constraints == "yoyc":
    rankings = rankings_yo_yc
else:
    print("INVALID CONSTRAINTS")

ranking_dict = {}
inv_ranking_dict = {}

for index, ranking in enumerate(rankings):
    inv_ranking_dict[tuple(ranking)] = index
    ranking_dict[index] = tuple(ranking)

ranking_count_dict_train = {}
ranking_count_dict_dev = {}
ranking_count_dict_test = {}

for index in range(len(rankings)):
    ranking_count_dict_train[index] = 0
    ranking_count_dict_dev[index] = 0
    ranking_count_dict_test[index] = 0


train_lang_list = load_languages(args.ranking_prefix + ".train_keys")
dev_lang_list = load_languages(args.ranking_prefix + ".dev_keys")
test_lang_list = load_languages(args.ranking_prefix + ".test_keys")

if args.constraints == "nonc":
    all_input_outputs = load_io("no_nc_io_correspondences.txt")
elif args.constraints == "noyc":
    all_input_outputs = load_io("no_yc_io_correspondences.txt")
elif args.constraints == "yonc":
    all_input_outputs = load_io("yo_nc_io_correspondences.txt")
elif args.constraints == "yoyc":
    all_input_outputs = load_io("yo_yc_io_correspondences.txt")

if args.correspondences is not None:
    all_input_outputs = load_io(args.correspondences)

if args.aio_shuffle is not None:
    all_input_outputs_list = []

    all_input_outputs_files = args.aio_shuffle.split(",")

    for fi in all_input_outputs_files:
        all_input_outputs_list.append(load_io(fi))

train_set = []
dev_set = []
test_set = []

if not periods:
    args.ranking_prefix += "_no_periods"


if args.output_prefix is None:
    fo_train = open(args.ranking_prefix + ".train", "w")
    fo_dev = open(args.ranking_prefix + ".dev", "w")
    fo_test = open(args.ranking_prefix + ".test", "w")
else:
    fo_train = open(args.output_prefix + ".train", "w")
    fo_dev = open(args.output_prefix + ".dev", "w")
    fo_test = open(args.output_prefix + ".test", "w")

for index, elt in enumerate(train_lang_list):
    if args.n_train_tasks is not None:
        if index >= args.n_train_tasks:
            break

    ranking = elt[0]
    v_list = elt[1]
    c_list = elt[2]

    if args.n_v is not None and args.n_c is not None:
        (v_list, c_list) = phoneme_inventory(v_min=args.n_v, v_max=args.n_v, c_min=args.n_c, c_max=args.n_c)

    if args.n_train_tasks_per_ranking is not None:
        if ranking_count_dict_train[inv_ranking_dict[tuple(ranking)]] == args.n_train_tasks_per_ranking:
            continue
    
    new_io_list = None 
    if args.aio_shuffle is not None:
        all_input_outputs = None
        aio = all_input_outputs_list[0]

        inputs = [x[0] for x in aio[aio.keys()[0]]]

        new_io_list = []

        for inp in inputs:

            aio = random.choice(all_input_outputs_list)

            rand_ranking = random.choice(aio.keys())

            for elt in aio[rand_ranking]:
                if elt[0] == inp:
                    new_io_list.append(elt)
                    break



    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter_small=input_filter_function, output_filter_small=output_filter_function, abstract_input_filter_small=abstract_input_filter_function, abstract_output_filter_small=abstract_output_filter_function, input_filter_med=input_filter_function, output_filter_med=output_filter_function, abstract_input_filter_med=abstract_input_filter_function, abstract_output_filter_med=abstract_output_filter_function, input_filter_large=input_filter_function, output_filter_large=output_filter_function, abstract_input_filter_large=abstract_input_filter_function, abstract_output_filter_large=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large, artificial_io_list=new_io_list)

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


    if args.n_v is not None and args.n_c is not None:
        (v_list, c_list) = phoneme_inventory(v_min=args.n_v, v_max=args.n_v, c_min=args.n_c, c_max=args.n_c)



    if args.n_dev_tasks_per_ranking is not None:
        if ranking_count_dict_dev[inv_ranking_dict[tuple(ranking)]] == args.n_dev_tasks_per_ranking:
            continue

    new_io_list = None
    if args.aio_shuffle is not None:
        all_input_outputs = None
        aio = all_input_outputs_list[0]

        inputs = [x[0] for x in aio[aio.keys()[0]]]

        new_io_list = []

        for inp in inputs:

            aio = random.choice(all_input_outputs_list)

            rand_ranking = random.choice(aio.keys())

            for elt in aio[rand_ranking]:
                if elt[0] == inp:
                    new_io_list.append(elt)
                    break





    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter_small=input_filter_function, output_filter_small=output_filter_function, abstract_input_filter_small=abstract_input_filter_function, abstract_output_filter_small=abstract_output_filter_function, input_filter_med=input_filter_function, output_filter_med=output_filter_function, abstract_input_filter_med=abstract_input_filter_function, abstract_output_filter_med=abstract_output_filter_function, input_filter_large=input_filter_function, output_filter_large=output_filter_function, abstract_input_filter_large=abstract_input_filter_function, abstract_output_filter_large=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large, artificial_io_list=new_io_list)

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


    if args.n_v is not None and args.n_c is not None:
        (v_list, c_list) = phoneme_inventory(v_min=args.n_v, v_max=args.n_v, c_min=args.n_c, c_max=args.n_c)



    if args.n_test_tasks_per_ranking is not None:
        if ranking_count_dict_test[inv_ranking_dict[tuple(ranking)]] == args.n_test_tasks_per_ranking:
            continue

    new_io_list = None
 
    if args.aio_shuffle is not None:
        all_input_outputs = None
        aio = all_input_outputs_list[0]

        inputs = [x[0] for x in aio[aio.keys()[0]]]

        new_io_list = []

        for inp in inputs:
        
            aio = random.choice(all_input_outputs_list)
            
            rand_ranking = random.choice(aio.keys())

            for elt in aio[rand_ranking]:
                if elt[0] == inp:
                    new_io_list.append(elt)
                    break

   

    task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, v_list=v_list, c_list=c_list, periods=periods, input_filter_small=input_filter_function, output_filter_small=output_filter_function, abstract_input_filter_small=abstract_input_filter_function, abstract_output_filter_small=abstract_output_filter_function, input_filter_med=input_filter_function, output_filter_med=output_filter_function, abstract_input_filter_med=abstract_input_filter_function, abstract_output_filter_med=abstract_output_filter_function, input_filter_large=input_filter_function, output_filter_large=output_filter_function, abstract_input_filter_large=abstract_input_filter_function, abstract_output_filter_large=abstract_output_filter_function, replace_one_small=replace_one_small, replace_one_med=replace_one_med, replace_one_large=replace_one_large, replace_all_small=replace_all_small, replace_all_med=replace_all_med, replace_all_large=replace_all_large, artificial_io_list=new_io_list)

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

if args.n_train_tasks_per_ranking is not None:
    for ranking in ranking_count_dict_train:
        if ranking_count_dict_train[ranking] != args.n_train_tasks_per_ranking:
            print(ranking, ranking_count_dict_train[ranking])


if args.n_dev_tasks_per_ranking is not None:
    for ranking in ranking_count_dict_dev:
        if ranking_count_dict_dev[ranking] != args.n_dev_tasks_per_ranking:
            print(ranking, ranking_count_dict_dev[ranking])


if args.n_test_tasks_per_ranking is not None:
    for ranking in ranking_count_dict_test:
        if ranking_count_dict_test[ranking] != args.n_test_tasks_per_ranking:
            print(ranking, ranking_count_dict_test[ranking])







