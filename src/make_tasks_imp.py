# Create tasks that test for implicational universals
import sys
import argparse
import random

from phonology_task_creation import *
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="number of training examples to generate per language", type=int, default=100)
parser.add_argument("--n_dev", help="number of dev examples to generate per language", type=int, default=0)
parser.add_argument("--n_test", help="number of test examples to generate per language", type=int, default=100)
parser.add_argument("--output_prefix", help="prefix for the output files", type=str, default=None)
parser.add_argument("--constraints", help="which constraint set to use", type=str, default="yonc")
parser.add_argument("--implication_type", help="whether the implication is based on one or three", type=str, default=None)

args = parser.parse_args()

def filtering_function(allowed_lengths=None, must_be_first=None, cant_be_first=None, must_be_last=None, cant_be_last=None, allowed_strings=None, disallowed_strings=None):
    f_list = []
    if allowed_lengths is not None:
        f1 = lambda x: len(x) in allowed_lengths
        f_list.append(f1)
    if must_be_first is not None:
        f2 = lambda x: (len(x) > 0 and x[0] in must_be_first)
        f_list.append(f2)
    if cant_be_first is not None:
        f3 = lambda x: (len(x) > 0 and x[0] not in cant_be_first)
        f_list.append(f3)
    if must_be_last is not None:
        f4 = lambda x: (len(x) > 0 and x[-1] in must_be_last)
        f_list.append(f4)
    if cant_be_last is not None:
        f5 = lambda x: (len(x) > 0 and x[-1] not in cant_be_last)
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




random.seed(12345)

alphabet = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
        '.']

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


if args.constraints == "nonc":
    all_input_outputs = load_io("no_nc_io_correspondences.txt")
elif args.constraints == "noyc":
    all_input_outputs = load_io("no_yc_io_correspondences.txt")
elif args.constraints == "yonc":
    all_input_outputs = load_io("yo_nc_io_correspondences.txt")
elif args.constraints == "yoyc":
    all_input_outputs = load_io("yo_yc_io_correspondences.txt")


train_set = []
dev_set = []
test_set = []

fo_train = open(args.output_prefix + ".train", "w")
fo_dev = open(args.output_prefix + ".dev", "w")
fo_test = open(args.output_prefix + ".test", "w")

if args.implication_type == "one":
    for line in open("imp_show_one.txt", "r"):
        parts = line.strip().split("\t")
        ranking = [int(x) for x in parts[0].split(",")]
        train_abstract = parts[1]
        test_abstract = parts[2]

        train_filter_function = filtering_function(allowed_strings=[train_abstract])
        test_filter_function = filtering_function(allowed_strings=[test_abstract])

        blank_f = filtering_function()

        (v_list, c_list) = phoneme_inventory(v_min=10, v_max=10, c_min=20, c_max=20)

        task = make_task(ranking, all_input_outputs, v_list=v_list, c_list=c_list, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, abstract_input_filter_small=train_filter_function, abstract_input_filter_med=test_filter_function, abstract_input_filter_large=train_filter_function, abstract_output_filter_small=blank_f, abstract_output_filter_med=blank_f, abstract_output_filter_large=blank_f, input_filter_small=blank_f, input_filter_med=blank_f, input_filter_large=blank_f, output_filter_small=blank_f, output_filter_med=blank_f, output_filter_large=blank_f)

        test_set.append(task)

elif args.implication_type == "three":
    for line in open("imp_withhold_one.txt", "r"):
        parts = line.strip().split("\t")
        ranking = [int(x) for x in parts[0].split(",")]
        test_abstract = parts[1]

        all_syll_types = ["V", "CV", "VC", "CVC"]
        train_abstract = []
        for syll_type in all_syll_types:
            if syll_type != test_abstract:
                train_abstract.append(syll_type)


        train_filter_function = filtering_function(allowed_strings=train_abstract)
        test_filter_function = filtering_function(allowed_strings=[test_abstract])

        blank_f = filtering_function()

        (v_list, c_list) = phoneme_inventory(v_min=10, v_max=10, c_min=20, c_max=20)

        task = make_task(ranking, all_input_outputs, v_list=v_list, c_list=c_list, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, abstract_input_filter_small=train_filter_function, abstract_input_filter_med=test_filter_function, abstract_input_filter_large=train_filter_function, abstract_output_filter_small=blank_f, abstract_output_filter_med=blank_f, abstract_output_filter_large=blank_f, input_filter_small=blank_f, input_filter_med=blank_f, input_filter_large=blank_f, output_filter_small=blank_f, output_filter_med=blank_f, output_filter_large=blank_f)

        test_set.append(task)

elif args.implication_type == "first":
    for ranking in rankings:
        for _ in range(10):
            (v_list, c_list) = phoneme_inventory()
            v_first = random.choice(v_list)
            c_first = random.choice(c_list)

            train_filter_function = filtering_function(cant_be_first=[v_first, c_first])
            test_filter_function = filtering_function(must_be_first=[v_first, c_first])

            blank_f = filtering_function()


            task = make_task(ranking, all_input_outputs, v_list=v_list, c_list=c_list, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, abstract_input_filter_small=blank_f, abstract_input_filter_med=blank_f, abstract_input_filter_large=blank_f, abstract_output_filter_small=blank_f, abstract_output_filter_med=blank_f, abstract_output_filter_large=blank_f, input_filter_small=train_filter_function, input_filter_med=test_filter_function, input_filter_large=train_filter_function, output_filter_small=blank_f, output_filter_med=blank_f, output_filter_large=blank_f)


            test_set.append(task)

elif args.implication_type == "last":
    for ranking in rankings:
        print(ranking)
        for i in range(10):
            print(i)
            (v_list, c_list) = phoneme_inventory()
            v_last = random.choice(v_list)
            c_last = random.choice(c_list)

            train_filter_function = filtering_function(cant_be_last=[v_last, c_last])
            test_filter_function = filtering_function(must_be_last=[v_last, c_last])

            blank_f = filtering_function()


            task = make_task(ranking, all_input_outputs, v_list=v_list, c_list=c_list, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, abstract_input_filter_small=blank_f, abstract_input_filter_med=blank_f, abstract_input_filter_large=blank_f, abstract_output_filter_small=blank_f, abstract_output_filter_med=blank_f, abstract_output_filter_large=blank_f, input_filter_small=train_filter_function, input_filter_med=test_filter_function, input_filter_large=train_filter_function, output_filter_small=blank_f, output_filter_med=blank_f, output_filter_large=blank_f)

            test_set.append(task)

elif args.implication_type == "length":
    for ranking in rankings:
        for i in range(10):
            train_filter_function = filtering_function(allowed_lengths=[0,1,2,3,4])
            test_filter_function = filtering_function(allowed_lengths=[5])

            blank_f = filtering_function()


            task = make_task(ranking, all_input_outputs, n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test, abstract_input_filter_small=train_filter_function, abstract_input_filter_med=test_filter_function, abstract_input_filter_large=train_filter_function, abstract_output_filter_small=blank_f, abstract_output_filter_med=blank_f, abstract_output_filter_large=blank_f, input_filter_small=blank_f, input_filter_med=blank_f, input_filter_large=blank_f, output_filter_small=blank_f, output_filter_med=blank_f, output_filter_large=blank_f)

            test_set.append(task)





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






