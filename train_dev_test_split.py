# Creates non-overlapping train, development, and test sets of languages
import sys
import argparse
import random

from phonology_task_creation import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="number of training languages to generate", type=int, default=20000)
parser.add_argument("--n_dev", help="number of dev languages to generate", type=int, default=500)
parser.add_argument("--n_test", help="number of test languages to generate", type=int, default=1000)
parser.add_argument("--n_vowels_min", help="minimum number of vowels per language", type=int, default=2)
parser.add_argument("--n_vowels_max", help="maximum number of vowels per language", type=int, default=4)
parser.add_argument("--n_consonants_min", help="minimum number of consonants per language", type=int, default=2)
parser.add_argument("--n_consonants_max", help="maximum number of consonats per language", type=int, default=4)
parser.add_argument("--output_prefix", help="prefix for the output files", type=str, default="phonology")
args = parser.parse_args()


rankings = [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,3,1],
    [0,3,2,1],
    [2,3,0,1],
    [3,2,0,1],
    [1,2,3,0],
    [1,3,2,0]
]

vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']


seen_already = {}

list_languages = []

while len(list_languages) < args.n_train + args.n_dev + args.n_test:
    this_ranking = random.choice(rankings)

    v_inv, c_inv = phoneme_inventory(v_min=args.n_vowels_min, v_max=args.n_vowels_max, c_min=args.n_consonants_min, c_max=args.n_consonants_max, vowels=vowels, consonants=consonants)

    sorted_v_inv = sorted(v_inv)
    sorted_c_inv = sorted(c_inv)

    key = (tuple(this_ranking), tuple(sorted_v_inv), tuple(sorted_c_inv))
    key_unsorted = (tuple(this_ranking), tuple(v_inv), tuple(c_inv))


    if key not in seen_already:
        seen_already[key] = 1
        list_languages.append(key_unsorted)


random.shuffle(list_languages)
train_set = list_languages[:args.n_train]
dev_set = list_languages[args.n_train:args.n_train + args.n_dev]
test_set = list_languages[args.n_train + args.n_dev:]

fo_train = open(args.output_prefix + ".train_keys", "w")
fo_dev = open(args.output_prefix + ".dev_keys", "w")
fo_test = open(args.output_prefix + ".test_keys", "w")

for train_lang in train_set:
    fo_train.write("\t".join([",".join([str(x) for x in elt]) for elt in train_lang]) + "\n")

for dev_lang in dev_set:
    fo_dev.write("\t".join([",".join([str(x) for x in elt]) for elt in dev_lang]) + "\n")

for test_lang in test_set:
    fo_test.write("\t".join([",".join([str(x) for x in elt]) for elt in test_lang]) + "\n")













