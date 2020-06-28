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
parser.add_argument("--constraints", help="which constraints are being used", type=str, default="yonc")
parser.add_argument("--rankings", help="set of rankings to consider", type=str, default=None)
args = parser.parse_args()

# A list of the unique sets of rankings for each set of constraints
# By "unique sets of rankings", we mean the rankings that produce distinct
# sets of mappings

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

# If we want to only include certain rankings
if args.rankings is not None:
    indices = [int(x) for x in args.rankings.split(",")]
    new_rankings = []

    for index, ranking in enumerate(rankings):
        if index in indices:
            new_rankings.append(ranking)

    rankings = new_rankings



vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']


seen_already = {}

list_languages = []

# Generate enough unique languages to cover all 3 datasets
# Each language is represented with a key, which will be enough to then
# generate that language's training set, dev set, and test set
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

# Randomly assign languages to training, dev, or test
random.shuffle(list_languages)
train_set = list_languages[:args.n_train]
dev_set = list_languages[args.n_train:args.n_train + args.n_dev]
test_set = list_languages[args.n_train + args.n_dev:]

# Save the languages
fo_train = open("../data/" + args.output_prefix + ".train_keys", "w")
fo_dev = open("../data/" + args.output_prefix + ".dev_keys", "w")
fo_test = open("../data/" + args.output_prefix + ".test_keys", "w")

for train_lang in train_set:
    fo_train.write("\t".join([",".join([str(x) for x in elt]) for elt in train_lang]) + "\n")

for dev_lang in dev_set:
    fo_dev.write("\t".join([",".join([str(x) for x in elt]) for elt in dev_lang]) + "\n")

for test_lang in test_set:
    fo_test.write("\t".join([",".join([str(x) for x in elt]) for elt in test_lang]) + "\n")





