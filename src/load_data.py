# Scripts for loading data from saved text files

from random import shuffle


# Load a list of abstract language descriptors
# The descriptor is a 3-tuple:
# 1) The constraint ranking for the language, expressed in
#    shorthand where 0 = Onset (or NoOnset), 1 = Coda (or NoCoda),
#    2 = NoDeletion, and 3 = NoInsertion
# 2) The language's vowel inventory. The first vowel listed is the
#    vowel used for insertion.
# 3) The language's consonant inventory. The first consonant listed
#    the consonant used for insertion.
# The input is the name of a file, with one such descriptor per line.
# The output is a list, where each element is one of these descriptors
# expressed as a list with 3 elements.
def load_languages(language_file):
    fi = open("../data/" + language_file, "r")
    lang_list = []

    for line in fi:
        parts = line.strip().split("\t")

        ranking = [int(x) for x in parts[0].split(",")]
        vowel_inventory = parts[1].split(",")
        consonant_inventory = parts[2].split(",")

        lang = [ranking, vowel_inventory, consonant_inventory]

        lang_list.append(lang)

    return lang_list

# Load a file of input/output correspondences,
# which tell you what sorts of outputs go with what
# sorts of inputs (e.g., "CCV involves deleting the
# first C to yield CV"). Returns a dictionary, where the
# key is a constraint ranking and the value is a list
# of all such input-output pairs for that constraint ranking,
# plus a list of the steps (insertions and deletions) needed to 
# transform that input to that output
def load_io(io_file):
    fi = open("../io_correspondences/" + io_file, "r")

    io_correspondences = {}

    for line in fi:
        parts = line.strip().split("\t")
        ranking = tuple([int(x) for x in parts[0].split(",")])

        value = parts[1]
        value_groups = value.split("&")

        value_list = []

        for group in value_groups:
            components = group.split("#")
            inp = components[0]
            outp = components[1]
            steps = components[2].split(",")

            value_list.append([inp, outp, steps])

        io_correspondences[ranking] = value_list

    return io_correspondences

# Load a dataset
# The input is a filename, where each line in the file
# contains one language expressed with the following 
# tab-separated values:
# 1) Training set (a list of input-output pairs, where the input
#    and output are separated by a comma, and each input-output 
#    pair is separated by a space)
# 2) Dev set (formatted like the training set)
# 3) Test set (formatted like the training set)
# 4) The vocabulary of the language (delimited by spaces)
# 5) A keystring describing the language's vowels, consonants,
#    and constraint ranking
# Returns a list of all these languages in the dataset
def load_dataset(dataset_file):
    fi = open("../data/" + dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")
        
        train_set = [elt.split(",") for elt in parts[0].split()]
        dev_set = [elt.split(",") for elt in parts[1].split()]
        test_set = [elt.split(",") for elt in parts[2].split()]
        vocab = parts[3].split()
        key_string = parts[4].split(",")

        v_list = key_string[0].split()
        c_list = key_string[1].split()
        ranking = [int(x) for x in key_string[2].split()]

        key = [v_list, c_list, ranking]

        langs.append([train_set, dev_set, test_set, vocab, key])

    return langs


# Load a simple dataset that consists entirely of C's and V's
def load_dataset_cv(dataset_file):
    fi = open("../data/" + dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")
        
        train_set = [elt.split(",") for elt in parts[0].split()]
        test_set = [elt.split(",") for elt in parts[1].split()]
        vocab = parts[2].split()

        langs.append([train_set, test_set, vocab])

    return langs






