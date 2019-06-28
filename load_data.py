# Scripts for loading data from saved text files

# Load a list of abstract language descriptors
def load_languages(language_file):
    fi = open(language_file, "r")
    lang_list = []

    for line in fi:
        parts = line.strip().split("\t")

        ranking = [int(x) for x in parts[0].split(",")]
        vowel_inventory = parts[1].split(",")
        consonant_inventory = parts[2].split(",")

        lang = [ranking, vowel_inventory, consonant_inventory]

        lang_list.append(lang)

    return lang_list

# Load the file input/output correspondences
def load_io(io_file):
    fi = open(io_file, "r")

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

# Load a language with phonemes
def load_dataset(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")
        
        train_set = [elt.split(",") for elt in parts[0].split()]
        test_set = [elt.split(",") for elt in parts[1].split()]
        v_inv = parts[2].split(",")
        c_inv = parts[3].split(",")

        langs.append([train_set, test_set, v_inv, c_inv])

    return langs

# Load a language that is just Cs and Vs
def load_dataset_cv(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")
        
        train_set = [elt.split(",") for elt in parts[0].split()]
        test_set = [elt.split(",") for elt in parts[1].split()]
        vocab = parts[2].split()

        langs.append([train_set, test_set, vocab])

    return langs






