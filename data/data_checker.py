
from load_data import *



fi_train = "yonc.train"
fi_dev = "yonc.dev"
fi_test = "yonc.test"

train_set = load_dataset(fi_train)
dev_set = load_dataset(fi_dev)
test_set = load_dataset(fi_test)

key_dict = {}

for task in train_set:
    key = tuple([tuple(x) for x in task[-1]])
    if key in key_dict:
        print("ERROR! REPEATED LANGUAGE!")
    key_dict[key] = 1

    train_inputs = [x[0] for x in task[0]]
    train_inputs_dict = {}
    for inp in train_inputs:
        train_inputs_dict[inp] = 1

    dev_inputs = [x[0] for x in task[1]]
    dev_inputs_dict = {}
    for inp in dev_inputs:
        dev_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN DEV OVERLAP!")

    test_inputs = [x[0] for x in task[2]]
    test_inputs_dict = {}
    for inp in test_inputs:
        test_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN TEST OVERLAP!")
        if inp in train_inputs_dict:
            print("DEV TEST OVERLAP!")





for task in dev_set:
    key = tuple([tuple(x) for x in task[-1]])
    if key in key_dict:
        print("ERROR! REPEATED LANGUAGE!")
    key_dict[key] = 1

    train_inputs = [x[0] for x in task[0]]
    train_inputs_dict = {}
    for inp in train_inputs:
        train_inputs_dict[inp] = 1

    dev_inputs = [x[0] for x in task[1]]
    dev_inputs_dict = {}
    for inp in dev_inputs:
        dev_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN DEV OVERLAP!")

    test_inputs = [x[0] for x in task[2]]
    test_inputs_dict = {}
    for inp in test_inputs:
        test_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN TEST OVERLAP!")
        if inp in train_inputs_dict:
            print("DEV TEST OVERLAP!")




for task in test_set:
    key = tuple([tuple(x) for x in task[-1]])
    if key in key_dict:
        print("ERROR! REPEATED LANGUAGE!")
    key_dict[key] = 1

    train_inputs = [x[0] for x in task[0]]
    train_inputs_dict = {}
    for inp in train_inputs:
        train_inputs_dict[inp] = 1

    dev_inputs = [x[0] for x in task[1]]
    dev_inputs_dict = {}
    for inp in dev_inputs:
        dev_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN DEV OVERLAP!")

    test_inputs = [x[0] for x in task[2]]
    test_inputs_dict = {}
    for inp in test_inputs:
        test_inputs_dict[inp] = 1
        if inp in train_inputs_dict:
            print("TRAIN TEST OVERLAP!")
        if inp in train_inputs_dict:
            print("DEV TEST OVERLAP!")







