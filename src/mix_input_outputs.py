import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--io_files", help="file(s) from which to draw input/output pairs", type=str, default=None)
parser.add_argument("--prefix", help="prefix for the file to save to", type=str, default=None) 

args = parser.parse_args()

# Creating a set of input/output correspondences that draw from various different rankings
def create_input_output_list(all_inputs_outputs_list):
    inputs = [x[0] for x in all_inputs_outputs_list[0][(0,1,2,3)]]
    ranking_list = all_inputs_outputs_list[0].keys

    this_dict = []

    for this_ranking in ranking_list:
        this_io_list = []
        for inp in inputs:
            io_dict = random.choice(all_inputs_outputs_list)
            ranking = random.choice(io_dict.keys)

            io_list = io_dict[ranking]

            for elt in io_list:
                if elt[0] == inp:
                    this_io_list.append(elt)

        this_dict[this_ranking] = this_io_list

    return this_dict



io_file_list = args.io_files.split("")
io_dict_list = [load_io(io_file) for io_file in io_file_list]


new_io = create_input_output_list(io_dict_list)


fo = open(args.prefix + "mixed_io_correspondences.txt", "w")

for key in all_input_outputs:
    key_string = ",".join([str(x) for x in key])

    io_list = all_input_outputs[key]
    io_list_string = "&".join(["#".join([elt[0], elt[1], ",".join(elt[2])]) for elt in io_list])

    fo.write(key_string + "\t" + io_list_string + "\n")

