# Computes which abstract output wins for a given
# abstract input ("abstract input" means something like CVCCV,
# and "abstract output" means something like ".CV.CV.") and saves
# that info, as it is slow to compute otherwise
from phonology_task_creation import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onset", help="whether onsets are favored or penalized", type=str, default="yes")
parser.add_argument("--coda", help="whether onsets are favored or penalized", type=str, default="no")
parser.add_argument("--prefix", help="output prefix", type=str, default="")
parser.add_argument("--max_input_length", help="maximum length of the underlying representation", type=int, default=5)
args = parser.parse_args()

# Whether onsets and codas are favored
# or disfavored in the set of constraints being assumed
yes_onset = args.onset=="yes"
yes_coda = args.coda=="yes"

inputs = generate_cv_cumul(args.max_input_length)
if yes_onset and yes_coda:
    max_output_length = 3 * args.max_input_length
else:
    max_output_length = 2 * args.max_input_length

# All possible rankings of the 4 constraints
rankings = [
    [0,1,2,3],
    [0,1,3,2],
    [0,2,1,3],
    [0,2,3,1],
    [0,3,1,2],
    [0,3,2,1],
    [1,0,2,3],
    [1,0,3,2],
    [1,2,0,3],
    [1,2,3,0],
    [1,3,0,2],
    [1,3,2,0],
    [2,1,0,3],
    [2,1,3,0],
    [2,0,1,3],
    [2,0,3,1],
    [2,3,1,0],
    [2,3,0,1],
    [3,1,2,0],
    [3,1,0,2],
    [3,2,1,0],
    [3,2,0,1],
    [3,0,1,2],
    [3,0,2,1]
]

# Computing the winners; this is pretty slow
all_input_outputs = {}

# Loop over all 24 rankings
for ranking in rankings:
    print(ranking)
    io_list = []
    outputs = []


    # The set of possible outputs is the set of all syllabifiable strings
    # bounded by the maximum output length
    for inp in generate_cv_cumul(max_output_length):
        if ranking.index(0) < ranking.index(1):
            onset_over_coda = True
        else:
            onset_over_coda = False

        if syllabifiable(inp):
            outputs.append(syllabify(inp, yes_onset=yes_onset, yes_coda=yes_coda, onset_over_coda=onset_over_coda))

    # For each possible input, find what output among the candidate outputs
    # wins, based on Optimality Theory and the provided constraints
    for inp in inputs:
        output = winner(inp, outputs, ranking, yes_onset=yes_onset, yes_coda=yes_coda)[0][0]
        print(inp,output)
        
        outp_no_periods = output.replace(".", "")
        ep = edit_path(inp, outp_no_periods)

        # We also have to save which particular insertions and deletions must occur
        # to transform the input to the output, in order to know which input symbols
        # correspond to which output symbols
        steps = edit_steps(ep[0], ep[1])
        
        io_list.append([inp, output, steps])
    
    all_input_outputs[tuple(ranking)] = io_list

# Now save the input-output correspondences that we have computed
fo = open(args.prefix + "io_correspondences.txt", "w")

for key in all_input_outputs:
    key_string = ",".join([str(x) for x in key])

    io_list = all_input_outputs[key]
    io_list_string = "&".join(["#".join([elt[0], elt[1], ",".join(elt[2])]) for elt in io_list])

    fo.write(key_string + "\t" + io_list_string + "\n")








