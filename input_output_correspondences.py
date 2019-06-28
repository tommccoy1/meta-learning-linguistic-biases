# Saves info about which abstract string wins for a given
# abstract input ("abstract" = CVCCV and the like) and saves
# that info, as it is slow to compute otherwise
from phonology_task_creation import *


inputs = generate_cv_cumul(5)
outputs = []

for inp in generate_cv_cumul(10):
    if syllabifiable(inp):
        outputs.append(syllabify(inp))


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

# This is pretty slow
all_input_outputs = {}

for ranking in rankings:
    print(ranking)
    io_list = []
    
    for inp in inputs:
        output = winner(inp, outputs, ranking)[0][0]
        print(inp,output)
        
        outp_no_periods = output.replace(".", "")
        ep = edit_path(inp, outp_no_periods)
        steps = edit_steps(ep[0], ep[1])
        
        io_list.append([inp, output, steps])
    
    all_input_outputs[tuple(ranking)] = io_list


fo = open("input_output_correspondences.txt", "w")

for key in all_input_outputs:
    key_string = ",".join([str(x) for x in key])

    io_list = all_input_outputs[key]
    io_list_string = "&".join(["#".join([elt[0], elt[1], ",".join(elt[2])]) for elt in io_list])

    fo.write(key_string + "\t" + io_list_string + "\n")








