# Miscellaneous functions

# Break a list into batches of the desired size
def batchify_list(lst, batch_size=100):
    batches = []
    this_batch_in = []
    this_batch_out = []
   
    for index, elt in enumerate(lst):
        this_batch_in.append(elt[0])
        this_batch_out.append(elt[1])
        
        if (index + 1) % batch_size == 0:
            batches.append([this_batch_in, this_batch_out])
            this_batch_in = []
            this_batch_out = []
            
    if this_batch_in != []:
        batches.append([this_batch_in, this_batch_out])
        
    return batches
        
# Trim the excess from the end of an output string
def process_output(output):
    if "EOS" in output:
        return output[:output.index("EOS")]
    else:
        return output





