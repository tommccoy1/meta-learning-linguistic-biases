# Methods for creating phonology tasks
import random
from random import shuffle
import copy

# Creates a phoneme inventory for a language
def phoneme_inventory(v_min=2, v_max=4, c_min=2, c_max=4, vowels=None, consonants=None):
    if vowels is None:
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    
    if consonants is None:
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
    
    shuffle(vowels)
    shuffle(consonants)
    
    v_inv = vowels[:random.randint(v_min,v_max)]
    c_inv = consonants[:random.randint(c_min,c_max)]
    
    return (v_inv, c_inv)

# Generates all sequences of Cs and Vs of a certain length
def generate_cv(length):
    if length == 0:
        return [""]

    else:
        previous = generate_cv(length - 1)
        new = []

        for elt in previous:
            new.append(elt + "V")
            new.append(elt + "C")

        return new

# Generates all sequences of Cs and Vs up to a certain length
def generate_cv_cumul(max_length):
    output = []
    for i in range(max_length + 1):
        output += generate_cv(i)

    return output

# Determines whether a sequence of Cs and Vs can be syllabified
def syllabifiable(word):
    if word[:2] == "CC":
        return False
    if word[-2:] == "CC":
        return False
    if "CCC" in word:
        return False
    if word == "C":
        return False

    return True

# Replace all instances of "old" in "string" with "new"
def replace_iter(string, old, new):
    if old not in string:
        return string
    else:
        return replace_iter(string.replace(old, new), old, new)

# Syllabifies a sequence of Cs and Vs
# yes_onset is whether onsets are favored
# yes_coda is whether codas are favored
# onset_over_coda is whether a consonant will be treated
# as an onset or coda when either is possible. If this is True, then aba 
# is syllabified as .a.ba.; if false, it is .ab.a.
# input and output are both strings. The input must be syllabifiable.
def syllabify(word, yes_onset=True, yes_coda=False, onset_over_coda=True):
    word = replace_iter(word, "VV", "V.V")
    word = replace_iter(word, "CCV", "C.CV")


    if yes_onset and not yes_coda:
        word = replace_iter(word, "VCV", "V.CV")
    elif not yes_onset and yes_coda:
        word = replace_iter(word, "VCV", "VC.V")
    elif yes_onset and yes_coda and onset_over_coda:
        word = replace_iter(word, "VCV", "V.CV")
    elif yes_onset and yes_coda and not onset_over_coda:
        word = replace_iter(word, "VCV", "VC.V")
    elif not yes_onset and not yes_coda and onset_over_coda:
        word = replace_iter(word, "VCV", "VC.V")
    elif not yes_onset and not yes_coda and not onset_over_coda:
        word = replace_iter(word, "VCV", "V.CV")
    else:
        print("SYLLABIFICATION ERROR")

    if word != "":
        word = "." + word + "."

    return word

# Counts how many times a pair of an underlying representation
# and a surface representation violate each of the 4 constraints
def violations(ur, sr, yes_onset=True, yes_coda=False):
    onset = 0
    nocoda = 0
    mx = 0
    dep = 0

    if len(sr) > 0:
        if sr[0] == ".":
            sr = sr[1:]
        if sr[-1] == ".":
            sr = sr[:-1]

        syllables = sr.split(".")


        # Onset, NoCoda
        for syllable in syllables:
            parts = syllable.split("V")
            ons = parts[0]
            cod = parts[1]
            if yes_onset:
                if ons == "":
                    onset += 1
            else:
                if ons != "":
                    onset += 1

            if yes_coda:
                if cod == "":
                    nocoda += 1
            else:
                if cod != "":
                    nocoda += 1

    # Max, Dep
    edit_paths, _ = edit_path(ur,sr.replace(".",""))

    all_violations = []
    for path in edit_paths:
        #print(path)
        all_violations.append([onset, nocoda, path[1], path[0]])

    return all_violations

# Returns the edit path that transforms w1 into w2
def edit_path(w1,w2):
    l1 = len(w1) + 1
    l2 = len(w2) + 1
    
    grid = [[0 for i in range(l2)] for j in range(l1)]
    
    for ind in range(l1):
        grid[ind][0] = [[0,ind,[ind-1,0],[ind,0]]]
    for ind in range(l2):
        grid[0][ind] = [[ind,0,[0,ind-1],[0,ind]]]
    grid[0][0] = [[0,0,[-1,-1],[0,0]]]
        
        
    for i1 in range(1,l1):
        for i2 in range(1,l2):
            p1 = grid[i1-1][i2]
            p2 = grid[i1][i2-1]
            
            if w1[i1-1] == w2[i2-1]:
                possibles = copy.deepcopy(grid[i1-1][i2-1])
            else:
                new_poss = []
                
                possibles = copy.deepcopy(p1)
                for poss in possibles:
                    new_poss.append([poss[0], 1 + poss[1],poss[2],poss[3]])
                    
                possibles = copy.deepcopy(p2)
                for poss in possibles:
                    new_poss.append([1 + poss[0], poss[1],poss[2],poss[3]])
                    
                possibles = new_poss
            
            grid[i1][i2] = min_cands(possibles)[:]
            for index, elt in enumerate(grid[i1][i2]):
                grid[i1][i2][index][2] = grid[i1][i2][index][3][:]
                grid[i1][i2][index][3] = [i1,i2]
            
    return grid[l1-1][l2-1], grid

# Determines what steps are used to maneuver through an edit distance chart
def edit_steps(end_pt,grid):
    path = [end_pt[0][3]]
    prev = end_pt[0][2]
    
    done = False
    while not done:
        if prev == [-1,-1]:
            done = True
        else:
            path = [prev] + path
            prev = grid[prev[0]][prev[1]][0][2]
    
    #print(path)
    steps = []
    prev = None
    for stop in path:
        if prev is None:
            prev = [stop[0], stop[1]]
        else:
            if stop[0] == prev[0] + 1 and stop[1] == prev[1] + 1:
                steps.append("next")
            elif stop[0] == prev[0] + 1 and stop[1] == prev[1]:
                steps.append("del")
            elif stop[0] == prev[0] and stop[1] == prev[1] + 1:
                steps.append("ins")
            else:
                print("WRONG")
                14/0
            prev = [stop[0], stop[1]]
            
        
            
    return steps
        
# Creates concrete strings based on abstract CV notation
# inp is CVCV-format
# outp is CVCV-format
# returns both input and output in specific phonemes (e.g., abab)
def output_string(inp, outp, v_list, c_list, steps=None, change_one=False, change_all=False):
    
    outp_no_periods = outp.replace(".", "")
    
    if steps is None:
        ep = edit_path(inp, outp_no_periods)
        steps = edit_steps(ep[0], ep[1])
    
    inp_phonemes = []
    for char in inp:
        if char == "C":
            inp_phonemes.append(random.choice(c_list))
        else:
            inp_phonemes.append(random.choice(v_list))
    
    inp_word = "".join(inp_phonemes)
    if change_one:
        inp_word = replace_one(inp_word, v_list, c_list)
    elif change_all:
        inp_word = replace_all(inp_word, v_list, c_list)
   
    inp_phonemes = list(inp_word)    
 
    outp_phonemes = []
    indi = 0
    indo = 0
    for step in steps:
        if step == "next":
            outp_phonemes.append(inp_phonemes[indi])
            indi += 1
            indo += 1
        if step == "del":
            indi += 1
        if step == "ins":
            if outp_no_periods[indo] == "C":
                outp_phonemes.append(c_list[0])
            else:
                outp_phonemes.append(v_list[0])
            indo += 1
            
    outp_alignment = []
    ind_al = 0
    for char in outp:
        if char == ".":
            outp_alignment.append(char)
        else:
            outp_alignment.append(ind_al)
            ind_al += 1
            
    outp_chars = []
    for elt in outp_alignment:
        if elt == ".":
            outp_chars.append(".")
        else:
            outp_chars.append(outp_phonemes[elt])
    
    outp_word = "".join(outp_chars)
    
    
    return [inp_word, outp_word]
    
# Determines the candidates that minimize each constraint
def min_cands(cands):
    min_first = 1000000
    min_second = 1000000
    
    for cand in cands:
        first = cand[0]
        second = cand[1]
        
        if first < min_first:
            min_first = first
            
        if second < min_second:
            min_second = second
            
    firsts = []
    seconds = []
    
    for cand in cands:
        if cand[0] == min_first:
            firsts.append(cand)
        if cand[1] == min_second:
            seconds.append(cand)
            
    min_second_firsts = 1000000
    best_first = []
    for first in firsts:
        if first[1] < min_second_firsts:
            best_first = first
            
    min_first_seconds = 1000000
    best_second = []
    for second in seconds:
        if second[1] < min_first_seconds:
            best_second = second
            
    if best_first[0] ==  best_second[0] and best_first[1] == best_second[1]:
        return [best_first]
    else:
        return [best_first, best_second]
    
            
# Determines the optimal output given an unput, a set of candidates, and a ranking
def winner(ur, candidates, ranking, yes_onset=True, yes_coda=False):
    all_violations = []
    for cand in candidates:
        viols = violations(ur, cand, yes_onset=yes_onset, yes_coda=yes_coda)
        for viol in viols:
            all_violations += [[cand, viol]] 
            
    for constraint in ranking:
        min_viols = 1000000
        for candidate in all_violations:
            #print(all_violations, candidate)
            this_constraint_viols = candidate[1][constraint]
            if this_constraint_viols < min_viols:
                min_viols = this_constraint_viols
                
        filtered_cands = []
        for candidate in all_violations:
            if candidate[1][constraint] == min_viols:
                filtered_cands.append(candidate)
                
        all_violations = filtered_cands
        
    return all_violations

# Creates a phonology task (a training, dev, and test set for a single language)
# ranking = a constraint ranking
# all_input_outputs = the set of all abstract input/output pairs
# n_train, n_dev, n_test = number of train, dev, and test set examples
# v_list, c_list = list of vowels and consonants, If None, one will be generated randomly
# periods = whether to include periods in the output indicating syllable boundaries
# All of the "filters" filter allowable input-output pairs: the function takes the pair as input,
#   and if it returns True the pair is kept, while if it returns False the pair is rejected.
# input_filters filter the pair's input's form
# output_filters filter tha pair's output's form
# abstract filters filter the abstract form (expressed as C's and V's)
# replace_one says to replace one phoneme per input with a  phoneme not in the language
# replace_all says to replace all phonemes in each input with a phoneme not in the language
# small, med, and large refer to the smallest, middle, and largest dataset out of training, dev and test.
# typically, dev is smallest, then test, then train is largest - but it could be in any order, and it's determined
#   based on the n_train, n_dev, and n_test arguments.
# artificial_io_list is an alternatively-generated list of input-output pairs to use instead of all_input_outputs
def make_task(ranking, all_input_outputs, n_train=10, n_dev=10, n_test=10, v_list=None, c_list=None, periods=True, input_filter_small=None, output_filter_small=None, input_filter_med=None, output_filter_med=None, input_filter_large=None, output_filter_large=None, abstract_input_filter_small=None, abstract_output_filter_small=None, abstract_input_filter_med=None, abstract_output_filter_med=None, abstract_input_filter_large=None, abstract_output_filter_large=None, replace_one_small=False, replace_one_med=False, replace_one_large=False, replace_all_small=False, replace_all_med=False, replace_all_large=False, artificial_io_list=None):
    if all_input_outputs is not None:
        io_list = all_input_outputs[tuple(ranking)][:]
    else:
        io_list = artificial_io_list

    shuffle(io_list)

    if v_list is None or c_list is None:
        v_list, c_list = phoneme_inventory()

    large_pairs = []
    small_pairs = []
    med_pairs = []
   
    large_dict = {}  
    small_dict = {}
    med_dict = {}

    n_small, n_med, n_large = sorted([n_train, n_dev, n_test])

    for i in range(n_small):
        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)

            if (replace_one_large or replace_all_large) and abstract[0] == '':
                continue

            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_large, change_all=replace_all_large)
            if tuple(candidate) not in small_dict and tuple(candidate) not in med_dict and abstract_input_filter_large(abstract[0]) and abstract_output_filter_large(abstract[1].replace(".", "")) and input_filter_large(candidate[0]) and output_filter_large(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    large_pairs.append(candidate)
                else:
                    large_pairs.append([candidate[0], candidate[1].replace(".", "")])
                large_dict[tuple(candidate)] = 1

        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)

            if (replace_one_small or replace_all_small) and abstract[0] == '':
                continue

            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_small, change_all=replace_all_small)
            if tuple(candidate) not in large_dict and tuple(candidate) not in med_dict and abstract_input_filter_small(abstract[0]) and abstract_output_filter_small(abstract[1].replace(".", "")) and input_filter_small(candidate[0]) and output_filter_small(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    small_pairs.append(candidate)
                else:
                    small_pairs.append([candidate[0], candidate[1].replace(".", "")])
                small_dict[tuple(candidate)] = 1


        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)

            if (replace_one_med or replace_all_med) and abstract[0] == '':
                continue

            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_med, change_all=replace_all_med)
            if tuple(candidate) not in large_dict and tuple(candidate) not in small_dict and abstract_input_filter_med(abstract[0]) and abstract_output_filter_med(abstract[1].replace(".", "")) and input_filter_med(candidate[0]) and output_filter_med(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    med_pairs.append(candidate)
                else:
                    med_pairs.append([candidate[0], candidate[1].replace(".", "")])
                med_dict[tuple(candidate)] = 1


    for i in range(n_med - n_small):

        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)

            if (replace_one_large or replace_all_large) and abstract[0] == '':
                continue

            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_large, change_all=replace_all_large)
            if tuple(candidate) not in small_dict and tuple(candidate) not in med_dict and abstract_input_filter_large(abstract[0]) and abstract_output_filter_large(abstract[1].replace(".", "")) and input_filter_large(candidate[0]) and output_filter_large(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    large_pairs.append(candidate)
                else:
                    large_pairs.append([candidate[0], candidate[1].replace(".", "")])
                large_dict[tuple(candidate)] = 1

        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)


            if (replace_one_med or replace_all_med) and abstract[0] == '':
                continue


            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_med, change_all=replace_all_med)
            if tuple(candidate) not in large_dict and tuple(candidate) not in small_dict and abstract_input_filter_med(abstract[0]) and abstract_output_filter_med(abstract[1].replace(".", "")) and input_filter_med(candidate[0]) and output_filter_med(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    med_pairs.append(candidate)
                else:
                    med_pairs.append([candidate[0], candidate[1].replace(".", "")])
                med_dict[tuple(candidate)] = 1


    for i in range(n_large - n_med):
        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)


            if (replace_one_large or replace_all_large) and abstract[0] == '':
                continue


            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2], change_one=replace_one_large, change_all=replace_all_large)
            if tuple(candidate) not in small_dict and tuple(candidate) not in med_dict and abstract_input_filter_large(abstract[0]) and abstract_output_filter_large(abstract[1].replace(".", "")) and input_filter_large(candidate[0]) and output_filter_large(candidate[1].replace(".", "")):
                satisfied = True
                if periods:
                    large_pairs.append(candidate)
                else:
                    large_pairs.append([candidate[0], candidate[1].replace(".", "")])
                large_dict[tuple(candidate)] = 1

    if n_train >= n_test and n_test >= n_dev:
        train_pairs = large_pairs
        test_pairs = med_pairs
        dev_pairs = small_pairs
    elif n_train >= n_dev and n_dev >= n_test:
        train_pairs = large_pairs
        dev_pairs = med_pairs
        test_pairs = small_pairs
    elif n_test >= n_train and n_train >= n_dev:
        test_pairs = large_pairs
        train_pairs = med_pairs
        dev_pairs = small_pairs
    elif n_test >= n_dev and n_dev >= n_train:
        test_pairs = large_pairs
        dev_pairs = med_pairs
        train_pairs = small_pairs
    elif n_dev >= n_train and n_train >= n_test:
        dev_pairs = large_pairs
        train_pairs = med_pairs
        test_pairs = small_pairs
    elif n_dev >= n_test and n_test >= n_train:
        dev_pairs = large_pairs
        test_pairs = med_pairs
        train_pairs = small_pairs
                                    

    return train_pairs, dev_pairs, test_pairs, v_list, c_list, ranking



# Creates a simple task using only C's and V's
def make_task_cv(ranking, all_input_outputs, n=10):
    io_list = all_input_outputs[tuple(ranking)][:]
    shuffle(io_list)
    
    train_pairs = io_list[:n]
    test_pairs = io_list[n:]
    vocab = ["C", "V", "."]
       
    return train_pairs, test_pairs, vocab
    
# Replace one symbol with a symbol not in the vocabulary
def replace_one(inp, v_list, c_list):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']

    index = random.choice([x for x in range(len(inp))])
    inp = list(inp)

    satisfied = False
    while not satisfied:
        if inp[index] in vowels:
            new_segment = random.choice(vowels)

            if new_segment not in v_list:
                inp[index] = new_segment
                satisfied = True
        else:
            new_segment = random.choice(consonants)
            if new_segment not in c_list:
                inp[index] = new_segment
                satisfied = True
    return "".join(inp)

# Replace all symbols with a symbol not in the vocabulary
def replace_all(inp, v_list, c_list):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']

    inp = list(inp)

    for index in range(len(inp)):
        satisfied = False
        while not satisfied:
            if inp[index] in vowels:
                new_segment = random.choice(vowels)

                if new_segment not in v_list:
                    inp[index] = new_segment
                    satisfied = True
            else:
                new_segment = random.choice(consonants)
                if new_segment not in c_list:
                    inp[index] = new_segment
                    satisfied = True
    return "".join(inp)


