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

# Syllabifies a sequence of Cs and Vs
def syllabify(word):
    prev = "#"
    syll = ""

    rev = word[::-1]

    for char in rev:
        if prev == "#":
            syll += "."
            syll += char
            prev = char
        elif prev == "C" and char == "V":
            syll += char
            prev = char
        elif prev == "V" and char == "C":
            syll += char
            syll += "."
            prev = "."
        elif prev == "V" and char == "V":
            syll += "."
            syll += char
            prev = char
        elif prev == ".":
            syll += char
            prev = char
        else:
            print(word)

    syll = syll[::-1]
    if len(syll) > 0:
        if syll[0] == "V" or syll[0] == "C":
            syll = "." + syll

    return syll

# Counts how many times a pair of an underlying representation
# and a surface representation violate each of the 4 constraints
def violations(ur, sr):
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
            if ons == "":
                onset += 1
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
# returns both in phonemes
def output_string(inp, outp, v_list, c_list, steps=None):
    
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
def winner(ur, candidates, ranking):
    all_violations = []
    for cand in candidates:
        viols = violations(ur, cand)
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

# Creates a phonology task
def make_task(ranking, all_input_outputs, n_train=10, n_test=10, v_list=None, c_list=None):
    io_list = all_input_outputs[tuple(ranking)][:]
    shuffle(io_list)
    
    if v_list is None or c_list is None:
        v_list, c_list = phoneme_inventory()

    train_pairs = []
    test_pairs = []
   
    train_dict = {}   

    for i in range(n_train):
        abstract = random.choice(io_list)
        train_pairs.append(output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2]))
        
    for elt in train_pairs:
        train_dict[tuple(elt)] = 1

    for i in range(n_test):
        satisfied = False
        while not satisfied:
            abstract = random.choice(io_list)
            candidate = output_string(abstract[0],abstract[1], v_list, c_list, steps=abstract[2])
            if tuple(candidate) not in train_dict:
                satisfied = True
                test_pairs.append(candidate)
        
    return train_pairs, test_pairs, v_list, c_list

# Creates a CV task
def make_task_cv(ranking, all_input_outputs, n=10):
    io_list = all_input_outputs[tuple(ranking)][:]
    shuffle(io_list)
    
    train_pairs = io_list[:n]
    test_pairs = io_list[n:]
    vocab = ["C", "V", "."]
       
    return train_pairs, test_pairs, vocab
    
    





