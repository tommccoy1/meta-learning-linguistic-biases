
# Sanity check to confirm that there is no overlap between
# train, dev, and test sets

fi1 = open("phonology.train", "r")
fi2 = open("phonology.dev", "r")
fi3 = open("phonology.test", "r")

all_dict = {}

for line in fi1:
    parts = line.strip().split("\t")
    parts[1] = ",".join(sorted(parts[1].split(",")))
    parts[2] = ",".join(sorted(parts[2].split(",")))

    full_key = "&".join([parts[1], parts[2]])

    if full_key not in all_dict:
        all_dict[full_key] = [parts[0]]
    else:
        #print(parts[0])
        #print(all_dict[full_key])
        #print("")

        if parts[0] in all_dict[full_key]:
            print("ERROR!!")


        all_dict[full_key].append(parts[0])
        

