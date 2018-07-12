"""
Translate simulation data from MS^2 to pairwise difference sequence

For msms: java -jar msms3.2rc-b163.jar -N 10000 -ms 10 1 -t 40 -r 40 100000 -SAA 100 -SAa 50 -Saa 0 -Sp 0.5 -SF 0 > simNat.txt

Example command: python3 msms_to_dif.py -t sim1.txt -l 10000
"""

import numpy as np
import os
import optparse

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='msms simulator to difference sequence')
    parser.add_option('-t', '--input_file', type='string', help='path to input text file')
    parser.add_option('-o', '--out_folder', type='string', help='path to output files folder')
    (opts, args) = parser.parse_args()

    mandatories = ['input_file'] 

    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_folder == None:
        opts.out_folder = "dif"

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def msms_to_dif(input_file):
    input_param = ""
    dif_string_list = []
    SEQ_D = ""
    trueTMRCA = []
    with open(input_file) as msms_file:
        input_string = next(msms_file).replace("\n","")
        length = int(input_string.split(" ")[9])
        print("The length of the sequence is %d." % length)
        input_param = "_".join(input_string.split(" ")[:-3])
        next(msms_file) # rand number
        next(msms_file) # blank
        next(msms_file) # "//"
        """
        TMRCA
        """
        #for line in msms_file:
        #    if line[0] == '[':
        #        seq_length = int(line[line.find("[")+1:line.find("]")])
        #        TMRCA = float(line[line.find(":")+1:line.find(",")])
        #        for i in range(seq_length):
        #            trueTMRCA.append(TMRCA)
        #    else:
        #        break # time: ______ ________
        _ = next(msms_file) # segsites: __
        pos_string_list = next(msms_file).split(" ")[1:-1]
        pos_set = set()
        for pos_string in pos_string_list:
            position = int(float(pos_string) * length)
            while position in pos_set:
                position += 1
            pos_set.add(position)
        pos_list = sorted(list(pos_set))
        num_samples = 0
        seg_sequence_list = [] # segregating sites sequence for each individual
        for line in msms_file:
            num_samples += 1
            seg_sequence_list.append(line.replace("\n",""))
            dif_string_list.append("")
        for idx, seq in enumerate(seg_sequence_list):
            if seq == "":
                break
        seg_sequence_list = seg_sequence_list[0:idx] # gets rid of empty line
        prev_pos = 0
        for idx0, pos in enumerate(pos_list):
            for idx1, dif_string in enumerate(dif_string_list):
                for i in range(prev_pos, pos-1):
                    dif_string += "0"
                dif_string_list[idx1] = dif_string
            for i in range(prev_pos, pos-1):
                SEQ_D += "0"
            for idx2, seg_string in enumerate(seg_sequence_list):
                seg = seg_string[idx0]
                dif_string_list[idx2] += seg
            SEQ_D += "1"
            prev_pos = pos
        for idx1, dif_string in enumerate(dif_string_list):
            for i in range(prev_pos, length):
                dif_string += "0"
            dif_string_list[idx1] = dif_string
        for i in range(prev_pos, length):
            SEQ_D += "0"
        return input_param, SEQ_D, trueTMRCA

def main():
    opts = parse_args()
    input_param, SEQ_D, trueTMRCA = msms_to_dif(opts.input_file)
    out_filename = input_param
    # with open(opts.out_folder + "/" + out_filename + ".txt", 'w') as outputFile:
    #     outputFile.write(">> " + out_filename.replace("_"," ") + "\n")
    #     outputFile.write(SEQ_D + "\n")
    #     print("Output pairwise difference sequence file: %s" % (opts.out_folder + "/" + out_filename + ".txt"))
    with open(opts.input_file.replace('.txt','_dif.txt'), 'w') as outputFile:
        outputFile.write(">> " + out_filename.replace("_"," ") + "\n")
        outputFile.write(SEQ_D + "\n")
        print("Output pairwise difference sequence file: %s" % (opts.out_folder + "/" + out_filename + ".txt"))

    # with open(opts.out_folder + "/" + "TMRCA_" + out_filename + ".txt", 'w') as outputFile:
    #     outputFile.write(">> " + "TMRCA " + out_filename.replace("_", " ") + "\n")
    #     for i in trueTMRCA:
    #         outputFile.write(str(i) + "\n")
    #     print("Output TMRCA                        file: %s" % (opts.out_folder + "/" + "TMRCA_" + out_filename + ".txt"))

    # with open(opts.out_folder + "/" + "TMRCA_" + out_filename + ".txt", 'w') as outputFile:
    #     outputFile.write(">> " + "TMRCA " + out_filename.replace("_", " ") + "\n")
    #     for i in TMRCA:
    #         outputFile.write(str(i) + "\n")

if __name__ == '__main__':
    main()
