"""
Translate simulation data from MS^2 to pairwise difference sequence

Example command: python3 msms_to_dif.py -t sim1.txt -l 10000
"""

import numpy as np
import os
import optparse

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='msms simulator to difference sequence')
    parser.add_option('-t', '--input_file', type='string', help='path to input text file')
    parser.add_option('-l', '--length', type='string', help='desired length of sequence')
    parser.add_option('-o', '--out_folder', type='string', help='path to output files folder')
    (opts, args) = parser.parse_args()

    mandatories = ['input_file', 'length']

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

def msms_to_dif(input_file, length):
    input_param = ""
    dif_string_list = []
    with open(input_file) as msms_file:
        input_param = next(msms_file).replace("\n","")
        input_param = "_".join(input_param.split(" ")[:-3])
        next(msms_file) # rand number
        next(msms_file) # blank
        next(msms_file) # "//"
        next(msms_file) # segsites: __
        pos_string_list = next(msms_file).split(" ")[1:-1]
        pos_list = []
        for pos_string in pos_string_list:
            pos_list.append(int(float(pos_string) * length))
        pos_list.append(-1)
        num_samples = 0
        seg_sequence_list = [] # segregating sites sequence for each individual
        for line in msms_file:
            num_samples += 1
            seg_sequence_list.append(line.replace("\n","END"))
            dif_string_list.append("")
        for idx, seq in enumerate(seg_sequence_list):
            if seq == "":
                break
        seg_sequence_list = seg_sequence_list[0:idx]
        prev_pos = 0
        for pos in pos_list:
            if pos != -1:
                for idx1, dif_string in enumerate(dif_string_list):
                    for i in range(prev_pos, pos-1):
                        dif_string += "0"
                    dif_string_list[idx1] = dif_string
                prev_pos = pos
                for idx2, seg_string in enumerate(seg_sequence_list):
                    seg, new_seg_string = seg_string[0], seg_string[1:]
                    dif_string_list[idx2] += seg
                    seg_sequence_list[idx2] = new_seg_string
        for idx1, dif_string in enumerate(dif_string_list):
            for i in range(prev_pos, length):
                dif_string += "0"
            dif_string_list[idx1] = dif_string
        import pdb; pdb.set_trace()
        return input_param, dif_string_list

def main():
    opts = parse_args()
    input_param, dif_string_list = msms_to_dif(opts.input_file, int(opts.length))
    out_filename = input_param
    with open(opts.out_folder + "/" + out_filename + ".txt", 'w') as outputFile:
        outputFile.write(">> " + out_filename.replace("_"," ") + "\n")
        for dif_string in dif_string_list:
            outputFile.write(dif_string + "\n")

    # with open(opts.out_folder + "/" + "TMRCA_" + out_filename + ".txt", 'w') as outputFile:
    #     outputFile.write(">> " + "TMRCA " + out_filename.replace("_", " ") + "\n")
    #     for i in TMRCA:
    #         outputFile.write(str(i) + "\n")

if __name__ == '__main__':
    main()
