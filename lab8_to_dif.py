"""
lab8_to_dif.py
Hunter Lee

Converts lab8 FASTA sequences into a difference sequence

Example command:
python3 lab8_to_dif.py -f input/sequences_2mu.fasta
"""


import optparse
import os

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='HMM for Tmrca')
    parser.add_option('-f', '--fasta_filename', type='string', help='path to FASTA file')
    parser.add_option('-n', '--out_filename', type='string', help='path to output file, if desired')
    parser.add_option('-o', '--out_folder', type='string', help='path to folder for output files')
    (opts, args) = parser.parse_args()

    mandatories = ['fasta_filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_filename == None:
        opts.out_filename = opts.fasta_filename.replace(".fasta", ".txt").replace("input/","")

    if opts.out_folder == None:
        opts.out_folder = 'dif'

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def lab8_to_dif(fasta_filename):
    seq_1 = ""
    seq_2 = ""
    # take in two sequences
    with open(fasta_filename) as f:
        next(f)
        for line in f:
            new_line = line.strip("\n")
            if new_line[0] == ">":
                break
            else:
                seq_1 += new_line
        for line in f:
            new_line = line.strip("\n")
            seq_2 += new_line
    # compare the two
    dif_string = ""
    for i in range(0, len(seq_1)):
        dif_string += "0" if seq_1[i] == seq_2[i] else "1"
    return dif_string

def main():
    opts = parse_args()
    dif_seq = lab8_to_dif(opts.fasta_filename)

    with open(opts.out_folder + "/" + opts.out_filename, 'w') as outputFile:
        outputFile.write(">> \n")
        for i in range(0, len(dif_seq), 100):
            outputFile.write(dif_seq[i:i+100] + "\n")

if __name__ == "__main__":
    main()
