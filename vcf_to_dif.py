"""
vcf_to_dif.py
Hunter Lee

Input:  -v path to vcf file, -s starting locus, -e ending locus, -w window
Output: FASTA format file, first line includes the starting and ending locus,
        difference sequence begins in the second sequence
        Example:
        >>> 0 10000000 100
        0001010101100100101010011010101010101010101000010...

Example command:
python3 vcf_to_dif.py -v input/chr12_aldh2.vcf -s 112131266 -e 112352266 -w 1 -o dif
"""

import os
import optparse

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='VCF to difference sequence')
    parser.add_option('-v', '--vcf_filename', type='string', help='path to input vcf file')
    parser.add_option('-n', '--out_filename', type='string', help='path to output file, if desired')
    parser.add_option('-s', '--start', type='string', help='starting locus')
    parser.add_option('-e', '--end', type='string', help='ending locus, inclusive')
    parser.add_option('-w', '--window', type='string', help='window')
    parser.add_option('-o', '--out_folder', type='string', help='path to folder for output files')
    (opts, args) = parser.parse_args()

    mandatories = ['vcf_filename']

    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_filename == None:
        opts.out_filename = opts.vcf_filename.replace('.vcf', '.txt').replace("input/","")

    if opts.out_folder == None:
        opts.out_folder = "dif"

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def vcf_to_dif(vcf_file, start, end, window):
    # This links the SNP to its location
    pos_set = set()
    data_list = []
    dif_seq = ""
    n = 0
    with open(vcf_file) as infile:
        data = 0
        pos = -1
        for line in infile:
            if not line.startswith('#'):
                stuff = line.split()
                prev_pos = pos
                pos = int(stuff[1])
                prev_data = data
                data = stuff[9][0]
                if pos not in pos_set:
                    n += 1
                if data == 1:
                    pos_set.add(pos)
                    data_list.append(data)
                else:
                    if pos not in pos_set:
                        if prev_pos != pos:
                            pos_set.add(pos)
                            data_list.append(data)
    pos_list = sorted(pos_set)

    # This creates the difference sequence with all the above mind
    for i in range(start, end+1, window):
        for j in range(i, i+window):
            if j in pos_list:
                if data_list[pos_list.index(j)] == '1':
                    dif_seq += '1'
                    break
        dif_seq += '0'
    #TODO: MAKE THE BELOW TRUE
    print(len(list(range(start, end+1, window))) == len(dif_seq))
    return dif_seq

def main():
    opts = parse_args()
    dif_seq = vcf_to_dif(opts.vcf_filename,
                            int(opts.start), int(opts.end), int(opts.window))

    with open(opts.out_folder + "/" + opts.out_filename, 'w') as outputFile:
        outputFile.write(">> " + opts.start + " " +
                                opts.end + " " + opts.window + "\n")
        for i in range(0, len(dif_seq), 100):
            outputFile.write(dif_seq[i:i+100] + "\n")

if __name__ == "__main__":
    main()
