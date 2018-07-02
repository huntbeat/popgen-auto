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
python3 vcf_to_dif.py -v input/chr12_aldh2.vcf -s 112131266 -e 112352266 -w 100 -o dif
"""

import os
import optparse
import vcf

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='VCF to difference sequence')
    parser.add_option('-v', '--vcf_filename', type='string', help='path to input vcf file')
    parser.add_option('-n', '--out_filename', type='string', help='path to output file, if desired')
    parser.add_option('-1', '--individual_1', type='string', help='first individual')
    parser.add_option('-2', '--individual_2', type='string', help='second individual')
    parser.add_option('-a', '--start', type='string', help='starting locus')
    parser.add_option('-z', '--end', type='string', help='ending locus, inclusive')
    parser.add_option('-w', '--window', type='string', help='window')
    parser.add_option('-o', '--out_folder', type='string', help='path to folder for output files')
    (opts, args) = parser.parse_args()

    mandatories = ['vcf_filename', 'individual_1']

    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_filename == None:
        opts.out_filename = "outputVCF.txt"

    if opts.out_folder == None:
        opts.out_folder = "dif"

    if opts.start == None:
        opts.start = "1"

    if opts.end == None:
        opts.end = '-1'

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def vcf_to_dif(vcf_reader, individual_1, start, end, window):
    # This links the SNP to its location
    pos_set = set()
    data_list = []
    dif_seq = ""
    begin = False
    n = 0
    first = -1
    last = -1
    for record in vcf_reader:
        pos = record.POS
        if pos < start:
            continue
        if not begin:
            print("START!")
            first = pos
            begin = True
        if end != -1:
            if pos > end:
                print("END!")
                last = pos
                break
        genotypes = record.genotype(individual_1)['GT'].split("|")
        left = int(genotypes[0])
        right = int(genotypes[1])
        if (left - right) % 2 == 1:
            n += 1
            print("HIT!")
            while pos in pos_set:
                pos += 1
            pos_set.add(pos)
    pos_list = sorted(pos_set)
    # This creates the difference sequence with all the above mind
    for i in range(start, end, window):
        found = False
        for j in range(i, i+window):
            if j in pos_list:
                dif_seq += '1'
                found = True
                break
        if not found:          
            dif_seq += '0'
    print("Number of segregating sites : %d" % n)
    print("LAST SNP loci               : %d" % pos)
    print("Length of sequence          : %d" % len(dif_seq))
    return dif_seq, first, last

def main():
    opts = parse_args()
    print('File: %s' % opts.vcf_filename)
    vcf_reader = vcf.Reader(filename=opts.vcf_filename)
    dif_seq, first, last = vcf_to_dif(vcf_reader, opts.individual_1,
                            int(opts.start), int(opts.end), int(opts.window))
    print("Begin loci : %d" % first)
    print(" Last loci : %d" % last)
    with open(opts.out_folder + "/" + opts.out_filename, 'w') as outputFile:
        outputFile.write(">> " + opts.start + " " +
                                opts.end + " " + opts.window + "\n")
        for i in range(0, len(dif_seq), 100):
            outputFile.write(dif_seq[i:i+100] + "\n")

if __name__ == "__main__":
    main()
