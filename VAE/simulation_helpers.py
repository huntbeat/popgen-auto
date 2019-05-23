# helpers for all our simulations
# ssheehan, June 2015


import sys


#---------
# HELPERS
#---------


# get all the demo sizes from the demo file (for consistency)
def get_demo_sizes(demo_filename):
    demo_file  = file(demo_filename,'r')
    demo_line = demo_file.read().replace('\n','')

    return [demo_line]


# from one msms file, return the frequency of the selected allele
def examine_selection(msms_string):
    begin_end = msms_string.split('positions: ')
    
    header = begin_end[0].split('\n')[0].split()
        
    # get the location of the selected site
    sp_idx = header.index('-Sp')
    sel_location = "%.5f" % float(header[sp_idx+1])

    s_idx = header.index('-SAA')
    s = float(header[s_idx+1])

    # get the sample size
    n = int(header[3])

    # skip until the mutation positions and get the selected index
    lines = begin_end[1].split('\n')

    mut_pos_lst = lines[0].strip().split()
    if sel_location not in mut_pos_lst:
        sys.exit('selected site *not* in mutation list! ' + msms_filename)
    sel_idx = mut_pos_lst.index(sel_location)

    # read all the haplotypes and get the frequency
    sel_alleles = []
    for i in range(n):
        haplotype = lines[i+1].strip()
        assert len(haplotype) == len(mut_pos_lst)
        sel_alleles.append(haplotype[sel_idx])
    frequency = (n-sel_alleles.count('0'))/float(n) # count 0's since msms records different instances of allele as different numbers/letters
    return frequency
