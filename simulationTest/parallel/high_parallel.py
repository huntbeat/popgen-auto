# simulator for HMM CNN, parallelized
# ssheehan, June 2015
# example command:
# python high_parallel.py -d /scratch/saralab/second/MXL_demo.txt -m /scratch/saralab/third/ -c 8

from multiprocessing import Pool, Lock, Value
import optparse
import random
import subprocess
import sys
import time

import helper_parallel

#---------
# GLOBALS
#---------

NUM_TARGET = 25000 # this is how many datasets we want per demo

# we want the commandline to look something like this
# msms -N 100000 -ms 100 1 -t 336 -r 336 100000 -eN 0 4.754958 -eN 0.05 0.668148 -eN 0.2 3.163320 -Sp 0.5 -SI 0.003 1 0.0001 -SAA 10000 -SAa 5000 -Saa 0 -Smark > msms/hs/demo0/data0.msms

# DROSOPHILA SPECIFIC PARAMETERS

Ne = 10000 # baseline effective population size
L = 100000  # region length: this fits with how we are doing the selection stats
MU = 1.25e-8 # from "Direct estimation of per nucleotide and genomic deleterious mutation rates in Drosophila" (2007)
R =  1.25e-8 # set to same as theta, as inferred by PMSC
THETA = 4*Ne*MU
RHO   = 4*Ne*R
THETA_REGION = int(L * THETA)
RHO_REGION   = int(L * RHO)
SAMPLE_SIZE  = 20 # we have up to 197 ZI individuals

# HARDSWEEP PARAMETERS, fixed for now, except selection location

# time the selection starts
#SEL_START = 0.005
SEL_START = 0.05
#SEL_START = 0.5

# selection strength
S_LST = 0.1/2 # selection strengths - low, mid, high : [0.001/2, 0.01/2, 0.1/2]

# hard sweep occurs in one individual
SEL_FREQ = float(1)/Ne

# selection location
MIN_SEL_LOC = 0.4 
MAX_SEL_LOC = 0.6


#------------------------------
# PARSE COMMAND LINE ARGUMENTS
#------------------------------


parser = optparse.OptionParser(description='simulate neutral case')

parser.add_option('-d', '--demo_filename', type='string', help='path to file of demography information')
parser.add_option('-m', '--msms_folder',   type='string', help='path to output msms folder, will create folders for each demo')
parser.add_option('-c', '--num_cores',     type='int',    help='number of cores, default 1', default=1)

(opts, args) = parser.parse_args()

mandatories = ['demo_filename','msms_folder']
for m in mandatories:
    if not opts.__dict__[m]:
        print('mandatory option ' + m + ' is missing\n')
        parser.print_help()
        sys.exit()


#---------
# HELPERS
#---------

# select_type is now always the same
def msms_commandline(p):

    loc_R = random.random()
    sel_loc = (loc_R*(MAX_SEL_LOC - MIN_SEL_LOC) + MIN_SEL_LOC)

    msms_str = 'java -jar msms3.2rc-b163.jar -N ' + str(Ne) + ' -ms ' + str(SAMPLE_SIZE) + ' 1 '
    msms_str += '-t ' + str(THETA_REGION) + ' '
    msms_str += '-r ' + str(RHO_REGION) + ' ' + str(L) + ' '
    msms_str += DEMO

    # hard sweep 
    S = S_LST
    SI  = str(SEL_START) + ' 1 ' + str(SEL_FREQ)
    SAA_BAL = 2*Ne*S # homozygote with selected allele
    SAa_BAL =   Ne*S # heterozygote half double selection strength of homozygote
    Saa_BAL =      0 # homozygote with unselected allele
    msms_str += ' -Sp ' + str(sel_loc) + ' -SAA ' + str(SAA_BAL) + ' -SAa ' + str(SAa_BAL) + ' -Saa ' + str(Saa_BAL) + ' -SI ' + SI + ' -Smark'
    
    return msms_str

# msms runs for this demo_idx
def run_msms((i, interval)):

    global counter
    msms_list = []

    for p in range(i*interval, (i+1)*interval):
        msms_filename = demo_path + 'data' + str(p) + '.msms'
        msms_command = msms_commandline(DEMO) + ' > ' + msms_filename
        #print(msms_command)

        # keep doing the simulation unti we get freq > 0
        try_num = 1
        freq = 0
        while freq == 0:
            #print(msms_filename + ', try' + str(try_num))
            subprocess.call(msms_command, shell=True) # here is where we call msms
            msms_file = open(msms_filename,'r')
            msms_string = msms_file.read()
            #msms.file.close()
            freq = helper_parallel.examine_selection(msms_string)
            #print(msms_string[:msms_string.find('\n')])
            #print(freq)
            try_num += 1
        cut = find_nth(msms_string, '\n', 3) + len('\n')
        msms_list.append(msms_string[cut:])
        #print('final freq: ' + str(freq))
        with counter.get_lock():
            counter.value += 1
            print(counter.value)
            print( "Est. time till compl : %f hours" % ((time.time() - BEGIN)*(NUM_TARGET-counter.value)/(counter.value)/60/60) ) 

    return "".join(msms_list)

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
        

#--------------
# MAIN
#--------------

def init(args):
    global counter
    counter = args

BEGIN = time.time()
DEMO = helper_parallel.get_demo_sizes(opts.demo_filename)

demo_path = opts.msms_folder + 'high/'
subprocess.call('mkdir ' + demo_path, shell=True)
msms_total_filename = demo_path + 'compiled.msms'
msms_total = open(msms_total_filename, 'w')

print('num target: ' + str(NUM_TARGET) + ', num cores: ' + str(opts.num_cores))
print(demo_path)
print("sel strength : %f" % S_LST)

# testing
#run_msms((0,1))
#import pdb; pdb.set_trace()
# 1/0

# begin parallelization
v = Value('i', 0)
pool = Pool(processes=opts.num_cores, initializer=init, initargs=(v,))

arguments = [(i, int(NUM_TARGET/opts.num_cores)) for i in range(0, opts.num_cores)]

# note that the function async is applied to ('run_msms' in this case) cannot have printing
# and runtime error messages within this function will also not be displayed
msms_strings = pool.map(run_msms, arguments)
pool.close()
pool.join()

msms_string_list = [msms_strings[0] for msms_string in msms_strings]

msms_final_string = "".join(msms_string_list)
msms_total.write("ms -N 10000 20 " +str(NUM_TARGET)+ " -t 400 -r 400 100000 -SAA -SAa -Saa" + "\n")
msms_total.write("random\n")
msms_total.write("\n")
msms_total.write(msms_final_string)
msms_total.close()

print("DONE!")
