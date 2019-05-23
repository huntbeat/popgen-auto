# simulator for HMM CNN, parallelized
# ssheehan, June 2015
# example command:
# python low_rDemo.py -d /scratch/saralab/second/MXL_demo.txt -m /scratch/saralab/second/ -c 8

from multiprocessing import Pool
import optparse
import random
import subprocess
import sys
import time
from tqdm import tqdm

import simulation_helpers

#---------
# GLOBALS
#---------

START_DEMO = 0

NUM_TARGET = 25000 # this is how many datasets we want per demo
NUM_PER_S  = 25000 # using 4 selection coefficients

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
def msms_commandline(demo, p):
    msms_str = 'java -jar msms3.2rc-b163.jar -N ' + str(Ne) + ' -ms ' + str(SAMPLE_SIZE) + ' 1 '
    msms_str += '-t ' + str(THETA_REGION) + ' '
    msms_str += '-r ' + str(RHO_REGION) + ' ' + str(L) + ' '
    msms_str += demo
    return msms_str


# for parallelization (I believe this is so we can still use keyboard input to kill a parallel run)
def init_worker():
    simulation_helpers.signal.signal(simulation_helpers.signal.SIGINT, simulation_helpers.signal.SIG_IGN)


# msms runs for this demo_idx
def run_msms(demo_idx, demo):
    demo_path = opts.msms_folder + 'neu/'
    subprocess.call('mkdir ' + demo_path, shell=True)

    msms_total_filename = demo_path + 'compiled.msms'
    msms_total = open(msms_total_filename, 'w')
    
    printed_header = False
    
    for p in tqdm(range(NUM_TARGET)):

        msms_filename = demo_path + 'data' + str(p) + '.msms'
        msms_command = msms_commandline(demo, p) + ' > ' + msms_filename
        #print(msms_command)

        subprocess.call(msms_command, shell=True) # here is where we call msms
        msms_file = open(msms_filename,'r')
        msms_string = msms_file.read()
        #print(msms_string[:msms_string.find('\n')])
        if not printed_header:
            msms_total.write(msms_string)
            printed_header = True
        else:
            cut = find_nth(msms_string, '\n', 3) + len('\n')
            msms_total.write(msms_string[cut:])
        #print('final freq: ' + str(freq))

    msms_total.close()

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
        

#--------------
# MAIN
#--------------


all_demos = simulation_helpers.get_demo_sizes(opts.demo_filename)
num_demo  = len(all_demos)
print('num demos: ' + str(num_demo-START_DEMO) + ', num cores: ' + str(opts.num_cores))

# testing
#run_msms(0,all_demos[0])
#1/0

# begin parallelization
pool = Pool(opts.num_cores)

# note that the function async is applied to ('run_msms' in this case) cannot have printing
# and runtime error messages within this function will also not be displayed
#results = [pool.apply_async(run_msms, [i, all_demos[i]]) for i in range(START_DEMO,num_demo)]
run_msms(0, all_demos[0])

try:
    print "Waiting 10 seconds"
    time.sleep(10)

except KeyboardInterrupt:
    print "Caught KeyboardInterrupt, terminating workers"
    pool.terminate()
    pool.join()

else:
    print "Quitting normally"
    pool.close()
    pool.join()
# end parallelization
