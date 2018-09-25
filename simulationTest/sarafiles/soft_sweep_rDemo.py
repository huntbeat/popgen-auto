# soft sweep class for ZI data, now parallelized
# ssheehan, June 2015

# example command
# python soft_sweep_rDemo.py -d demo_param_one.txt -m ss/ -c 8


from multiprocessing import Pool
import optparse
import random
import subprocess
import sys
import time

import simulation_helpers


#---------
# GLOBALS
#---------


NUM_TARGET = 40 # this is how many datasets we want per demo
NUM_PER_S  = 10 # using 4 selection coefficients

# we want the commandline to look something like this
# msms -N 100000 -ms 100 1 -t 336 -r 336 100000 -eN 0 4.754958 -eN 0.05 0.668148 -eN 0.2 3.163320 -Sp 0.5 -SI 0.003 1 0.0001 -SAA 10000 -SAa 5000 -Saa 0 -Smark > msms/hs/demo0/data0.msms

# DROSOPHILA SPECIFIC PARAMETERS

Ne = 10000 # baseline effective population size
L = 100000  # region length: this fits with how we are doing the selection stats
MU = 8.4e-9 # from "Direct estimation of per nucleotide and genomic deleterious mutation rates in Drosophila" (2007)
R =  8.4e-9 # set to same as theta, as inferred by PMSC
THETA = 4*Ne*MU
RHO   = 4*Ne*R
THETA_REGION = int(L * THETA)
RHO_REGION   = int(L * RHO)
SAMPLE_SIZE  = 20 # we have up to 197 ZI individuals

# HARDSWEEP PARAMETERS, fixed for now, except selection location

# time the selection starts (going to do 0.001, 0.005, 0.01, 0.1, 1)
#SEL_START = 0.005
#SEL_START = 0.05
SEL_START = 0.5

# selection strength (going to do 0.01, 0.02, 0.05, 0.1)
S_LST = [0.01, 0.02, 0.05, 0.1]

# soft sweep frequency (going to do 0.0001, 0.001, 0.01, 0.1) 
SEL_FREQ = 0.001

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
def msms_commandline(demo, sel_loc, p):
    assert len(demo) == 3
    msms_str = 'java -jar msms3.2rc-b163.jar -N ' + str(Ne) + ' -ms ' + str(SAMPLE_SIZE) + ' 1 '
    msms_str += '-t ' + str(THETA_REGION) + ' '
    msms_str += '-r ' + str(RHO_REGION) + ' ' + str(L) + ' '
    msms_str += '-eN 0 ' + str(demo[0]) + ' -eN ' + str(T1) + ' ' + str(demo[1]) + ' -eN ' + str(T2) + ' ' + str(demo[2])

    # soft sweep
    S = S_LST[p/NUM_PER_S]
    SI  = str(SEL_START) + ' 1 ' + str(SEL_FREQ)
    SAA_DIR = 2*Ne*S # strength (multiplied by 2Ne)
    SAa_DIR =   Ne*S # heterozygote half the strength of a homozygote
    Saa_DIR =      0 # homozygote with unselected allele
    msms_str += ' -Sp ' + str(sel_loc) + ' -SAA ' + str(SAA_DIR) + ' -SAa ' + str(SAa_DIR) + ' -Saa ' + str(Saa_DIR) + ' -SI ' + SI + ' -Smark'
    
    return msms_str


# for parallelization (I believe this is so we can still use keyboard input to kill a parallel run)
def init_worker():
    simulation_helpers.signal.signal(simulation_helpers.signal.SIGINT, simulation_helpers.signal.SIG_IGN)


# msms runs for this demo_idx
def run_msms(demo_idx, demo):
    demo_path = opts.msms_folder + 'demo' + str(demo_idx) + '/'
    subprocess.call('mkdir ' + demo_path, shell=True)
    
    for p in range(NUM_TARGET):

        # sample the selection location
        loc_R = random.random()
        sel_loc = (loc_R*(MAX_SEL_LOC - MIN_SEL_LOC) + MIN_SEL_LOC)
        
        msms_filename = demo_path + 'data' + str(p) + '.msms'
        msms_command = msms_commandline(demo, sel_loc, p)
        print(msms_command)

        # keep doing the simulation unti we get freq > 0
        try_num = 1
        freq = 0
        while freq == 0:
            print(msms_filename + ', try' + str(try_num))
            test = subprocess.Popen(msms_command.split(" "), stdout=subprocess.PIPE) # here is where we call msms
            output = test.stdout.read()
            freq = simulation_helpers.examine_selection(output)
            try_num += 1
        hellofile = open(msms_filename,'w')
        hellofile.write(test)

        print('final freq: ' + str(freq))
        

#--------------
# MAIN
#--------------


all_demos = simulation_helpers.get_demo_sizes(opts.demo_filename)
num_demo  = len(all_demos)
print('num demos: ' + str(num_demo) + ', num cores: ' + str(opts.num_cores))

# testing
#run_msms(0,all_demos[0])
#1/0

# begin parallelization
pool = Pool(opts.num_cores)

# note that the function async is applied to ('run_msms' in this case) cannot have printing
# and runtime error messages within this function will also not be displayed
results = [pool.apply_async(run_msms, [i, all_demos[i]]) for i in range(num_demo)]

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
