# neutral class for ZI data, now parallelized
# ssheehan, June 2015


from multiprocessing import Pool
import optparse
import subprocess
import sys
import time

import simulation_helpers

#---------
# GLOBALS
#---------


NUM_TARGET = 40      # this is how many datasets we want per demo

# we want the commandline to look something like this
# msms -N 100000 -ms 100 1 -t 336 -r 336 100000 -eN 0 4.754958 -eN 0.05 0.668148 -eN 0.2 3.163320 > msms/neutral/demo0/data0.msms

# DROSOPHILA SPECIFIC PARAMETERS

Ne = 100000 # baseline effective population size
L = 100000  # region length: this fits with how we are doing the selection stats
MU = 8.4e-9 # from "Direct estimation of per nucleotide and genomic deleterious mutation rates in Drosophila" (2007)
R =  8.4e-9 # set to same as theta, as inferred by PMSC
THETA = 4*Ne*MU
RHO   = 4*Ne*R
THETA_REGION = int(L * THETA)
RHO_REGION   = int(L * RHO)
SAMPLE_SIZE  = 100 # we have up to 197 ZI individuals

# from PSMC **UPDATED**
T1_years =  10000 # 10^4
T2_years = 100000 # 10^5
g = 0.1 # number of years per generation
T1 = float(T1_years) / (2*Ne*g) # time bottleneck ends in coalescent units
T2 = float(T2_years) / (2*Ne*g) # time bottleneck starts


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
def msms_commandline(demo):
    assert len(demo) == 3
    msms_str = 'java -jar msms3.2rc-b163.jar -N ' + str(Ne) + ' -ms ' + str(SAMPLE_SIZE) + ' 1 '
    msms_str += '-t ' + str(THETA_REGION) + ' '
    msms_str += '-r ' + str(RHO_REGION) + ' ' + str(L) + ' '
    msms_str += '-eN 0 ' + str(demo[0]) + ' -eN ' + str(T1) + ' ' + str(demo[1]) + ' -eN ' + str(T2) + ' ' + str(demo[2])
    return msms_str


# for parallelization (I believe this is so we can still use keyboard input to kill a parallel run)
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# msms runs for this demo_idx
def run_msms(demo_idx, demo):
    demo_path = opts.msms_folder + '/demo' + str(demo_idx) + '/'
    subprocess.call('mkdir ' + demo_path, shell=True)
    
    for p in range(NUM_TARGET):
        
        msms_filename = demo_path + 'data' + str(p) + '.msms'
        msms_command = msms_commandline(demo) + ' > ' + msms_filename
        #print(msms_command)

        # run the msms_command
        subprocess.call(msms_command, shell=True) # here is where we call msms
        

#--------------
# MAIN
#--------------


all_demos = simulation_helpers.get_demo_sizes(opts.demo_filename)
num_demo  = len(all_demos)
print('num demos: ' + str(num_demo) + ', num cores: ' + str(opts.num_cores))

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
