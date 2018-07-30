import subprocess as sp
from random import uniform
from tqdm import tqdm
# java -jar msms3.2rc-b163.jar -N 10000 -ms 20 20000 -t 400 -r 400 100000 -SAA 1000 -SAa 50 -Saa 0 -Sp 0.5 -SF 0 > stat/simNat.txt;

s0    = 'java -jar msms3.2rc-b163.jar -N 10000 -ms 100 1 -t 80 -r 80 20000'
s10   = 'java -jar msms3.2rc-b163.jar -N 10000 -ms 100 1 -t 80 -r 80 20000 -SAA 10 -SAa 1 -Saa 0 -Sp 0.5 -SI 0.1 1 10'
s100  = 'java -jar msms3.2rc-b163.jar -N 10000 -ms 100 1 -t 80 -r 80 20000 -SAA 100 -SAa 10 -Saa 0 -Sp 0.5 -SI 0.1 1 10'
s1000 = 'java -jar msms3.2rc-b163.jar -N 10000 -ms 100 1 -t 80 -r 80 20000 -SAA 1000 -SAa 100 -Saa 0 -Sp 0.5 -SI 0.1 1 10'

demo  = ' -eN 0 8.047254 -eN 0.5 2.972169 -eN 5.0 9.64412'

strength = [s0, s10, s100, s1000]

name_start = 'statsZI/example/data/demo0/data'
demo_start = 'statsZI/example/data/demo1/data'

total_sim = 300

for i in tqdm(range(total_sim)):
    index = i / (total_sim / len(strength))
    index = int(index)
    full_command = strength[index].split(" ")
    sp.call(full_command, stdout=open(name_start+str(i)+'.msms','wb'))

if demo != None:
    for i in tqdm(range(total_sim)):
        index = i / (total_sim / len(strength))
        index = int(index)
        full_command = (strength[index]+demo).split(" ")
        sp.call(full_command, stdout=open(demo_start+str(i)+'.msms','wb'))
