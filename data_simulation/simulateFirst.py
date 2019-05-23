import subprocess as sp
from random import uniform
# java -jar msms3.2rc-b163.jar -N 10000 -ms 20 20000 -t 400 -r 400 100000 -SAA 1000 -SAa 50 -Saa 0 -Sp 0.5 -SF 0 > stat/simNat.txt;
# java -jar msms3.2rc-b163.jar -N 10000 -ms 20 20000 -t 400 -r 400 100000 -SAA 1000 -SAa 50 -Saa 0 -Sp 0.5 -SF 0 > stat/simNat.txt;
# java -jar msms3.2rc-b163.jar -N 10000 -ms 20 20000 -t 400 -r 400 100000 -SAA 1000 -SAa 50 -Saa 0 -Sp 0.5 -SF 0 > stat/simNat.txt;

# java -jar msms3.2rc-b163.jar -N 10000 -ms 20 8000 -t 400 -r 400 100000 -SAA 100 -SAa 50.0 -Saa 0 -Sp 0.5546 -SF 0 > /scratch/saralab/first/sim100_0.txt

command = "java -jar msms3.2rc-b163.jar -N 10000 -ms 20 20000 -t 400 -r 400 100000 -SAA"
demo_history = ''
num_computers = 1
strength_list = [1000,100,10,0]

for i in range(num_computers):
    for strength in strength_list:
        filename = "/scratch/saralab/first/strength"+str(strength)+".msms"
        full_command = ["java","-jar","msms3.2rc-b163.jar","-N","10000","-ms","20","1","-t","400","-r","400","100000",
        "-SAA",str(strength),"-SAa",str(strength/2),"-Saa","0","-Sp","{0:.4f}".format(uniform(0.33333333,0.666666666)),"-SF","0.0","-threads","8"]
        import pdb; pdb.set_trace()
        hello = sp.Popen(full_command, stdout=sp.PIPE)
