Hi there! This is the simulation command script for the msms -- these scripts specifically deal with the issue where setting population history parameters cannot require the -SF parameter to be set. It resolves the issue by checking whether the naturally selected allele still exists in the present time. This also means that there are many simulations that fail this criterion for each simulation that succeeds, resulting in longer runtimes.

The name of the scripts speak for themselves. For example, 'high' means simulations with high sweep, and so on.

The scripts are parallelized, and I recommend that you run each script on each computer.

Do remember to specify where to output the files and change the parameters as you wish!
