import numpy as np
import h5py

STAT_FILENAMES =    ['stats_0.txt',
                     'stats_1.txt']

def stat2input(stat_filename):
    stat_file = open(stat_filename, 'r')
    header = next(stat_file).split(" ")
    stats = []
    for line in stat_file:
        stats.append(list(map(float,line.split(" ")))) 
    stats = np.array(stats)
    stats -= np.amin(stats, axis=1).reshape(-1, 1)
    stats *= 1/np.amax(stats, axis=1).reshape(-1, 1)
    stat_file.close()
    return stats

def main():
    for stat_filename in STAT_FILENAMES:
        VAE_input = stat2input(stat_filename)
        f = h5py.File("/scratch/saralab/VAE/input/" +
             stat_filename.replace('.txt','.h5'), "w")
        dset = f.create_dataset("VAE", data=VAE_input)
        f.close()

if __name__== '__main__':
    main()
