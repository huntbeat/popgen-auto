import numpy as np
import h5py

STAT_FILENAMES =    ['/scratch/saralab/VAE/statsZI/example/stats/stats_0.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_1.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_2.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_3.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_4.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_5.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_6.txt',
                     '/scratch/saralab/VAE/statsZI/example/stats/stats_7.txt']

def stat2list(stat_filename):
    stat_file = open(stat_filename, 'r')
    header = next(stat_file).split(" ")
    stats = []
    for line in stat_file:
        stats.append(list(map(float,line.split(" ")))) 
    stat_file.close()
    return stats

def main():
    VAE_input = []
    for stat_filename in STAT_FILENAMES:
        VAE_input.extend(stat2list(stat_filename))
    import pdb; pdb.set_trace()
    stats = VAE_input
    stats = np.array(stats)
    stats -= np.amin(stats, axis=1).reshape(-1, 1)
    stats *= 1/np.amax(stats, axis=1).reshape(-1, 1)
    #f = h5py.File("/scratch/saralab/VAE/input/" +
    #      stat_filename[stat_filename.find('stats'):.replace('.txt','.h5'), "w")
    f = h5py.File("/scratch/saralab/VAE/input/" +
          "test_0.h5", "w")
    dset = f.create_dataset("VAE", data=VAE_input)
    f.close()

if __name__== '__main__':
    main()
