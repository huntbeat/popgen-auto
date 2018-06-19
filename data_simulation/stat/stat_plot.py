"""
Use the msms simulation data to plot summary statistics across the sequence
"""

from tajima import parse_msms, calculate_D
import matplotlib.pyplot as plt

def main():
    simulation_list = ['simNone','simNat','simChange','simBoth']
    for sim in simulation_list:
        filename = sim + ".txt"
        bp_buckets, genomic_locations, num_indivs, sample_size, window, input_string, pos_start, pos_end = parse_msms(filename)
        calculate_D(bp_buckets, genomic_locations, num_indivs, sample_size, window, input_string, pos_start, pos_end)

    ## LCT Gene
    # pop = '/home/smathieson/public/cs68/1000g/EAS_135-136Mb.chr2.vcf'
    # bp_buckets, genomic_locations, num_indivs, n, window, super_pop = parse_vcf(pop)
    # calculate_D(bp_buckets, genomic_locations, num_indivs, n, window, super_pop)
    # plt.legend(super_pop_list)
    # plt.savefig('figs/tajimas_d_allpop_1000_humans.png')

if __name__ == "__main__":
    main()
