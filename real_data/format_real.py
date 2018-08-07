'''
format real data to network and trmca friendly structure
'''

import numpy as np
import vcf

class RealData():
  def __init__(self,data_file, region, n=20, window_size=100000):
    self.vcf_reader = vcf.Reader(filename=data_file)
    self.region = region
    self.n = n
    self.window_size = window_size
    self.all_positions, self.all_snps_by_site = self.parse_real()
    self.windows = self.split_into_windows()

  def parse_real(self):
    positions = []
    snps_by_site = []
    for record in self.vcf_reader:
      positions.append(record.POS)
      snps = self.snps_at_site(record)
      snps_by_site.append(snps)
    return positions, snps_by_site

  def snps_at_site(self, record, both_chrom=False):
    snps = []
    for sample in record.samples:
      gt = sample['GT'].split('|')
      if both_chrom: snps.extend([int(x) for x in gt])
      else: snps.append(int(gt[0]))
      if len(snps) == self.n: break
    return np.array(snps)

  '''
  note that last window could have an incomplete set of SNPs, based on region selection
  '''
  def split_into_windows(self):
    '''
    window_ranges = []
    for i in int((self.region[1]-self.region[0])/self.window_size):
      start = self.region[0] + self.window_size*i
      window_ranges.append((start,start+self.window_size))
    '''
    chromosome_windows = []
    start = self.region[0]
    remaining_pos = np.array(self.all_positions)
    remaining_sites = np.array(self.all_snps_by_site)
    last_position_index = 0
    while remaining_pos.shape[0] != 0:
      window_end = start + self.window_size
      in_window = np.where(remaining_pos < window_end)
      window_positions = remaining_pos[in_window[0]]
      window_snps = remaining_sites[in_window[0]]
      chrom_window = ChromosomeWindow(start,start+self.window_size,window_positions,window_snps,1500)
      chromosome_windows.append(chrom_window)
      print("num windows:",len(chromosome_windows))
      start += self.window_size
      last_position_index = in_window[0][-1] + 1
      remaining_pos = remaining_pos[last_position_index:]
      remaining_sites = remaining_sites[last_position_index:]
    return chromosome_windows

  def write_network_data(self, filename):
    data = []
    for win in self.windows:
      sample = win.network_sample
      data.append(sample)
    np_data = np.array(data) #TODO check shape of this
    print(np_data.shape)
    with h5py.File(filename, 'w') as f:
      f.create_dataset('SNP_pos', data=np_data)

class ChromosomeWindow():
  def __init__(self,start, end, positions, snps_by_site, L):
    self.start = start
    self.end = end
    self.range = end-start
    self.L = L
    self.absolute_positions, self.SNPs = self.remove_non_variations(positions, snps_by_site)
    self.relative_positions = self.compute_relative_positions()
    self.relative_position_distances_matrix = self.compute_distance_between(self.relative_positions, self.SNPs.shape[0])
    self.network_sample = self.center_network_sample()

  def remove_non_variations(self, positions, snps_by_site):
    rm_indices = []
    for i in range(snps_by_site.shape[0]):
      if np.all(snps_by_site[i] == snps_by_site[i][0]):
        rm_indices.append(i)
    remaining_snps = np.delete(snps_by_site, rm_indices, axis=0)
    remaining_positions = np.delete(positions, rm_indices, axis=0)
    #print("number of SNPs in this window:", remaining_positions.shape[0])
    return remaining_positions, remaining_snps.T # returns snps by n, not by site

  def compute_relative_positions(self):
    relative_positions = []
    for ap in self.absolute_positions:
      rp = ((ap - self.start) * self.range) / self.range
      relative_positions.append(rp)
    return np.array(relative_positions, dtype=int)

  def compute_distance_between(self, positions, n):
    distances = []
    for i in range(positions.shape[0] - 1): #pad last column with zeros
      dist = positions[i+1] - positions[i]
      distances.append(dist)
    distances.append(0)
    dist_mat = np.tile(distances,(n,1))
    return dist_mat

  # add tmrca in later
  def center_network_sample(self):
    space = self.L - self.SNPs.shape[1]
    half_space = int(space/2)
    padding = np.zeros((self.SNPs.shape[0],space), dtype=int)
    padded_position_dists = np.concatenate((padding[:,:half_space],self.relative_position_distances_matrix,padding[:,half_space:]), axis=1)
    padded_snps = np.concatenate((padding[:,:half_space],self.SNPs,padding[:,half_space:]), axis=1)
    return padded_snps, padded_position_dists

