"""
Test
"""

import numpy as np
from scipy.stats import expon


def main():
  pos_set = set()
  data_list = ""
  n = 0
  with open('vcf_example.vcf') as infile:
      data = 0
      pos = -1
      for line in infile:
          if not line.startswith('#'):
              stuff = line.split()
              prev_pos = pos
              pos = int(stuff[1])
              prev_data = data 
              data = stuff[9][0]
              if pos not in pos_set:
                  n += 1
              if data == 1:
                  pos_set.add(pos)
                  data_list += data
              else:
                  if pos not in pos_set:
                      if prev_pos != pos:
                          pos_set.add(pos)
                          data_list += data
  print(len(data_list))
  print(data_list)
  print(n)
  return

if __name__ == "__main__":
    main()
