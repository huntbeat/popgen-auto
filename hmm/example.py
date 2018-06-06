"""
Test
"""

import numpy as np
from scipy.stats import expon


def main():
  for i in range(0,100,10):
      for j in range(i,i+10):
          print("a")
          break

if __name__ == "__main__":
    main()
