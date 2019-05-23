from distutils.core import setup
from Cython.Build import cythonize

setup(name='Baum Welch in C',
      ext_modules=cythonize("bw.pyx"),
      annotate=True)
