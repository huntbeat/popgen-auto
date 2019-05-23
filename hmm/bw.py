import numpy as np
from libc.math cimport log, exp
from libc.stdlib cimport atoi

DTYPE = np.floatc

cdef class FB:

    def __init__(self, dif_seq, log_init, log_tran, log_emit, state):
        self.dif_seq = dif_seq
        self.log_init = log_init
        self.log_tran = log_tran
        self.log_emit = log_emit
        self.state = np.array(state)
        self.K = len(self.log_init)
        self.L = len(self.dif_seq)
        self.F = np.zeros((self.K,self.L)) # K x L
        self.F[:,0] = self.log_init + self.log_emit[:,int(self.dif_seq[0])]
        self.B = np.zeros((self.K,self.L))
        self.B[:,-1] = 0
        self.P = np.zeros((self.K,self.L)) # posterior probability table
        self.P_mean = np.zeros((self.L)) # posterior mean
        self.P_decoded = np.zeros((self.L)) # posterior decoding
        self.X_p = 0

        self.fb()
"""
C++ version
"""
cdef float log_sum(self, float log_p, float log_q):
    return log_p + log(1+exp(log_q-log_p))
"""
C++ version, assumes that array is longer than 1
"""
cdef float log_sum_all(self, float array[], float length):
    sig = self.log_sum(array[0],array[1])
    for i in range(2,length):
        sig = self.log_sum(sig,array[i])
    return sig

def forward(self):

    cdef char c_dif_string[len(self.dif_seq)] = self.dif_seq.encode("UTF-8")

    cdef float [:,:] c_F = self.F

    cdef float [:,:] c_log_init = self.log_init
    cdef float [:,:] c_log_tran = self.log_tran
    cdef float [:,:] c_log_emit = self.log_emit

    cdef Py_ssize_t c_K, c_L
    cdef Py_ssize_t c_K = self.F.shape[0]
    cdef Py_ssize_t c_L = self.F.shape[1]

    for col in range(1, c_L):
        for row in range(c_K):
            cdef float c_likelihood[c_K]
            for prev_state in range(c_K):
                c_likelihood[prev_state] = c_F[prev_state, col-1] + c_log_tran[prev_state, row]
            likelihood = self.log_sum_all(c_likelihood)
            c_F[row, col] = likelihood + c_log_emit[row, atoi(c_dif_seq[col])]

def backward(self):

    cdef float [:,:] c_B = self.B

    cdef float [:,:] c_log_init = self.log_init
    cdef float [:,:] c_log_tran = self.log_tran
    cdef float [:,:] c_log_emit = self.log_emit

    cdef Py_ssize_t c_K, c_L
    cdef Py_ssize_t c_K = self.F.shape[0]
    cdef Py_ssize_t c_L = self.F.shape[1]

    for col in range(c_L-2, -1, -1):
        for row in range(c_K):
            cdef float c_likelihood[c_K]
            for next_state in range(c_K):
                c_likelihood[next_state] = c_F[next_state, col+1] + c_log_tran[row, next_state] + c_log_emit[next_state, atoi(self.dif_seq[col+1])]
            c_B[row, col] = self.log_sum_all(c_likelihood)

def fb(self):
    self.forward()
    self.backward()
    self.X_p = self.log_sum_all(self.F[:,-1])
    self.P = self.F+self.B-self.X_p # still log-space
    self.find_mean()
    self.decode()

"""
Posterior mean calculation
"""
def find_mean(self):
    self.P_mean = np.sum(np.multiply(self.state,np.exp(self.P[:,col])), axis=0)
"""
Posterior probability decoding
"""
def decode(self):
    self.P_decoded = [self.state[i] for i in np.argmax(self.P[:,col], axis=0)]
