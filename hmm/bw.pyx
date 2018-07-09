import numpy as np
from libc.math cimport log, exp
from libc.stdlib cimport atoi, malloc, free, strtol
cimport cython

"""
C++ version
"""
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double log_sum(double log_p, double log_q):
    return log_p + log(1+exp(log_q-log_p))
"""
C++ version, assumes that array is longer than 1
"""
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double log_sum_all(double [:] things, int length):
    cdef double sig
    sig = log_sum(things[0],things[1])
    for i in range(2,length):
        sig = log_sum(sig, things[i])
    return sig

class FB:

    def __init__(self, dif_seq, log_init, log_tran, log_emit, state):
        self.dif_seq = dif_seq
        self.log_init = log_init
        self.log_tran = log_tran
        self.log_emit = log_emit
        self.state = np.array(state)
        self.K = len(self.log_init)
        self.L = len(self.dif_seq)
        self.likelihood = np.zeros(self.K)
        self.F = np.zeros((self.K,self.L)) # K x L
        self.F[:,0] = self.log_init + self.log_emit[:,int(self.dif_seq[0])]
        self.B = np.zeros((self.K,self.L))
        self.B[:,-1] = 0
        self.P = np.zeros((self.K,self.L)) # posterior probability table
        self.P_mean = np.zeros((self.L)) # posterior mean
        self.P_decoded = np.zeros((self.L)) # posterior decoding
        self.X_p = 0

        self.fb()

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def forward(self):

        cdef double [:,:] c_F = self.F

        cdef Py_ssize_t indice

        cdef double [:] c_log_init = self.log_init
        cdef double [:,:] c_log_tran = self.log_tran
        cdef double [:,:] c_log_emit = self.log_emit

        cdef Py_ssize_t c_K = self.K
        cdef Py_ssize_t c_L = self.L

        cdef double [:] c_likelihood = self.likelihood

        for col in range(c_L-1):
            col += 1
            for row in range(c_K):
                for prev_state in range(c_K):
                    c_likelihood[prev_state] = c_F[prev_state, col-1] + c_log_tran[prev_state, row]
                likelihood = log_sum_all(c_likelihood, c_K)
                indice = int(self.dif_seq[col])
                c_F[row, col] = likelihood + c_log_emit[row, indice]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def backward(self):

        cdef double [:,:] c_B = self.B

        cdef Py_ssize_t indice

        cdef double [:] c_log_init = self.log_init
        cdef double [:,:] c_log_tran = self.log_tran
        cdef double [:,:] c_log_emit = self.log_emit

        cdef Py_ssize_t c_K = self.F.shape[0]
        cdef Py_ssize_t c_L = self.F.shape[1]

        cdef double [:] c_likelihood = self.likelihood

        for col in range(c_L-1):
            col = c_L - 2 - col
            for row in range(c_K):
                for next_state in range(c_K):
                    indice = int(self.dif_seq[col+1])
                    c_likelihood[next_state] = c_B[next_state, col+1] + c_log_tran[row, next_state] + c_log_emit[next_state, indice]
                c_B[row, col] = log_sum_all(c_likelihood, c_K)

    def fb(self):
        self.forward()
        self.backward()
        self.X_p = log_sum_all(self.F[:,-1], len(self.F[:,-1]))
        self.P = self.F+self.B-self.X_p # still log-space
        self.find_mean()
        self.decode()

    """
    Posterior mean calculation
    """
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def find_mean(self):
        cdef double [:] c_state = self.state
        cdef double [:,:] c_P = self.P
        cdef double [:] c_P_mean = self.P_mean
        cdef double hold

        cdef Py_ssize_t c_K = self.P.shape[0]
        cdef Py_ssize_t c_L = self.P.shape[1]

        for col in range(c_L):
            hold = 0
            for row in range(c_K):
                hold += c_state[row] * exp(c_P[row,col])
            c_P_mean[col] = hold
    """
    Posterior probability decoding
    """
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def decode(self):
        self.P_decoded = np.array([self.state[i] for i in np.argmax(self.P, axis=0)])


class BW:

    def __init__(self, dif_seq, log_init, log_tran, log_emit, state, update=False):
        self.dif_seq = dif_seq
        self.fb = FB(dif_seq=dif_seq, log_init=log_init,
                   log_tran=log_tran, log_emit=log_emit, state=state)
        self.X_p_list = [self.fb.X_p]   # the optimization marker
                                        # starts at 0 for initial
        self.u_log_init = None
        self.u_log_tran = None
        self.u_log_emit = None

        if update:
            self.update_init()
            self.update_tran()
            self.update_emit()

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update_tran(self):
        self.u_log_tran = np.zeros(np.shape(self.fb.log_tran))

        cdef double [:,:] c_u_log_tran = self.u_log_tran

        cdef double [:,:] c_log_tran = self.fb.log_tran
        cdef double [:,:] c_log_emit = self.fb.log_emit
        cdef double [:,:] c_F = self.fb.F
        cdef double [:,:] c_B = self.fb.B

        cdef Py_ssize_t c_K = self.fb.K
        cdef Py_ssize_t c_L = self.fb.L

        cdef double c_X_p = self.fb.X_p

        for col in range(c_L-1):
            for prev_state in range(c_K):
                for cur_state in range(c_K):
                    indice = int(self.dif_seq[col+1])
                    prob = (c_F[prev_state, col]+
                            c_log_tran[prev_state, cur_state]+
                            c_log_emit[cur_state, indice]+
                            c_B[cur_state, col] - c_X_p)
                    if not c_u_log_tran[prev_state, cur_state] == 0:
                        c_u_log_tran[prev_state, cur_state] = log_sum(c_u_log_tran[prev_state, cur_state], prob)
                    else:
                        c_u_log_tran[prev_state, cur_state] = prob
        # Normalize
        normalize = np.zeros((self.fb.K))
        cdef double [:] c_normalize = normalize
        for prev_state in range(c_K):
            c_normalize[prev_state] = log_sum_all(c_u_log_tran[prev_state,:], c_K)
        for cur_state in range(c_K):
            for next_state in range(c_K):
                c_u_log_tran[cur_state, next_state] = c_u_log_tran[cur_state, next_state] - c_normalize[cur_state]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update_emit(self):
        self.u_log_emit = np.zeros(np.shape(self.fb.log_emit))

        cdef double [:,:] c_u_log_emit = self.u_log_emit

        cdef double [:,:] c_log_tran = self.fb.log_tran
        cdef double [:,:] c_log_emit = self.fb.log_emit
        cdef double [:,:] c_F = self.fb.F
        cdef double [:,:] c_B = self.fb.B
        cdef double [:,:] c_P = self.fb.P

        cdef Py_ssize_t c_K = self.fb.K
        cdef Py_ssize_t c_L = self.fb.L

        cdef double c_X_p = self.fb.X_p

        for col in range(c_L):
            for cur_state in range(c_K):
                indice = int(self.dif_seq[col])
                if not self.u_log_emit[cur_state, indice] == 0:
                    self.u_log_emit[cur_state, indice] = log_sum(c_P[cur_state, col], c_u_log_emit[cur_state, indice])
                else:
                    self.u_log_emit[cur_state, indice] = c_P[cur_state, col]
        normalize = np.zeros((self.fb.K))
        cdef double [:] c_normalize = normalize
        for prev_state in range(c_K):
            c_normalize[prev_state] = log_sum_all(c_u_log_emit[prev_state, :], 2)
        for cur_state in range(c_K):
            for obs in range(2):
                c_u_log_emit[cur_state, obs] = c_u_log_emit[cur_state, obs] - c_normalize[cur_state]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update_init(self):
        self.u_log_init = self.fb.F[:,0] + self.fb.B[:,1] - self.fb.X_p
