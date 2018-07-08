import numpy as np
from libc.math cimport log, exp
from libc.stdlib cimport atoi

DTYPE = np.floatc

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
    """
    C++ version
    """
    def log_sum(self, float log_p, float log_q):
        return log_p + log(1+exp(log_q-log_p))
    """
    C++ version, assumes that array is longer than 1
    """
    def log_sum_all(self, array, int length):
        sig = self.log_sum(array[0],array[1])
        for i in range(2,length):
            sig = self.log_sum(sig,array[i])
        return sig

    def forward(self):

        cdef float [:,:] c_F = self.F

        cdef float [:,:] c_log_init = self.log_init
        cdef float [:,:] c_log_tran = self.log_tran
        cdef float [:,:] c_log_emit = self.log_emit

        cdef Py_ssize_t c_K = self.K
        cdef Py_ssize_t c_L = self.L

        cdef float [:] c_likelihood = self.likelihood

        for col in range(1, c_L):
            for row in range(c_K):
                for prev_state in range(c_K):
                    c_likelihood[prev_state] = c_F[prev_state, col-1] + c_log_tran[prev_state, row]
                likelihood = self.log_sum_all(c_likelihood, c_K)
                c_F[row, col] = likelihood + c_log_emit[row, atoi(self.dif_seq[col])]

    def backward(self):

        cdef float [:,:] c_B = self.B

        cdef float [:,:] c_log_init = self.log_init
        cdef float [:,:] c_log_tran = self.log_tran
        cdef float [:,:] c_log_emit = self.log_emit

        cdef Py_ssize_t c_K = self.F.shape[0]
        cdef Py_ssize_t c_L = self.F.shape[1]

        cdef float [:] c_likelihood = self.likelihood

        for col in range(c_L-2, -1, -1):
            for row in range(c_K):
                for next_state in range(c_K):
                    c_likelihood[next_state] = c_B[next_state, col+1] + c_log_tran[row, next_state] + c_log_emit[next_state, atoi(self.dif_seq[col+1])]
                c_B[row, col] = self.log_sum_all(c_likelihood, c_K)

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
        self.P_mean = np.sum(np.multiply(self.state,np.exp(self.P)), axis=0)
    """
    Posterior probability decoding
    """
    def decode(self):
        self.P_decoded = [self.state[i] for i in np.argmax(self.P, axis=0)]


class BW:

    def __init__(self, dif_seq, log_init, log_tran, log_emit, state, update=False):
        self.dif_seq = dif_seq
        self.fb = FB(dif_seq=dif_seq, log_init=log_init,
                   log_tran=log_tran, log_emit=log_emit, state=state)
        self.X_p_list = [self.fb.X_p]   # the optimization marker
                                        # starts at 0 for initial

"""
    def log_sum(self, float log_p, float log_q):
        return log_p + log(1+exp(log_q-log_p))

    def log_sum_all(self, float array[], int length):
        sig = self.log_sum(array[0],array[1])
        for i in range(2,length):
            sig = self.log_sum(sig,array[i])
        return sig

    def update_tran(self):
        self.u_log_tran = np.zeros(np.shape(self.fb.log_tran))
        cdef float [:,:] c_u_log_tran = self.u_log_tran

        cdef float [:,:] c_log_tran = self.fb.log_tran
        cdef float [:,:] c_log_emit = self.fb.log_emit
        cdef float [:,:] c_F = self.fb.F
        cdef float [:,:] c_B = self.fb.B

        cdef Py_ssize_t c_K = self.fb.K
        cdef Py_ssize_t c_L = self.fb.L
        
        cdef float c_X_P = self.fb.X_p

        for col in range(c_L-1):
            for prev_state in range(c_K):
                for cur_state in range(c_K):
                    prob = (c_F[prev_state, col]+
                            c_log_tran[prev_state,cur_state]+
                            c_log_emit[cur_state, atoi(self.dif_seq[col+1])]+
                            c_B[cur_state, col] - c_X_p)
                            # F*B*... -> log F + log B
                            # log sum F*B -> log_sum (log F + log B)
                    if not c_u_log_tran[prev_state][cur_state] == 0:
                        c_u_log_tran[prev_state, cur_state] = self.log_sum(c_u_log_tran[prev_state, cur_state], prob)
                    else:
                        c_u_log_tran[prev_state, cur_state] = prob
        # Normalize
        normalize = np.zeros((self.fb.K))
        cdef float [:] c_normalize = normalize
        for prev_state in range(c_K):
            c_normalize[prev_state] = self.log_sum_all(c_u_log_tran[prev_state,:])
        for cur_state in range(c_K):
            for next_state in range(c_K):
                c_u_log_tran[cur_state, next_state] = c_u_log_tran[cur_state, next_state] - c_normalize[cur_state]

    def update_emit(self):
        self.u_log_emit = np.zeros(np.shape(self.fb.log_emit))
        for col in range(self.fb.L):
            for cur_state in range(self.fb.K):
                if not self.u_log_emit[cur_state][int(self.fb.dif_seq[col])] == 0:
                    self.u_log_emit[cur_state][int(self.fb.dif_seq[col])] = self.log_sum(self.fb.P[cur_state][col], self.u_log_emit[cur_state][int(self.fb.dif_seq[col])])
                else:
                    self.u_log_emit[cur_state][int(self.fb.dif_seq[col])] = self.fb.P[cur_state][col]
        normalize = np.zeros((self.fb.K))
        for prev_state in range(self.fb.K):
            normalize[prev_state] = self.log_sum_all(self.u_log_emit[prev_state][:])
        for cur_state in range(self.fb.K):
            for obs in range(np.shape(self.fb.log_emit)[1]):
                self.u_log_emit[cur_state][obs] = self.u_log_emit[cur_state][obs] - normalize[cur_state]

    def update_init(self):
        self.u_log_init = np.zeros(np.shape(self.fb.log_init))
        for state in range(self.fb.K):
            self.u_log_init[state] = self.fb.F[state][0] + self.fb.B[state][1] - self.fb.X_p
"""
