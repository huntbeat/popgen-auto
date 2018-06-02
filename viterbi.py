"""
Viterbi class
1. Create a difference string of two sequences
2. Create a 2D DP table dimensions KxL
3. Initialize the first column with log_init probabilities
4. Run Viterbi on the table
5. Return backtrace of hidden state sequence
"""

import numpy as np
import math

class Viterbi:

    def __init__(self, seq_file, log_init, log_tran, log_emit, state):
        self.seq_dif = self.find_dif(seq_file)
        self.log_init = log_init
        self.log_tran = log_tran
        self.log_emit = log_emit
        self.state = state
        self.K = len(self.log_init)
        self.L = len(self.seq_dif)
        self.dp = np.zeros((self.K,self.L)) # K x L
        self.dp[:,0] = self.log_init
        self.backtrace = np.zeros((self.K,self.L))
        self.path = []

        self.viterbi()
        self.find_path()

    def find_dif(self, seq_file):
        seq_1 = ""
        seq_2 = ""
        # take in two sequences
        with open(seq_file) as f:
            next(f)
            for line in f:
                new_line = line.strip("\n")
                if new_line[0] == ">":
                    break
                else:
                    seq_1 += new_line
            for line in f:
                new_line = line.strip("\n")
                seq_2 += new_line
        # compare the two
        diff_string = ""
        for i in range(0, len(seq_1)):
            diff_string += "0" if seq_1[i] == seq_2[i] else "1"
        return diff_string

    """
    For every element in a column, link up the elements of the previous column
    to the current element by multiplying each element of the previous column
    with the transition probability.
    dp[k_i][l_i] is the probability to transition into state k_i given that
    you were at k_(i-1) before
    """
    def viterbi(self):
        for col in range(1,self.L):
            for row in range(self.K):
                likelihood = [] # probabilities from column-1
                for prev_state in range(self.K):
                    likelihood.append(self.dp[prev_state][col-1] + self.log_tran[prev_state][row])
                self.backtrace[row][col] = likelihood.index(max(likelihood))
                # likelihood + log_emission of emitting: curr hidden (row) -> curr observation (col)
                self.dp[row][col] = max(likelihood) + self.log_emit[row][int(self.seq_dif[col])]
    """
    After running Viterbi, this method will find the path using the max_prob
    at the last column of self.dp, then self.backtrace to trace back from there.
    """
    def find_path(self):
        last_col = list(self.dp[:,-1])
        begin = last_col.index(max(last_col))
        self.path.append(begin)
        for col in range(self.L-1,0,-1):
            self.path.append(self.backtrace[self.path[-1]][col])
        self.path.reverse()
        viterbi_path = []
        for ind in self.path:
            viterbi_path.append(self.state[ind])
        self.path = viterbi_path


"""
Forward-Backward
1.
2.
"""
class FB:

    def __init__(self, seq_file, log_init, log_tran, log_emit, state):
        self.seq_dif = self.find_dif(seq_file)
        self.log_init = log_init
        self.log_tran = log_tran
        self.log_emit = log_emit
        self.state = np.array(state)
        self.K = len(self.log_init)
        self.L = len(self.seq_dif)
        self.F = np.zeros((self.K,self.L)) # K x L
        self.F[:,0] = self.log_init + self.log_emit[:,int(self.seq_dif[0])]
        self.B = np.zeros((self.K,self.L))
        self.B[:,-1] = 0
        self.P = np.zeros((self.K,self.L)) # posterior probability table
        self.P_mean = np.zeros((self.L)) # posterior mean
        self.P_decoded = np.zeros((self.L)) # posterior decoding
        self.X_p = 0

        self.fb()

    def find_dif(self, seq_file):
        seq_1 = ""
        seq_2 = ""
        # take in two sequences
        with open(seq_file) as f:
            next(f)
            for line in f:
                new_line = line.strip("\n")
                if new_line[0] == ">":
                    break
                else:
                    seq_1 += new_line
            for line in f:
                new_line = line.strip("\n")
                seq_2 += new_line
        # compare the two
        diff_string = ""
        for i in range(0, len(seq_1)):
            diff_string += "0" if seq_1[i] == seq_2[i] else "1"
        return diff_string
    """
    log_sum
    """
    def log_sum(self, log_p, log_q):
        return log_p + math.log(1+math.exp(log_q-log_p))

    """
    log_sum_all
    """
    def log_sum_all(self, array):
        sig = self.log_sum(array[0],array[1])
        for i in range(2,len(array)):
            sig = self.log_sum(sig,array[i])
        return sig

    """
    Forward
    """
    def forward(self):
        for col in range(1,self.L):
            for row in range(self.K):
                likelihood = []
                for prev_state in range(self.K):
                    likelihood.append(self.F[prev_state][col-1] + self.log_tran[prev_state][row])
                likelihood = self.log_sum_all(likelihood)
                self.F[row][col] = likelihood + self.log_emit[row][int(self.seq_dif[col])]

    """
    Backward
    """
    def backward(self):
        for col in range(self.L-2,-1,-1):
            for row in range(self.K):
                likelihood = []
                for next_state in range(self.K):
                    likelihood.append(self.B[next_state][col+1] + self.log_tran[row][next_state] + self.log_emit[next_state][int(self.seq_dif[col+1])])
                self.B[row][col] = self.log_sum_all(likelihood)

    """
    FB
    Runs forward and backward iterations and the following definitions
    """
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
        for col in range(0,self.L):
            self.P_mean[col] = sum(np.multiply(self.state,np.exp(self.P[:,col])))
    """
    Posterior probability decoding
    """
    def decode(self):
        for col in range(0,self.L):
            self.P_decoded[col] = self.state[np.argmax(self.P[:,col])]
