"""
Baum-Welch ---- This runs the Baum-Welch algorithm for the HMM
                and outputs the appropriate posterior statistics.

Arguments: seq_file -

Hunter Lee
Sam Shih
"""

from viterbi import FB
import numpy as np
import math

class BW:

    def __init__(self, seq_file, log_init, log_tran, log_emit, state, i):
        self.seq_file = seq_file
        self.fb = FB(seq_file=self.seq_file, log_init=log_init,
                   log_tran=log_tran, log_emit=log_emit, state=state)
        self.i = i
        self.X_p_list = [self.fb.X_p]   # the optimization marker
                                        # starts at 0 for initial
        self.u_log_tran = None
        self.u_log_emit = None
        self.u_log_init = None

        for rep in range(i):
            self.update()

    def log_sum(self, log_p, log_q):
        return log_p + math.log(1+math.exp(log_q-log_p))

    def log_sum_all(self, array):
        sig = self.log_sum(array[0],array[1])
        for i in range(2,len(array)):
            sig = self.log_sum(sig,array[i])
        return sig

    def update_tran(self):
        self.u_log_tran = np.zeros(np.shape(self.fb.log_tran))
        for col in range(self.fb.L-1):
            for prev_state in range(self.fb.K):
                for cur_state in range(self.fb.K):
                    prob = (self.fb.F[prev_state][col]+
                            self.fb.log_tran[prev_state][cur_state]+
                            self.fb.log_emit[cur_state][int(self.fb.seq_dif[col+1])]+
                            self.fb.B[cur_state][col] - self.fb.X_p)
                            # F*B*... -> log F + log B
                            # log sum F*B -> log_sum (log F + log B)
                    if not self.u_log_tran[prev_state][cur_state] == 0:
                        self.u_log_tran[prev_state][cur_state] = self.log_sum(self.u_log_tran[prev_state][cur_state], prob)
                    else:
                        self.u_log_tran[prev_state][cur_state] = prob
        # Normalize
        normalize = np.zeros((self.fb.K))
        for prev_state in range(self.fb.K):
            normalize[prev_state] = self.log_sum_all(self.u_log_tran[prev_state][:])
        for cur_state in range(self.fb.K):
            for next_state in range(self.fb.K):
                self.u_log_tran[cur_state][next_state] = self.u_log_tran[cur_state][next_state] - normalize[cur_state]

    def update_emit(self):
        self.u_log_emit = np.zeros(np.shape(self.fb.log_emit))
        for col in range(self.fb.L):
            for cur_state in range(self.fb.K):
                if not self.u_log_emit[cur_state][int(self.fb.seq_dif[col])] == 0:
                    self.u_log_emit[cur_state][int(self.fb.seq_dif[col])] = self.log_sum(self.fb.P[cur_state][col], self.u_log_emit[cur_state][int(self.fb.seq_dif[col])])
                else:
                    self.u_log_emit[cur_state][int(self.fb.seq_dif[col])] = self.fb.P[cur_state][col]
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

    def update(self):
        self.update_tran()
        self.update_emit()
        self.update_init()
        self.fb = FB(seq_file=self.seq_file, log_init=self.u_log_init, log_tran=self.u_log_tran, log_emit=self.u_log_emit, state=self.fb.state)
        self.X_p_list.append(self.fb.X_p)
