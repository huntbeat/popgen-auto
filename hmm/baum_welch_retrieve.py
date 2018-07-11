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

    def __init__(self, dif_seq, log_init, log_tran, log_emit, state, update=True):
        self.dif_seq = dif_seq
        self.fb = FB(dif_seq=dif_seq, log_init=log_init,
                   log_tran=log_tran, log_emit=log_emit, state=state)
        self.X_p_list = [self.fb.X_p]   # the optimization marker
                                        # starts at 0 for initial
        self.u_log_tran = None
        self.u_log_emit = None
        self.u_log_init = None

    def log_sum(self, log_p, log_q):
        return log_p + math.log(1+math.exp(log_q-log_p))

    def log_sum_all(self, array):
        sig = self.log_sum(array[0],array[1])
        for i in range(2,len(array)):
            sig = self.log_sum(sig,array[i])
        return sig
