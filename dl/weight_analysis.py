from analyze_network import get_weights
import numpy as np
import matplotlib.pyplot as plt

wd_T = get_weights('/home/nhoang1/saralab/popgen-hmm-dl/dl/TD_pop/TD_pop_model.hdf5')
wd_D = get_weights('/home/nhoang1/saralab/popgen-hmm-dl/dl/dupl_pop/dupl_pop_model.hdf5')
wd_R = get_weights('/home/nhoang1/saralab/popgen-hmm-dl/dl/rand_pop/rand_pop_model.hdf5')
wm_pop_origT = wd_T['pop'] # weight matrix for population classification
wm_pop_origD = wd_D['pop_1'] # weight matrix for population classification
wm_pop_origR = wd_R['pop_2'] # weight matrix for population classification
wm_TD  = wd_T['TD']  # for Tajima's D classification
wm_DP  = wd_D['TD_1']  # for Tajima's D classification
wm_RN  = wd_R['TD_2']  # for Tajima's D classification

constant = wm_pop_origT[:,0]
bottleneck = wm_pop_origD[:,1]
nat_sel = wm_pop_origR[:,2]

wm_pop = np.zeros(wm_pop_orig.shape)

wm_pop[:,0] = nat_sel
wm_pop[:,1] = constant
wm_pop[:,2] = bottleneck

print(np.sum(wm_pop[0,:]))
print(np.sum(wm_pop[1,:]))

wm_difTD = wm_TD - wm_pop_origT
wm_difDP = wm_DP - wm_pop_origD
wm_difRN = wm_RN - wm_pop_origR
wm_abs_difTD = np.absolute(wm_difTD)
wm_abs_difDP = np.absolute(wm_difDP)
wm_abs_difRN = np.absolute(wm_difRN)

plt.figure(1)
plt.plot(np.arange(1,257),np.sort(wm_abs_difTD, axis=0))
plt.legend(['< 0, nat_sel','= 0, constant','> 0, bottleneck'])
plt.title('Weights for TD, separate')
plt.show()

plt.figure(2)
plt.plot(np.arange(1,257),np.sort(np.sum(wm_abs_difTD, axis=1)/3.0))
plt.title('Weights for TD, combined')
plt.show()

plt.figure(3)
plt.plot(np.arange(1,257),np.sort(wm_abs_difDP, axis=0))
plt.legend(['< 0, nat_sel','= 0, constant','> 0, bottleneck'])
plt.title('Weights for DP, separate')
plt.show()

plt.figure(4)
plt.plot(np.arange(1,257),np.sort(np.sum(wm_abs_difDP, axis=1)/3.0))
plt.title('Weights for DP, combined')
plt.show()

plt.figure(5)
plt.plot(np.arange(1,257),np.sort(wm_abs_difRN, axis=0))
plt.legend(['< 0, nat_sel','= 0, constant','> 0, bottleneck'])
plt.title('Weights for RN, separate')
plt.show()

plt.figure(6)
plt.plot(np.arange(1,257),np.sort(np.sum(wm_abs_difRN, axis=1)/3.0))
plt.title('Weights for RN, combined')
plt.show()
