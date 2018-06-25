from analyze_network import get_weights
import numpy as np
import matplotlib.pyplot as plt

wm_abs_dif_list = []

for arguments in [('/home/nhoang1/saralab/popgen-hmm-dl/dl/TD_pop/TD_pop_model.hdf5', 'pop', 'TD', 'TD'), 
                  ('/home/nhoang1/saralab/popgen-hmm-dl/dl/dupl_pop/dupl_pop_model.hdf5', 'pop_1', 'TD_1', 'duplicate'),
                  ('/home/nhoang1/saralab/popgen-hmm-dl/dl/rand_pop/rand_pop_model.hdf5', 'pop_2', 'TD_2', 'random')]:
    wd = get_weights(arguments[0])
    wm_pop_orig = wd[arguments[1]]
    wm = wd[arguments[2]]

    constant = wm_pop_orig[:,0]
    bottleneck = wm_pop_orig[:,1]
    nat_sel = wm_pop_orig[:,2]

    wm_pop = np.zeros(wm_pop_orig.shape)

    wm_pop[:,0] = nat_sel
    wm_pop[:,1] = constant
    wm_pop[:,2] = bottleneck

    wm_dif = wm - wm_pop_orig
    wm_abs_dif = np.absolute(wm_dif)

    wm_abs_dif_list.append(np.sort(np.sum(wm_abs_dif, axis=1)/3.0))

    plt.figure(1)
    plt.plot(np.arange(1,257),np.sort(wm_abs_dif, axis=0))
    plt.legend(['< 0, nat_sel','= 0, constant','> 0, bottleneck'])
    plt.title('Weights for '+arguments[3] +', separate')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(1,257),np.sort(np.sum(wm_abs_dif, axis=1)/3.0))
    plt.title('Weights for '+arguments[3] +', combined')
    plt.show()
    
    plt.close()

plt.plot(np.arange(1,257),wm_abs_dif_list[0]) 
plt.plot(np.arange(1,257),wm_abs_dif_list[1])
plt.plot(np.arange(1,257),wm_abs_dif_list[2])
plt.title('Summed weight abs dif for three outputs')
plt.legend(['TD', 'DP', 'RN'])
plt.show()
