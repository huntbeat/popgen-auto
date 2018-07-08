'''
parse msms files and format info for natural selection strength network with TD
'''
import numpy as np
import sys
import h5py

def main():
    if len(sys.argv) != 2:
        print("USAGE: python3 generate_msms.py filename_for_new_data")
        sys.exit()

    # can be changed
    strengths_dict = {}
    for file_code in range(4):
        strength_label = np.zeros((4), dtype='int32')
        strength_label[file_code] = 1
        if file_code==0: key = 0
        else: key = 10**file_code
        strengths_dict[key] = strength_label

    sim_dir = '/scratch/saralab/first/'
    TMRCA_dir = '/scratch/saralab/first/TMRCA/'
    files = ['sim0_0.txt','sim10_0.txt','sim100_0.txt','sim1000_0.txt']

    # for the network
    xSNPs = []
    xTMRCAs = []
    xPositions = []
    yNS_strengths = []

    for f in files:
        strength, SNPs, positions = parse_sim(sim_dir+f,20,100000,strengths_dict)
        SNP_lengths = [mat.shape[1] for mat in SNPs]
        TMRCAs = parse_TMRCA(TMRCA_dir+f,20,SNP_lengths)
        SNPs_matrices, TMRCAs_matrices, position_matrices = centered_padding(SNPs,TMRCAs,positions,1500)
        xSNPs.extend(SNPs_matrices)
        xTMRCAs.extend(TMRCAs_matrices)
        xPositions.extend(position_matrices)
        #yNS_strengths.extend([strength for i in range(len(positions))])
        yNS_strengths.extend([strength for i in range(len(position_matrices))])
    assert(len(xSNPs)==len(xPositions)==len(yNS_strengths))
    xSNPs = np.array(xSNPs)
    xTMRCAs = np.array(xTMRCAs)
    xPositions = np.array(xPositions)
    yNS_strengths = np.array(yNS_strengths)

    # 3 channel input
    dims = xSNPs.shape
    snp_tmrca_pos = np.concatenate((np.reshape(xSNPs,(dims[0],1,dims[1],dims[2])),
                                    np.reshape(xTMRCAs,(dims[0],1,dims[1],dims[2])),
                                    np.reshape(xPositions,(dims[0],1,dims[1],dims[2])),
                                    ), axis=1)
    assert(snp_tmrca_pos.shape[1]==3)

    '''
    # 2 channel input
    snp_pos = np.concatenate((np.reshape(xSNPs,(dims[0],1,dims[1],dims[2])),
                            np.reshape(xPositions,(dims[0],1,dims[1],dims[2])),
                            ), axis=1)
    assert(snp_pos.shape[1]==2)
    '''

    dset_name = sys.argv[1]
    path = '/scratch/saralab/'+dset_name+'.hdf5'
    with h5py.File(path,'w') as ns:
        ns.create_dataset('SNPs', data=xSNPs)
        ns.create_dataset('TMRCAs', data=np.array(xTMRCAs))
        ns.create_dataset('positions', data=xPositions)
        ns.create_dataset('SNP_TMRCA_pos', data=snp_tmrca_pos)
        #ns.create_dataset('SNP_pos', data=snp_pos)
        ns.create_dataset('NS_strength', data=yNS_strengths)

###################################################################################

'''
@param filename - msms simulation file
@param n - number of individuals per simulation
@param L - length of sequence
@return natsel strength, SNPs matrices, position vectors
'''
def parse_sim(filename, n, L, strengths_dict):
    SNPs_matrices = [] # each element is an n by [num sites] SNPs matrix
    position_matrices = [] # each element is an n by [num sites] matrix of the corresponding [num sites] positions PLUS an end zero column
    num_sites_list = [] # corresponds to num sites per position vector
    count = 0 # to track parsing progress

    file_code = int(filename.split('/')[4][3:].split('_')[0]) # example filename: scratch/saralab/first/sim10_1.txt
    ns_strength = strengths_dict[file_code]

    print("nat sel data: parsing SNPs matrices...")
    file_ = open(filename,'r')
    lines = file_.readlines()
    for i in range(3,len(lines),n+4): # excluding header, moving through n SNP seqs
        assert(lines[i].strip()=='//')
        num_sites = int(lines[i+1].split(' ')[1])
        num_sites_list.append(num_sites)
        if num_sites == 0: # no seg sites
            position_matrices.append(np.zeros((n,0), dtype='int32'))
            SNPs_matrices.append(np.zeros((n,0), dtype='int32'))
        else:
            positions = lines[i+2].strip().split(' ')
            position_vector = [int(float(p)*L) for p in positions[1:]]
            position_vector = check_positions(position_vector)
            position_dists = position_distances(position_vector)
            position_matrix = np.tile(position_dists,(n,1))
            position_matrices.append(position_matrix)
            SNPs = []
            for j in range(n):
                str_SNPs = lines[i+3+j].strip()
                int_SNPs = np.array(list(str_SNPs), dtype='int32')
                SNPs.append(int_SNPs)
            SNPs = np.array(SNPs)
            assert(SNPs[0].shape[0]==num_sites==position_matrix[0].shape[0])
            assert(SNPs.shape[0]==n==position_matrix.shape[0])
            SNPs_matrices.append(SNPs)
        count += 1
        if count%100==0: print(count)
    file_.close()

    print("nat sel data: stats")
    print("min sites:",min(num_sites_list))
    print("max sites:",max(num_sites_list))
    print("avg sites:",sum(num_sites_list)/len(num_sites_list))
    over1500 = 0
    for s in num_sites_list:
        if s > 1500: over1500 += 1
    print("over 1500 sites:",over1500)

    return ns_strength, SNPs_matrices, position_matrices

'''
@param SNPs_matrices - seg sites
@param TMRCAs_matrices - respective TMRCA values
@param position_matrices - respective position vectors
@param length - length to extend all vectors to
'''
def centered_padding(SNPs_matrices, TMRCAs_matrices, position_matrices, length):
    print("nat sel data: padding sequence matrices...")
    count = 0
    uniform_SNPs = []
    uniform_TMRCAs = []
    uniform_pos_vecs = []
    #for i in range(len(SNPs_matrices)):
    for i in range(3900): # TMRCA cut short
        snp = SNPs_matrices[i]
        tmrca = TMRCAs_matrices[i]
        position_matrix = position_matrices[i]
        h, w = snp.shape
        if w >= length:
            reduced_snp = snp[:,:length]
            uniform_SNPs.append(reduced_snp)
            reduced_tmrca = tmrca[:,:length]
            uniform_TMRCAs.append(reduced_tmrca)
            reduced_pos_vec = position_matrix[:,:length]
            uniform_pos_vecs.append(reduced_pos_vec)
        else:
            padding_width = length-w
            zeros = np.zeros((h,padding_width), dtype='int32')
            half = int(padding_width/2)
            padded_snp = np.concatenate((zeros[:,:half],snp,zeros[:,half:]),axis=1)
            uniform_SNPs.append(padded_snp)
            padded_tmrca = np.concatenate((zeros[:,:half],tmrca,zeros[:,half:]),axis=1)
            uniform_TMRCAs.append(padded_tmrca)
            padded_pos_vec = np.concatenate((zeros[:,:half],position_matrix,zeros[:,half:]),axis=1)
            uniform_pos_vecs.append(padded_pos_vec)
        if count%100==0: print(count)
        count += 1
    return uniform_SNPs, uniform_TMRCAs, uniform_pos_vecs

'''
'''
def parse_TMRCA(filename, n, SNP_lengths):
    TMRCAs_matrices = [] # each element is an n by [num sites] TMRCAs matrix
    count = 0 # to track parsing progress and for checking TMRCA lengths

    print("nat sel data: parsing TMRCAs matrices...")
    file_ = open(filename,'r')
    lines = file_.readlines()
    for i in range(1,len(lines),n+1): # excluding header, moving through n TMRCA lines
        assert('>>' in lines[i])
        TMRCAs = []
        for j in range(1,n+1):
            str_TMRCAs = lines[i+j].strip()
            float_TMRCAs = np.array([float(x) for x in str_TMRCAs.split(' ')])
            TMRCAs.append(float_TMRCAs)
        TMRCAs = np.array(TMRCAs)

        # BANDAGE
        if TMRCAs[0].shape[0] > SNP_lengths[count]:
            TMRCAs = TMRCAs[:,:SNP_lengths[count]]
        if TMRCAs[0].shape[0] < SNP_lengths[count]:
            diff = SNP_lengths[count] - TMRCAs[0].shape[0]
            TMRCAs = np.concatenate((TMRCAs, np.zeros((n,diff))), axis=1)
        #

        assert(TMRCAs[0].shape[0]==SNP_lengths[count])
        assert(TMRCAs.shape[0]==n)
        TMRCAs_matrices.append(TMRCAs)
        if count%100==0: print(count)
        count += 1
    file_.close()
    return TMRCAs_matrices

'''
rare, but if two mutations are really close to each other,
they could be settling into the same position number. check
for that and separate them by a value of 1
'''
def check_positions(position_vector):
    for p in range(len(position_vector)-1):
        while position_vector[p] >= position_vector[p+1]:
            position_vector[p+1] += 1
    return position_vector

'''
calc distance between site positions
@param position_vector
@return position_distances_vector
'''
def position_distances(position_vector):
    distances = []
    for i in range(len(position_vector)-1):
        distance = position_vector[i+1] - position_vector[i]
        distances.append(distance)
    distances.append(0) # last column zero padding
    return np.array(distances)


###################################################################################

main()
