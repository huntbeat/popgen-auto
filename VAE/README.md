Variational auto-encoding for genomic data

1. Get real (or simulated) data in msms format -- use vcf2msms.py
2. Feed it to the STAT java script -- statsZI from evoNet
3. Format STAT txt output to VAE input data -- use stat2input.py
4. Run! -- VAE.py

Pipeline

0. Get desired real subsequence (within sequence trained by autoencoder) and find label values
1. Simulate msms sequences based on fixed and variable parameters (we're trying to train the variable parameters)
2. Get summary statistics
3. Translate summary statistics to autoencoder data
4. Optimize (Adam) and return new variable parameters
5. Loop to step 1. 
