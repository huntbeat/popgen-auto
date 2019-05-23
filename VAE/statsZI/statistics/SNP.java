package statistics;

import java.util.Arrays;

/**
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Basic struct for holding a SNP.
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */

public class SNP {
	
	public final int index;
	public final int[] alleles;

	// the index of the snp (starting from 0)
	// a list of binary alleles for each haplotype (i.e. [0,1,0,1,1])
	public SNP(int index, int[] alleles) {
		this.index = index;
	    this.alleles = alleles;
	}
	
	@Override
	public String toString() {
		return this.index + ": " + Arrays.toString(this.alleles);
	}
}
