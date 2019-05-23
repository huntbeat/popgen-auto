package statistics;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import utility.Util;

/**
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Computes our specific set of statistics (these methods could be swapped in and out
 * to create a set of tailored summary statistics).
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */
public class HapConfigZI {
	
	// globals about msms file
	private final int N_IDX = 3; // index of the number of haplotypes
	private final int NUM_FILL_LINES = 3; // number of "filler" lines between first line and seg site positions
	
	private final int n;                   // number of haplotypes (i.e. 197)
	private final int nThresh;             // number of individuals in simulated data (i.e. 100)
	private final List<SNP> snpList;       // list of SNPs for all 197 haplotypes
	private final List<SNP> snpListThresh; // list of SNPs for the "best" sfsThresh individuals
	
	
	// L is the length of the sequence, which doesn't matter so much right now, but might later on
	public HapConfigZI(String msmsFilename, int L, int nThresh) throws IOException {
		
		BufferedReader reader = new BufferedReader(new FileReader(msmsFilename));
		this.nThresh = nThresh;
		
		// parse first line 
		String[] tokens = Util.readLine(reader).trim().split(" ");
		this.n = Integer.parseInt(tokens[N_IDX]);
		
		// parse intermediate lines
		for (int k=0; k < NUM_FILL_LINES; k++) {
			Util.readLine(reader);
		}
		
		// parse line of seg site positions to create the segSitesMap (dictionary)
		Map<Integer,Integer> segSiteMap = new HashMap<Integer,Integer>();
		String segSiteStr = Util.readLine(reader).split(":")[1].trim(); // this is to get rid of the word positions
		String[] segSiteRatios = segSiteStr.split(" ");
		int[] segSitePositions = new int[segSiteRatios.length];
		
		for (int p=0; p < segSitePositions.length; p++) {
			double siteRatio = Double.parseDouble(segSiteRatios[p]);
	        int sitePos = Math.min(L-1, (int)Math.round(siteRatio * L));

	        if (segSiteMap.containsKey(sitePos)) {
	        	Integer val = segSiteMap.get(sitePos);
	        	segSiteMap.put(sitePos, val + 1);

	        } else {
	        	segSiteMap.put(sitePos, 1);
	        }   
		}
		
		// final step for the positions is to create a sorted list of them (using a map might have mixed them up)
		List<Integer> sortedIndices = new ArrayList<Integer>(segSiteMap.keySet());
		Collections.sort(sortedIndices);
		
		// now parse all the haplotype lines
		List<String[]> hapList = new ArrayList<String[]>();
		for (int i=0; i < this.n; i++) {
			String[] segHap = Util.readLine(reader).trim().split(""); // string[] of seg alleles for this haplotype
			hapList.add(segHap);
		}
		
		// sort the haplotypes by number of Ns so we can easily get the best ones
		Collections.sort(hapList, new UnknownComparator());
		
		// invert the hapList into a list organized by SNP
		List<List<Integer>> hapInvertedList = new ArrayList<List<Integer>>();
		for (int i=0; i < this.n; i++) {
			String[] segHap = hapList.get(i);
			List<Integer> hapInverted = new ArrayList<Integer>();
			
			int segSiteCount = 0; // smathieson: change from 1 to 0, might be fix for Java version 1.8
		    for (Integer p : sortedIndices) {
		    	if (segSiteMap.get(p) == 1) {
		    		hapInverted.add(convertAllele(segHap[segSiteCount]));
		            segSiteCount += 1;
		            
		        // more than one mutation at this locus
		    	} else {
                    int numMut = segSiteMap.get(p);
                    //System.out.println("more than one mut " + numMut);
                    
                    String[] muts = Util.subset(segHap, segSiteCount, numMut); // args: array, startIdx, length
                    int toAdd = Util.count(muts, "1") % 2; // mod by 2 (i.e. it "flips" from 0->1 or 1->0 each time)
                    hapInverted.add(toAdd);
                    segSiteCount += numMut;
		    	}
		    }
		    hapInvertedList.add(hapInverted);
		}
		
		// go through hapInvertedList to create our list of SNPs, and "thresh" list of SNPs
		this.snpList = new ArrayList<SNP>();
		this.snpListThresh = new ArrayList<SNP>();
		int numSnps = hapInvertedList.get(0).size();
		
		for (int j=0; j < numSnps; j++) {
			int[] alleles = new int[this.n];
			int[] allelesThresh = new int[this.nThresh];
			
			for (int k=0; k < this.n; k++) { 
				alleles[k] = hapInvertedList.get(k).get(j);
			}
			for (int k=0; k < this.nThresh; k++) {
				allelesThresh[k] = hapInvertedList.get(k).get(j);
			}
			
			// add SNP if is is segregating
		    if (isSegregating(alleles)) {
		    	SNP newSnp = new SNP(sortedIndices.get(j),alleles);
		    	this.snpList.add(newSnp);
		    }
		    
		    if (isSegregating(allelesThresh)) {
		    	SNP newSnp = new SNP(sortedIndices.get(j),allelesThresh);
		    	this.snpListThresh.add(newSnp);
		    }
		}
	}
	
	// convert the msms format ('0' ancestral, '1' derived, 'N' unknown)
	// into our integer format
	private int convertAllele(String allele) {
		if (allele.equals("0")) {
			return 0;
		} else if (allele.equals("N")) {
			return -1;
		} else {
			return 1; // this includes letters/numbers in msms that represent different backgrounds for selected allele
		}
	}
	
	// sort haps according to number of unknown SNP data (Ns)
	class UnknownComparator implements Comparator<String[]> {
	    @Override
	    public int compare(String[] s1, String[] s2) {
	    	int count1 = Util.count(s1, "N");
	    	int count2 = Util.count(s2, "N");
	        return count1 < count2 ? -1 : count1 == count2 ? 0 : 1;
	    }
	}
	
	// determine whether a list of alleles is segregating
	private boolean isSegregating(int[] alleles) {
		int count0 = Util.count(alleles, 0);
		int count1 = Util.count(alleles, 1);
		if (count0 > 0 && count1 > 0) { return true; } 
		return false;
	}
	
	// -----------------------0--------------------------
	// number of segregating sites (use first sfsThresh)
	public int numSegSites(int regionStart, int regionEnd) {
		int snpCount = 0;
		for (SNP snp : this.snpListThresh) {
			if (snp.index >= regionStart && snp.index < regionEnd) {
				snpCount += 1;
			}
		}
		return snpCount;
	}
	
	// Tajima's D (takes in S, number of segregating sites, as computed above)
	public double tajimasD(int S, double kHat) {
		double a1 = 0; double a2 = 0;
		for (int i=1; i < this.nThresh; i++) { 
			a1 += 1/((double)i);
			a2 += 1/((double)i*i);
		}
		
		double b1 = (this.nThresh + 1)/((double)3*(this.nThresh-1));
		double b2 = 2*(this.nThresh*this.nThresh + this.nThresh + 3)/((double)9*this.nThresh*(this.nThresh-1));
		
		double c1 = b1 - 1/a1;
		double c2 = b2 - (this.nThresh+2)/(a1*this.nThresh) + a2/(a1*a1);
		
		double e1 = c1/a1;
		double e2 = c2/(a1*a1 + a2);
		
		double D = (kHat - S/a1)/Math.sqrt(e1*S + e2*S*(S-1));
		if (Double.isNaN(D)) {
			return 0;
		}
		return D;
	}
	
	// compute average number of SNPs when considering all pairs of individuals
	public double computeKhat(int regionStart, int regionEnd) {
		int sumKij = 0;
		for (int h1=0; h1 < this.nThresh; h1++) { // hap1
			for (int h2=h1+1; h2 < this.nThresh; h2++) { // hap2
				for (SNP snp : this.snpListThresh) { // use thresh here for consistency w/ simulated data
		        	if (snp.index >= regionStart && snp.index < regionEnd) {
		        		if (!alleleEquals(snp.alleles[h1], snp.alleles[h2])) {
		        			sumKij += 1;
		        		}
		        	}
				}
			}
		}
		return ((double)2*sumKij) / (this.nThresh*(this.nThresh-1));
	}
	
	// -----------------------1--------------------------
	// folded site frequency spectrum, raw counts
	// to become later: P(k derived alleles | seg site) for k in [1,n-1]
	// note: this is where we *don't* know the ancestral allele
	public int[] sfsFolded(int regionStart, int regionEnd) {
		int[] sfs = new int[this.nThresh/2]; // half as many entries here, don't need ceiling or anything since diploid
		//int badCount = 0;
		//int allCount = 0;
		    
		for (SNP snp : this.snpList) {
			if (snp.index >= regionStart && snp.index < regionEnd) {
			
				// if we have data for at least 100 individuals
				if ((this.n - Util.count(snp.alleles, -1)) >= this.nThresh) {
				
					int knownCount = 0;
					int count0 = 0;
					int count1 = 0;
					for (int i=0; i < this.n; i++) {
						if (snp.alleles[i] != -1) {
							knownCount += 1;
						}
						if (snp.alleles[i] == 0 && knownCount <= this.nThresh) {
							count0 += 1;
						} else if (snp.alleles[i] == 1 && knownCount <= this.nThresh) {
							count1 += 1;
						}
					}
				
					// if segregating for these 100 individuals, add to our list
					if (count0 > 0 && count1 > 0) {
						assert count0 + count1 == this.nThresh;
						int count = Math.min(count0, count1);
						sfs[count-1] += 1;
					}
				} else {
					//badCount += 1;
				}
				//allCount += 1;
			}
		}
		//System.out.println("folded: " + badCount + "/" + allCount + " = " + (double)badCount/allCount);
		return sfs;
	}
	
	// -----------------------2--------------------------
	// distances between segregating sites, counts in bins
	// to become later: P(distance between seg sites is between x and x+delta)
	// note: this is unaffected by the change to incorporate unknown data
	public int[] lengthCounts(int regionStart, int regionEnd, int[] binStarts) {

		// from the spike train, compute the length distribution
		List<Integer> lenCounts = new ArrayList<Integer>();
		int prevSnpIdx = regionStart;
		int currSnpIdx = regionStart;
		for (SNP snp : this.snpListThresh) { // use thresh here for consistency w/ simulated data
			
			if (snp.index >= regionStart && snp.index < regionEnd) {
		        currSnpIdx = snp.index;

		        lenCounts.add(currSnpIdx-prevSnpIdx);
		        prevSnpIdx = currSnpIdx;
			}
		}

		// check on counts
		int regionLength = regionEnd - regionStart;
		assert regionLength == Util.sumArray(lenCounts)+(regionEnd-prevSnpIdx);

		// sort lengths into bins
		int[] binCounts = new int[binStarts.length];
		for (int l : lenCounts) {
			for (int i=0; i < binStarts.length-1; i++) {
				if (l >= binStarts[i] && l < binStarts[i+1]) {
					binCounts[i] += 1;
				}
			}
			if (l > binStarts[binStarts.length-1]) {
				binCounts[binCounts.length-1] += 1;
			}
		}
		return binCounts;
	}

	// -----------------------3--------------------------
	// when comparing two alleles from the same SNP, take into account missing data
	private boolean alleleEquals(int allele1, int allele2) {
		if (allele1 == allele2) {
			return true;
		} else if (allele1 == -1 || allele2 == -1) {
			return true;
		} return false;
	}
	
	// IBS tract lengths, counts in bins
	// to become: P(IBS tract length is between x and x+delta)
	public int[] ibsCounts(int regionStart, int regionEnd, double[] ibsStarts) {
		List<Integer> ibsTracts = new ArrayList<Integer>();
		//int endSums = 0;
		
		// get the IBS distribution, including distance to first diff
		for (int h1=0; h1 < this.nThresh; h1++) { // hap1
			for (int h2=h1+1; h2 < this.nThresh; h2++) { // hap2

				// for each pair
		        int prevSnpIdx = regionStart;
		        int currSnpIdx = regionStart;
		        int length = 0;
		        for (SNP snp : this.snpListThresh) { // use thresh here for consistency w/ simulated data
		        	if (snp.index >= regionStart && snp.index < regionEnd) {
		        		currSnpIdx = snp.index;
		        		length += (currSnpIdx-prevSnpIdx); // add on this length where they are the same
		        		
		        		// alleles are different at this snp, terminate IBS tract and start a new one
		        		if (!alleleEquals(snp.alleles[h1], snp.alleles[h2])) {
		        			ibsTracts.add(length);
		        			length = 0;
		        		}
		        		prevSnpIdx = currSnpIdx;
		        	}
		        }
		        length += (regionEnd-prevSnpIdx);
		        ibsTracts.add(length);
			}
		}

	    // check on counts
	    assert (regionEnd - regionStart)*this.nThresh*(this.nThresh-1)/2 == Util.sumArray(ibsTracts);

	    // sort lengths into bins
	    int[] ibsCounts = new int[ibsStarts.length];
	    for (int l : ibsTracts) {
	        for (int i=0; i < ibsStarts.length-1; i++) {
	            if (l >= ibsStarts[i] && l < ibsStarts[i+1]) {
	                ibsCounts[i] += 1;
	            }
	        }
	        if (l > ibsStarts[ibsStarts.length-1]) {
	            ibsCounts[ibsCounts.length-1] += 1;
	        }
	    }
	    return ibsCounts;
	}
	
	//-----------------------4--------------------------
	// compute the LD between two SNPs
	// note: this ignores missing data, but shouldn't be too much affected since it's 
	// relative frequency of different allele combinations
	private double computeLD(SNP snp1, SNP snp2) {
		
	    // first find the 4 different haplotypes (00,01,10,11)
	    double[] hapCounts = new double[4];
	    double unknownHapCount = 0d; // if either allele is unknown, add to this count
	    for (int i=0; i < this.nThresh; i++) {
	        int a1 = snp1.alleles[i];
	        int a2 = snp2.alleles[i];
	        if      (a1 == 0 && a2 == 0) { hapCounts[0] += 1d; }
	        else if (a1 == 0 && a2 == 1) { hapCounts[1] += 1d; }
	        else if (a1 == 1 && a2 == 0) { hapCounts[2] += 1d; }
	        else if (a1 == 1 && a2 == 1) { hapCounts[3] += 1d; }
	        else { unknownHapCount += 1d; }
	    }

	    // then get the most frequently occuring hap
	    int bestHapIdx = Util.maxIndex(hapCounts);
	    double x11;
	    double x12;
	    double x21;

	    double normalizer = Util.sumArray(hapCounts);
	    // if we have a high fraction of unknown haps, reject
	    if (unknownHapCount/normalizer >= 0.9) {
	    	return Double.NEGATIVE_INFINITY;
	    }
	 
	    // cases
	    if (bestHapIdx == 0) { // i.e. 00
	        x11 = hapCounts[0]/normalizer; // 00
	        x12 = hapCounts[1]/normalizer; // 01
	        x21 = hapCounts[2]/normalizer; // 10
	    }
	    else if (bestHapIdx == 1) { // i.e. 01
	        x11 = hapCounts[1]/normalizer; // 01
	        x12 = hapCounts[0]/normalizer; // 00
	        x21 = hapCounts[3]/normalizer; // 11
	    }
	    else if (bestHapIdx == 2) { // i.e. 10
	        x11 = hapCounts[2]/normalizer; // 10
	        x12 = hapCounts[3]/normalizer; // 11
	        x21 = hapCounts[0]/normalizer; // 00
	    }
	    else { // i.e. 11
	        x11 = hapCounts[3]/normalizer; // 11
	        x12 = hapCounts[2]/normalizer; // 10
	        x21 = hapCounts[1]/normalizer; // 01
	    }

	    // compute and return LD
	    double p1 = x11 + x12;
	    double q1 = x11 + x21;
	    double D = x11 - p1*q1;
	    return D;
	}

	// compute the LD dist between selected site and sites in given region
	public int[] ldCounts(int regionStart1, int regionEnd1, int regionStart2, int regionEnd2, double[] ldBins) {

	    // get all the LD values between each pair of snps in given region
	    List<Double> ldList = new ArrayList<Double>();
	    for (SNP snp1 : this.snpListThresh) {
	    	if (snp1.index >= regionStart1 && snp1.index < regionEnd1) {
	    		for (SNP snp2 : this.snpListThresh) {
	    			if (snp2.index >= regionStart2 && snp2.index < regionEnd2 && snp1.index != snp2.index) {
	    				double LD = computeLD(snp1,snp2);
	    				if (LD != Double.NEGATIVE_INFINITY) {
	    					ldList.add(computeLD(snp1,snp2));
	    				}
	    			}
	    		}
	    	}
	    }

	    // sort list of LD values into bins
	    int[] ldCounts = new int[ldBins.length+1];
	    for (double ld : ldList) {
	        for (int i=1; i < ldBins.length; i++) {
	            if (ld >= ldBins[i-1] && ld < ldBins[i]) {
	            	ldCounts[i] += 1;
	            }
	    	}
	        if (ld < ldBins[0]) {
	        	ldCounts[0] += 1;
	        }
	        if (ld >= ldBins[ldBins.length-1]) {
	        	ldCounts[ldCounts.length-1] += 1;
	        }
	    }
	    return ldCounts;
	}
	
	//-----------------------5--------------------------
	// computer H1, H12, and H2 (to help with the standing variation case)
	public List<Double> Hstats(int regionStart, int regionEnd) {
		
		// go through the haplotypes, adding them to a class or creating a new one
		// the first index is the hap index, and the second is the number of times it appears
		Map<Integer, Integer> hapDict = new HashMap<Integer, Integer>();
		
		for (int i=0; i < this.nThresh; i++) {
			boolean hapAdded = false;
			for (int j : hapDict.keySet()) {
				if (hapEquals(i, j, regionStart, regionEnd)) {
					hapDict.put(j, hapDict.get(j) + 1);
					hapAdded = true;
					break; // only add once
				}
			}
			
			// if it's not equal to any other haplotypes, add it as a new one
			if (!hapAdded) {
				hapDict.put(i, 1);
			}
		}
		
		// convert to frequencies
		int numClasses = hapDict.keySet().size();
		Double[] frequencies = new Double[numClasses];
		int classIdx = 0;
		for (int j : hapDict.keySet()) {
			frequencies[classIdx] = hapDict.get(j)/((double) this.nThresh);
			classIdx += 1;
		}
		
		// sort with the most frequent haplotype first
		Arrays.sort(frequencies, Collections.reverseOrder());
		
		// H1
		double H1 = 0;
		for (double f : frequencies) {
			H1 += Math.pow(f,2);
		}
		
		// H12
		double H12 = H1;
		if (frequencies.length > 1) { // this makes sure we actually have multiple types of haplotypes
			H12 += 2*frequencies[0]*frequencies[1]; 
		}
		
		// H2
		double H2 = H1 - Math.pow(frequencies[0],2);
		
		List<Double> Hstats = new ArrayList<Double>();
		Hstats.add(H1); Hstats.add(H12); Hstats.add(H2);
		return Hstats;
	}
	
	// whether or not two haplotypes are equal to each other, up to missing data (-1: unknown ('N'))
	private boolean hapEquals(int hapIdx1, int hapIdx2, int regionStart, int regionEnd) {
		
		for (SNP snp : this.snpListThresh) {
			if (snp.index >= regionStart && snp.index < regionEnd) {
				
				// if both are known and they are not equal to each other, return false
				if (snp.alleles[hapIdx1] != -1 && snp.alleles[hapIdx2] != -1 && snp.alleles[hapIdx1] != snp.alleles[hapIdx2]) {
					return false;
				}
			}
		}
		
		// if they share the same SNPs in this region, return true
		return true;
	}
	
	//-----------------------6--------------------------
	// compute Kim and Nielsen's omega statistics (2004)
	public double omega(int middleBase, int S) {
		
		// first find the middle SNP (putative selected site)
		int l = middleSnp(middleBase);
		
		// average correlation coefficient within L or R
		double avgWithin = 0;
		int withinCount = 0; // need to count, since r^2 cannot be computed for every pair
		
		// within L
		for (int i=0; i < l; i++) {
			SNP snp1 = this.snpListThresh.get(i);
			for (int j=i+1; j < l; j++) {
				SNP snp2 = this.snpListThresh.get(j);
				double r2 = computeR2(snp1, snp2);
				
				if (r2 != Double.POSITIVE_INFINITY) {
					avgWithin += r2;
					withinCount += 1;
				}
			}
		}
		
		// within R
		for (int i=l; i < S; i++) {
			SNP snp1 = this.snpListThresh.get(i);
			for (int j=i+1; j < S; j++) {
				SNP snp2 = this.snpListThresh.get(j);
				double r2 = computeR2(snp1, snp2);
				
				if (r2 != Double.POSITIVE_INFINITY) {
					avgWithin += r2;
					withinCount += 1;
				}
			}
		}
		
		// normalize within
		avgWithin = avgWithin/withinCount;
		
		// average correlation coefficient between L and R
		double avgBetween = 0;
		int betweenCount = 0;
		
		// between L and R
		for (int i=0; i < l; i++) {
			SNP snp1 = this.snpListThresh.get(i);
			for (int j=l; j < S; j++) {
				SNP snp2 = this.snpListThresh.get(j);
				double r2 = computeR2(snp1, snp2);
				
				if (r2 != Double.POSITIVE_INFINITY) {
					avgBetween += r2;
					betweenCount += 1;
				}
				
			}
		}
		
		// normalize between
		avgBetween = avgBetween/betweenCount;
		
		return avgWithin/avgBetween;
	}
	
	// helper function to find the index of the SNP closest to the center (putative selected site)
	private int middleSnp(int middleBase) {
		int minDist = Integer.MAX_VALUE;
		int minIdx  = 0;
		
		for (int i=0; i < this.snpListThresh.size(); i++) {
			SNP snp = this.snpListThresh.get(i);
			int dist = Math.abs(snp.index - middleBase);
			if (dist < minDist) {
				minDist = dist;
				minIdx = i;
			}
		}
		return minIdx + 1; // this is "l" from the paper
	}
	
	// helper to compute correlation coefficient between two snps
	private double computeR2(SNP snp1, SNP snp2) {
		double D = computeLD(snp1, snp2); // first compute LD
		
		if (D != Double.NEGATIVE_INFINITY) {
		
			int c0 = Util.count(snp1.alleles, 0);
			int c1 = Util.count(snp1.alleles, 1);
			int d0 = Util.count(snp2.alleles, 0);
			int d1 = Util.count(snp2.alleles, 1);
			
			double p0 = c0/((double) (c0+c1));
			double q0 = d0/((double) (d0+d1));
			
			return D*D/(p0*(1-p0)*q0*(1-q0));
			
		} else {
			return Double.POSITIVE_INFINITY;
		}
	}
}
