package statistics;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import utility.Util;

/** 
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Implements "Stats" class, takes in an msms filename (real or simulated) and returns a list of stats
 * this is for the ZI scenario with n=100 (real data has n=197 and down-sampled, simulated data has n=100)
 *
 * stat types:
 * 1) folded sfs
 * 2) length distribution between seg sites
 * 3) IBS length distribution
 * 4) LD stats
 * 5) H1/2 stats
 * 6) omega (omitting for now)
 *
 * regions:
 * 1) close to selected site (within 10kb on either side)
 * 2) moderately close (within 10kb and 30kb on either side)
 * 3) faraway (from 30kb to 50kb on either side)
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */
public class ZIstats implements Stats {

	private static final int L = 100000; // length of the region (kind of arbitrary now since we're using infinite sites mutation and recombination)
	private static final int SEL_SITE = 50000; // location of the selected site
	private static final int FLANK_REGION1 = 10000; // either side of region1 (around the selected site)
	private static final int REGION23 = 20000; // length of regions 2 and 3 (mid range from selected site)
	
	// for seg site and D statistics
	private static final int MAX_S = 5000;  // max number of segregating sites, for normalizing only
	private static final double MIN_D = -3; // minimum tajima's D, for normalizing only
	private static final double MAX_D =  3; // maximum tajima's D, for normalizing only (TODO should eventually increase this I think)

	// for length distribution between seg sites
	private static final int SFS_THRESH = 10; // we must have data for at least 18 individuals to include and entry in the SFS
	private static final int NUM_DIST_BINS = 15; // number of bins we will divide len dist into
	private static final int MAX_SNP_DIST = 300; // 20 bins will be below 500, remaining bin will be above
	private static final int[] BIN_STARTS = new int[NUM_DIST_BINS+1];

	// for IBS tract length distribution
	private static final int NUM_IBS_BINS = 29;
	private static final int MAX_IBS_DIST = 5000;
	private static final double[] IBS_STARTS = new double[NUM_IBS_BINS+1];
	
	// for LD
	private static final int NUM_LD_BINS = 16; // number of bins for LD distribution
	private static final double LD_MIN = -0.05; // min LD value
	private static final double LD_MAX = 0.2; // max LD value
	private static final double[] LD_STARTS = new double[NUM_LD_BINS-1];
	
	// header info
	private final String header;
	private final int numStats;
	
	// for omega TODO add back in
	//private static final double MIN_OMEGA = 1d;
	//private static final double MAX_OMEGA = 2d;
	
	// constructor, initialize distributions
	public ZIstats() {
		// initialize distributions
		for (int i=0; i < NUM_DIST_BINS+1; i++) {
			BIN_STARTS[i] = (int) i*MAX_SNP_DIST/NUM_DIST_BINS;
		}
		for (int j=0; j < NUM_IBS_BINS+1; j++) {
			IBS_STARTS[j] = 10*Math.exp(((double) j)/NUM_IBS_BINS * Math.log(1+0.1*MAX_IBS_DIST))-10;
		}
		for (int k=0; k < NUM_LD_BINS-1; k++) {
			LD_STARTS[k] = k*(LD_MAX - LD_MIN) / (NUM_LD_BINS-2) + LD_MIN;
		}
		
		this.header = makeHeader();
		this.numStats = this.header.split(" ").length;
	}
	
	private static String makeHeader() {
		String statsHeader = "";
		
		// close
		statsHeader += "Sclose Dclose ";
		for (int i=1; i <= SFS_THRESH/2; i++) { statsHeader += "fsfsClose:" + i + " "; }
		for (int i : BIN_STARTS) { statsHeader += "binClose:" + i + " "; }
		for (double i : IBS_STARTS) { statsHeader += "ibsClose:" + i + " "; }
		statsHeader += "ldClose:-inf ";
		for (double i: LD_STARTS) { statsHeader += "ldClose:" + i + " "; }
		
		// mid-range
		statsHeader += "Smid Dmid ";
		for (int i=1; i <= SFS_THRESH/2; i++) { statsHeader += "fsfsMid:" + i + " "; }
		for (int i : BIN_STARTS) { statsHeader += "binMid:" + i + " "; }
		for (double i : IBS_STARTS) { statsHeader += "ibsMid:" + i + " "; }
		statsHeader += "ldMid:-inf ";
		for (double i: LD_STARTS) { statsHeader += "ldMid:" + i + " "; }
		
		// far
		statsHeader += "Sfar Dfar ";
		for (int i=1; i <= SFS_THRESH/2; i++) { statsHeader += "fsfsFar:" + i + " "; }
		for (int i : BIN_STARTS) { statsHeader += "binFar:" + i + " "; }
		for (double i : IBS_STARTS) { statsHeader += "ibsFar:" + i + " "; }
		statsHeader += "ldFar:-inf ";
		for (double i: LD_STARTS) { statsHeader += "ldFar:" + i + " "; }
		
		// H1, H12, H2, omega
		statsHeader += "H1 H12 H2"; // TODO add back in omega";
		
		return statsHeader.trim();
	}  
	
	public String getHeader() { return this.header; }
	    
	public List<Double> msms2stats(String msmsFilename) throws IOException {
		        
        List<Double> stats = new ArrayList<Double>();
        
        // get haplotype configuration from the msms file
        HapConfigZI hapConfig = new HapConfigZI(msmsFilename, L, SFS_THRESH);

        // region 1 ------------------------------------------------------------
        int regionStart1 = SEL_SITE - FLANK_REGION1;
        int regionEnd1   = SEL_SITE + FLANK_REGION1;
        
        int Sregion1 = hapConfig.numSegSites(regionStart1, regionEnd1);
        double SnormalizedRegion1 = Sregion1/((double) MAX_S);
        stats.add(SnormalizedRegion1);
        //System.out.println(SnormalizedRegion1);
        
        double kHat1 = hapConfig.computeKhat(regionStart1, regionEnd1);
        double tajimasD1 = hapConfig.tajimasD(Sregion1, kHat1);
        double normalizedD1 = (tajimasD1 - MIN_D)/(MAX_D - MIN_D);
        stats.add(normalizedD1);
        //System.out.println(normalizedD1);
    
        int[] sfsCountsRegion1 = hapConfig.sfsFolded(regionStart1, regionEnd1);
        //System.out.println(Arrays.toString(sfsCountsRegion1));
        List<Double> sfsProbsRegion1 = Util.normalize(sfsCountsRegion1);
        stats.addAll(sfsProbsRegion1);

        int[] binCountsRegion1 = hapConfig.lengthCounts(regionStart1, regionEnd1, BIN_STARTS);
        //System.out.println(Arrays.toString(binCountsRegion1));
        List<Double> binProbsRegion1 = Util.normalize(binCountsRegion1);
        stats.addAll(binProbsRegion1);

        int[] ibsCountsRegion1 = hapConfig.ibsCounts(regionStart1, regionEnd1, IBS_STARTS);
        //System.out.println(Arrays.toString(ibsCountsRegion1));
        List<Double> ibsProbsRegion1  = Util.normalize(ibsCountsRegion1);
        stats.addAll(ibsProbsRegion1);
        
        int[] ldCountsRegion1 = hapConfig.ldCounts(regionStart1, regionEnd1, regionStart1, regionEnd1, LD_STARTS);
        //System.out.println(Arrays.toString(ldCountsRegion1));
        List<Double> lsProbsRegion1 = Util.normalize(ldCountsRegion1);
        stats.addAll(lsProbsRegion1);
        
        // region 2 and 3 ------------------------------------------------------
        int regionStart2 = SEL_SITE - (FLANK_REGION1 + REGION23);
        int regionEnd2   = SEL_SITE - FLANK_REGION1;
        int regionStart3 = SEL_SITE + FLANK_REGION1;
        int regionEnd3   = SEL_SITE + (FLANK_REGION1 + REGION23);
        
        int Sregion23 = hapConfig.numSegSites(regionStart2, regionEnd2) + hapConfig.numSegSites(regionStart3, regionEnd3);
        double SnormalizedRegion23 = Sregion23/((double) MAX_S);
        stats.add(SnormalizedRegion23);
        //System.out.println(SnormalizedRegion23);
        
        double kHat23 = hapConfig.computeKhat(regionStart2, regionEnd2) + hapConfig.computeKhat(regionStart3, regionEnd3);
        double tajimasD23 = hapConfig.tajimasD(Sregion23, kHat23);
        double normalizedD23 = (tajimasD23 - MIN_D)/(MAX_D - MIN_D);
        stats.add(normalizedD23);
        //System.out.println(normalizedD23);

        int[] sfsCountsRegion2 = hapConfig.sfsFolded(regionStart2, regionEnd2);
        int[] sfsCountsRegion3 = hapConfig.sfsFolded(regionStart3, regionEnd3);
        int[] sfsCountsRegion23 = Util.addArrays(sfsCountsRegion2, sfsCountsRegion3);
        //System.out.println(Arrays.toString(sfsCountsRegion23));
        List<Double> sfsProbsRegion23 = Util.normalize(sfsCountsRegion23);
        stats.addAll(sfsProbsRegion23);

        int[] binCountsRegion2 = hapConfig.lengthCounts(regionStart2, regionEnd2, BIN_STARTS);
        int[] binCountsRegion3 = hapConfig.lengthCounts(regionStart3, regionEnd3, BIN_STARTS);
        int[] binCountsRegion23 = Util.addArrays(binCountsRegion2, binCountsRegion3);
        //System.out.println(Arrays.toString(binCountsRegion23));
        List<Double> binProbsRegion23 = Util.normalize(binCountsRegion23);
        stats.addAll(binProbsRegion23);

        int[] ibsCountsRegion2 = hapConfig.ibsCounts(regionStart2, regionEnd2, IBS_STARTS);
        int[] ibsCountsRegion3 = hapConfig.ibsCounts(regionStart3, regionEnd3, IBS_STARTS);
        int[] ibsCountsRegion23 = Util.addArrays(ibsCountsRegion2, ibsCountsRegion3);
        //System.out.println(Arrays.toString(ibsCountsRegion23));
        List<Double> ibsProbsRegion23 = Util.normalize(ibsCountsRegion23);
        stats.addAll(ibsProbsRegion23);

        int[] ldCountsRegion2 = hapConfig.ldCounts(regionStart1,regionEnd1,regionStart2,regionEnd2, LD_STARTS);
        int[] ldCountsRegion3 = hapConfig.ldCounts(regionStart1,regionEnd1,regionStart3,regionEnd3, LD_STARTS);
        int[] ldCountsRegion23 = Util.addArrays(ldCountsRegion2, ldCountsRegion3);
        //System.out.println(Arrays.toString(ldCountsRegion23));
        List<Double> ldProbsRegion23 = Util.normalize(ldCountsRegion23);
        stats.addAll(ldProbsRegion23);

        // region 4 and 5 ------------------------------------------------------
        int regionStart4 = 0;
        int regionEnd4   = SEL_SITE - (FLANK_REGION1 + REGION23);
        int regionStart5 = SEL_SITE + (FLANK_REGION1 + REGION23);
        int regionEnd5   = L;
        
        int Sregion45 = hapConfig.numSegSites(regionStart4, regionEnd4) + hapConfig.numSegSites(regionStart5, regionEnd5);
        double SnormalizedRegion45 = Sregion45/((double) MAX_S);
        stats.add(SnormalizedRegion45);
        //System.out.println(SnormalizedRegion45);

        double kHat45 = hapConfig.computeKhat(regionStart4, regionEnd4) + hapConfig.computeKhat(regionStart5, regionEnd5);
        double tajimasD45 = hapConfig.tajimasD(Sregion45, kHat45);
        double normalizedD45 = (tajimasD45 - MIN_D)/(MAX_D - MIN_D);
        stats.add(normalizedD45);
        //System.out.println(normalizedD45);
        
        int[] sfsCountsRegion4 = hapConfig.sfsFolded(regionStart4, regionEnd4);
        int[] sfsCountsRegion5 = hapConfig.sfsFolded(regionStart5, regionEnd5);
        int[] sfsCountsRegion45 = Util.addArrays(sfsCountsRegion4, sfsCountsRegion5);
        //System.out.println(Arrays.toString(sfsCountsRegion45));
        List<Double> sfsProbsRegion45 = Util.normalize(sfsCountsRegion45);
        stats.addAll(sfsProbsRegion45);

        int[] binCountsRegion4 = hapConfig.lengthCounts(regionStart4, regionEnd4, BIN_STARTS);
        int[] binCountsRegion5 = hapConfig.lengthCounts(regionStart5, regionEnd5, BIN_STARTS);
        int[] binCountsRegion45 = Util.addArrays(binCountsRegion4, binCountsRegion5);
        //System.out.println(Arrays.toString(binCountsRegion45));
        List<Double> binProbsRegion45 = Util.normalize(binCountsRegion45);
        stats.addAll(binProbsRegion45);

        int[] ibsCountsRegion4 = hapConfig.ibsCounts(regionStart4, regionEnd4, IBS_STARTS);
        int[] ibsCountsRegion5 = hapConfig.ibsCounts(regionStart5, regionEnd5, IBS_STARTS);
        int[] ibsCountsRegion45 = Util.addArrays(ibsCountsRegion4, ibsCountsRegion5);
        //System.out.println(Arrays.toString(ibsCountsRegion45));
        List<Double> ibsProbsRegion45 = Util.normalize(ibsCountsRegion45);
        stats.addAll(ibsProbsRegion45);

        int[] ldCountsRegion4 = hapConfig.ldCounts(regionStart1,regionEnd1,regionStart4,regionEnd4, LD_STARTS);
        int[] ldCountsRegion5 = hapConfig.ldCounts(regionStart1,regionEnd1,regionStart5,regionEnd5, LD_STARTS);
        int[] ldCountsRegion45 = Util.addArrays(ldCountsRegion4, ldCountsRegion5);
        //System.out.println(Arrays.toString(ldCountsRegion45));
        List<Double> ldProbsRegion45 = Util.normalize(ldCountsRegion45);
        stats.addAll(ldProbsRegion45);
        
        // entire region ------------------------------------------------------
        
        // H1/H12/H2 stats
        List<Double> Hstats = hapConfig.Hstats(regionStart1, regionEnd1); // region around selected site
        //System.out.println(Arrays.toString(Hstats.toArray()));
        stats.addAll(Hstats);
        
        // omega TODO commenting out omega for now, but should add back in later
        /*int S = hapConfig.numSegSites(0, L); // total number of segregating sites
        double omega = hapConfig.omega(L/2, S);
        double omegaNormalized = (omega-MIN_OMEGA)/(MAX_OMEGA-MIN_OMEGA);
        System.out.println(omega);
        stats.add(omegaNormalized);*/

        assert stats.size() == this.numStats;

        // make sure our stats are between 0 and 1
        int statIdx = 0;
        for (int i=0; i < this.numStats; i++) {
        	Double s = stats.get(i);
        	if (s < 0) {
        		String statLabel = header.split(" ")[statIdx];
        		System.out.println(msmsFilename + ": statistic " + statLabel + " " + s + ", changing to 0");
        		stats.set(i, 0d); // changing for now
        	} else if (s > 1) {
        		String statLabel = header.split(" ")[statIdx];
        		System.out.println(msmsFilename + ": statistic " + statLabel + " " + s + ", changing to 1");
        		stats.set(i, 1d); // changing for now
        	} 
        	statIdx += 1;
        }
		
        // return list of stats for this msms file
		return stats;
	}
}
