package statistics;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPException;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;

/** 
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Wrapper around the ZIstats class for the simulated data scenario (ZI, n=100)
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */
public class ZIstatsSimData {

	public static void main(String[] args) throws JSAPException, IOException {
		SimpleJSAP jsap = new SimpleJSAP("StatsDemoSelect", "From msms datasets, compute and write out summary statistics.",
			new Parameter[] {
					
				// the input folder of folders of msms files
		        new FlaggedOption( "msmsFolder", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, 'm', "msmsFolder", "The input folder of (folders of) msms files."),
		            
		        // the output folder of statistics files, within each file there should be one line for each dataset (plus a header)
		        new FlaggedOption( "statsFolder", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, 's', "statsFolder", "The output file for statistics."),
		        
		        // start demography
		        new FlaggedOption( "beginDemo", JSAP.INTEGER_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, 'b', "beginDemo", "The start demography."),
		        
		        // number of demographies
		        new FlaggedOption( "endDemo", JSAP.INTEGER_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, 'e', "endDemo", "The end demography."),
		            
		        // the file of statistics, one line for each dataset (including train and validation, but not test)
		        new FlaggedOption( "numPerDemo", JSAP.INTEGER_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, 'p', "numPerDemo", "The number of datasets per demography."),
			}
		);
		
		// get commandline options
		JSAPResult config = jsap.parse(args);
		
		String msmsFolder = config.getString("msmsFolder");
		String statsFolder = config.getString("statsFolder");
		int beginDemo = config.getInt("beginDemo");
		int endDemo = config.getInt("endDemo");
		int numPerDemo = config.getInt("numPerDemo");
		
		// initialize ZIstats class
		ZIstats statsComputer = new ZIstats();
		String header = statsComputer.getHeader();
	    
	    for (int f=beginDemo; f < endDemo; f++) {
	    	System.out.println(msmsFolder + "demo" + f);

		    // for each file within the folder
	    	String statsString = "";
		    for (int d=0; d < numPerDemo; d++) {
		    	String msmsFilename = msmsFolder + "demo" + f + "/data" + d + ".msms";
		        //System.out.println(msmsFilename);
		        
		        List<Double> stats = statsComputer.msms2stats(msmsFilename);
	
		        // build stats string
		        for (Double s : stats) {
		        	statsString += String.format("%.10f", s) + " ";
		        }
		        statsString = statsString.trim() + "\n";
			}
		    
		    // create the stats file for this demography
		    File statsFile = new File(statsFolder + "stats_" + f + ".txt");
	        FileWriter writer = new FileWriter(statsFile);
		    writer.write(header + "\n");
		    writer.write(statsString);
		    writer.close();
	    }
	}
}