package statistics;

import java.io.IOException;
import java.util.List;

/**
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Interface for all statistics classes, should take in an
 * msms file and return a list of doubles (statistics).
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */
public interface Stats {

	public List<Double> msms2stats(String msmsFilename) throws IOException;
	public String getHeader();
}
