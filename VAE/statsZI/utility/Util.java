package utility;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 
 * License: FreeBSD (Berkeley Software Distribution)
 * Copyright (c) 2016, Sara Sheehan and Yun Song
 * 
 * Util class, mainly for matrix algebra and basic statistics.
 * 
 * @author Sara Sheehan
 * @version March 11, 2016
 */
public class Util {
		
	// util for printing matrix
	public static String getMatrixString(Double[][] matrix) {
		String printstr = "";
		for (int i=0; i < matrix.length; i++) {
			printstr += Arrays.toString(matrix[i]) + "\n";
		}
		return printstr;
	}

	// util for adding two lxw matrices
	public static Double[][] addMatrices(Double[][] m1, Double[][] m2) {
		
		// get length
		int l = m1.length;
		assert l == m2.length;
		
		// get width
		assert l > 0;
		int w = m1[0].length;
		assert w == m2[0].length;
		
		Double[][] total = new Double[l][w];
		for (int i=0; i < l; i++) {
			for (int j=0; j < w; j++) {
				total[i][j] = m1[i][j] + m2[i][j];
			}
		}
		return total;
	}
	
	// util for multiplying two lxw matrices element-wise
	public static double[][] multMatrices(Double[][] m1, Double[][] m2) {
		
		// get length
		int l = m1.length;
		assert l == m2.length;
		
		// get width
		assert l > 0;
		int w = m1[0].length;
		assert w == m2[0].length;
		
		double[][] total = new double[l][w];
		for (int i=0; i < l; i++) {
			for (int j=0; j < w; j++) {
				total[i][j] = m1[i][j] * m2[i][j];
			}
		}
		return total;
	}
		
	// util for adding up an array
	public static double sumArray(Double[] a) {
		double sum = 0;
		for (int i=0; i < a.length; i++) {
			sum += a[i];
		}
		return sum;
	}
	public static double sumArray(double[] a) {
		double sum = 0;
		for (int i=0; i < a.length; i++) {
			sum += a[i];
		}
		return sum;
	}
	public static int sumArray(int[] a) {
		int sum = 0;
		for (int i=0; i < a.length; i++) {
			sum += a[i];
		}
		return sum;
	}
	public static int sumArray(List<Integer> a) {
		int sum = 0;
		for (int i=0; i < a.size(); i++) {
			sum += a.get(i);
		}
		return sum;
	}

	// util for adding up all entries in a matrix
	public static double sumMatrix(double[][] mat) {
		double sum = 0;
		for (int i=0; i < mat.length; i++) {
			for (int j=0; j < mat[0].length; j++) {
				sum += mat[i][j];
			}
		}
		return sum;
	}
	
	// util for adding two arrays element-wise
	public static Double[] addArrays(Double[] a1, Double[] a2) {
		
		// get length
		int l = a1.length;
		assert l == a2.length;
		
		Double[] total = new Double[l];
		for (int i=0; i < l; i++) {
			total[i] = a1[i] + a2[i];
		}
		return total;
	}
	public static int[] addArrays(int[] a1, int[] a2) {
		
		// get length
		int l = a1.length;
		assert l == a2.length;
		
		int[] total = new int[l];
		for (int i=0; i < l; i++) {
			total[i] = a1[i] + a2[i];
		}
		return total;
	}
	
	// util for computing a1 - a2, element-wise
	public static Double[] subtractArrays(Double[] a1, Double[] a2) {
			
		// get length
		int l = a1.length;
		assert l == a2.length;
			
		Double[] total = new Double[l];
		for (int i=0; i < l; i++) {
			total[i] = a1[i] - a2[i];
		}
		return total;
	}
	
	// util for multiplying two arrays element-wise
	public static Double[] multArrays(Double[] a1, Double[] a2) {
		
		// get length
		int l = a1.length;
		assert l == a2.length;
		
		Double[] total = new Double[l];
		for (int i=0; i < l; i++) {
			total[i] = a1[i] * a2[i];
		}
		return total;
	}
	
	// normalize a list so it sums to 1
	public static List<Double> normalize(int[] a) {
		List<Double> normal = new ArrayList<Double>();
		double sum = sumArray(a);
		
		// if the sum is 0, just return the original array
		if (sum == 0) {
			for (int i=0; i < a.length; i++) {
				normal.add((double) a[i]);
			}
		} else {
			for (int i=0; i < a.length; i++) {
				normal.add(a[i]/sum);
			}
		}
		return normal;
	}
	
	// compute the squared error (element-wise) between two arrays
	public static double squaredError(Double[] a1, Double[] a2) {
			
		// get length
		int l = a1.length;
		assert l == a2.length;
			
		double total = 0;
		for (int i=0; i < l; i++) {
			total += 0.5 * Math.pow(a1[i] - a2[i], 2);
		}
		return total;
	}
	
	// util for multiplying a vector by a constant
	public static Double[] multConstArray(double c, Double[] vec) {
		int l = vec.length;
		Double[] newVec = new Double[l];
		for (int i=0; i < l; i++) {
			newVec[i] = vec[i] * c;
		}
		return newVec;
	}
	
	// util for multiplying a vector by a constant
	public static Double[][] multConstMatrix(double c, Double[][] mat) {
		int l = mat.length;
		int w = mat[0].length;
		Double[][] newMat = new Double[l][w];
		for (int i=0; i < l; i++) {
			for (int j=0; j < w; j++) {
				newMat[i][j] = mat[i][j] * c;
			}
		}
		return newMat;
	}
	
	// util for multiplying a matrix by a vector (normal matrix multiplication)
	public static Double[] multMatrixVector(Double[][] mat, Double[] vec) {
		int l = mat.length;
		int w = mat[0].length;
		assert vec.length == w;
		Double[] newVec = new Double[l];
		for (int i=0; i < l; i++) {
			newVec[i] = sumArray(multArrays(mat[i], vec));
		}
		return newVec;
	}
	
	// util for multiplying a matrix by a vector for the per label case
	public static Double[] multMatrixVectorPerLabel(Double[][] mat, Double[] vec) {
		int weightsPerLabel = mat.length;
		int numLabels = mat[0].length;
		assert vec.length == numLabels;
		Double[] newVec = new Double[weightsPerLabel*numLabels];
		for (int i=0; i < weightsPerLabel; i++) {
			for (int j=0; j < numLabels; j++) {
				newVec[j*weightsPerLabel+i] = mat[i][j]*vec[j];
			}
		}
		return newVec;
	}
	
	// utility for multiplying two vectors to create a matrix
	public static Double[][] multVectors(Double[] vec1, Double[] vec2) {
		Double[][] matrix = new Double[vec1.length][vec2.length];
		
		for (int i=0; i < vec1.length; i++) {
			for (int j=0; j < vec2.length; j++) {
				matrix[i][j] = vec1[i] * vec2[j];
			}
		}
		return matrix;
	}
	
	// utility for multiplying two vectors in the per-label case (for the last layer)
	/*public static Double[][] multVectorsPerLabel(Double[] vec1, Double[] vec2) {
		int weightsPerLabel = vec1.length/vec2.length; ensure this is an integer
		Double[][] matrix = new Double[weightsPerLabel][vec2.length];
		
		for (int i=0; i < weightsPerLabel; i++) {
			for (int j=0; j < vec2.length; j++) {
				matrix[i][j] = vec1[j*weightsPerLabel+i] * vec2[j];
			}
		}
		return matrix;
	}*/
	
	// function for reshaping a vector into a matrix
	public static Double[][] reshape(double[] vector, int startIndex, int row, int col) {
		Double[][] newMatrix = new Double[row][col];
		
		for (int i=0; i < row; i++) {
			for (int j=0; j < col; j++) {
				newMatrix[i][j] = vector[startIndex + i*col + j];
			}
		}
		
		return newMatrix;
	}
	
	// function for obtaining a subset of a vector
	public static Double[] subset(double[] vector, int startIndex, int length) {
		Double[] newVector = new Double[length];
		
		for (int i=0; i < length; i++) {
			newVector[i] = vector[startIndex+i];
		}
		return newVector;
	}
	public static Double[] subset(Double[] vector, int startIndex, int length) {
		Double[] newVector = new Double[length];
	
		for (int i=0; i < length; i++) {
			newVector[i] = vector[startIndex+i];
		}
		return newVector;
	}
	public static String[] subset(String[] vector, int startIndex, int length) {
		String[] newVector = new String[length];
	
		for (int i=0; i < length; i++) {
			newVector[i] = vector[startIndex+i];
		}
		return newVector;
	}
	
	// count the number of occurrences of a value in an array
	public static int count(String[] array, String value) {
		int count = 0;
		for (int i=0; i < array.length; i++) {
			if (array[i].equals(value)) {
				count += 1;
			}
		}
		return count;
	}
	public static int count(int[] array, int value) {
		int count = 0;
		for (int i=0; i < array.length; i++) {
			if (array[i] == value) {
				count += 1;
			}
		}
		return count;
	}
	
	// combine two arrays to create one long one
	public static Double[] combine(Double[] vector1, Double[] vector2) {
		int len = vector1.length + vector2.length;
		Double[] combinedVector = new Double[len];
		
		for (int i=0; i < vector1.length; i++) {
			combinedVector[i] = vector1[i];
		}
		for (int j=0; j < vector2.length; j++) {
			combinedVector[vector1.length+j] = vector2[j];
		}
		return combinedVector;
	}
	
	// convert a Double[][] into an int[] for classification scenario
	public static int[] convertDoubleArray(Double[][] Ydouble) {
		int M = Ydouble.length;
		int[] Yint = new int[M];
		for (int m=0; m < M; m++) {
			Yint[m] = (int) (double) Ydouble[m][0];
		}
		return Yint;
	}

	// compute the standard deviation of an array (unwind the matrix first)
	public static double stddevArray(Double[] array) {
		int l = array.length;
		double mean = sumArray(array)/l;
			
		double stddev = 0;
		for (int i=0; i < l; i++) {
			stddev += Math.pow(array[i] - mean, 2);
		}	
		return Math.sqrt(stddev/l);
	}
	
	// compute the standard deviation of an matrix (unwind the matrix first)
	public static double stddevMatrix(double[][] mat) {
		assert mat.length > 0; 
		int r = mat.length;
		int c = mat[0].length;
		double mean = sumMatrix(mat)/(r*c);
		
		double stddev = 0;
		for (int i=0; i < r; i++) {
			for (int j=0; j < c; j++) {
				stddev += Math.pow(mat[i][j] - mean, 2);
			}
		}
		
		return Math.sqrt(stddev/(r*c));
	}
	
	// from a visibleSize, list of hiddenSizes and finalSize, create int[] of sizes
	public static int[] formatLayerSizes(int visibleSize, int[] hiddenSizes, int finalSize) {
		int[] sizes = new int[hiddenSizes.length + 2];
		
		sizes[0] = visibleSize;
		for (int l=0; l < hiddenSizes.length; l++) {
			sizes[l+1] = hiddenSizes[l];
		}
		sizes[hiddenSizes.length+1] = finalSize;
		
		return sizes;
	}
	
	// compute the number of weights we need to find over the whole network
	public static int numWeights(int visibleSize, int[] hiddenSizes, int finalSize) {
		int[] sizes = formatLayerSizes(visibleSize, hiddenSizes, finalSize);
		
		int numWeights = 0;
		for (int i=0; i < sizes.length-1; i++) {
			numWeights += sizes[i]*sizes[i+1];
			numWeights += sizes[i+1];
		}
		return numWeights;
	}
	
	// see if we are out of the lambda bounds
    public static boolean outOfBounds(double[] lambdaList, double minLambda, double maxLambda) {
    	for (int s=0; s < lambdaList.length; s++) {
    		if (lambdaList[s] < minLambda || lambdaList[s] > maxLambda) {
    			return true;
    		}
    	}
    	return false;
    }
    
    // get the index of the minimum value of a double[]
    public static int minIndex(double[] array) {
    	int minIndex = 0;
    	double minValue = Double.POSITIVE_INFINITY;
    	for (int i=0; i < array.length; i++) {
    		if (array[i] < minValue) {
    			minIndex = i;
    			minValue = array[i];
    		}
    	}
    	return minIndex;
    }
    
    // get the index of the maximum value of a double[]
    public static int maxIndex(double[] array) {
    	int maxIndex = 0;
    	double maxValue = Double.NEGATIVE_INFINITY;
    	for (int i=0; i < array.length; i++) {
    		if (array[i] > maxValue) {
    			maxIndex = i;
    			maxValue = array[i];
    		}
    	}
    	return maxIndex;
    }
    
    // read one line of a file
    public static String readLine(BufferedReader reader) throws IOException {
		String readString = null;
		while ((readString = reader.readLine()) != null) {
			readString = readString.trim();
			
			if (readString.startsWith("#")) continue; // for parameter format
			if (readString.startsWith(">")) continue; // for fasta
			if (readString.isEmpty()) continue;
			
			return readString;
		}
		
		return "";
	}
    
    public static double[][] confusionMatrix(int[] labels, int[] predictions, int numClasses) {
		double[][] confusionMatrix = new double[numClasses][numClasses];
		double[] classTotals = new double[numClasses];
		assert labels.length == predictions.length;
		
		for (int i=0; i < labels.length; i++) {
			confusionMatrix[labels[i]][predictions[i]] += 1;
			classTotals[labels[i]] += 1; // get the total count of datasets that (truly) belong to each class
		}
		
		// normalize confusion matrix
		for (int c=0; c < numClasses; c++) {
			for (int d=0; d < numClasses; d++) {
				confusionMatrix[c][d] = confusionMatrix[c][d]/classTotals[c];
			}
		}
		return confusionMatrix;
	}
    
    // Kullback-Leibler (KL) divergence between two Bernoulli random variables with means p1 and p2 
 	private static double kullbackLeibler(double p1, double p2) {
 		return p1 * Math.log(p1/p2) + (1-p1) * Math.log((1-p1)/(1-p2));
 	}
 		
 	// KL divergence for a constant and an array
 	public static double[] kullbackLeiblerArray(double c, Double[] rho) {
 		double[] KLArray = new double[rho.length];
 			
 		for (int j=0; j < rho.length; j++) {
 			KLArray[j] = kullbackLeibler(c, rho[j]);
 		}
 		return KLArray;
 	}
 	
 	// compute sparsity param
 	public static Double[] computeSparsity(Double[] rho, double sparsityParam) {
 		Double[] sparsityArray = new Double[rho.length];
 			
 		for (int j=0; j < rho.length; j++) {
 			sparsityArray[j] = -sparsityParam/rho[j] + (1-sparsityParam)/(1-rho[j]);
 		}
 		return sparsityArray;
 	}
 	
 	// compute the sample variance of an array
 	public static double sampleVariance(double[] array) {
 		int n = array.length;
 		double mean = sumArray(array)/n;
 			
 		double stddev = 0;
 		for (int i=0; i < n; i++) {
 			stddev += Math.pow(array[i] - mean, 2);
 		}	
 		return stddev/(n-1);
 	}
 	
 	// compute a confidence interval for a list of size estimates
 	public static double[] confidenceInterval(double[] array) {
 		double z = 1.96; // for a 95% confidence interval
 		double n = array.length;
 		double Xbar = sumArray(array)/n;
 		double S2 = sampleVariance(array);
 		
 		double[] interval = new double[2];
 		interval[0] = Xbar - z*Math.sqrt(S2/n);
 		interval[1] = Xbar + z*Math.sqrt(S2/n);
 		return interval;
 	}
 	
 	// compute the 2.5th and 97.5 quantile of our data (size 160 for now)
 	public static double[] quantileInterval(double[] array) {
 		int n = array.length;
 		int lowerIdx = ((int)(0.025*n))-1;
 		int upperIdx = ((int)(0.975*n))-1;
 		
 		Arrays.sort(array); // sort array
 		double[] interval = new double[2];
 		interval[0] = array[lowerIdx];
 		interval[1] = array[upperIdx];
 		return interval;
 	}
}
