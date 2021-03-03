//Phase 2 by Corban Kenyon; Implementing K Means Clustering on several sample datasets
//Resource for java programming practices: https://www.baeldung.com/java-clean-code
package main.java; //may need to comment out without entire project file

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;

public class KMeansClustering {
	public static void main(String[] args) {
		try {
			//save clustering arguments
			String fileName = args[0];
			int numClusters = Integer.parseInt(args[1]); //K
			int maxIter = Integer.parseInt(args[2]); //I
			double convThreshold = Double.parseDouble(args[3]); //T
			int numRuns = Integer.parseInt(args[4]); //R
			
			if (numClusters < 2 || maxIter < 1 || numRuns < 1 || convThreshold < 0)
				throw new Exception("Invalid parameter value");
			
			//get the number of tuples/features from our dataset (the first line)
			File dataset = new File(fileName);
			Scanner dataReader = new Scanner(dataset);
			String dimensions[] = dataReader.nextLine().split(" ");
			int numTuples = Integer.parseInt(dimensions[0]);
			int numFeatures = Integer.parseInt(dimensions[1]);
			double clusters[][] = new double[numClusters][];
			double tuples[][] = new double[numTuples][];
			
			//parse through rest of the file and save values
			for (int i = 0; dataReader.hasNextLine(); i++) {
				tuples[i] = new double[numFeatures];
				String temp[] = dataReader.nextLine().split(" ");
				for (int j = 0; j < numFeatures; j++) {
					tuples[i][j] = Double.parseDouble(temp[j]);
				}
			}
			dataReader.close();
			
			//create a new file to output clustering performance (will be in /src/)
			String temp[] = fileName.split("/");
			String newFileName = temp[temp.length-1].split("\\.")[0] + "Performance.txt";
			FileWriter clusteringWrite = new FileWriter(newFileName);
			int bestRun = 0;
			double bestSSE = Double.MAX_VALUE;
			double totalSSE = 0; //for seeing how empty centroid checking may function
			double totalIter = 0;
			
			//--------Begin Algorithm--------
			
			//go through numRuns runs of the algorithm
			for (int run = 1; run <= numRuns; run++, clusteringWrite.write("\n")) {
				System.out.println("\nRun " + run + "\n-----");
				clusteringWrite.write("Run " + run + "\n-----\n");
				
				//random cluster selection
				List<Integer> rands = new Random().ints(0, numTuples).distinct().limit(numClusters).boxed().collect(Collectors.toList());
				//fetch the starting random tuples and store them
				for (int i = 0; i < numClusters; i++) {
					for (int j = 0; j < numFeatures; j++) {
						clusters[i] = new double[numFeatures];
						clusters[i][j] = tuples[rands.get(i)][j];
					}
				}			
	
				//go through up to maxIter iterations or until convergence is reached
				double sse = Double.MAX_VALUE;
				double improvement = Double.MAX_VALUE;
				for (int iteration = 0; iteration < maxIter+1 && improvement >= convThreshold; iteration++) {
					int closestCentroid[] = new int[numTuples]; //track each tuples closest centroid
					double errorValues[][] = new double[numTuples][numClusters]; //track the error for each tuple-cluster pairing
					int clusterMembers[] = new int[numClusters]; //track how many tuples each cluster has
					double prevSSE = sse;
					sse = 0;
					totalIter++;
					
					for (int t = 0; t < numTuples; t++) {
						//get the error for each tuple-cluster pairing
						for (int f = 0; f < numFeatures; f++) {
							double fValue = tuples[t][f];
							for (int c = 0; c < numClusters; c++) {
								double error = fValue - clusters[c][f];
								errorValues[t][c] += error * error; //square the error with multiplication
							}
						}
						//calculate the closest centroid based on distance
						closestCentroid[t] = 0;
						for (int c = 1; c < numClusters; c++) {
							if (errorValues[t][c] < errorValues[t][closestCentroid[t]]) {
								closestCentroid[t] = c;
							}
						}					
					}		
					
					//recalculate the centroids by minimizing error (averaging each feature value from each cluster member)
					double tupleSums[][] = new double[numFeatures][numClusters];
					for (int t = 0; t < numTuples; t++) {
						for (int f = 0; f < numFeatures; f++) {
							tupleSums[f][closestCentroid[t]] += tuples[t][f];
						}
						clusterMembers[closestCentroid[t]] += 1;
					}
					for (int c = 0; c < numClusters; c++) {
						for (int f = 0; f < numFeatures; f++) {
							clusters[c][f] = tupleSums[f][c]/clusterMembers[c];
						}
					}
					
					//recalculate error
					//first need to reset error
					for (int t = 0; t < numTuples; t++) {
						for (int c = 0; c < numClusters; c++) {
							errorValues[t][c] = 0;
						}
					}
					//get the error for each tuple-cluster pairing
					for (int t = 0; t < numTuples; t++) {
						for (int f = 0; f < numFeatures; f++) {
							double error = tuples[t][f] - clusters[closestCentroid[t]][f];
							double errorSq = error * error;
							errorValues[t][closestCentroid[t]] += errorSq;
							sse += errorSq;
						}
					}
					
					improvement = (prevSSE - sse) / prevSSE; //track how the sse is decreasing
					String truncatedSSE = String.format("%.4f", sse);
					System.out.println("Iteration " + (iteration) + ": SSE = " + truncatedSSE);
					clusteringWrite.write("Iteration " + (iteration) + ": SSE = " + truncatedSSE + "\n");

					
					//check for "empty" centroids
					for (int c = 0; c < numClusters; c++) {
						if (clusterMembers[c] == 0) {
							//get highest error and set as new cluster
							int worstTuple = 0;
							for (int t = 0; t < numTuples; t++) {
								if (errorValues[t][closestCentroid[t]] > worstTuple)
									worstTuple = t;
							}
							for (int f = 0; f < numFeatures; f++) {
								clusters[c][f] = tuples[worstTuple][f];
							}
							//now need to recalculate errors in case there is another empty cluster
							//we don't want the new centroids to be too similar
							for (int t = 0; t < numTuples; t++) {
								for (int c1 = 0; c1 < numClusters; c1++) {
									//resetting error values
									errorValues[t][c1] = 0;
								}
							}
							//recalculate error
							for (int t = 0; t < numTuples; t++) {
								for (int f = 0; f < numFeatures; f++) {
									double error = tuples[t][f] - clusters[closestCentroid[t]][f];
									double errorSq = error * error;
									errorValues[t][closestCentroid[t]] += errorSq;
								}
							}
						}
					}
				}
				//keep track of our best run and avg stats
				if (sse < bestSSE) {
					bestSSE = sse;
					bestRun = run;
				}
				totalSSE += sse;
			}
			//output best performance, we show average SSE and iterations to see how empty centroid checking may affect iris dataset
			System.out.println("\nAfter " + numRuns + " runs for " + dataset + " with K = " + numClusters);
			clusteringWrite.write("\nAfter " + numRuns + " runs for " + dataset + " with K = " + numClusters);
			System.out.println("AvgSSE = " + totalSSE/numRuns + ". AvgIterations: " + totalIter/numRuns);
			clusteringWrite.write("\nAvgSSE = " + totalSSE/numRuns + ". AvgIterations: " + totalIter/numRuns);
			System.out.println("Best Run: " + bestRun + ": SSE = " + String.format("%.4f", bestSSE));
			clusteringWrite.write("\nBest Run: " + bestRun + ": SSE = " + String.format("%.4f", bestSSE));
			clusteringWrite.close();
		}
		catch(FileNotFoundException e) {System.out.println("Error: Could not find file. "
				+ "Make sure relative path is correct if input files are in resources file");}
		catch(IOException e) {e.printStackTrace();}
		catch(Exception e) {
			e.printStackTrace();
			System.out.println("Possible issue with parameters given. "
					+ "Expected filename (string), number of clusters (int greater than 1), "
					+ "max iterations (pos. int), convergence threshold (non-neg double), runs (pos. int).");
		}
	}
}
