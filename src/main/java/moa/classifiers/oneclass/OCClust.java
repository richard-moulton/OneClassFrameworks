/*
 *    OCClust.java
 *    Copyright (C) 2018
 *    @author Richard Hugh Moulton
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

package moa.classifiers.oneclass;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.OneClassClassifier;
import moa.classifiers.meta.ThreadedEnsemble;
import moa.classifiers.meta.ThreadedEnsembleOC;
import moa.cluster.CFCluster;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.macro.NonConvexCluster;
import moa.core.AutoExpandVector;
import moa.core.FastVector;
import moa.core.FixedLengthList;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.InstanceStream;

// Implements the OCCluster framework.
public class OCClust extends AbstractClassifier implements Classifier, OneClassClassifier
{
	private static final long serialVersionUID = 1L;

	public String getPurposeString()
	{
        return "OCClust performs one class classification by learning different classifiers over the majority class's clusters.";
    }	
	
	public IntOption numClassifiersOption = new IntOption("numClassifiers", 'n', "Number of classifiers to train per cluster", 1);
	
	public ClassOption classifierOption = new ClassOption("classifier", 'C', "Base classifier to use for ensembles",
			OneClassClassifier.class, "oneclass.Autoencoder");
	
	public ClassOption clustererOption = new ClassOption("clusterer", 'c',
            "Clustering algorithm to apply to majority class.", AbstractClusterer.class, "clustree.ClusTree");
		
	public IntOption windowSizeOption = new IntOption("windowSize", 'w', 
			"Number of immediately past instances to store in memory; doubles as the initialization window.", 2000);
	
	public FloatOption trainingThresholdOption = new FloatOption("trainingThreshold", 't', "How close an instance must be to a cluster in "
			+ "order to be assigned to it.", 1.0, 0, Double.MAX_VALUE);
	
	public FloatOption clusterMovementThresholdOption = new FloatOption("clusterMovementThreshold", 'm', "How close two clusters must be in"
			+ " order to be considered the same cluster.", 0.2, 0, Double.MAX_VALUE);
	
	public IntOption randomSeedOption = new IntOption("randomSeed", 'r', "Seed for the random number generator", 1, 1, Integer.MAX_VALUE);
	
	/**
	 * The last clustering produced from the data stream.
	 */
	private AutoExpandVector<Cluster> memory;
	
	/**
	 * Old, inactive clusters produced earlier in the data stream.
	 */
	private FixedLengthList<Cluster> oldClusters;
	
	/**
	 * The clustering algorithm used by OCClust to cluster the data stream.
	 */
	private AbstractClusterer clusterAlgorithm;
	
	/**
	 * A vector consisting of the ensembles, in order of their clusters in memory.
	 * 
	 * @see memory
	 */
	private AutoExpandVector<ThreadedEnsembleOC> ensembles;
	
	/**
	 * Ensembles matched with the clusters in oldClusters.
	 * @see oldClusters
	 */
	private FixedLengthList<ThreadedEnsembleOC> oldEnsembles;
	
	/**
	 * A source of random numbers for building the model and its associated classifiers.
	 */
	private Random modelRandom;
	
	/**
	 * The number of classifiers to train per cluster.
	 */
	private int numClassifiers;
	
	/**
	 * The number of instances seen by the classifier to date.
	 */
	private int numInstances;
	
	/**
	 * The number of instances in each landmark window.
	 */
	private int windowSize;
	
	/**
	 * If the first 'windowSize' instances have been seen, <b>true</b>; else, <b>false</b>.
	 */
	private boolean initialized;
	
	/**
	 * The last 'windowSize' instances seen in the data stream.
	 * 
	 * @see windowSize
	 */
	private FixedLengthList<Instance> lastPoints;
	
	/**
	 * The dimensionality of the data stream.
	 */
	private int dimensions;
	
	/**
	 * How close an instance must be to a cluster in order to be assigned to it.
	 */
	private double trainingThreshold;
	
	/**
	 * How close two clusters must be in order to be considered the same cluster.
	 */
	private double clusterMovementThreshold;
	
	
	int instAssigned;
	int instUnassigned;
	
	public OCClust()
	{
	}

	@Override
	public void resetLearningImpl()
	{
		this.memory = new AutoExpandVector<Cluster>();
		this.clusterAlgorithm = (AbstractClusterer) getPreparedClassOption(this.clustererOption);
		this.clusterAlgorithm.resetLearning();
		this.ensembles = new AutoExpandVector<ThreadedEnsembleOC>();
		this.modelRandom = new Random(this.randomSeedOption.getValue());
		
		this.initialized = false;
		this.numClassifiers = this.numClassifiersOption.getValue();
		this.windowSize = this.windowSizeOption.getValue();
		this.lastPoints = new FixedLengthList<Instance>(this.windowSize*2);
		this.trainingThreshold = this.trainingThresholdOption.getValue();
		this.clusterMovementThreshold = this.clusterMovementThresholdOption.getValue();
		
		this.oldClusters = new FixedLengthList<Cluster>(10*this.numClassifiers);
		this.oldEnsembles = new FixedLengthList<ThreadedEnsembleOC>(10*this.numClassifiers);
		
		this.numInstances = 0;
		this.instAssigned = 0;
		this.instUnassigned = 0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst)
	{
		Instance ncmi = getNonConceptMarked(inst);
		this.clusterAlgorithm.trainOnInstance(ncmi);
		this.lastPoints.add(ncmi);
		this.numInstances++;
		
		//System.out.print(" This is instance number "+this.numInstances+". lastPoints is "+this.lastPoints.size()+" instances long.");
		
		//for(int i = 0 ; i < inst.numAttributes() ; i++)
		//{
		//	System.out.print(" "+inst.value(i));
		//}
		//System.out.print(" ");
		
		if(this.initialized)
		{
			if(this.numInstances % this.windowSize == 0)
			{
				//System.out.println("("+this.numInstances+")");
				updateMemory();
				//System.out.println(printClustering(this.memory));
				//System.out.println("In the last window there were "+this.instAssigned+" instances assigned to clusters and "+this.instUnassigned+" left unassigned.");
				//this.instAssigned = 0;
				//this.instUnassigned = 0;
			}
				
			InclusionResult incRes = getClusterAssignment(this.memory, ncmi);
			
			//System.out.println(ncmi.value(ncmi.classAttribute()) + ","+incRes.getClusterAssignment()+","+incRes.getInclusionProbability());
			
			// Train the associated ensemble on the argument instance
			if(incRes.getClusterAssignment() != -1 && incRes.getInclusionProbability() > this.trainingThreshold)
			{
				ThreadedEnsembleOC te = this.ensembles.get(incRes.getClusterAssignment());
				te.trainOnInstance(ncmi);
				this.ensembles.set(incRes.getClusterAssignment(), te);
				//instAssigned++;
			}
			//else
			//{
			//	instUnassigned++;
			//}
		}
		else if(this.numInstances == this.windowSize)
		{
			this.initialized = true;
			this.dimensions = getNonConceptMarked(inst).numAttributes() - 1;
			this.memory = pruneClusters(this.clusterAlgorithm.getClusteringResult().getClustering());

			AutoExpandVector<AutoExpandVector<Instance>> pointsByCluster = getPointsByCluster(this.memory, this.lastPoints);

			//System.out.println("(nI==wS) The following clustering was produced:");
			for(int i = 0 ; i < this.memory.size() ; i++)
			{
				AutoExpandVector<Instance> trainingPoints = new AutoExpandVector<Instance>();
				trainingPoints.addAll(pointsByCluster.get(i));
				trainingPoints.addAll(pointsByCluster.get(i));
				this.ensembles.set(i, trainClassifier(trainingPoints));
				//System.out.println(printCluster(this.memory.get(i), i)+" lP: "+pointsByCluster.get(i).size()+" "+this.ensembles.get(i).hashCode());
			}

		}
		//System.out.println("//");
	}

	/**
	 * Takes an argument clustering and returns a new clustering with only those clusters whose weight is above a threshold value.
	 * 
	 * @param clustering the clustering to prune
	 * @return the pruned clustering
	 */
	private AutoExpandVector<Cluster> pruneClusters(AutoExpandVector<Cluster> clustering)
	{
		//System.out.print(clustering.size()+"c");
		AutoExpandVector<Cluster> prunedClusters = new AutoExpandVector<Cluster>();
		double clusterSum = 0.0;
		double weightThreshold = Math.min(0.1, 1.0 / Math.pow(clustering.size(), 2.0));
		
		for(int i = 0 ; i < clustering.size() ; i++)
		{
			clusterSum += clustering.get(i).getWeight();
		}
		
		for(int i = 0 ; i < clustering.size() ; i++)
		{
			//System.out.print(" "+i+":"+(clustering.get(i).getWeight()/clusterSum));
			if((clustering.get(i).getWeight()/clusterSum) >= weightThreshold)
			{
				prunedClusters.add(clustering.get(i));
				//System.out.print("+");
			}
			//else
				//System.out.print("-");
		}
		
		//System.out.println(" "+prunedClusters.size()+"p");

		
		return prunedClusters;
	}

	private InclusionResult getClusterAssignment(AutoExpandVector<Cluster> clusters, Instance point)
	{
		double maxInclusion = this.trainingThreshold - Double.MIN_VALUE;
		double currInclusion;
		int maxCluster = -1;
		
		for(int i = 0 ; i < clusters.size() ; i++)
		{
			//System.out.print(" ("+i);
			currInclusion = clusters.get(i).getInclusionProbability(point);
			if(currInclusion == 0)
			{
				currInclusion = Math.pow(2.0,-1.0 * getDistance(point, clusters.get(i)));
			}

			//System.out.print(" "+currInclusion);
			
			if(currInclusion > maxInclusion)
			{
				maxInclusion = currInclusion;
				maxCluster = i;
			}
			//System.out.println(")");
		}
		
		return new InclusionResult(maxCluster, maxInclusion);
	}
	
	private AutoExpandVector<AutoExpandVector<Instance>> getPointsByCluster(AutoExpandVector<Cluster> clusters, List<Instance> points)
	{
		AutoExpandVector<AutoExpandVector<Instance>> pointsByCluster = new AutoExpandVector<AutoExpandVector<Instance>>();
		Iterator<Instance> pointsIterator = this.lastPoints.iterator();
		
		for(int i = 0 ; i < clusters.size() ; i++)
		{
			pointsByCluster.set(i, new AutoExpandVector<Instance>());
		}
		
		while (pointsIterator.hasNext())
		{
			Instance point = pointsIterator.next();
			//System.out.print("Point: [");
			//for(int i = 0 ; i < point.numAttributes()-1 ; i++)
			//{
			//	System.out.print(point.value(i)+" ");
			//}
			//System.out.print("] // Inclusion:");
			
			InclusionResult incRes = getClusterAssignment(clusters, point);
			
			//System.out.print("("+incRes.getClusterAssignment()+" - "+incRes.getInclusionProbability()+")");
			
			if(incRes.getInclusionProbability() > this.trainingThreshold)
			{
				AutoExpandVector<Instance> clusterPoints = pointsByCluster.get(incRes.getClusterAssignment());
				//System.out.print(" // cP1="+clusterPoints.size());
				clusterPoints.add(point);
				//System.out.print(" cP2="+clusterPoints.size());
				pointsByCluster.set(incRes.getClusterAssignment(), clusterPoints);
				//System.out.print(" pBC="+pointsByCluster.get(incRes.getClusterAssignment()).size());
			}
			//System.out.println();
		}
		
		//int unassignedPoints = this.lastPoints.size();
		//for(int i = 0 ; i < clusters.size() ; i++)
		//{
		//	System.out.print("C"+i+": "+pointsByCluster.get(i).size()+" ");
		//	unassignedPoints -= pointsByCluster.get(i).size();
		//}
		//System.out.println("U: "+unassignedPoints);
		
		return pointsByCluster;
	}
	
	/**
	 * Updates the assignment of classifiers to clusters when a new clustering is produced.
	 */
	private void updateMemory()
	{
		int stashed = 0;
		int pulled = 0;
		int carriedForward = 0;
		int trainedNew = 0;
		
		// Initialize data structures
		Clustering clustering = this.clusterAlgorithm.getClusteringResult();
		AutoExpandVector<Cluster> currClusters = clustering.getClustering();
		currClusters = pruneClusters(currClusters);
		double[][] clusterDistanceMatrix = new double[currClusters.size()][this.memory.size()];
		int[] clusterAssignment = new int[currClusters.size()];
		boolean[] assignedCluster = new boolean[this.memory.size()];
		
		//System.out.println("Clusters in Memory:");
		//System.out.println(printClustering(this.memory));
		//System.out.println("Current Clusters:");
		//System.out.println(printClustering(currClusters));
		
		// Calculate the Cluster Distance Function between all old clusters and all new clusters
		for(int i = 0 ; i < currClusters.size() ; i++)
		{
			for(int j = 0 ; j < this.memory.size() ; j++)
			{
				assignedCluster[j] = false;
				clusterDistanceMatrix[i][j] = getClusterDistance(currClusters.get(i), this.memory.get(j));
			}
		}
		
		// For each new clusters, determine which old cluster it is closest to
		//System.out.println("Cluster Distance Matrix ["+currClusters.size()+"x"+this.memory.size()+"] :");
		//int unassignedClusters = 0;
		for(int i = 0 ; i < currClusters.size() ; i++)
		{
			double currDistance = Double.MAX_VALUE;
			
			for(int j = 0; j < this.memory.size() ; j++)
			{
				if(clusterDistanceMatrix[i][j] < currDistance)
				{
					clusterAssignment[i] = j;
					currDistance = clusterDistanceMatrix[i][j];
				}
				//System.out.print(clusterDistanceMatrix[i][j]+" ");
			}
			
			if(currDistance > this.clusterMovementThreshold)
			{
				clusterAssignment[i] = -1;
				//unassignedClusters++;
			}
			
			//System.out.println("("+clusterAssignment[i]+" - "+currDistance+")");
		}
		//System.out.println("There were "+this.memory.size()+" clusters and there are now "+currClusters.size()+" clusters."+(currClusters.size() - unassignedClusters)+"/"+unassignedClusters);
		
		// Update classifier assignment and train new classifier as required
		AutoExpandVector<ThreadedEnsembleOC> newEnsembles = new AutoExpandVector<ThreadedEnsembleOC>();
		AutoExpandVector<AutoExpandVector<Instance>> pointsByCluster = getPointsByCluster(currClusters, this.lastPoints);
		for(int i = 0 ; i < currClusters.size() ; i++)
		{
			if(clusterAssignment[i] != -1)
			{
				assignedCluster[clusterAssignment[i]] = true;
				newEnsembles.set(i, (ThreadedEnsembleOC)this.ensembles.get(clusterAssignment[i]).copy());
				carriedForward++;
			}
			else
			{
				double currDist;
				double minDist = Double.MAX_VALUE;
				int minIndex = -1;
				for(int j = 0 ; j < this.oldClusters.size() ; j++)
				{
					currDist = getClusterDistance(currClusters.get(i),this.oldClusters.get(j));
					
					if(currDist < minDist)
					{
						minDist = currDist;
						minIndex = j;
					}
				}
				
				if(minIndex != -1)
				{
					ThreadedEnsembleOC te = (ThreadedEnsembleOC)this.oldEnsembles.get(minIndex).copy();
					AutoExpandVector<Instance> trainingPoints = pointsByCluster.get(i);
					for(int j = 0 ; j < trainingPoints.size() ; j++)
					{
						te.trainOnInstanceImpl(trainingPoints.get(j));
					}
					newEnsembles.set(i, te);
					this.oldClusters.remove(minIndex);
					this.oldEnsembles.remove(minIndex);
					pulled++;
				}
				else
				{
					AutoExpandVector<Instance> trainingPoints = pointsByCluster.get(i);
					trainingPoints.addAll(pointsByCluster.get(i));
					newEnsembles.set(i, trainClassifier(trainingPoints));
					trainedNew++;
				}
			}			
		}
		
		// Store unassigned clusters/ensembles in oldClusters/oldEnsembles
		for(int i = 0 ; i < this.memory.size() ; i++)
		{
			if(!assignedCluster[i])
			{
				oldClusters.add(this.memory.get(i));
				oldEnsembles.add(this.ensembles.get(i));
				stashed++;
			}
		}
		
		//System.out.println("There were "+this.memory.size()+" clusters in memory. "+(this.memory.size() - stashed)+" were carried forward and "+stashed+" were stored away.");
		//System.out.println("There are now "+currClusters.size()+" clusters. "+carriedForward+" were carried forward, "+pulled+" were pulled from storage and "+trainedNew+" were trained from scratch.");

		this.memory = currClusters;
		this.ensembles = newEnsembles;
		
		//for(int i = 0 ; i < this.memory.size() ; i++)
		//{
		//	System.out.println(printCluster(this.memory.get(i), i)+" lP: "+pointsByCluster.get(i).size()+" "+this.ensembles.get(i).hashCode());
		//}
	}

	private String printClustering(AutoExpandVector<Cluster> clustering)
	{
		StringBuilder sb = new StringBuilder();
		
		for(int i = 0 ; i < clustering.size() ; i++)
		{
			sb.append(printCluster(clustering.get(i), i));
		}
		return sb.toString();
	}

	
	private String printCluster(Cluster c, int i)
	{
		StringBuilder sb = new StringBuilder();
		
		double[] centre = c.getCenter();
		sb.append("Cluster "+i+": [");
		for(int j = 0 ; j < this.dimensions ; j++)
		{
			sb.append(centre[j]+", ");
		}
		sb.delete(sb.length()-2, sb.length());
		if(c instanceof SphereCluster)
		{
			sb.append("] R: "+((SphereCluster)c).getRadius()+" S");
		}
		else if(c instanceof NonConvexCluster)
		{
			sb.append("] mC: "+((NonConvexCluster)c).getMicroClusters().size()+" N");
		}
		
		return sb.toString();
	}
	
	private double getClusterDistance(Cluster clusterA, Cluster clusterB)
	{
		double integrateRange = 0.0;
		double[] integrateCentre = new double[this.dimensions];
		Random monteCarloRandom = new Random(this.modelRandom.nextLong());
		
		for(int j = 0 ; j < 100 ; j++)
		{
			DenseInstance aSample = new DenseInstance(clusterA.sample(monteCarloRandom));
			DenseInstance bSample = new DenseInstance(clusterB.sample(monteCarloRandom));

			integrateRange = Math.max(integrateRange, 5.0*getDistance(aSample,bSample));
		}
		
		double clusterDistance = 0.0;
		double runningSum = 0.0;
		
		double volume = Math.pow(integrateRange,(double)this.dimensions);
		double error = Double.MAX_VALUE;
		double N = 0.0;
		double sampleVar = 0.0;
		double mean = 0.0;
		double M2 = 0.0;
		double delta1 = 0.0;
		double delta2 = 0.0;
		double x = 0.0;
		double[] point = new double[this.dimensions+1];

		//System.out.print("integrate:");
		for(int i = 0 ; i < this.dimensions ; i++)
		{
			integrateCentre[i] = (clusterA.getCenter()[i] + clusterB.getCenter()[i]) /2.0;
			//System.out.print(" "+integrateCentre[i]);
		}
		//System.out.println("("+integrateRange+") and V = "+volume);
		
		// Monte Carlo integration
		while(error > 0.05)
		{
			Instance testInst;
						
			if(N < 100)
			{
				if(N % 2 == 0)
					testInst = new DenseInstance(clusterA.sample(monteCarloRandom));
				else
					testInst = new DenseInstance(clusterB.sample(monteCarloRandom));
			}
			else
			{
				// Randomly generate the point at which to evaluate the function
				for(int i = 0 ; i < this.dimensions ; i++)
				{
					point[i] = ((monteCarloRandom.nextDouble() - 0.5)*2.0*integrateRange) + integrateCentre[i];
					//System.out.print(point[i]+" ");
				}
				point[this.dimensions] = 0.0;
				testInst = new DenseInstance(1.0, point);
			}
			
			// Evaluate the function at point and add the result to the running sum
			x = Math.abs(clusterA.getInclusionProbability(testInst) - clusterB.getInclusionProbability(testInst));
			
			runningSum += x;
			N++;

			// Adjust other values of interest
			delta1 = x - mean;
			mean += delta1/N;
			delta2 = x - mean;
			M2 += delta1*delta2;
			clusterDistance = volume*runningSum/N;

			// Once a sufficient base of samples has been built, calculate the sample variance and estimate the error
			if (N > 100000)
			{
				sampleVar = M2/(N-1);
				error = volume*Math.sqrt(sampleVar)/Math.sqrt(N);
				
				if(Math.abs(clusterDistance - this.clusterMovementThresholdOption.getValue()) > error)
					break;
			}
		}
		
		//System.out.println("clusterDistance = "+clusterDistance+", runningSum = "+runningSum+", volume = "+volume+", N = "+N+", error = "+error);
		
		return clusterDistance;
	}

	/**
	 * Calculates the Euclidean distance between two instances.
	 * 
	 * @param instOne the first instance
	 * @param instTwo the second instance
	 * @return the Euclidean distance between the two instances
	 */
	private double getDistance(Instance instOne, Instance instTwo)
	{
		double distance = 0.0;
		
		//System.out.print("instOne:");
		for(int i = 0 ; i < this.dimensions ; i++)
		{
			//System.out.print(" "+instOne.value(i));
			distance += Math.pow(instOne.value(i) - instTwo.value(i), 2.0);
		}
		//System.out.print(" // instTwo:");
		//for(int i = 0 ; i < this.dimensions ; i++)
		//{
		//	System.out.print(" "+instTwo.value(i));
		//}
		//System.out.println(" // dist = "+Math.sqrt(distance));
			
		return Math.sqrt(distance);
	}

	private double getDistance(Instance inst, Cluster cluster)
	{
		double distance = 0.0;
		
		if(cluster instanceof SphereCluster)
		{
			double[] clusterCentre = cluster.getCenter();
			//System.out.print(" sd:");
			for(int i = 0 ; i < clusterCentre.length ; i++)
			{
				distance += Math.pow((inst.value(i) - clusterCentre[i]), 2.0);
				//System.out.print(" "+distance);
			}
			distance = (Math.sqrt(distance) - ((SphereCluster) cluster).getRadius());
			//System.out.print(" "+distance);
		}
		else if(cluster instanceof NonConvexCluster)
		{
			NonConvexCluster ncc = (NonConvexCluster) cluster;
			List<CFCluster> microClusters = ncc.getMicroClusters();
			distance = Double.MAX_VALUE;
			double currDistance;
			
			//System.out.print(" nd:");
			for(int i = 0 ; i < microClusters.size() ; i++)
			{
				currDistance = 0.0;
				double[] clusterCentre = microClusters.get(i).getCenter();
				for(int j = 0 ; j < clusterCentre.length ; j++)
				{
					currDistance += Math.pow((inst.value(j) - clusterCentre[j]), 2.0);
				}
				//System.out.print(" "+currDistance);
				currDistance = (Math.sqrt(currDistance) - microClusters.get(i).getRadius());
				//System.out.print("/"+currDistance);
				
				if(currDistance < distance)
					distance = currDistance;
			}
		}
		
		return distance;
	}

	/**
	 * Trains a ThreadedEnsemble on the argument list of instances
	 * 
	 * @param points list of training instances
	 * 
	 * @return A ThreadedEnsemble that has been trained on all instances in 'points'
	 */
	private ThreadedEnsembleOC trainClassifier(AutoExpandVector<Instance> points)
	{
		ThreadedEnsembleOC te = new ThreadedEnsembleOC(this.numClassifiers, this.classifierOption.getValueAsCLIString());
		Iterator<Instance> pointsIterator = points.iterator();
		
		while(pointsIterator.hasNext())
		{
			te.trainOnInstance(pointsIterator.next());
		}
		
		return te;
	}

	/**
	 * Polls each of the ensembles for their vote on the class label to assign to the argument instance.
	 * 
	 * @param inst the instance to assign a label to.
	 */
	@Override
	public double[] getVotesForInstance(Instance inst)
	{
		double[] votes = {0,0};
		votes[0] = this.getAnomalyScore(inst);
		return votes;
	}
	
	/**
	 * OCClust is not randomizable.
	 */
	@Override
	public boolean isRandomizable()
	{
		return false;
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl()
	{
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent)
	{
	}

	@Override
	public void initialize(Collection<Instance> trainingPoints) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * Returns the argument instance without its concept indicator (the first attribute, at index 0)
	 * 
	 * @param inst the instance from which to remove the concept indicator
	 * @return the argument instance without its concept indicator
	 */
	private Instance getNonConceptMarked(Instance inst)
	{
		double[] newAttributeValues = new double[inst.numAttributes()-1];
		FastVector<Attribute> attributes = new FastVector<Attribute>();
        FastVector<String> classLabels = new FastVector<String>();
		
		for(int i = 0 ; i < (newAttributeValues.length - 1) ; i++)
		{
			newAttributeValues[i] = inst.value(i+1);
			attributes.addElement(new Attribute("att" + (i + 1)));
		}
		for (int i = 0; i < 2; i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class", classLabels));
        
        InstancesHeader newHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        newHeader.setClassIndex(newHeader.numAttributes() - 1);
        
        DenseInstance ncmDI = new DenseInstance(1.0, newAttributeValues);
        ncmDI.setDataset(newHeader);
       //System.out.println(newHeader.numAttributes()+"/"+ncmDI.classIndex());
        
		return ncmDI;
	}
	
	
	@Override
	public double getAnomalyScore(Instance inst)
	{
		double anomalyScore = Double.MAX_VALUE;
		Instance ncmi = getNonConceptMarked(inst);
		
		for(int i = 0 ; i < this.ensembles.size() ; i++)
		{
			anomalyScore = Math.min(anomalyScore, this.ensembles.get(i).getAnomalyScore(ncmi));	
		}
		
		return anomalyScore;
	}

}
