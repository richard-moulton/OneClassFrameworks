/*
 *    OCComplete.java
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

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.OneClassClassifier;
import moa.classifiers.meta.ThreadedEnsembleOC;
import moa.core.AutoExpandVector;
import moa.core.FastVector;
import moa.core.FixedLengthList;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.InstanceStream;

public class OCComplete extends AbstractClassifier implements Classifier, OneClassClassifier
{

	private static final long serialVersionUID = 8354445934235673084L;
	
	public String getPurposeString()
	{
        return "OCComplete performs one class classification by learning different classifiers over the majority class's concepts. "+
        		"These concepts are known with complete domain knowledge.";
    }	
	
	public IntOption numClassifiersOption = new IntOption("numClassifiers", 'n', "Number of classifiers to train per concept", 1);
	
	public ClassOption classifierOption = new ClassOption("classifier", 'c', "Base classifier to use for ensembles",
			OneClassClassifier.class, "oneclass.Autoencoder");
	
	public IntOption initInstancesOption = new IntOption("initOption", 'i', "The number of instances to use for framework "+
			"initialization.",1000, 1, Integer.MAX_VALUE);
		
	private boolean initialized;
	private int numInstances, numConcepts, numClassifiers;
	private FixedLengthList<Instance> initialPoints;
	private AutoExpandVector<ThreadedEnsembleOC> classifiers;

	/**
	 * Passes the test instance to the classifier trained on its concept to get the proper vote.
	 * 
	 * @param inst the test instance to classify
	 * 
	 * @return the votes for the test instance
	 */
	@Override
	public double[] getVotesForInstance(Instance inst)
	{
		//System.out.print(this.numInstances);
		//for(int i = 0 ; i <= this.numConcepts ; i++)
		//{
		//	System.out.print(", "+this.classifiers.get(i).trainingWeightSeenByModel());
		//}
		//System.out.print((int)inst.value(0)+", "+this.thresholds[(int)inst.value(0)]+","+
		//		this.stdDevs[(int)inst.value(0)]+", "+(int)inst.classValue());
		
		double[] votes = {this.getAnomalyScore(inst), 0};
		
		//votes[0] += (anomalyScore) / (6.0 * this.stdDevs[(int)inst.value(0)]);
		//if(votes[0] > 1)
		//	votes[0] = 1;
		//else if (votes[0] < 0)
		//	votes[0] = 0;
		//votes[1] = 1- votes[0];
		//System.out.println(", "+anomalyScore+","+votes[0]);
		
		//System.out.println("\n("+this.numInstances+") Concept: "+(int)inst.value(0)+", Inst Weight: "+inst.weight()+
		//		", Classifier "+this.classifiers.get((int)inst.value(0)).hashCode()+", Cl. Inst.: "+this.classifiers.get((int)inst.value(0)).trainingWeightSeenByModel());
		//return this.classifiers.get((int)inst.value(0)).getVotesForInstance(getNonConceptMarked(inst));
		
		return votes;
	}

	
	public double getAnomalyScore(Instance inst)
	{
		int concept = (int)inst.value(0);
		double anomalyScore = this.classifiers.get(concept).getAnomalyScore(getNonConceptMarked(inst));
		
		return anomalyScore;
	}
	
	/**
	 * Resets all of the framework's internal variables.
	 */
	@Override
	public void resetLearningImpl()
	{
		this.initialized = false;
		this.numInstances = 0;
		this.numConcepts = -1;
		
		this.classifiers = new AutoExpandVector<ThreadedEnsembleOC>();
		
		this.initialPoints = new FixedLengthList<Instance>(this.initInstancesOption.getValue());
		this.numClassifiers = this.numClassifiersOption.getValue();
	}

	/**
	 * During the initialization period, stores instances in <b>initialPoints</b>.
	 * At initialization, calls <b>initializeFramework.</b>
	 * After initialization, gives the training instance to the classifier trained on its concept.
	 * 
	 * @param inst the training instance
	 * 
	 * @see initialPoints
	 * @see initializeFramework
	 */
	@Override
	public void trainOnInstanceImpl(Instance inst)
	{
		this.numInstances++;
		
		if(this.initialized)
		{
			int conceptIndex = (int)inst.value(0);
			ThreadedEnsembleOC te = this.classifiers.get(conceptIndex);
			te.trainOnInstance(getNonConceptMarked(inst));
			this.classifiers.set(conceptIndex, te);
		}
		else
		{
			this.initialPoints.add(inst);
			
			if(this.numInstances == this.initInstancesOption.getValue())
			{
				initializeFramework();
				this.initialized = true;
			}
		}
	}

	/**
	 * Determines the number of concepts present and trains a ThreadedEnsemble on each.
	 */
	private void initializeFramework()
	{
		AutoExpandVector<AutoExpandVector<Instance>> pointsByConcept = new AutoExpandVector<AutoExpandVector<Instance>>();
		
		// Determine the number of concepts. Concepts much be numbered sequentially starting from 0.
		for(Instance inst : this.initialPoints)
		{
			int thisConcept = (int)inst.value(0);
			if(thisConcept > this.numConcepts)
			{
				for(int i = this.numConcepts ; i <= thisConcept ; i++)
				{
					pointsByConcept.add(new AutoExpandVector<Instance>());
				}
				
				this.numConcepts = thisConcept;				
			}
			
			AutoExpandVector<Instance> currConcept = pointsByConcept.get(thisConcept);
			currConcept.add(inst);
			pointsByConcept.set(thisConcept, currConcept);
		}
					
		// Train a ThreadedEnsemble for each concept.
		for(int i = 0 ; i <= this.numConcepts ; i++)
		{
			FixedLengthList<Instance> trainingPoints = new FixedLengthList<Instance>(this.initInstancesOption.getValue()/2);
			trainingPoints.addAll(pointsByConcept.get(i));
			
			this.classifiers.set(i, trainClassifier(trainingPoints));
		}
	}
	
	/**
	 * Trains and returns a ThreadedEnsemble using the training points from a given concept.
	 * 
	 * @param concept the specific concept to train the ThreadedEnsemble on
	 * @param trainingPoints a list of all available training points
	 * @return a trained ThreadedEnsemble
	 */
	private ThreadedEnsembleOC trainClassifier(FixedLengthList<Instance> trainingPoints)
	{
		ThreadedEnsembleOC te = new ThreadedEnsembleOC(this.numClassifiers, this.classifierOption.getValueAsCLIString());
		
		FixedLengthList<Instance> ncmTrainingPoints = new FixedLengthList<Instance>(trainingPoints.getMaxSize());
		
		for(Instance inst : trainingPoints)
		{
			ncmTrainingPoints.add(getNonConceptMarked(inst));
		}
		
		te.initialize(ncmTrainingPoints);
			
		return te;
	}

	
	/**private Instance oversample(Instance baseInst, Instance neighbourInst)
	{
		double[] attributesOne = baseInst.toDoubleArray();
		double[] attributesTwo = neighbourInst.toDoubleArray();
		double[] newAttributes = new double[baseInst.numAttributes()];
		double gap = this.modelRandom.nextDouble();
		
		// Generate a new instance somewhere between the argument instances
		for(int i = 0 ; i < newAttributes.length - 1 ; i++)
		{
			newAttributes[i] = attributesOne[i] + ((attributesTwo[i] - attributesOne[i]) * gap);
		}

		// Add the class value
		newAttributes[newAttributes.length-1] = attributesOne[newAttributes.length-1];
		Instance newInstance = new DenseInstance(1.0, newAttributes);
		newInstance.setDataset(baseInst.dataset());
		
		//for(int i = 0 ; i < newAttributes.length ; i++)
		//{
		//	System.out.print(", "+newAttributes[i]);
		//}
		
		return newInstance;
	}*/
	
	/**
	 * Returns the argument instance without its concept indicator (the first attribute, at index 0)
	 * 
	 * @param inst the instrument from which to remove the concept indicator
	 * @return the argument instance without its concept indicator
	 */
	private Instance getNonConceptMarked(Instance inst)
	{
		double[] newAttributeValues = new double[inst.numAttributes()-1];
		FastVector<Attribute> attributes = new FastVector<Attribute>();
        FastVector<String> classLabels = new FastVector<String>();
		
		for(int i = 0 ; i < (newAttributeValues.length-1) ; i++)
		{
			newAttributeValues[i] = inst.value(i+1);
			attributes.addElement(new Attribute("att" + (i + 1)));
		}
		newAttributeValues[newAttributeValues.length-1] = inst.classValue();
		for (int i = 0; i < 2; i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class", classLabels));
        
        InstancesHeader newHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        newHeader.setClassIndex(newHeader.numAttributes() - 1);
        
        DenseInstance ncmDI = new DenseInstance(1.0, newAttributeValues);
        ncmDI.setDataset(newHeader);
        
		return ncmDI;
	}
	
	/**
	 * OCComplete is not randomizable.
	 * 
	 * @return false
	 */
	@Override
	public boolean isRandomizable()
	{
		return false;
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent)
	{
		// TODO Auto-generated method stub

	}

	@Override
	public void initialize(Collection<Instance> trainingPoints) {
		// TODO Auto-generated method stub
		
	}

}
