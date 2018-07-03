/*
 *    SMOTE.java
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

package moa.core;

import java.util.Collection;
import java.util.Random;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * This code implements the Synthetic Minority Over-sampling Technique from Chawla et al.
 * 
 * N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: 
 * Synthetic minority over-sampling technique,” J. Artif. Intell. Res., 
 * vol. 16, pp. 321–357, 2002.
 * 
 * @author Richard Hugh Moulton
 *
 */
public class SMOTE
{
	AutoExpandVector<Instance> dataset;
	int numNN;
	int[][] nearestNeighbours;
	Random smoteRandom;

	/**
	 * Constructor. Populates the dataset and determines the appropriate number of nearest neighbours.
	 * 
	 * @param coll the instances with which to populate dataset
	 * @param nN the number of nearest neighbours to calculate for each instance
	 * @param randomSeed the seed for the random number generator
	 */
	public SMOTE(Collection<Instance> coll, int nN, int randomSeed)
	{
		this.dataset = new AutoExpandVector<Instance>();
		this.dataset.addAll(coll);
		this.smoteRandom = new Random(randomSeed);

		this.numNN = Math.min(nN, coll.size());

		if(!this.dataset.isEmpty())
		{
			computeNearestNeighbours();			
		}
	}

	/**
	 * Produces the next SMOTE generated synthetic instance based on dataset.
	 * @return
	 */
	public Instance next()
	{
		int chosenIndex = smoteRandom.nextInt(this.dataset.size());
		int chosenNeighbour = this.nearestNeighbours[chosenIndex][smoteRandom.nextInt(this.numNN)];

		double[] inst = this.dataset.get(chosenIndex).toDoubleArray();
		double[] instNN = this.dataset.get(chosenNeighbour).toDoubleArray();
		double[] newAttributes = new double[inst.length];
		double gap = this.smoteRandom.nextDouble();

		// Generate a new instance somewhere between the argument instances
		for(int i = 0 ; i < newAttributes.length - 1 ; i++)
		{
			newAttributes[i] = inst[i] + ((instNN[i] - inst[i]) * gap);
		}
		
		// Add the class value
		newAttributes[newAttributes.length-1] = inst[newAttributes.length-1];
		Instance smoteInst = new DenseInstance(1.0, newAttributes);
		smoteInst.setDataset(this.dataset.get(chosenIndex).dataset());

		return smoteInst;
	}

	/**
	 * Computes the 'this.numNN' nearest neighbours of the instances in 'this.dataset.'
	 */
	private void computeNearestNeighbours()
	{					
		// Initialize data structures
		this.nearestNeighbours = new int[this.dataset.size()][this.numNN];
		double[] distances = new double[this.numNN];

		// For every instance in dataset...
		for(int i = 0 ; i < this.dataset.size() ; i++)
		{
			//...grab the instance...
			Instance instOne = this.dataset.get(i);

			//...initialize distances...
			for(int k = 0 ; k < this.numNN ; k++)
			{
				distances[k] = Integer.MAX_VALUE;
			}

			//...check all instances in dataset...
			for(int j = 0 ; j < this.dataset.size() ; j++)
			{
				//...that aren't the current dataset...
				if(i!=j)
				{
					double distance = distance(instOne, this.dataset.get(j));
					int index = -1;

					//...check if they are in the top 'nN' nearest neighbours...
					for(int k = 0 ; k < this.numNN ; k++)
					{
						if(distance < distances[k])
						{
							index = k;
							break;
						}
					}

					//...if they make the cut, update this.nearestNeighbours and distances.
					if(index != -1)
					{
						for(int l = (this.numNN-1) ; l > index ; l--)
						{
							this.nearestNeighbours[i][l] = this.nearestNeighbours[i][l-1];
							distances[l] = distances[l-1];
						}

						this.nearestNeighbours[i][index] = j;
						distances[index] = distance;
					}
				}
			}
		}
	}

	/**
	 * Calculates the Euclidean distance between two argument instances.
	 * 
	 * @param instOne the first instance
	 * @param instTwo the second instance
	 * 
	 * @return the distance between instOne and instTwo
	 */
	private double distance(Instance instOne, Instance instTwo)
	{
		double dist = 0.0;

		for(int i = 0 ; i < instOne.numAttributes() ; i++)
		{
			dist += Math.pow(instOne.value(i)-instTwo.value(i), 2.0);
		}

		return Math.sqrt(dist);
	}


}
