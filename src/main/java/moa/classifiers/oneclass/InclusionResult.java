/*
 *    InclusionResult.java
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

import java.io.Serializable;

public class InclusionResult implements Serializable
{
	private static final long serialVersionUID = -7591826457348263943L;
	private int clusterAssignment;
	private double inclusionProbability;
	
	/**
	 * Constructor 1
	 */
	public InclusionResult()
	{
		
	}
	
	/**
	 * Constructor 2
	 * 
	 * @param cA the cluster assignment
	 * @param iP the inclusion probability
	 */
	public InclusionResult(int cA, double iP)
	{
		this.clusterAssignment = cA;
		this.inclusionProbability = iP;
	}
	
	/**
	 * @return the cluster to which the instance was assigned.
	 */
	public int getClusterAssignment() {
		return clusterAssignment;
	}
	/**
	 * @param clusterAssignment the cluster to which the instance is assigned.
	 */
	public void setClusterAssignment(int clusterAssignment) {
		this.clusterAssignment = clusterAssignment;
	}
	/**
	 * @return the instance's inclusion probability in the assigned cluster.
	 */
	public double getInclusionProbability() {
		return inclusionProbability;
	}
	/**
	 * @param inclusionProbability the inclusion probability of the instance in its assigned cluster.
	 */
	public void setInclusionProbability(double inclusionProbability) {
		this.inclusionProbability = inclusionProbability;
	}
}
