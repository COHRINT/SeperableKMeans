"""
***************************************************
File: testArena.py
Author: Luke Burks
Date: October 2017

Exploring alternative metrics with which to cluster
gaussians using K-means and runnals method

***************************************************
"""


from __future__ import division
from gaussianMixtures import GM,Gaussian
from copy import deepcopy
import numpy as np
from testFunctions import *


#distance functions
def euclidianMeanDistance(mix_i,mix_j):
	#General N-dimensional euclidean distance
	dist = 0;
	a = mix_i.mean; 
	b = mix_j.mean; 

	for i in range(0,len(a)):
		dist += (a[i]-b[i])**2;
	dist = np.sqrt(dist);
	return dist;

def euclidSquared(mix_i,mix_j):
	#General N-dimensional euclidean distance
	dist = 0;
	a = mix_i.mean; 
	b = mix_j.mean; 

	for i in range(0,len(a)):
		dist += (a[i]-b[i])**2;
	#dist = np.sqrt(dist);
	return dist;


#main testing function
def theArena(mix,kmeansFunc,numClusters = 4,finalNum = 5,verbose = False):
	startMix = deepcopy(mix); 

	#separate
	[posMix,negMix,posNorm,negNorm] = separateAndNormalize(startMix); 

	#cluster
	posClusters = cluster(posMix,kmeansFunc,numClusters); 
	negClusters = cluster(negMix,kmeansFunc,numClusters);

	#condense
	posCon = conComb(posClusters,finalNum); 
	negCon = conComb(negClusters,finalNum); 

	#recombine
	newMix = GM(); 
	posCon.scalerMultiply(posNorm);
	newMix.addGM(posCon); 
	negCon.scalerMultiply(negNorm)
	newMix.addGM(negCon); 

	if(verbose):
		plotResults(mix,newMix); 
	return newMix


def plotResults(start,end):
	[xBefore,yBefore,cBefore] = start.plot2D(low=[0,0],high=[10,10],vis=False); 
	[xAfter,yAfter,cAfter] = end.plot2D(low=[0,0],high=[10,10],vis=False); 


	fig,axarr = plt.subplots(2); 

	im1 = axarr[0].contourf(xBefore,yBefore,cBefore); 
	axarr[0].set_title('Original')

	im2 = axarr[1].contourf(xAfter,yAfter,cAfter); 
	axarr[1].set_title('ISD:{}'.format(start.ISD(end))); 
	  
	plt.suptitle("Condensation from 200 to 40 mixands"); 

	plt.show(); 


def createRandomMixture(size = 200,dims = 2):
	testMix = GM(); 

	lowInit = [0]*dims; 
	highInit = [5]*dims;

	for i in range(0,200):
		tmp = []; 
		#get a random mean
		for j in range(0,dims):
			tmp.append(np.random.random()*(highInit[j]-lowInit[j]) + lowInit[j]); 
		#get a random covariance
		#MUST BE POSITIVE SEMI DEFINITE and symmetric
		a = scirand.rand(dims,dims)*1;

		#keep things from getting near singular
		for j in range(0,len(a)):
			a[j][j] = (a[j][j]+1)**2; 

		b = np.dot(a,a.transpose()); 
		c = (b+b.T)/2; 
		
		d = c.flatten().tolist(); 
		for j in range(0,len(d)):
			tmp.append(d[j]); 
		weight = np.random.random()*100-50; 
		g = convertListToNorm(tmp); 
		g.weight = weight; 
		testMix.addG(g); 
		
	return testMix; 


if __name__ == '__main__':


	#Testing Parameters:
	dims = 2; 
	startNum = 100; 
	distanceMeasure = euclidianMeanDistance;
	intermediate_mixture_size = 4; 
	finalNum = 5; 

	#Create Test Mixture from params
	testMix = createRandomMixture(startNum,dims); 

	#Run tests
	results = [];
	results.append(theArena(testMix,distanceMeasure,intermediate_mixture_size,finalNum));  

	#Save/display results
	plotResults(testMix,results[0]);
	print(testMix.ISD(results[0])); 
