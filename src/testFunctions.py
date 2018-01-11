#!/usr/bin/env python
"""
***************************************************
File: testFunctions.py
Author: Luke Burks
Date: October 2017

Houses functions to be called by testArena.py

***************************************************
"""

from __future__ import division
from gaussianMixtures import GM,Gaussian
from copy import deepcopy
import numpy as np
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import inv,det
import matplotlib.pyplot as plt
from scipy import random as scirand


def separateAndNormalize(mix):
	#Takes a mixture and returns two mixtures
	#and two anti-normalizing constants

	posMix = GM(); 
	negMix = GM(); 

	posSum = 0; 
	negSum = 0; 

	for g in mix:
		if(g.weight < 0):
			negSum += g.weight; 
			negMix.addG(deepcopy(g)); 
		else:
			posSum += g.weight; 
			posMix.addG(deepcopy(g)); 

	posMix.normalizeWeights(); 
	negMix.normalizeWeights(); 


	return [posMix,negMix,posSum,negSum]; 

def cluster(mixture,distanceFunc,k=4,maxIter = 100):
	'''
	Condenses mixands by first clustering them into K groups, using
	k-means. Then each group is condensed to a predefined number N using 
	runnals method, such that K*N = M, the desired number of mixands

	Inputs:
	k: number of mixand groups
	N: the condensed size of each group

	lowInit: lower bound on the placement of initial grouping means
	highInit: upper bound on placement of initial grouping means

	'''

	if(mixture.size <= k):
		return mixture;

	mixDims = len(mixture[0].mean) + len(mixture[0].var)*len(mixture[0].var); 

	lowInit = [0]*mixDims; 
	highInit = [10]*mixDims; 

	means = [1]*k; 
	for i in range(0,k):
		tmp = []; 
		#get a random mean
		for j in range(0,len(mixture[0].mean)):
			tmp.append(np.random.random()*(highInit[j]-lowInit[j]) + lowInit[j]); 
		#get a random covariance
		#MUST BE POSITIVE SEMI DEFINITE and symmetric
		a = scirand.rand(len(mixture[0].var),len(mixture[0].var))*5; 
		b = np.dot(a,a.transpose()); 
		c = (b+b.T)/2; 
		d = c.flatten().tolist(); 
		for j in range(0,len(d)):
			tmp.append(d[j]); 	
		means[i] = tmp; 


	converge = False; 
	count = 0; 
	newMeans = [1]*k; 

	while(not converge and count < maxIter):
		clusters = [GM() for i in range(0,k)]; 
		for g in mixture:
			#put the gaussian in the cluster which minimizes the distance between the distribution mean and the cluster mean
			clusters[np.argmin([distanceFunc(g,convertListToNorm(means[j])) for j in range(0,k)])].addG(g);

		#find the new mean of each cluster
		newMeans = [0]*k; 

		for i in range(0,k):
			newMeans[i] = np.array([0]*mixDims); 
			for g in clusters[i]:
				newMeans[i] = np.add(newMeans[i],np.divide(convertNormToList(g),clusters[i].size));
		
		#check for convergence
		if(np.array_equal(means,newMeans)):
			converge = True;
		count = count+1;

		for i in range(0,k):
			if not newMeans[i].all() == 0: 
				means[i] = newMeans[i]
			else:
				tmp = []; 
				#get a random mean
				for j in range(0,len(mixture[0].mean)):
					tmp.append(np.random.random()*(highInit[j]-lowInit[j]) + lowInit[j]); 
				#get a random covariance
				#MUST BE POSITIVE SEMI DEFINITE and symmetric
				a = scirand.rand(len(mixture[0].var),len(mixture[0].var))*5; 
				b = np.dot(a,a.transpose()); 
				c = (b+b.T)/2; 
				d = c.flatten().tolist(); 
				for j in range(0,len(d)):
					tmp.append(d[j]); 
				# print('tmp: {}'.format(tmp))	
				means[i] = tmp;
	return clusters;

def conComb(mixtures,max_num_mixands):
	newMix = GM(); 
	for gm in mixtures:
		d = deepcopy(condense(gm,max_num_mixands)); 
		# print(type(d))
		# NOTE: this comment apparently needs to be here to not make d an int...
		try:
			if(d.size > 0):
				newMix.addGM(d); 
		except AttributeError as e:
			# print('throwing out')
			# print e
			pass
	return newMix; 


def condense(mixture,max_num_mixands):


	#Check if any mixands are small enough to not matter
	#specifically if they're weighted really really low
	dels = [];
	for g in mixture.Gs:
		if(abs(g.weight) < 0.000001):
			dels.append(g);

	for rem in dels:
		if(rem in mixture.Gs):
			mixture.Gs.remove(rem);
			mixture.size = mixture.size-1;


	#Check if any mixands are identical
	dels = [];
	for i in range(0,mixture.size):
		for j in range(0,mixture.size):
			if(i==j):
				continue;
			g1 = mixture.Gs[i];
			g2 = mixture.Gs[j];
			if(g1.fullComp(g2) and g1 not in dels):
				dels.append(g2);
				g1.weight = g1.weight*2;
	for rem in dels:
		if(rem in mixture.Gs):
			mixture.Gs.remove(rem);
			mixture.size = mixture.size-1;


	#Check if merging is useful <>NOTE: this seems to be the spot where things are going south
	# for a large final number of mixands, condensing is not needed often, and condense returns
	# 0, an int, which is decidedly not a gaussian mixture
	# <>TODO: ask Luke what is suppose to happen when codensation is not needed 
	if mixture.size <= max_num_mixands:
		# print('returning')
		# return 0;
		return mixture

	# Create lower-triangle of dissimilarity matrix B
	#<>TODO: this is O(n ** 2) and very slow. Speed it up! parallelize?
	B = np.zeros((mixture.size, mixture.size))

	for i in range(mixture.size):
	    #mix_i = (mixture.Gs[i].weight, mixture.Gs[i].mean, mixture.Gs[i].var)
	    mix_i = mixture[i]; 
	    for j in range(i):
	        if i == j:
	            continue
	        #mix_j = (mixture.Gs[j].weight, mixture.Gs[j].mean, mixture.Gs[j].var)
	        mix_j = mixture.Gs[j]; 
	        B[i,j] = KLD_UpperBound(mix_i, mix_j)



	# Keep merging until we get the right number of mixands
	deleted_mixands = []
	toRemove = [];
	while mixture.size > max_num_mixands:
	    # Find most similar mixands

		try:
			min_B = B[abs(B)>0].min()
		except:
			#mixture.display();
			#raise;
			return;



		ind = np.where(B==min_B)
		i, j = ind[0][0], ind[1][0]

		# Get merged mixand
		#mix_i = (mixture.Gs[i].weight, mixture.Gs[i].mean, mixture.Gs[i].var)
		#mix_j = (mixture.Gs[j].weight, mixture.Gs[j].mean, mixture.Gs[j].var)
		mix_i = mixture[i]; 
		mix_j = mixture[j]
		w_ij, mu_ij, P_ij = merge_mixands(mix_i, mix_j)

		# Replace mixand i with merged mixand
		ij = i
		mixture.Gs[ij].weight = w_ij
		mixture.Gs[ij].mean = mu_ij.tolist();
		mixture.Gs[ij].var = P_ij.tolist();



		# Fill mixand i's B values with new mixand's B values
		#mix_ij = (w_ij, mu_ij, P_ij)
		mix_ij = Gaussian(mu_ij,P_ij,w_ij); 
		deleted_mixands.append(j)
		toRemove.append(mixture.Gs[j]);

		#print(B.shape[0]);

		for k in range(0,B.shape[0]):
		    if k == ij or k in deleted_mixands:
		        continue

		    # Only fill lower triangle
		   # print(self.size,k)
		    #mix_k = (mixture.Gs[k].weight, mixture.Gs[k].mean, mixture.Gs[k].var)
		    mix_k = mixture[k];
		    if k < i:
		        B[ij,k] = KLD_UpperBound(mix_k, mix_ij)
		    else:
		        B[k,ij] = KLD_UpperBound(mix_k, mix_ij)

		# Remove mixand j from B
		B[j,:] = np.inf
		B[:,j] = np.inf
		mixture.size -= 1

	#print(mixture)


	# Delete removed mixands from parameter arrays
	for rem in toRemove:
		if(rem in mixture.Gs):
			mixture.Gs.remove(rem);

	#Make sure everything is positive semidefinite
	#TODO: dont just remove them, fix them?
	dels = [];
	for g in mixture.Gs:
		if(det(np.matrix(g.var)) <= 0):
			dels.append(g);
	for rem in dels:
		if(rem in mixture.Gs):
			mixture.Gs.remove(rem);
			mixture.size -= 1
	# print(type(mixture))
	return mixture; 

def merge_mixands(mix_i, mix_j):
	"""Use moment-preserving merge (0th, 1st, 2nd moments) to combine mixands.
	"""
	# Unpack mixands
	w_i,mu_i,P_i = mix_i.weight,mix_i.mean,mix_i.var
	w_j,mu_j,P_j = mix_j.weight,mix_j.mean,mix_j.var

	mu_i = np.array(mu_i);
	mu_j = np.array(mu_j);

	P_j = np.matrix(P_j);
	P_i = np.matrix(P_i);

	# Merge weights
	w_ij = w_i + w_j
	w_i_ij = w_i / (w_i + w_j)
	w_j_ij = w_j / (w_i + w_j)

	# Merge means

	mu_ij = w_i_ij * mu_i + w_j_ij * mu_j

	P_j = np.matrix(P_j);
	P_i = np.matrix(P_i);


	# Merge covariances
	P_ij = w_i_ij * P_i + w_j_ij * P_j + \
	    w_i_ij * w_j_ij * np.outer(subMu(mu_i,mu_j), subMu(mu_i,mu_j))



	return w_ij, mu_ij, P_ij

def subMu(a,b):

	if(isinstance(a,np.ndarray)):
		return a-b;
	if(isinstance(a,(float,int))):
		return a-b;
	else:
		c = [0]*len(a);
		for i in range(0,len(a)):
			c[i] = a[i]-b[i];
		return c;


def KLD_UpperBound(mix_i, mix_j):
	"""Calculate KL descriminiation-based dissimilarity between mixands.
	"""
	# Get covariance of moment-preserving merge
	#w_i, mu_i, P_i = mix_i
	#w_j, mu_j, P_j = mix_j
	w_i,mu_i,P_i = mix_i.weight,mix_i.mean,mix_i.var
	w_j,mu_j,P_j = mix_j.weight,mix_j.mean,mix_j.var
	_, _, P_ij = merge_mixands(mix_i, mix_j)


	if(P_ij.ndim == 1 or len(P_ij.tolist()[0]) == 1):
			if(not isinstance(P_ij,(int,list,float))):
				P_ij = P_ij.tolist()[0];
			while(isinstance(P_ij,list)):
				P_ij = P_ij[0];

			if(not isinstance(P_i,(int,list,float))):
				P_i = P_i.tolist()[0];
			while(isinstance(P_i,list)):
				P_i = P_i[0];
			if(not isinstance(P_j,(int,list,float))):
				P_j = P_j.tolist()[0];
			while(isinstance(P_j,list)):
				P_j = P_j[0];



			logdet_P_ij = P_ij;
			logdet_P_i = P_i;
			logdet_P_j = P_j;


	else:
	    # Use slogdet to prevent over/underflow
	    _, logdet_P_ij = np.linalg.slogdet(P_ij)
	    _, logdet_P_i = np.linalg.slogdet(P_i)
	    _, logdet_P_j = np.linalg.slogdet(P_j)

	    # <>TODO: check to see if anything's happening upstream
	    if np.isinf(logdet_P_ij):
	        logdet_P_ij = 0
	    if np.isinf(logdet_P_i):
	        logdet_P_i = 0
	    if np.isinf(logdet_P_j):
	        logdet_P_j = 0

	#print(logdet_P_ij,logdet_P_j,logdet_P_i)

	b = 0.5 * ((w_i + w_j) * logdet_P_ij - w_i * logdet_P_i - w_j * logdet_P_j)
	
	return b;

def convertNormToList(g):
	#converts to [mean,var]
	l = []; 
	for i in range(0,len(g.mean)):
		l.append(g.mean[i]); 
	flatVar = np.array(g.var).flatten(); 
	for i in range(0,len(flatVar)):
		l.append(flatVar[i]); 
	return l; 

def convertListToNorm(l):
	#coverts to a gaussian
	g = Gaussian(); 

	meanLen = 0; 
	if(len(l) == 2):
		meanLen = 1; 
	elif(len(l) == 6):
		meanLen = 2; 
	elif(len(l) == 12):
		meanLen = 3; 
	elif(len(l) == 20):
		meanLen = 4; 
	elif(len(l) == 30):
		meanLen = 5; 

	newMean = []; 
	for i in range(0,meanLen):
		newMean.append(l[i]); 

	h = l[meanLen:]; 
	newVar = []; 
	for i in range(0,meanLen):
		line = []; 
		for j in range(0,meanLen):
			line.append(h[i*meanLen + j]); 
		newVar.append(line); 
	g = Gaussian(newMean,newVar,1); 
	return g; 


def euclidianMeanDistance(mix_i,mix_j):
	#General N-dimensional euclidean distance
	dist = 0;
	a = mix_i.mean;
	b = mix_j.mean;

	for i in range(0,len(a)):
		dist += (a[i]-b[i])**2;
	dist = np.sqrt(dist);
	return dist;

if __name__ == '__main__':
	testMix = GM();

	for i in range(0,200):
		w = np.random.random()*100 - 50; 
		mean = [np.random.random()*10,np.random.random()*10]
		sides = np.random.random(); 
		var = [[np.random.random()*5+1,sides],[sides,np.random.random()*5+1,]];
		testMix.addG(Gaussian(mean,var,w));
	c = fixedCluster(testMix,euclidianMeanDistance);

	for i in range(0,len(c)):
		c[i].display();



	
	
