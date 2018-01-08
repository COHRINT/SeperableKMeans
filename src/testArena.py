#!/usr/bin/env python
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
import time
import timeit
import pickle
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


def KLD(mix_i,mix_j):
	"""
	Computes the Kullback-Leibler Divergence between two multivariate normal distributions
	"""

	new_mix_i = deepcopy(mix_i)
	new_mix_j = deepcopy(mix_j)

	# term1 = np.trace(np.m)

	# print('det mix j var = {}'.format(np.linalg.det(mix_j.var)))
	# print('det mix i var = {}'.format(np.linalg.det(mix_i.var)))
	# try:
		# val = np.linalg.inv(mix_j.var)
	# except np.linalg.linalg.LinAlgError as e:
		# print('mix j means: {}'.format(mix_j.mean))
		# print('mix j var: {}'.format(mix_j.var))
	# print(np.linalg.inv(mix_j.var))

	div = 0.5*(np.trace(np.dot(np.linalg.inv(new_mix_j.var),new_mix_i.var)) + np.dot(np.dot(np.transpose(np.subtract(new_mix_j.mean,new_mix_i.mean)) \
				,np.linalg.inv(new_mix_j.var)),(np.subtract(new_mix_j.mean,new_mix_i.mean))) \
				- len(new_mix_j.mean) + np.log(np.linalg.det(new_mix_j.var)/np.linalg.det(new_mix_i.var)))

	return div

def symKLD(mix_i,mix_j):
	"""
	Computes the symmetric Kullback-Leibler Divergence between two multivarite normal distributions.
	Calls fxn KLD in this file, testArena.py

	D = { KLD(P || Q) + KLD(Q || P) } / 2 
	""" 
	
	new_mix_i = deepcopy(mix_i)
	new_mix_j = deepcopy(mix_j)

	dist = 0.5*( KLD(new_mix_i,new_mix_j) + KLD(new_mix_j,new_mix_i) )

	return dist 


def JSD(mix_i,mix_j):
	"""
	Computes the Jensen-Shannon divergence between two multivarite normal distributions using
	the Kullback-Leibler Divergence

	JSD(I || J) = 0.5*D(I || M) + 0.5*D(J || M)
	M = 0.5*(I + J)
	"""

	new_mix_i = deepcopy(mix_i)
	new_mix_j = deepcopy(mix_j)

	# compute M = 0.5 * (I + J)
	new_mean = np.multiply(0.5,np.add(new_mix_i.mean, new_mix_j.mean))
	new_var = np.multiply(0.25,np.add(new_mix_i.var, new_mix_j.var))
	new_mix_m = Gaussian(new_mean,new_var)


	# D(I || M)
	div1 = 0.5*(np.trace(np.dot(np.linalg.inv(new_mix_m.var),new_mix_i.var)) + np.dot(np.dot(np.transpose(np.subtract(new_mix_m.mean,new_mix_i.mean)) \
				,np.linalg.inv(new_mix_m.var)),(np.subtract(new_mix_m.mean,new_mix_i.mean))) \
				- len(new_mix_m.mean) + np.log(np.linalg.det(new_mix_m.var)/np.linalg.det(new_mix_i.var)))

	# D(J || M)
	div2 = 0.5*(np.trace(np.dot(np.linalg.inv(new_mix_m.var),new_mix_j.var)) + np.dot(np.dot(np.transpose(np.subtract(new_mix_m.mean,new_mix_j.mean)) \
				,np.linalg.inv(new_mix_m.var)),(np.subtract(new_mix_m.mean,new_mix_j.mean))) \
				- len(new_mix_m.mean) + np.log(np.linalg.det(new_mix_m.var)/np.linalg.det(new_mix_j.var)))

	div = (0.5*div1) + (0.5*div2)
	return div

def EMD(mix_i,mix_j):
	"""
	Computes the Earth Mover's Distance or 2-Wasserstein distance between two normal distributions
	with means m1 and m2 and covariance matrices C1 and C2

	W_2(mu1,mu2)^2 = ||m1-m2||_2^2 - Tr( sqrt( C1 + C2 - 2( sqrt(C2) C1 sqrt(C2) ) ) )

	See:
	https://en.wikipedia.org/wiki/Wasserstein_metric
	https://en.wikipedia.org/wiki/Earth_mover%27s_distance
	"""
	new_mix_i = deepcopy(mix_i)
	new_mix_j = deepcopy(mix_j)

	m1 = new_mix_i.mean
	m2 = new_mix_j.mean
	C1 = new_mix_i.var
	C2 = new_mix_i.var

	# compute 2-norm of means
	norm2 = np.square( np.linalg.norm( np.subtract(m1,m2) ) )

	mult1 = np.dot( C1,np.sqrt(C2) )
	mult2 = np.dot( np.sqrt(C2),mult1 )

	# dist = norm2 - np.trace( np.sqrt( np.subtract( np.add(C1,C2), np.dot( 2,np.dot( np.sqrt(C2),np.dot( C1,np.sqrt(C2) ) ) ) ) ) )

	dist = norm2 - np.trace( np.subtract(np.add(C1,C2),np.dot(2,np.sqrt(mult2)) ) ) 

	return dist

def bhatt(mix_i,mix_j):
	"""
	Computes the Bhattacharyya distance between two multivariate normal distributions
	using the Bhattacharyya coefficient.

	D_B = (1/8)(m1-m2)^T Cov (m1-m2) + (1/2)ln( det(C) / sqrt( det(C1)det(C2) ) )
	Cov = (Cov1 + Cov2)/2

	See:
	https://en.wikipedia.org/wiki/Bhattacharyya_distance
	"""
	new_mix_i = deepcopy(mix_i)
	new_mix_j = deepcopy(mix_j)

	m1 = new_mix_i.mean
	m2 = new_mix_j.mean
	C1 = new_mix_i.var
	C2 = new_mix_j.var
	C = np.divide(np.add(C1,C2),2)

	m = np.subtract(m1,m2)
	m_trans = np.transpose(m)
	C_det = np.linalg.det(C)
	C1_det = np.linalg.det(C1)
	C2_det = np.linalg.det(C2)

	dist = (1/8)*(np.dot( m_trans,np.dot(np.linalg.inv(C),m) )) + \
			(0.5*np.log(C_det / np.sqrt(C1_det*C2_det)))

	# print(dist)
	return dist



#main testing function
def theArena(mix,kmeansFunc,numClusters = 4,finalNum = 5,verbose = False):
	"""
	numClusters: number if intermediate clusters
	finalNum: final number of mixands per cluster
	"""
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
	[xBefore,yBefore,cBefore] = start.plot2D(low=[0,0],high=[50,50],vis=False); 
	[xAfter,yAfter,cAfter] = end.plot2D(low=[0,0],high=[50,50],vis=False); 


	fig,axarr = plt.subplots(2); 

	im1 = axarr[0].contourf(xBefore,yBefore,cBefore,cmap='viridis'); 
	axarr[0].set_title('Original')

	im2 = axarr[1].contourf(xAfter,yAfter,cAfter,cmap='viridis'); 
	axarr[1].set_title('ISD:{}'.format(start.ISD(end))); 
	  
	plt.suptitle("Condensation from 200 to 40 mixands"); 

	plt.show(); 


def createRandomMixture(size = 200,dims = 2):
	testMix = GM(); 

	lowInit = [0]*dims; 
	highInit = [50]*dims;

	for i in range(0,size):
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
		weight = np.random.random()*5; 
		g = convertListToNorm(tmp); 
		g.weight = weight; 
		testMix.addG(g); 
		
	return testMix;

# def get_data(param_list):



# def iter_results(d):
# 		for key, val in d.iteritems():
# 			if isinstance(val,dict):
# 				yield from iter_results(val)
# 			else:
# 				yield val

def get_pos_mix(mix):
	pos_mix = GM()
	for g in mix:
		if(g.weight >= 0):
			pos_mix.addG(deepcopy(g))
	return pos_mix


if __name__ == '__main__':


	#Testing Parameters:
	dims = [1]; 
	# startNum = [100,400,700];
	# srating number of mixands
	startNum = [100] 
	# distanceMeasure = [euclidianMeanDistance];
	distanceMeasure = [symKLD,JSD,euclidianMeanDistance,EMD,bhatt]
	# distanceMeasure = [euclidianMeanDistance,euclidSquared,KLD,JSD];	
	intermediate_mixture_size = [4]; 
	# finalNum = [10,30,50]; # 10 30 50
	finalNum = [3];

	save = False

	results = []
	isd = []
	times = []

	total_results = {}

	# for dim in dims:
	# 	for num in startNum:
	# 		testMix = createRandomMixture(num,dim)
	# 		for mid_num in intermediate_mixture_size:
	# 			for fin_num in finalNum:
	# 				for dist in distanceMeasure:
	# 					t = timeit.default_timer()
	# 					tmp_result = theArena(testMix,dist,mid_num,fin_num)
	# 					elapsed = timeit.default_timer() - t
	# 					results_dict = {}
	# 					isd_val = testMix.ISD(tmp_result)
	# 					result = {'dim': dim, 'starting num': num, 'intermediate num': mid_num, 'final num': fin_num, 'measure': dist.__name__, 'time elapsed': elapsed, \
	# 								'ISD': isd_val,  'test mix': {'means': testMix.getMeans(),'vars': testMix.getVars(),'weights': testMix.getWeights()}, \
	# 								'result mix': {'means': tmp_result.getMeans(),'vars': tmp_result.getVars(), 'weights': tmp_result.getWeights()}}
	# 					total_results[dist.__name__][dim][num][mid_num][fin_num] = (elapsed,isd_val)
	# 					# total_results.append(result)
	# 					# print(result)
	# 					plotResults(testMix,tmp_result)
	# 					print('{} ISD: {}'.format(dist.__name__,isd_val))
	for start_num in startNum:
		total_results[start_num] = {}
		for dim in dims:
			testMix = createRandomMixture(start_num,dim)
			total_results[start_num][dim] = {}
			for dist in distanceMeasure:
				total_results[start_num][dim][dist.__name__] = {}
				for mid_num in intermediate_mixture_size:
					total_results[start_num][dim][dist.__name__][mid_num] = {}
					for fin_num in finalNum:
						# total_results[start_num][dist.__name__][dim][mid_num][fin_num] = {}
						t = timeit.default_timer()
						tmp_result = theArena(testMix,dist,mid_num,fin_num)
						elapsed = timeit.default_timer() - t
						isd_val = testMix.ISD(tmp_result)
						total_results[start_num][dim][dist.__name__][mid_num][fin_num] = \
							{'time': elapsed,'ISD': isd_val}#, \
							# 'test mix': {'means': testMix.getMeans(),'vars': \
							# 	testMix.getVars(),'weights': testMix.getWeights()}, \
							# 'result mix': {'means': tmp_result.getMeans(),'vars': \
							# 	tmp_result.getVars(), 'weights': tmp_result.getWeights()}}
						# plotResults(testMix,tmp_result)

	# print total_results
	print(total_results)

	for dist in distanceMeasure:
		for dim in dims:
			for start_num in startNum:
				for mid_num in intermediate_mixture_size:
					for fin_num in finalNum:
						print('dist: {} \t ISD: {} \t time: {}'.format(dist.__name__,\
							total_results[start_num][dim][dist.__name__][mid_num][fin_num]['ISD'],\
							total_results[start_num][dim][dist.__name__][mid_num][fin_num]['time']))

	# print('-----')
	# print('lowest ISD: {}'.format())
	

	if save == True:
		with open('test_file.pickle','wb') as handle:
			pickle.dump(total_results,handle, protocol=pickle.HIGHEST_PROTOCOL)


	#Create Test Mixture from params
	# testMix = createRandomMixture(startNum,dims); 

	# #Run tests
	# results = [];
	# # t = time.time()
	# t = timeit.default_timer()
	# tmp_result = theArena(testMix,distanceMeasure,intermediate_mixture_size,finalNum)
	# elapsed = timeit.default_timer() - t
	# results.append(tmp_result); 
	# print('Time elapsed: {:.4f} seconds'.format(elapsed)) 
	# print('Time elapsed: {:.4f} seconds'.format(elapsed1)) 

	#Save/display results
	# plotResults(testMix,tmp_result);
	# print('JSD ISD: {}'.format(testMix.ISD(results[0]))); 

	# distanceMeasure = KLD;

	# #Run tests
	# results = [];
	# results.append(theArena(testMix,distanceMeasure,intermediate_mixture_size,finalNum));  

	# #Save/display results
	# plotResults(testMix,results[0]);
	# print('KLD ISD: {}'.format(testMix.ISD(results[0]))); 

	# print('--------------------')
