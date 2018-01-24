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
import datetime
import timeit
import pickle
import sqlite3
import json
import sys
from testFunctions import *
from data_handle import *


#distance functions
def euclid(mix_i,mix_j):
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

	del new_mix_i
	del new_mix_j
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

	del new_mix_i
	del new_mix_j
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

	del new_mix_i
	del new_mix_j
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

	del new_mix_i
	del new_mix_j
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

	del new_mix_i
	del new_mix_j
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
	posClusters = cluster(posMix,kmeansFunc,k=numClusters); 
	#negClusters = cluster(negMix,kmeansFunc,k=numClusters);

	#condense
	posCon = conComb(posClusters,finalNum); 
	#negCon = conComb(negClusters,finalNum); 

	#recombine
	newMix = GM(); 
	posCon.scalerMultiply(posNorm);
	newMix.addGM(posCon); 
	#negCon.scalerMultiply(negNorm)
	#newMix.addGM(negCon); 

	del startMix
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


def createRandomMixture(size = 200,dims = 2,df=8):
	testMix = GM(); 

	for i in range(0,size):
		testMix.addG(sampleWishart(dims,df)); 
	testMix.normalizeWeights();
	return testMix;


def sampleWishart(dims = 2,df = 8):
	sigPrior = np.diag(np.ones(dims))*1;
	#cholesky = np.linalg.cholesky(sigPrior); 
	X = np.dot(sigPrior,np.random.normal(size=(dims,df))); 
	sigma = np.linalg.inv(np.dot(X,X.T)); 

	weight = np.sqrt(np.random.random());  

	# muPrior = np.random.random(size=dims)*10;
	# fudgeFactor = 10; #higher causes more spread
	# mu = np.random.multivariate_normal(muPrior,fudgeFactor*sigma); 

	lowInit = [0]*dims; 
	highInit = [10]*dims;
	mu = []; 
	for i in range(0,dims):
		mu.append(np.random.random()*(highInit[i]-lowInit[i]) + lowInit[i]); 

	return Gaussian(mu,sigma,weight); 

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

	test_num = sys.argv[1]
	try:
		test_num = int(test_num)
	except ValueError as e:
		print('Command line argument must be integer')
		test_num = 0

	while (test_num > 6 and test_num != 99) or (test_num < 1):
		print('Command line argument must be a test case number between 1 and 6.')
		print('possible tests: test_num = 1: dim = 1, startNum = 100, 1000\n \
			test_num = 2: 	dim = 1, startNum = 400, 700\n \
			test_num = 3:	dim = 2, startNum = 100, 1000\n \
			test_num = 4:	dim = 2, startNum = 400, 700\n \
			test_num = 5:	dim = 4, startNum = 100, 1000\n \
			test_num = 6:	dim = 4, startNum = 400, 700\n \
			test_num = 99:	all paramters')
		test_num = raw_input('Enter new test case number between one and six: ')
		test_num = int(test_num)
	# possible tests: test_num = 1: dim = 1, startNum = 100, 1000
	#		test_num = 2: 	dim = 1, startNum = 400, 700 
	#		test_num = 3:	dim = 2, startNum = 100, 1000 
	#		test_num = 4:	dim = 2, startNum = 400, 700 
	#		test_num = 5:	dim = 4, startNum = 100, 1000 
	#		test_num = 6:	dim = 4, startNum = 400, 700 

	# split up parameters for multiple processes
	if test_num == 1:
		dims = [1]
		startNum = [100,1000]
	elif test_num == 2:
		dims = [1]
		startNum = [400,700]
	elif test_num == 3:
		dims = [2]
		startNum = [100,1000]
	elif test_num == 4:
		dims = [2]
		startNum [400,700]
	elif test_num == 5:
		dims = [4]
		startNum = [100,1000]
	elif test_num == 6:
		dims = [4]
		startNum = [400,700]
	elif test_num == 99:
		dims = [1,2,4]
		startNum = [100,400,700,1000]

	# dims = [4]; 
	# starting number of mixands
	# startNum = [400,700] 

	# common for all experiments
	distanceMeasure = [symKLD,JSD,euclid,EMD,bhatt]
	intermediate_mixture_size = [4,10,15]
	finalNum = [5,10,25]
	repeat = 10

	dim_string = ''
	for d in dims:
		dim_string = dim_string + str(d)
	start_num_string = ''
	for s in startNum:
		start_num_string = start_num_string + str(s)
	filename = 'dim{}_{}_db_{}.sqlite'.format(dim_string,start_num_string,datetime.date.today())
	print('\nCreating new database {} with table \'{}\'\n'.format(filename,'data'))
	sys.exit(0)
	# connect to sqlite database
	c,conn = connect(filename)
	# create new table
	create_table(c,conn,'data')

	loop_count = 1

	try:
		for start_num in startNum:
			for dim in dims:
				for i in range(0,repeat):
					testMix = createRandomMixture(start_num,dim)

					# create serialized str from dictionary with means, variances, weights
					testMix_dict = {'means': testMix.getMeans(),'vars': \
									testMix.getVars(),'weights': testMix.getWeights()}
					testMix_ser = json.dumps(testMix_dict)

					
					for mid_num in intermediate_mixture_size:
						for fin_num in finalNum:
							if mid_num*fin_num > start_num:
								continue
							testMix_runnalls = deepcopy(testMix)
							t_r = timeit.default_timer()
							testMix_runnalls.condense(mid_num*fin_num)
							run_elapsed = timeit.default_timer() - t_r
							runnalls_isd = testMix.ISD(testMix_runnalls)
							runnalls_dict = {'means': testMix_runnalls.getMeans(),'vars': \
								testMix_runnalls.getVars(),'weights': testMix_runnalls.getWeights()}
							runnalls_ser = json.dumps(runnalls_dict)
							for dist in distanceMeasure:
								
								# run experiment with parameters
								t = timeit.default_timer()
								tmp_result = theArena(testMix,dist,mid_num,fin_num)
								elapsed = timeit.default_timer() - t
								isd_val = testMix.ISD(tmp_result)

								# create serialized str from dictionary with means, variances, weights
								tmp_result_dict = {'means': tmp_result.getMeans(),'vars': \
										tmp_result.getVars(), 'weights': tmp_result.getWeights()}
								tmp_result_ser = json.dumps(tmp_result_dict)

								# print experiment count for something to look at while running
								print('-----')
								print('Loop count: {}'.format(loop_count))
								loop_count += 1

								# add results of experiment to database
								add_result(c,dim,start_num,dist.__name__,mid_num,fin_num,
											i+1,isd_val,elapsed,testMix_ser,tmp_result_ser,
											runnalls_isd,run_elapsed,runnalls_ser)

					conn.commit() # commit additions to database after every distance function
	except KeyboardInterrupt:
		print('\nCommiting changes to database and exiting...')
		close(c,conn)
		sys.exit(0)

	close(c,conn)
