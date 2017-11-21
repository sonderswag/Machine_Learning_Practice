#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

import numpy as np
import matplotlib.pyplot as pl

class AHP:
	criteria = []
	alts = []

	cri_pw = np.array([[]])
	alts_pw = {}

	criteria_weights = {}
	alt_weights = np.array([])

	rank = {}

	def __init__ (self, cri, alternatives, criteria_pairwise, alt_pairwise):
		"""
			critieria and alternative must be arrays in the order they appear in the comparision matrix

		"""
		self.criteria = cri
		self.alts = alternatives
		self.cri_pw = np.copy(criteria_pairwise)
		self.alts_pw  = alt_pairwise
		self.alt_weights = np.zeros((len(alternatives),len(cri)))


	def run(self):

		w = self.EV(self.cri_pw)

		for i in range(0,len(w)):
			self.criteria_weights[self.criteria[i]] = w[i]


		for i in range(0,len(self.criteria)):
			if (self.alts_pw[self.criteria[i]].shape != (len(self.alts),len(self.alts))):
				self.alt_weights[:,i] = self.alts_pw[self.criteria[i]]
			else :
				self.alt_weights[:,i] = self.EV(self.alts_pw[self.criteria[i]])

		rank_values = np.dot(self.alt_weights,np.array(w))

		for i in range(0,len(rank_values)):
			self.rank[self.alts[i]] = rank_values[i]



	def EV(self, matrix):
		# normalize matrix columbs
		weights = []
		size = len(matrix)

		for i in range(0,size):
			matrix[:,i] = matrix[:,i] / sum(matrix[:,i])

		for i in range (0,size):
			weights.append(sum(matrix[i,:]))

		weights = map(lambda a: a / sum(weights), weights)

		return weights




if __name__ == '__main__' :

	cri = ['c1', 'c2', 'c3']
	alts = ['a0','a1','a2','a3']
	cri_pairwise = np.array([ [1.0,.5,3.0], [2.0,1.0,4.0], [(1/3.0),.25,1.0]])
	alt_pairwise = {'c1' : np.array([[1.0,.25,4.0,(1/6.0)], [4.0,1.0,4.0,.25], [.25,.25,1.0,.2], [6.0,4.0,5.0,1.0]]),
		'c2' : np.array([[1.0,2.0,5.0,1.0],[.5,1.0,3.0,2.0], [.2,(1/3.0),1.0,.25], [1.0,.5,4.0,1.0]]),
		'c3' : np.array([.3010,.2390,.2120,.2480])}

	ahp = AHP(cri,alts,cri_pairwise,alt_pairwise)
	ahp.run()
	print ahp.criteria_weights
	print ahp.alt_weights
	print ahp.rank
