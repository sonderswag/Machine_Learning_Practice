#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

import numpy as np
import matplotlib.pyplot as pl
import csv

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
		# print w
		for i in range(0,len(w)):
			self.criteria_weights[self.criteria[i]] = w[i]


		for i in range(0,len(self.criteria)):
			if (self.alts_pw[self.criteria[i]].shape != (len(self.alts),len(self.alts))):
				self.alt_weights[:,i] = normalize(self.alts_pw[self.criteria[i]])
			else :
				self.alt_weights[:,i] = self.EV(self.alts_pw[self.criteria[i]])

		# print self.alt_weights
		rank_values = normalize(np.dot(self.alt_weights,np.array(w)))

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


def normalize (array):
	norm = map(lambda a: a/reduce(lambda x,y: abs(x)+abs(y), array), array)
	return norm

def read_criteria(fileName):
	criteria = []
	cri_pw = []
	sub_criteria = []
	sub_cri_pw = []
	with open(fileName,'rb') as file:
		reader = csv.reader(file)
		for row in reader:
			# print row
			if ("#" in row[0]):
				pass
			elif (row[0] in ' criteria : '):
				criteria = row[1:]
			elif (row[0] in ' sub criteria : '):
				sub_criteria = row[1:]
			elif (row[0] in criteria):
				cri_pw.append(row[1:])
			elif (row[0] in sub_criteria):
				sub_cri_pw.append(row[1:])

	for i in range(0,len(cri_pw)) :
		cri_pw[i] = map(lambda x: float(x), cri_pw[i])

	for i in range(0,len(sub_cri_pw)) :
		sub_cri_pw[i] = map(lambda x: float(x), sub_cri_pw[i])


	return criteria, cri_pw, sub_criteria, sub_cri_pw

def read_tasks(fileName):
	tasks = {}
	criteria = []
	alt = []
	task_flag = 0
	current_task = ''
	with open(fileName,'rb') as file:
		reader = csv.reader(file)
		for row in reader:
			# print row
			if ((len(row) == 0) or ("#" in row[0]) ):
				task_flag = 0
				pass

			elif (('Task' in row[0]) and (task_flag == 0)):
				task_flag = 1
				tasks[row[0]] = {}
				current_task = row[0]

			elif ((task_flag == 1) and not(row[0] in criteria)):
				task_flag = 0

			elif ((task_flag == 1) and (row[0] in criteria)):
				tasks[current_task][row[0]] = np.array(map(lambda x: float(x), row[1:]))

			elif (row[0] in ' criteria : '):
				criteria = row[1:]
			elif (row[0] in ' alternatives : '):
				alt = row[1:]
	return tasks,alt




if __name__ == '__main__' :

	# first do AHP on sub certiera, use resualts as alt criteria pw weights,

	criteria, cri_pw, sub_criteria, sub_cri_pw = read_criteria('test_run.csv')
	print criteria
	# print cri_pw
	# print sub_criteria
	# print sub_cri_pw
	tasks,alt = read_tasks('alt_run.csv')
	# print tasks
	# print 'key', tasks.keys()

	for key,value in tasks.items() :

		sub_alt_pw = {}
		for sub in sub_criteria:
			sub_alt_pw[sub] = tasks[key][sub]

		sub_ahp = AHP(sub_criteria,alt,sub_cri_pw,sub_alt_pw)
		sub_ahp.run()
		# print sub_ahp.rank
		tasks[key]['Effectiveness'] = np.array([sub_ahp.rank['Human'],sub_ahp.rank['Robot']])

		alt_pw = {}
		for cri in criteria:
			alt_pw[cri] = tasks[key][cri]


		ahp = AHP(criteria,alt,cri_pw,alt_pw)
		ahp.run()
		print key,' ', ahp.rank
		print ahp.alt_weights, " ", ahp.criteria_weights
		print ahp.criteria
