#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

import numpy as np
import matplotlib.pyplot as pl
import csv

class AHP:
	criteria = []
	alts = []
	criteria_weights_dict = {}
	cri_pw = np.array([[]])


	alt_weights = np.array([])

	rank = {}

	def __init__ (self, cri, alternatives):
		"""
			critieria and alternative must be arrays in the order they appear in the comparision cri_pw

		"""
		self.criteria = cri
		self.alts = alternatives
		self.alt_weights = np.zeros((len(alternatives),len(cri)))



	def run(self,alts_pw,threshold=None):
		""" Function Run
		Purpose: will determine the rank of alternatives based off of weights
			from pairwise comparison of criteria
		Input:
			alts_pw -- a dictionary.
				keys = criteria names
				values = dict with keys = 'alt', value 'weight'
			threshold_on: bool to examine threshold for alt_weights.
				default = 0 (unless value is inputed)

		Return: nothing

		Notes: updates member variable of rank

		"""
		# this is only set up for 2  alts right now
		if (threshold != None):
			temp = self.check_threshold(alts_pw,threshold)
			if (temp != 0):
				self.rank['Human'] = 1
				self.rank["Robot"] = 1
				self.rank[temp] = 0
				return


		w = []
		for i in range(0,len(self.criteria)):
			# normalizing the weight ghts of alt cols
			self.alt_weights[:,i] = normalize(alts_pw[self.criteria[i]].values())

			# grabbing the weight for the criteria
			w.append(self.criteria_weights_dict[self.criteria[i]])


		# calculating the rank
		w = np.array(w)
		np.resize(w,(len(self.criteria),1))
		rank_values = normalize(np.dot(self.alt_weights,w))

		alt_order = alts_pw[self.criteria[0]].keys()
		for i in range(0,len(rank_values)):
			self.rank[alt_order[i]] = rank_values[i]


	def check_threshold(self,alts_pw,thresholds):
		for cri in thresholds.keys():
			for alt in self.alts:
				if (cri in self.criteria):
					if (alts_pw[cri][alt] < thresholds[cri][0]) or (alts_pw[cri][alt] > thresholds[cri][1]):
						# means that it is out of the threshold
						print "criteria: ",cri, " Agent : ", alt, ". out of threshold ragne"
						return alt
		return 0


	def EV(self, cri_pw):
		""" Fucntion EV == Eigen Values
		Purpose: calculate the EV of an matriwx, for this criteria PW
		input:
			cri_pw --- numpy matrix of the pairwise comparision in
			order of the critiera list
		Output:
			weights -- array of the criteria weights in order of criteria list
			criteria_weights_dict -- dict of weights with cri as keys
		Note: updates member variables criteria_weights

		"""
		# normalize cri_pw columbs
		weights = []
		criteria_weights_dict = {}
		size = len(cri_pw)
		cri_pw = np.array(cri_pw)

		# normalizing the col
		for i in range(0,size):
			cri_pw[:,i] = cri_pw[:,i] / sum(cri_pw[:,i])

		for i in range (0,size):
			weights.append(sum(cri_pw[i,:]))

		# norm the weights
		weights = map(lambda a: a / sum(weights), weights)

		for i in range(0,len(weights)):
			self.criteria_weights_dict[self.criteria[i]] = weights[i]

		# self.criteria_weights = weights
		return weights, criteria_weights_dict


def normalize (array):
	"""Function: normalize
	Purpose: normalize a list
	array: array -- list that is to be normalize
	return: norm -- the normalize list
	"""
	norm = map(lambda a: a/reduce(lambda x,y: abs(x)+abs(y), array), array)
	return norm

def read_criteria(file_name):
	"""Function: read_critieria
	Purpose: read the cvs file containing the criteria pairwise relations.
	input:
		 file_name -- the name of the .csv file
	return:
		criteria: list of the criteria
		cri_pw: matrix containing the pairwise relations
		sub_criteria: list of sub criteria
		sub_cri_pw: matrix with the sub criteria pairwise relations

	"""
	criteria = []
	cri_pw = []
	sub_criteria = []
	sub_cri_pw = []
	thresholds = {}
	with open(file_name,'rb') as file:
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
			elif (row[0] in " threshold : "):
				thresholds[row[1]] = [float(row[2]), float(row[3])]

	# cast numbers into float
	for i in range(0,len(cri_pw)) :
		cri_pw[i] = map(lambda x: float(x), cri_pw[i])

	for i in range(0,len(sub_cri_pw)) :
		sub_cri_pw[i] = map(lambda x: float(x), sub_cri_pw[i])


	return criteria, cri_pw, sub_criteria, sub_cri_pw, thresholds

def read_tasks(file_name):
	"""Function: read_task
	Purpose: read csv file contianing task values for alternatives w/r to
		critiera
	input:
		file_name: name of file
	return:
		tasks: multidimensional dict: {task: {criteria: [weights]}}
		alt: list of alternatives

	"""
	tasks = {}
	criteria = []
	alt = []
	task_flag = 0
	current_task = ''
	with open(file_name,'rb') as file:
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
				tasks[current_task][row[0]] = {}
				for i in range(0,len(row[1:])):
					tasks[current_task][row[0]][alt[i]] = float(row[1+i])


			elif (row[0] in ' criteria : '):
				criteria = row[1:]
			elif (row[0] in ' alternatives : '):
				alt = row[1:]
	return tasks,alt




if __name__ == '__main__' :

	# first do AHP on sub certiera, use resualts as alt criteria pw weights,

	criteria, cri_pw, sub_criteria, sub_cri_pw, thresholds_dict = read_criteria('criteria1.csv')
	# print thresholds_dict
	tasks,alt = read_tasks('alt_run.csv')
			# print 'values', alts_pw[self.criteria[i]].values()
	# print tasks

	# determining the weights for the critiera and sub critieria
	# sub criteria must be determine first
	sub_ahp = AHP(sub_criteria,alt)
	sub_cri_w = sub_ahp.EV(sub_cri_pw)
	# print sub_ahp.criteria_weights_dict


	ahp = AHP(criteria,alt)
	cri_pw = ahp.EV(cri_pw)
	# print ahp.criteria_weights_dict

	# looping over each task to run ahp with
	for key,value in tasks.items() :

		# extracting the pairwise alt for the sub criteria
		sub_alt_pw = {}
		for sub in sub_criteria:

			sub_alt_pw[sub] = tasks[key ][sub]

		# running ahp for the sub criteria

		sub_ahp.run(sub_alt_pw,thresholds_dict)

		# manually entering the alt_weights for the criteria with sub_criteria
		tasks[key]['Effectiveness'] = {'Robot' : sub_ahp.rank['Robot'], 'Human': sub_ahp.rank['Human']}

		alt_pw = {}
		for cri in criteria:
			alt_pw[cri] = tasks[key][cri]

		ahp.run(alt_pw,thresholds_dict)
		# print ahp.alt_weights
		print key,' ', ahp.rank
