from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm 
from gapp import gp
from scipy.special import gamma

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)

    poolIndices = list(range(n))
    
    if r > n:
        return
    indices = list(range(r))

     
    yield [list(pool[i] for i in poolIndices+indices if (poolIndices+indices).count(i)==1), list(pool[i] for i in indices)]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1

        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield [list(pool[i] for i in poolIndices+indices if (poolIndices+indices).count(i)==1), list(pool[i] for i in indices)]

def create_random_tasks(num_tasks, num_features,width): 
	"""	Create up to 26 different tasks with randomized feautre space 

		Parameters: 
			num_tasks: The number of tasks to be created.

			num_features: The size of the feature sapce. 

			width: the range of values for the feature space. (uniform in size)

		Returns: 
			tasks: A python dictionary of {task_name: [feature_space]}
	"""
	name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	tasks = {'A' : []}

	for i in range(0,num_tasks): 
		features = np.array([])
		for j in range(0,num_features): 
			features = np.append(features, np.random.uniform(width[0],width[1]))
		tasks[name[i]] = features
	return tasks 

def create_data_map(tasks_names):
	"""Create a python dictionary of all pairwise combinations of tasks 
			objects. It keeps track of what comparisions have been 
			observed.

		Arguments: s
			tasks: the list of tasks names

		Returns: 
			data: A dictionary containing all combinations of pariwise 
				comparison as the key. The value is intialized to -1. 
				i.e. {'AB': -1}

	"""
	data = {}
	for i in combinations(tasks_names,2):
		temp = "%s%s" %(i[1][0],i[1][1])
		data.update({temp:-1})
	return data 

def E_distence(x_0,x_1): 
	summation = sum(map(lambda a,b: (a-b)**2, x_0,x_1))
	return np.exp(-summation/2)

def getNeighbors(given_points, test_point, k): 
	distences_map = {} 
	neighbors = []

	if k  > len(given_points): 
		k = len(given_points )

	for task in given_points: 
		distences_map[task] = E_distence(given_points[task],test_point)

	neighbors = getMax_from_map(distences_map,k)
	return neighbors

def getMax_from_map(given_map,number):
	ans = []
	values = given_map.values() 
	keys = given_map.keys() 
	for i in range(0,number): 
		index = np.argmin(values)

		if (values[index] == 0): 
			k = k-1
		else: 
			ans.append(keys[index]) 

		values.remove(values[index])
		keys.remove(keys[index])
	return ans 

def quadratic_kernel(distence): 
	quad = 1 - ((2*np.abs(distence))/2)**2 
	return max([0,quad])

def data_scoring(tasks_map,data_map,k): 
	p_weight = 1 
	n_weight = .5

	tasks = tasks_map.keys()
	task_values = dict(map(lambda x:[x,0],tasks))
		
	for key in data_map: 
		if data_map[key] !=-1 : 
			if data_map[key] == 1: 
				task_A = key[0]; 
				task_B = key[1]; 
			else : 
				task_A = key[1]; 
				task_B = key[0]; 

			task_A_value = tasks_map[task_A]
			task_B_value = tasks_map[task_B]

			pos_neighbors = getNeighbors(tasks_map,task_A_value,k)
			neg_neighbors = getNeighbors(tasks_map,task_B_value,k)
			diff_between = E_distence(task_A_value,task_B_value)
			
			if task_B in pos_neighbors: #might be the case that of this happens the other must also happen 
				pos_neighbors.remove(task_B)
			
			if task_A in neg_neighbors: 
				neg_neighbors.remove(task_A)


			print pos_neighbors, task_A
			print neg_neighbors, task_B
			for i in range(0,len(pos_neighbors)):
				p_n_value = tasks_map[pos_neighbors[i]]
				n_n_value = tasks_map[neg_neighbors[i]]
				task_values[pos_neighbors[i]] += p_weight*diff_between*quadratic_kernel(E_distence(task_A_value,p_n_value))
				task_values[neg_neighbors[i]] -= n_weight*diff_between*quadratic_kernel(E_distence(task_B_value,n_n_value))

			task_values[task_A] += p_weight*diff_between
			task_values[task_B] -= n_weight*diff_between

	return task_values

def determine_preference_ordering(tasks_keys,tasks_score): 
	tasks_score = list(tasks_score)
	temp = tasks_score
	order = []
	for i in range(0,len(temp)): 
		index = np.argmax(temp)
		location = tasks_score.index(temp[index])
		order.append([tasks_keys[location],tasks_score[location]])
		temp[index] = -100
	return order 
		
def invgamma(theta, a, b): # is the prior distribution of the random varibles before the observations 
     s = theta[0]
     p = b**a/gamma(a) * s**(-1 - a) * np.exp(-b/s)
     return p

def mean(data_list): 
	N = len(data_list) - 1 
	summation = sum(data_list)
	return summation/N 

def varience(data_list): 
	mean_v = mean(data_list)
	N = len(data_list)
	summation = 0
	for i in data_list:
		summation += (i-mean_v)**2 
	return summation/N 

def normalize(data_list): 
	norm_data = []
	mean_v = mean(data_list)
	var = varience(data_list)
	norm_data = map(lambda x: (x-mean_v)/var, data_list)
	return norm_data

def normalize_data(data_map): 
	mean_v = []
	var = []
	norm_data_map = data_map
	data_v = np.array(data_map.values())
	for i in range(0,len(data_v[0,:])): 
		mean_v.append(mean(data_v[:,i]))
		var.append(varience(data_v[:,i]))

	for key in data_map: 
		for i in range(0,len(data_map[key])):
			norm_data_map[key][i] =  (data_map[key][i] - mean_v[i])/var[i]

	return norm_data_map


if __name__ == "__main__": 
	width = [-5,5]
	num_tasks = 8
	num_features = 5
	# tasks = create_random_tasks(num_tasks,num_features,width)
	tasks = {'A': [-0.40608332,  2.18405865,  1.24921397,  1.78371252,  4.5463986 ], 
	'C': [ 0.74243265, -1.11944769, -2.44058784, -1.75541207,  1.68052863], 
	'B': [-1.69389498, -2.5697072 , -2.47073355,  4.99015238,  2.0569257 ], 
	'E': [-0.80421139, -0.49990645,  2.27359508, -0.04031112,  4.49856496], 
	'D': [ 0.92989596, -3.39381534, -2.81862325,  0.89793112, -2.26533809], 
	'G': [ 3.88776375,  4.27727411, -4.54385268, -0.67578899, -2.07803292], 
	'F': [ 0.37243284,  2.37291976, -3.48032177,  4.75283332, -1.59625017], 
	'H': [-2.80235068,  3.23611657,  1.28498843, -3.58506868, -1.05289279]}

	norm_tasks =normalize_data(tasks)
	data_map = create_data_map(tasks.keys())
	data_map['DA'] = 1 
	data_map['AB'] = 1 
	# data_map['AF'] = 1 
	# data_map['AE'] = 1 
	data_map['CA'] = 1 
	# data_map['CD'] = 1 
	score_map = data_scoring(norm_tasks,data_map,2)
	print score_map
	X = norm_tasks.values()
	Y = score_map.values() 
	Sigma = map(lambda x:.1,Y)

	eval_points = []
	for key in tasks: 
		eval_points.append(tasks[key])

	
	g = gp.GaussianProcess(X, Y, Sigma, theta=[1, 1], Xstar = eval_points, prior = invgamma, priorargs=(0.2, 1), grad = 'False')

	(Xstar,fmean,fstd,theta1) = g.gp(unpack='True')
	order = determine_preference_ordering(tasks.keys(),fmean)
	print order
	print np.array(order)[:,0]
	# print Xstar









