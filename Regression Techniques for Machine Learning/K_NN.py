from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm

"""
 one possible means of simply computing preference rating is to do multiple runs of classificatoins:
I.E.
 tasks = [a,b,c,d,e,f]
 data = a > b
 run_classication()
 	-> resualt = [a,e,f] > [b,c,d]
 data = a > b && e > f
 run_classication()
 	-> resualt = [e] > [f,a] > [b,c,d]
 etc....

can run somethig like this to get some sort of ranking.
Will need to do some cross comparisons between groups in order to test the groupings

"""

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

def getNeighbors(given_points, test_point, k):
	distences_map = {}
	neighbors = []

	if k  > len(given_points):
		k = len(given_points )

	for task in given_points:
		distences_map[task] = E_distence(given_points[task],test_point)

	neighbors = getMax_from_map(distences_map,k)
	return neighbors

def getResponse(training_data, neighbors):
	class_vote = {}
	for i in range(0,len(neighbors)):
		response = neighbors[i]
		if response in class_vote:
			class_vote[i] += 1
		else :
			class_vote[neighbors[i]]+= 1
	winner = getMax_from_map(class_vote,1)
	return winner

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

def E_distence(x_1, x_2 ):
	diff_2 = sum(map(lambda x,y: (x-y)**2, x_1,x_2))
	return np.sqrt(diff_2)

def quadratic_kernel(distence):
	quad = 1 - ((2*np.abs(distence))/2)**2
	return max([0,quad])


if __name__ == "__main__":

	num_of_tasks = 10
	num_of_features = 3
	width = [0,2]
	tasks = create_random_tasks(num_of_tasks,num_of_features,width)
	data_map = create_data_map(tasks.keys())

	# observations:
	data_map['BD'] = 1
	# [x,y] = tasks.values()
	print getNeighbors(tasks, [0,0,0], 2)

	x = {'a' : 10 }
	x['b'] = 11
	print np.argmax(x)
