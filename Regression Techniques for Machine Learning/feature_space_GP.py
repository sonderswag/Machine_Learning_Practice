from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm 


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


def pairwise_compare(task_a, task_b, data): 
	"""
		To ask for user input of a pairwise comparison 

	"""
	print "Which of the two tasks do you favor more?"
	print "Enter 1 for %s and 0 for %s" %(task_a,task_b)
	ans = raw_input()
	return ans
	
	

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

def data_map_to_data_points(data_map,tasks,num_features): 
	""" Turn the dictionary of what comparisions have been preformed (data_map)
			and compute the data set for each feature space. Postive weights 
			are assigned to preferred tasks and negative to the other. 

		Pramaters: 
			data_map: Python dictionary of the comparisions that have been
				 preformed.  

			tasks: Python dictionary of the tasks and their feature space 
				values 

			num_features: easy access to the number of features.

		returns: 
			data_points: Numpy list of the following form [task,feature,x/y]
			

	"""
	data_points = np.array([])
	i = 0
	pos_weight = .5 
	neg_weight = .5 
	for key in data_map: 
		if data_map[key] != -1: 
			i += 1 
			task_0 = key[0]
			task_1 = key[1]
			if data_map[key] == 1 : 
				# comparisons between further apart tasks will have greater weight. 
				temp = map(lambda x,y: [x,pos_weight*abs(x-y)], tasks[task_0],tasks[task_1])
				data_points = np.append(data_points, np.array(temp).reshape(1,num_features,2))
				temp = map(lambda x,y: [x,-neg_weight*abs(x-y)], tasks[task_1],tasks[task_0])
				data_points = np.append(data_points, np.array(temp).reshape(1,num_features,2))
			else : 
				temp = map(lambda x,y: [x,pos_weight*abs(x-y)], tasks[task_1],tasks[task_0])
				data_points = np.append(data_points, np.array(temp).reshape(1,num_features,2))
				temp = map(lambda x,y: [x,-neg_weight*abs(x-y)], tasks[task_0],tasks[task_1])
				data_points = np.append(data_points, np.array(temp).reshape(1,num_features,2))
			

	data_points = data_points.reshape(i*2,num_features,2)

	return data_points

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1

    # print a.shape , b.shape, b.T.shape

    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

def gaussian_process(data,noise, Xtest):
	"""	data is a tuple array of a single feature 

	""" 
	X = data[:,0].reshape(-1,1)
	Y = data[:,1]
	s = noise 
	N = len(Y)
	#creating the kernel of the know data that we have 
	K = kernel(X, X) 
	L = np.linalg.cholesky(K + s*np.eye(N))


	# compute the mean 
	Lk = np.linalg.solve(L, kernel(X, Xtest)) 
	mu = np.dot(Lk.T, np.linalg.solve(L, Y))

	K_ = kernel(Xtest, Xtest)
	s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
	var = np.sqrt(s2)

	return mu,var

def fit_features(data,tasks,num_features,width,resolution): 
	#create test points for plotting the approximation 
	Xtest = np.linspace(width[0],width[-1],resolution).reshape(-1,1)
	mu = np.array([])
	var = np.array([])

	for i in range(0,num_features): 
		mu_temp,var_temp = gaussian_process(data[:,i,:],.0005,Xtest)

		mu = np.append(mu,mu_temp).reshape(i+1,len(Xtest))
		var = np.append(var,var_temp).reshape(i+1,len(Xtest))

	return mu, var

def plot_features(data,tasks,width,num_features,mu,var,resolution,aquistion): 
	Xtest = np.linspace(width[0],width[-1],resolution).reshape(-1,1)
	sub_plot_height = int(num_features/2)
	sub_plot_width = num_features - sub_plot_height
	fig, axs = pl.subplots(sub_plot_height,sub_plot_width)
	plot_row = 0
	plot_col = 0 

	for i in range(0,len(mu[:,0])): 
		if plot_col == sub_plot_width: 
				plot_row += 1
				plot_col = 0 
		plot_feature(mu[i,:],var[i,:],width,data[:,i,:],Xtest,axs[plot_row][plot_col])
		plot_tasks(tasks,axs[plot_row][plot_col],plot_row*sub_plot_width+plot_col)

		if aquistion: 
				EI = expected_improvement(mu[i,:],var[i,:],.0005,data[:,i,1])
				plot_aquistion_function(EI,Xtest,axs[plot_row][plot_col])

		plot_col += 1 


def plot_feature(mu,var,width,data,Xtest,fig):
	# print data
	fig.plot(data[:,0],data[:,1],'r+',ms=20)
	fig.fill_between(Xtest.flat, mu-3*var, mu+3*var, color="#dddddd")
	fig.plot(Xtest, mu, 'r--', lw=2)
	fig.axis([width[0],width[1],-10,10])

def plot_tasks(tasks,fig,feature): 
	for key in tasks.keys(): 
		fig.plot(tasks[key][feature],0,'gx',ms=10)
		fig.annotate(key, (tasks[key][feature],0,))

def plot_aquistion_function(AQ,Xtest,fig): 
	fig.plot(Xtest,AQ - 3, '--m')

def probability_of_improvement(mu,var,noise,observed_data): 
	max_observed = np.max(observed_data)
	z_score = (mu - max_observed - noise) / var
	PI = norm.cdf(z_score)
	return PI 

def expected_improvement(mu,var,noise,observed_data):
	max_observed = np.max(observed_data)
	z_score = (mu - max_observed - noise) / var
	EI = (mu - max_observed - noise)*norm.cdf(z_score)+noise*norm.pdf(z_score)
	return EI

def task_score(tasks,data,num_features): 
	scores = []
	for key in tasks.keys():
		task_score_temp = 0 
		for i in range(0,num_features):
			mean,var = gaussian_process(data[:,i,:],.0005,np.array([tasks[key][i]]).reshape(1,1))
			task_score_temp += mean 
		scores.append([key,task_score_temp])
	# print scores 
	return dict(scores)

def determine_preference_ordering(scores): 
	value = list(scores.values())
	keys = list(scores.keys())
	order = []
	for i in range(0,len(keys)): 
		index = np.argmax(value)
		order.append(keys[index])
		value.remove(value[index])
		keys.remove(keys[index])

	return order

			 


if __name__ == '__main__' : 
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
	# print tasks 
	data_map = create_data_map(tasks.keys())
	data_map['AB'] = 1 
	data_map['BC'] = 1 
	data_map['FH'] = 1 
	data_map['HA'] = 1 
	data = data_map_to_data_points(data_map,tasks,5)

	# print data
	# print 
	# print data[:,0,:]



	scores = task_score(tasks,data,num_features)
	print scores
	print determine_preference_ordering(scores)

	mu,var = fit_features(data,tasks,num_features,width,50)
	plot_features(data,tasks,width,num_features,mu,var,50,True) 
	pl.show()



