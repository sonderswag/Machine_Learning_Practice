#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

import random
import numpy as num

numAgents = 2
Tasks = ['A','B','C','D']
States = []
temp = []
# def createStateSpace(Agents, tasks):
# 	for i in reversed(range(tasks)-1):
# 		for left in combinations(tasks, i):


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

def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def makeStates(numAgents, tasksList):
	States = []
	for i in range(len(tasksList)):
		print i
		States.append([])
		for remaining in combinations(tasksList,i+1):
			# print "remaining =", remaining,
			if (i+1 > numAgents):
				for j in range((i+1)-numAgents):
					remaining[0].append('I')
			for active in combinations(remaining[0],numAgents):
				for activeAssigned in permutations(active[1 ]):

					States[i].append([activeAssigned,remaining[1]])
	return States


# for remaining in combinations(tasksList,3):
# 	if (i+1 > numAgents):
# 		for j in range(numAgents - i - 1):
# 			remaining[0].append('I')
# 	for active in combinations(remaining[0],numAgents):
# 		States[i].append([active[1],remaining[1]])
# print States

States = makeStates(numAgents, Tasks)
for i in States:
	print "----",i
