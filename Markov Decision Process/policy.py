#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

from grid_world import *
import numpy as _ny
import random

class Policy:
	valueIterationEepsilon = 0.1
	maxNumberOfIterations = 100 #for example the maps that have no exits
	_pe_maxk = 50 #for policy evaluation, max number of iteration

	world = None

	numOfIterations = 0
	utilities = None #memorized as the world grid [y][x]
	policy = None #created
	row = 0
	col = 0

	def __init__(self, world):
		self.world = world
		self.row,self.col = self.world.getsize()
		self.resetResults()

	def __createEmptyUtilityArray(self):
		return [ [ 0 for _ in range(self.col) ] for _ in range(self.row) ]

	def __createEmptyPolicyArray(self):
		code = {0 : "    ",
			1 : ' <> ',
			3 : '||||',
			2 : ' XX '}
		ans = []
		for y in range(self.row):
			ans.append([])
			for x in range(self.col):
				ans[y].append(code[self.world.cellTypeAt(y,x)])
		return ans


	def resetResults(self):
		self.numOfIterations = 0
		self.utilities = self.__createEmptyUtilityArray()

# -------------------VALUE ITERATION--------------------------------------------------
	def ValueIteration(self):
		numbOfIterations = 0
		row,col = self.world.getsize()
		# print row,col
		# print range(col)
		# print self.utilities

		iterate = True
		while(iterate):
			numbOfIterations += 1

			newUtil = self.__createEmptyUtilityArray() #creating 0 utility vector to hold the new values

			MaxNorm = 0 # max norm to tell when an iteration is over

			#for loop to update all of the utilities in all of the states
			for y in range(row):
				for x in range(col):
					# print "y =", y
					position = [y,x] # this is equalivalent to the current state

					#need to find the max value for a given action in a given state
					maxCellUtil = self.calMaxCellUtil(position)
					# print maxCellUtil,
					newUtil[y][x] = self.world.rewardAtState(y,x) + self.world.discFactor * maxCellUtil
					# print newUtil[x][y], " ", self.utilities[x][y]

					diff = abs(newUtil[y][x] - self.utilities[y][x])
					# print diff
					if  diff > MaxNorm:
						MaxNorm = diff
						# print MaxNorm

			self.utilities = newUtil # updating utilities
			print
			print self.utilities
			if (MaxNorm < ((self.valueIterationEepsilon*(1-self.world.discFactor)) / self.world.discFactor)):
				iterate = False

			if numbOfIterations > self.maxNumberOfIterations:
				iterate = False
				print "Reached max number of iterations", str(self.maxNumberOfIterations)

		return

	#function to calculate max utility of a cell
	def calMaxCellUtil(self,position):
		maxValue = 0
		for action in self.world.actionSet:
			sumUtilForAction = 0
			nextStateProbabilities = self.world.possiblePositionsFromAction(position,action)

			for dirr,state,p in nextStateProbabilities:
				# print position, action, p, state

				sumUtilForAction += p * self.utilities[state[0]][state[1]]
			if (sumUtilForAction >= maxValue):
				maxValue = sumUtilForAction

		# print position,maxValue
		return maxValue

	#function to print out the values on the grid world
	def printValues(self):

		line = "  |"
		row = ""

		for i in range(len(self.utilities[1])):
			line += "  " + str(i) + " |"
			row += '-----'
		row += "---"
		print line


		for i in range(len(self.utilities[:])):
			print row
			line = str(i) + " "
			for value in self.utilities[i]:
				iteam = '{:.4}'.format(str(value))
				if len(iteam) == 4:
					line += "|" + iteam
				elif len(iteam) == 3:
					line += "| " + iteam
				elif len(iteam) == 2:
					line += "| " + iteam + " "
				elif len(iteam) == 1:
					line += "|  " +  iteam + " "
			print line + '|'
# --------------- Policy Iteration  --------------------------------------------

	def creatRandomPolicy(self):
		ans = []
		for y in range(self.row):
			ans.append([])
			for x in range(self.col):
				state_type = self.world.cellTypeAt(y,x)
				if state_type != 0 :
					ans[y].append(state_type)
				else:
					ans[y].append(self.world.randomAction())
		return ans

	def stateEvaluationUnderAction(self,position,action):
		value = 0
		for dirr,newState,prob in self.world.possiblePositionsFromAction(position,action):
			value += prob*self.utilities[newState[0]][newState[1]]
		value *= self.world.discFactor
		return value


	def policyEvaluation(self):

		delta = 0
		numberOfIterations = 0
		iterate = True
		while(iterate):
			numberOfIterations += 1
			newUt = self.__createEmptyUtilityArray()

			for y in range(self.row):
				for x in range(self.col):
					newUt[y][x] = self.world.rewardAtState(y,x)
					newUt[y][x] += self.stateEvaluationUnderAction([y,x],self.policy[y][x])
					delta = max(delta,abs(self.utilities[y][x] - newUt[y][x]))

					self.utilities[y][x] = newUt[y][x] # updating utilities (may need to update while looking at wach state)

			if (delta < ((1-self.world.discFactor) / self.world.discFactor)):
				iterate = False
			if (numberOfIterations > self._pe_maxk):
				iterate = False
				print "reach max number of iterations for policy evaluation "

	def policyIteration(self):
		isStable = False
		numOfIterations = 0
		# create a random starting policy
		self.policy = self.creatRandomPolicy()
		self.utilities = self.__createEmptyUtilityArray()
		# main loop
		while (not isStable):
			#flag to tell when the policy has stablized
			unchanged = True
			numOfIterations += 1
			# print "------------------------------ ", numOfIterations, " ------------------"
			bestMove = []

			#Policy Evaluation:
			self.policyEvaluation()
			# print self.utilities
			#iterate over states and determine if the policy is optimal
			for y in range(self.row):
				for x in range(self.col):
					state = [y,x]
					# getting the best move from the utility
					bestMove = self.getStatePolicyFromUtility(state)
					# print "         state = ", state
					# print "bestMove", bestMove
					if ((self.policy[y][x] != bestMove[0]) and (bestMove[1] > self.utilities[y][x])):
						# print "policy change", state
						unchanged = False
						self.policy[y][x] = bestMove[0]

			# print
			# self.printPolicy()
			if (unchanged):
				isStable = True
				print "number of iterations for policy evaluation =", numOfIterations
			elif (numOfIterations > 10):
				isStable = True
				print "reach max number of iterations for policy iteration"




# --------------- Generating a policy -----------------------------------------

	# finds the optimal policy for a given state
	# returns the optimal action and it's utility maxArg = ['move',utility]
	def getStatePolicyFromUtility(self,state):
		maxArg = ['3',-10]
		# checking if cell type is void
		if (self.world.cellTypeAt(state[0],state[1]) != 0):
			return [self.world.cellTypeAt(state[0],state[1]),self.world.rewardAtState(state[0],state[1])]

		# finding the action that gives the best policy
		for action in self.world.actionSet:
			possibleMoves = self.world.possiblePositionsFromAction(state,action)
			sumFromMove = 0
			for _,move,p in possibleMoves:
				# print "util", self.utilities[move[0]][move[1]]
				sumFromMove += p*self.utilities[move[0]][move[1]]
				# print sumFromMove
			if sumFromMove > maxArg[1]:
				maxArg = [action,sumFromMove]
		# print "max arg = ", maxArg
		return maxArg

	def getPolicyFromUtility(self):
		row,col = self.world.getsize()
		self.policy = self.__createEmptyPolicyArray()
		move = ""
		for y in range(row):
			for x in range(col):
				move = self.getStatePolicyFromUtility([y,x])
				self.policy[y][x] = move

	def printPolicy(self):
		code = {'0' : "    ",
			'1' : ' <> ',
			'3' : '||||',
			'2' : ' XX ',
			"N": "  ^ ",
			"S": "  v ",
			"W": "  < ",
			"E": "  > "}
		line = "  |"
		row = ""

		for i in range(len(self.utilities[1])):
			line += "  " + str(i) + " |"
			row += '-----'
		row += "---"
		print line


		for i in range(len(self.policy[:])):
			print row
			line = str(i) + " "
			for value in self.policy[i]:
				line += "|" +  code[str(value)]
			print line + '|'








# ----------------------TESTING-------------------------------------------------
if __name__ == "__main__":
	# -------------------------------- setting up the world -----------------------------------------
	# This is where the user can create the world that the MDP will solve
	w = GridWorld([[GridWorld.CELL_VOID, GridWorld.CELL_VOID, GridWorld.CELL_VOID, GridWorld.CELL_EXIT],
	 			   [GridWorld.CELL_VOID, GridWorld.CELL_WALL, GridWorld.CELL_VOID, GridWorld.CELL_PIT],
	 			   [GridWorld.CELL_VOID, GridWorld.CELL_VOID, GridWorld.CELL_VOID, GridWorld.CELL_VOID]], discountFactor = 1 )

	# def setRewards(self, rewOfVoidCell, rewOfPitCell, rewOfExitCell):
	w.setRewards(-0.04, -1, 1)
	#def setProbabilities(self, probToGoForward, probToGoLeft, probToGoRight, probToGoBackward):
	w.setProbabilities(0.8, 0.1, 0.1, 0)

	w.setDiscountFactor(1)

	# ----------------------------------- Setting up value iteration --------------------------------

	vi = Policy(w)
	vi.ValueIteration()
	# vi.getPolicyFromUtility()

	w.printGridWorld()
	print

	print
	vi.printValues()
