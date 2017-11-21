#!/usr/bin/env python
#Author: Christian Wagner
#Date: July 2016

import random
import numpy as _ny

''' This Class handels the gird world. It keeps track of what the world looks like.
	Knows the probability of making a move. Sets the rewards for all state/actions.
	Has the Transistion function.


'''
class GridWorld:

	CELL_VOID = 0
	CELL_PIT  = 1
	CELL_EXIT = 2
	CELL_WALL = 3
	__cells = None # should be a 2d array. Each cell is a given state
	size = (0, 0) #(columns, rows)

	ACTION_NORTH = 'N'
	ACTION_SOUTH = 'S'
	ACTION_WEST  = 'W'
	ACTION_EAST  = 'E'
	actionSet = (ACTION_NORTH, ACTION_SOUTH, ACTION_WEST, ACTION_EAST)

	# Probabilities
	PROB_FORWARD  = 'F'
	PROB_BACKWARD = 'B'
	PROB_LEFT     = 'L'
	PROB_RIGHT    = 'R'
	prob = None

	#setting the default values of the reward
	rew = {CELL_VOID : 0,
				CELL_EXIT : 10,
				CELL_PIT  : -10,
				CELL_WALL : 0}

	discFactor = 0

	def __init__(self, cells, discountFactor = 1):
		'''the __cells is a matrix memorized in this way
			[[[cell 1 of first row, cell 2 of first row, ...]],[row2], ...]
		'''
		self.__cells = cells
		self.size = (len(self.__cells[0]), len(self.__cells))
		self.discFactor = discountFactor

	def printGridWorld(self):
		code = {0 : "    ",
			1 : ' <> ',
			3 : '||||',
			2 : ' XX '}
		line = "  |"
		row = ""

		for i in range(len(self.__cells[1])):
			line += "  " + str(i) + " |"
			row += '-----'
		row += "---"
		print line


		for i in range(len(self.__cells[:])):
			print row
			line = str(i) + " "
			for j in self.__cells[i]:
				line += '|' + code[j]
			print line + '|'

		print row

	def getsize(self):
		return len(self.__cells[:]),len(self.__cells[0])


	def cellTypeAt(self, y, x):
		return self.__cells[y][x]

	def cellAt(self, x, y):
		'''pos is a tuple (x,y)'''
		return self.__cells[y][x]

	# purpose: to set discount factor
	# notes: should by less then 1
	def setDiscountFactor(self, df):
		self.discFactor = df

	# Purpose: To set the reward values for the different type of grid cells
	def setRewards(self, rewOfVoidCell, rewOfPitCell, rewOfExitCell):
		self.rew = {self.CELL_VOID : rewOfVoidCell,
				self.CELL_EXIT : rewOfExitCell,
				self.CELL_PIT  : rewOfPitCell,
				self.CELL_WALL : 0}

	# Purpose: set probabilities that the action will suceed
	# Note: set all equal to 1 to be deterministic
	def setProbabilities(self, probToGoForward, probToGoLeft, probToGoRight, probToGoBackward):
		if probToGoForward + probToGoLeft + probToGoRight + probToGoBackward != 1:
			raise Exception('the prob must have 1 as sum')

		self.prob = {self.PROB_FORWARD  : probToGoForward,
				     self.PROB_LEFT     : probToGoLeft,
				     self.PROB_RIGHT    : probToGoRight,
				     self.PROB_BACKWARD : probToGoBackward}


	def transitionFunction(self, position, action):
		''' this function describes the movements that we can do (deterministic)
			if we are in a pit, in a exit or in a wall cell we can't do anything
			we can't move into a wall
			we can't move out the border of the grid
			returns the new position

			Note: position = [x,y]
		'''
		if action not in self.actionSet:
			raise Exception("unknown action")
		if self.__cells[position[0]][position[1]] != self.CELL_VOID:
			raise Exception("no action allowed")

		if action == self.ACTION_NORTH:
			ris = (max(0, position[0] - 1),position[1])
		elif action == self.ACTION_SOUTH:
			ris = (min(len(self.__cells) - 1, position[0] + 1),position[1])
		elif action == self.ACTION_WEST:
			ris = (position[0],max(0, position[1] - 1))
		else:
			ris = (position[0],min(len(self.__cells[0]) - 1, position[1] + 1))

		if self.__cells[ris[0]][ris[1]] == self.CELL_WALL: return position
		return ris

	def possiblePositionsFromAction(self, position, worldAction):
		'''
			given an action worldAction, return a dictionary D,
			where for each action a, D[a] is the probability to do the action a
		'''

		def getProbabilitiesFromAction(worldAction):
			if worldAction == self.ACTION_NORTH:
				return {self.ACTION_NORTH : self.prob[self.PROB_FORWARD],
						self.ACTION_SOUTH : self.prob[self.PROB_BACKWARD],
						self.ACTION_WEST  : self.prob[self.PROB_LEFT],
						self.ACTION_EAST  : self.prob[self.PROB_RIGHT]}
			elif worldAction == self.ACTION_SOUTH:
				return {self.ACTION_NORTH : self.prob[self.PROB_BACKWARD],
						self.ACTION_SOUTH : self.prob[self.PROB_FORWARD],
						self.ACTION_WEST  : self.prob[self.PROB_RIGHT],
						self.ACTION_EAST  : self.prob[self.PROB_LEFT]}
			elif worldAction == self.ACTION_WEST:
				return {self.ACTION_NORTH : self.prob[self.PROB_RIGHT],
						self.ACTION_SOUTH : self.prob[self.PROB_LEFT],
						self.ACTION_WEST  : self.prob[self.PROB_FORWARD],
						self.ACTION_EAST  : self.prob[self.PROB_BACKWARD]}
			else:
				return {self.ACTION_NORTH : self.prob[self.PROB_LEFT],
						self.ACTION_SOUTH : self.prob[self.PROB_RIGHT],
						self.ACTION_WEST  : self.prob[self.PROB_BACKWARD],
						self.ACTION_EAST  : self.prob[self.PROB_FORWARD]}

		if (self.__cells[position[0]][position[1]] != self.CELL_VOID):
			return [] #we can't do anything in the wall, in a pit or in a exit

		prob = getProbabilitiesFromAction(worldAction)
		result = []
		for a in self.actionSet:
			result.append((a, self.transitionFunction(position, a), prob[a]))
		return result

	# need a function to give the instantaneuos reward for the current state
	def rewardAtState(self,y,x):
		return self.rew[self.__cells[y][x]]

	def getCells(self):
		return self.__cells;

	def randomAction(self):
		return self.actionSet[int(random.random() * 4)]

# ------------------------------------------------------------------------------------------------------
''' create grid world cells:
	Purpose: To generate the cell 2-d array to be given in creating an instence of a GridWorld
	Notes: It randomly produces location of walls and pits.
		Position (2,2) is always kept clear as the generall starting point.
		There is always an exit at position (row-2, col-2).

'''
def createGridWorldCells(row, col, void_prob, pit_prob):
	assert row > 1, " need to have more then 1 row"
	assert col > 1, " need to have more then 1 column"
	assert void_prob < 1, "can't have probability greater then 1 "
	cells = []
	rand = 0
	cell_type = 0
	for i in range(row):
		cells.append([])
		for j in range(col):
			rand = random.random()
			if rand < void_prob:
				cell_type = 0
			else :
				if random.random() < pit_prob :
					cell_type = 1
				else:
					cell_type = 3
			if (i%10 == 0 and j%15 == 0):
				cell_type = 2
			elif (i == 2 and j == 2): # this is to make sure that I have a clear starting place
				cell_type = 2
			cells[i].append(cell_type)
	return cells

if __name__ == '__main__' :
	w = GridWorld(createGridWorldCells(4,3,.99,.1))
	w.printGridWorld()

	w.setProbabilities(.8,.1,.1,0)
	print w.rewardAtState(9,0)
	r = w.possiblePositionsFromAction([0,0],'N')
	print r

	print("\nSome transitions:")
	print(w.transitionFunction((1,1), GridWorld.ACTION_NORTH))
	print(w.transitionFunction((0,0), GridWorld.ACTION_EAST))
	# print(w.transitionFunction((1,1), GridWorld.ACTION_WEST))
	# print(w.transitionFunction((1,0), GridWorld.ACTION_SOUTH))

	# w.setRewards(-0.04, -1, 1)
	# w.setProbabilities(0.7, 0.1, 0.2, 0)

	# print(w.possiblePositionsFromAction((0,0), GridWorld.ACTION_NORTH))
	# print(w.possiblePositionsFromAction((0,0), GridWorld.ACTION_SOUTH))
