# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import math
import numpy
import copy
import multiprocessing

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt
import pygraphviz as pgv

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2):
    return partial(progn, out1, out2)

def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [ self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1), self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def doNothing(self):
        pass

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return( self.hit )

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return( self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) )

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    # Sense wall
    def sense_wall_left(self):
        return self.body[0][0] == 1

    def sense_wall_right(self):
        return self.body[0][1] == (XSIZE - 2)

    def sense_wall_down(self):
        return self.body[0][0] == (YSIZE - 2)

    def sense_wall_up(self):
        return self.body[0][1] == 1

    # Sense food
    def sense_food_up(self):
        return self.body[0][0] > self.food[0][0]

    def sense_food_down(self):
        return self.body[0][0] < self.food[0][0]

    def sense_food_left(self):
        return self.body[0][1] > self.food[0][1]

    def sense_food_right(self):
        return self.body[0][1] < self.food[0][1]

    # Sense body
    def sense_body_up(self):
        for i in range(1, len(self.body)):
            if ((self.body[0][0] == (self.body[i][0] + 1)) and (self.body[0][1] == self.body[i][1])):
                return True
        return False

    def sense_body_down(self):
        for i in range(1, len(self.body)):
            if ((self.body[0][0] == (self.body[i][0] - 1)) and (self.body[0][1] == self.body[i][1])):
                return True
        return False

    def sense_body_right(self):
        for i in range(1, len(self.body)):
            if ((self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] - 1)):
                return True
        return False

    def sense_body_left(self):
        for i in range(1, len(self.body)):
            if ((self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] + 1)):
                return True
        return False

    # Sense obstacle
    def sense_obstacle_left(self):
        return self.sense_wall_left() or self.sense_body_left()

    def sense_obstacle_right(self):
        return self.sense_wall_right() or self.sense_body_right()

    def sense_obstacle_up(self):
        return self.sense_wall_up() or self.sense_body_up()

    def sense_obstacle_down(self):
        return self.sense_wall_down() or self.sense_body_down()

    # Check food position
    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    # Check obstacle around
    def if_obstacle_left(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_left, out1, out2)

    def if_obstacle_right(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_right, out1, out2)

    def if_obstacle_up(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_up, out1, out2)

    def if_obstacle_down(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_down, out1, out2)

    # Check movement
    def if_direction_left(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_LEFT, out1, out2)

    def if_direction_right(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_RIGHT, out1, out2)

    def if_direction_up(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_UP, out1, out2)

    def if_direction_down(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_DOWN, out1, out2)

# This function places a food item in the environment
def placeFood(snake):
	food = []
	while len(food) < NFOOD:
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake.body) and not (potentialfood in food):
			food.append(potentialfood)
	snake.food = food  # let the snake know where the food is
	return( food )

snake = SnakePlayer()

# Add an input to the function
def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset=pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False
    while not collided and not timer == ((2*XSIZE) * YSIZE):
        # Add a routine functio to run the game again
        routine()

		# Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1 # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2*XSIZE) * YSIZE))

    curses.endwin()

    print("Collided: " + str(collided))
    print("HitBounds: " + str(hitBounds))
    print("Simulate Score:" + str(snake.score))
    input("Press to continue...")

    return snake.score,

# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
def runGame(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    totalScore = 0

    scores = []

    snake._reset()
    food = placeFood(snake)
    timer = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:
        routine()
        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1 # timesteps since last eaten

        totalScore += snake.score

    scores.append(snake.score)
    avgScore = numpy.mean(scores)

    # if snake moves many steps without eating a food
    if timer == XSIZE * YSIZE:
        avgScore = 0
        totalScore = 0

    return totalScore, avgScore

## Fitness function
def evaluateGame(individual):

    new_totalScore = 0
    new_avgScore = 0

    ## This function is used to avoid lucky placement of food
    N = 3

    for i in range(N):
        totalScore, avgScore = runGame(individual)
        new_totalScore += totalScore
        new_avgScore += avgScore

    new_avgScore = new_avgScore/N
    new_totalScore = new_totalScore/N

    ## Although we calculate the average score per each game, but we only use this value
    ## to see how the snake evolves
    ## For the fitness function, only return the total score

    return new_totalScore,

# Initial pset
pset = gp.PrimitiveSet("main", 0)

pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_food_right, 2)
pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_down, 2)

pset.addPrimitive(snake.if_obstacle_left, 2)
pset.addPrimitive(snake.if_obstacle_right, 2)
pset.addPrimitive(snake.if_obstacle_up, 2)
pset.addPrimitive(snake.if_obstacle_down, 2)

pset.addPrimitive(snake.if_direction_left, 2)
pset.addPrimitive(snake.if_direction_right, 2)
pset.addPrimitive(snake.if_direction_up, 2)
pset.addPrimitive(snake.if_direction_down, 2)

pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.doNothing)

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initial toolbox
toolbox = base.Toolbox()

## gp.genFull: each leaf has the same depth between min and max.
#toolbox.register("expr_init", gp.genFull, pset=pset, min_=0, max_=5)

## gp.genGrow: each leaf might have a different depth between min and max
toolbox.register("expr_init", gp.genGrow, pset=pset, min_=0, max_=5)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

## Use the runGame() function to evaluate
toolbox.register("evaluate", evaluateGame)

## Choose doubletournament as selection method
#toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("select", tools.selDoubleTournament, fitness_size=6, parsimony_size=1.05, fitness_first=True)

## gp.cxOnePoint: executes a one point crossover on the input sequence individuals
#toolbox.register("mate", gp.cxOnePoint)

## gp.cxOnePointLeafBiased: randomly select crossover point in each individual
## and exchange each subtree with the point as root between each individual
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.2)

#toolbox.register("expr_mut", gp.genGrow, min_=0, max_=3)

## gp.mutUniform: randomly select a point in the tree individual
## then replace the subtree at that point as a root by the expression generated using method expr()
#toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

## limit the depth
## because python cannot evaluate a tree higher than 90
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), max_value=8))

# Shows the fitness value
mstats = tools.Statistics(lambda ind: ind.fitness.values[0])
#stats_score = tools.Statistics(lambda ind: ind.fitness.values[1])
#mstats = tools.MultiStatistics(Total_Score=stats_total_score, Score=stats_score)
#mstats = tools.Statistics(lambda ind: ind.fitness.values[0])

## Multiprocessing
## code from: https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

def main():

    global snake
    global pset

    random.seed(128)

    ## Initial parameters

    NGEN = 100  ## generation
    CXPB = 0.5  ## crossover probability
    MUTPB = 0.7  ## mutation probability
    POP = 1500  ## population size

    ## output tree
    OUTPUT_TREE = False
    ## run simulation?
    SIMULATE_AFTER_EVALUATION = False
    ## print evaluation result for analysis
    OUTPUT_RESULT = False

    ## generate population
    population = toolbox.population(n = POP)

    ## store best individual
    ## contains the best individual that ever lived in the population during the evolution
    hof = tools.HallOfFame(3)

    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    population, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats = mstats, halloffame=hof, verbose=True)

    ## select the best individual
    expr = tools.selBest(population, 1)
    #print(expr[0])

    ## Output data for analysis
    if OUTPUT_RESULT == True:
        fd = open('result.txt', 'a')

        ## Output the title
        row_title = ("Generation, Maximum Fitness, Average Fitness" + "\n")
        fd.write(row_title)

        ## Output each generation in separate line
        for i in range(NGEN):
            gen = logbook.select('gen')[i] + 1
            avg_fit = logbook.select('avg')[i]
            max_fit = logbook.select('max')[i]
            row = (gen, ", ", max_fit, ", ", avg_fit, "\n")
            fd.write("".join(map(str, row)))
        fd.close()

    #logbook.header = 'gen', 'nevals', "avg", "std"
    #print(logbook)

    ## Print the best solution
    #index = numpy.argmax([ind.fitness for ind in result_population])
    #x = evaluateGame(result_population[index])
    #print(str(x) + '  ' + str(result_population[index].fitness))

    ## draw the tree
    ## code from: http://deap.gel.ulaval.ca/doc/default/tutorials/advanced/gp.html
    ## -------------------------------------
    nodes, edges, labels = gp.graph(expr[0])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    if OUTPUT_TREE == True:
        g.draw("syntax_tree.pdf")
    ## -------------------------------------

    ## Simulate the game to identify the strategy of snake movement
    if SIMULATE_AFTER_EVALUATION == True:
        displayStrategyRun(expr[0])

    return population, hof, mstats

## If running in Windows, the main function should be modified
## In order to use the multiprocessing module 
if __name__ == "__main__":
    main()
