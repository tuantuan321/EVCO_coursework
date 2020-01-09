# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# My extra import
import math
import matplotlib.pyplot as plt
import numpy

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
# NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

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

    # Extra functions
    # TODO: Add extra functions

    # Sense wall
    def sense_wall_left(self):
        return self.body[0][1] == 1

    def sense_wall_right(self):
        return self.body[0][1] == (XSIZE - 2)

    def sense_wall_up(self):
        return self.body[0][0] == 1

    def sense_wall_down(self):
        return self.body[0][0] == (YSIZE - 2)

    # Sense food
    def sense_food_left(self):
        if ((self.body[0][0] == self.food[0][0]) and (self.body[0][1] == self.food[0][1] + 1)):
            return True
        else:
            return False

    def sense_food_right(self):
        if ((self.body[0][0] == self.food[0][0]) and (self.body[0][1] == self.food[0][1] - 1)):
            return True
        else:
            return False

    def sense_food_up(self):
        if ((self.body[0][0] == self.food[0][0] + 1) and (self.body[0][1] == self.food[0][1])):
            return True
        else:
            return False

    def sense_food_down(self):
        if ((self.body[0][0] == self.food[0][0] - 1) and (self.body[0][1] == self.food[0][1])):
            return True
        else:
            return False

    # Sense tail
    def sense_tail_left(self):
        for n in range(1, len(self.body)):
            if ((self.body[0][1] == self.body[n][1] + 1) and (self.body[0][0] == self.body[n][0])):
                return True
            else:
                return False

    def sense_tail_right(self):
        for n in range(1, len(self.body)):
            if ((self.body[0][1] == self.body[n][1] - 1) and (self.body[0][0] == self.body[n][0])):
                return True
            else:
                return False

    def sense_tail_up(self):
        for n in range(1, len(self.body)):
            if ((self.body[0][1] == self.body[n][1]) and (self.body[0][0] == self.body[n][0] + 1)):
                return True
            else:
                return False

    def sense_tail_down(self):
        for n in range(1, len(self.body)):
            if ((self.body[0][1] == self.body[n][1]) and (self.body[0][0] == self.body[n][0] - 1)):
                return True
            else:
                return False

    # Sense obstacle
    def sense_obstacle_left(self):
        return self.sense_wall_left() or self.sense_tail_left()

    def sense_obstacle_right(self):
        return self.sense_wall_right() or self.sense_tail_right()

    def sense_obstacle_up(self):
        return self.sense_wall_up() or self.sense_tail_up()

    def sense_obstacle_down(self):
        return self.sense_wall_down() or self.sense_tail_down()

    # If food
    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    # If wall
    def if_wall_left(self, out1, out2):
        return partial(if_then_else, self.sense_wall_left, out1, out2)

    def if_wall_right(self, out1, out2):
        return partial(if_then_else, self.sense_wall_right, out1, out2)

    def if_wall_up(self, out1, out2):
        return partial(if_then_else, self.sense_wall_up, out1, out2)

    def if_wall_down(self, out1, out2):
        return partial(if_then_else, self.sense_wall_down, out1, out2)

    # If tail
    def if_tail_left(self, out1, out2):
        return partial(if_then_else, self.sense_tail_left, out1, out2)

    def if_tail_right(self, out1, out2):
        return partial(if_then_else, self.sense_tail_right, out1, out2)

    def if_tail_up(self, out1, out2):
        return partial(if_then_else, self.sense_tail_up, out1, out2)

    def if_tail_down(self, out1, out2):
        return partial(if_then_else, self.sense_tail_down, out1, out2)

    # If obstacle
    def if_obstacle_left(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_left, out1, out2)

    def if_obstacle_right(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_right, out1, out2)

    def if_obstacle_up(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_up, out1, out2)

    def if_obstacle_down(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_down, out1, out2)

    # If move
    def if_left(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_RIGHT, out1, out2)

    def if_right(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_LEFT, out1, out2)

    def if_up(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_UP, out1, out2)

    def if_down(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_DOWN, out1, out2)

    # If obstacle


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

# Add a input to the function
def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

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

    print(collided)
    print(hitBounds)
    input("Press to continue...")

    return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
def runGame(individual):

    global snake
    global pset

    totalScore = 0

    snake._reset()
    food = placeFood(snake)
    timer = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        playGame = gp.compile(individual, pset)
        playGame()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1 # timesteps since last eaten

        totalScore += snake.score

    return totalScore,

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

#pset.addPrimitive(snake.if_wall_left, 2)
#pset.addPrimitive(snake.if_wall_right, 2)
#pset.addPrimitive(snake.if_wall_up, 2)
#pset.addPrimitive(snake.if_wall_down, 2)

#pset.addPrimitive(snake.if_tail_left, 2)
#pset.addPrimitive(snake.if_tail_right, 2)
#pset.addPrimitive(snake.if_tail_up, 2)
#pset.addPrimitive(snake.if_tail_down, 2)

pset.addPrimitive(snake.if_left, 2)
pset.addPrimitive(snake.if_right, 2)
pset.addPrimitive(snake.if_up, 2)
pset.addPrimitive(snake.if_down, 2)

pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionDown)

creator.create("FitnessMax", base.Fitness, weights=(1.0,0.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initial toolbox
toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genGrow, pset=pset, min_=1, max_=3)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", runGame)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

def main():

    global snake
    global pset

    #random.seed(128)

    # Initial some parameters
    NGEN = 20
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n = 500)
    hof = tools.HallOfFame(1)

    # Initial stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame=hof, verbose=True)

    best = tools.selBest (pop, 1)

    input("Press to continue")

    for i in range(100):
        displayStrategyRun(best[0])

main()
