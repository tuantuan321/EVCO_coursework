# This code defines the agent (as in the playable version) in a way that can be called and executed from an
# evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an
# evolutionary algorithm that evolves and executes a snake agent.

import curses
import math
import multiprocessing
import operator
import random
import numpy
import pygraphviz as pgv
import matplotlib.pyplot as plt

from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
NFOOD = 1

toolbox = base.Toolbox()
stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_size = tools.Statistics(key=len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        super().__init__()
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8]]
        self.score = 0
        self.ahead = []
        self.food = []
        self.hit = False

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8]]
        self.score = 0
        self.ahead = []
        self.food = []
        self.hit = False

    def get_ahead_location(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def update_position(self):
        self.get_ahead_location()
        self.body.insert(0, self.ahead)

    def change_direction_up(self):
        # if self.direction != S_DOWN:
        self.direction = S_UP

        # else:
        #     self.direction = S_UP

    def change_direction_right(self):
        # if self.direction != S_LEFT:
        self.direction = S_RIGHT

        # else:if
        #     self.direction = S_RIGHT

    def change_direction_down(self):
        # if self.direction != S_UP:
        self.direction = S_DOWN

        # else:
        #     self.direction = S_DOWN

    def change_direction_left(self):
        # if self.direction != S_RIGHT:
        self.direction = S_LEFT

        # else:
        #     self.direction = S_LEFT

    def snake_has_collided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return self.hit

    def sense_wall_ahead(self):
        self.get_ahead_location()

        return self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE - 1)

    def sense_food_ahead(self):
        self.get_ahead_location()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.get_ahead_location()
        return self.ahead in self.body

        # Additional functions:

    def sense_food_direction_left(self):
        if len(self.food) == 0:
            return False
        return self.body[0][1] > self.food[0][1]

    def sense_food_direction_right(self):
        if len(self.food) == 0:
            return False
        return self.body[0][1] < self.food[0][1]

    def sense_food_direction_up(self):
        if len(self.food) == 0:
            return False
        return self.body[0][0] > self.food[0][0]

    def sense_food_direction_down(self):
        if len(self.food) == 0:
            return False
        return self.body[0][0] < self.food[0][0]

    def sense_tail_up(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] + 1)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_right(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] - 1):
                return True
        return False

    def sense_tail_down(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] - 1)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_left(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] + 1):
                return True
        return False

    def sense_wall_up(self):
        return self.body[0][0] == 1

    def sense_wall_right(self):
        return self.body[0][1] == (XSIZE - 2)

    def sense_wall_down(self):
        return self.body[0][0] == (YSIZE - 2)

    def sense_wall_left(self):
        return self.body[0][1] == 1

    def sense_danger_up(self):
        return self.sense_wall_up() or self.sense_tail_up()

    def sense_danger_right(self):
        return self.sense_wall_right() or self.sense_tail_right()

    def sense_danger_down(self):
        return self.sense_wall_down() or self.sense_tail_down()

    def sense_danger_left(self):
        return self.sense_wall_left() or self.sense_tail_left()

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_right, out1, out2)

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_down, out1, out2)

    def if_danger_up(self, out1, out2):
        return partial(if_then_else, self.sense_danger_up, out1, out2)

    def if_danger_right(self, out1, out2):
        return partial(if_then_else, self.sense_danger_right, out1, out2)

    def if_danger_down(self, out1, out2):
        return partial(if_then_else, self.sense_danger_down, out1, out2)

    def if_danger_left(self, out1, out2):
        return partial(if_then_else, self.sense_danger_left, out1, out2)

    def if_moving_right(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_RIGHT, out1, out2)

    def if_moving_left(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_LEFT, out1, out2)

    def if_moving_up(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_UP, out1, out2)

    def if_moving_down(self, out1, out2):
        return partial(if_then_else, lambda: self.direction == S_DOWN, out1, out2)

    def if_trapped(self, out1, out2):
        return partial(if_then_else, lambda: self.is_trapped(), out1, out2)

    def is_trapped(self):
        # (number of rooms, (point closest to head, room coords))
        info = get_room_info()

        if info[0]:
            v = len(self.body) >= len(info[1][1])
            return v
        else:
            return False


# This function places a food item in the environment
def place_food():
    food = []
    while len(food) < NFOOD:
        potential_food = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        if not (potential_food in snake.body) and not (potential_food in food):
            food.append(potential_food)
    snake.food = food  # let the snake know where the food is
    return food


snake = SnakePlayer()


def translate_grid():
    grid = []

    for i in range(XSIZE):
        grid.append([0] * YSIZE)

    for i in snake.body:
        grid[i[0]][i[1]] = 2

    grid.insert(0, [2] * XSIZE)
    grid.insert(YSIZE + 1, [2] * XSIZE)

    for r in grid:
        r.insert(0, 2)
        r.insert(XSIZE + 1, 2)

    return grid


def point_difference(point):
    # Increment required as the grid's boundary are manually added in when processing the grid
    a = math.pow(point[0] - (snake.body[0][0] + 1), 2)
    b = math.pow(point[1] - (snake.body[0][1] + 1), 2)

    return math.sqrt(a + b)


# Adapted from: http://inventwithpython.com/blog/2011/08/11/recursion-explained-with-the-flood-fill-algorithm-and-zombies-and-cats/
def flood_fill(grid, x, y, old, new, info=None):
    if info is None:
        info = ((-1.0, -1.0), [])

    dx = len(grid)
    dy = len(grid[0])

    if old is None:
        old = grid[x][y]

    if grid[x][y] != old:
        return info

    grid[x][y] = new
    t = info[1]
    t.append((x, y))

    val = point_difference((x, y))

    if val == 1:
        coord_next_to_head = (x, y)
        info = (coord_next_to_head, t)
    else:
        info = (info[0], t)

    if x > 0:  # left
        info = flood_fill(grid, x - 1, y, old, new, info)
    if y > 0:  # up
        info = flood_fill(grid, x, y - 1, old, new, info)
    if x < dx - 1:  # right
        info = flood_fill(grid, x + 1, y, old, new, info)
    if y < dy - 1:  # down
        info = flood_fill(grid, x, y + 1, old, new, info)

    return info


def get_room_info():
    grid = translate_grid()
    grid_width = len(grid)
    grid_height = len(grid[0])
    rooms_found = 0
    results = []

    for x in range(grid_width):
        for y in range(grid_height):
            if grid[x][y] == 0:
                coord_next_to_head, coords = flood_fill(grid, x, y, 0, 1)
                results.append((coord_next_to_head, coords))

    room_info = 0

    for ival in results:
        # Adjust to new coordinates because of manual boundary insertion
        next_point = [snake.body[0][0] + 1, snake.body[0][1] + 1]

        if snake.direction == S_RIGHT:
            next_point = (next_point[0] + 1, next_point[1])
        elif snake.direction == S_LEFT:
            next_point = (next_point[0] - 1, next_point[1])
        elif snake.direction == S_UP:
            next_point = (next_point[0], next_point[1] - 1)
        elif snake.direction == S_DOWN:
            next_point = (next_point[0], next_point[1] + 1)

        if next_point in ival[1] and point_difference(next_point) == 1:
            # (point closest to head, room coordinates)
            rooms_found += 1
            room_info = (ival[0], ival[1])

    return rooms_found, room_info


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def display_strategy_run(routine):
    global snake
    global pset

    routine = gp.compile(routine, pset=pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.scrollok(0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = place_food()

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0

    while not snake.snake_has_collided() and not timer == ((2 * XSIZE) * YSIZE):
        routine()

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(snake.score) + ' ')
        win.getch()

        snake.update_position()

        if snake.body[0] in food:
            # Set coord to out of the map boundaries
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = place_food()
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # time steps since last eaten

        win.addch(snake.body[0][0], snake.body[0][1], 'o')
        hitBounds = (timer == ((2 * XSIZE) * YSIZE))

    input("Press to continue")
    curses.endwin()

    return snake.score


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def run_game(individual):
    global snake
    routine = gp.compile(individual, pset)
    snake._reset()

    food = place_food()
    timer = 0
    steps = 0

    while not snake.snake_has_collided() and not timer == XSIZE * YSIZE:
        routine()
        snake.update_position()
        steps += 1

        if snake.body[0] in food:
            snake.score += 1
            food = place_food()
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # time steps since last eaten

    if snake.snake_has_collided() and snake.score == 0:
        return -20, steps

    if timer == XSIZE * YSIZE:
        return -10, -10

    return math.pow(snake.score, 2), steps


def evaluate(ind):
    steps = 0
    score = 0
    nrun = 1
    for i in range(nrun):
        t_score, t_steps = run_game(ind)
        score += t_score
        steps += t_steps

    return (score / nrun), (steps / nrun)


def init():
    global snake
    global pset
    global stats
    global halloffame

    snake = SnakePlayer()
    pset = gp.PrimitiveSet("main", 0)
    pset.addPrimitive(snake.if_food_up, 2)
    pset.addPrimitive(snake.if_food_right, 2)
    pset.addPrimitive(snake.if_food_down, 2)
    pset.addPrimitive(snake.if_food_left, 2)

    pset.addPrimitive(snake.if_danger_up, 2)
    pset.addPrimitive(snake.if_danger_right, 2)
    pset.addPrimitive(snake.if_danger_down, 2)
    pset.addPrimitive(snake.if_danger_left, 2)

    pset.addPrimitive(snake.if_moving_down, 2)
    pset.addPrimitive(snake.if_moving_up, 2)
    pset.addPrimitive(snake.if_moving_right, 2)
    pset.addPrimitive(snake.if_moving_left, 2)

    pset.addPrimitive(snake.if_trapped, 2)

    pset.addTerminal(snake.change_direction_up)
    pset.addTerminal(snake.change_direction_right)
    pset.addTerminal(snake.change_direction_down)
    pset.addTerminal(snake.change_direction_left)

    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Attribute generator
    toolbox.register("expr_init", gp.genGrow, pset=pset, min_=1, max_=3)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.05, fitness_first=True)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    halloffame = tools.HallOfFame(1)

    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)


def main():
    global toolbox

    random.seed(128)

    ngen, cxpb, mutpb, pop = 500, 0.8, 0.2, 500

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "size"
    logbook.chapters["fitness"].header = "min", "avg", "max"
    logbook.chapters["size"].header = "min", "avg", "max"

    population = toolbox.population(n=pop)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)

    record = mstats.compile(population)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = mstats.compile(population)

        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        print(logbook.stream)

    gen = logbook.select("gen")
    fit_max = logbook.chapters["fitness"].select("max")
    size_avgs = logbook.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.savefig('population-tests/pop-' + str(pop) + '.png')

    expr = tools.selBest(population, 1)
    nodes, edges, labels = gp.graph(expr[0])

    gen = pgv.AGraph(landscape='false', ranksep='1', nodesep='2')
    gen.add_nodes_from(nodes)
    gen.add_edges_from(edges)
    gen.layout(prog="dot")

    for i in nodes:
        n = gen.get_node(i)
        n.attr["label"] = labels[i]

    gen.draw("tree.png")

    input("Press to continue")

    for i in range(100):
        display_strategy_run(expr[0])


if __name__ == "__main__":
    init()
    main()
