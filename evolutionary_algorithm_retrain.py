import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from retrain import retrain_model

# this code was constructed with the aid of an article on hyper-parameter optimization by Conor Rothwell
# information on DEAP GA construction found at
# https://www.linkedin.com/pulse/hyper-parameter-optimisation-using-genetic-algorithms-conor-rothwell

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximise the fitness function value
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# the parameters that we tune for the pruned model
# min_angle for minimum angle, max_angle for maximum angle and lr for learning rate
lr_low, lr_max = 0.001, 1
min_angle_low, min_angle_max = 1, 89
max_angle_low, max_angle_max = 90, 180

# registering these as attributes and how they will be generated
# angles are random ints in the specified range
# learning rate is a random float in the specified range
toolbox.register("attr_lr", random.uniform, lr_low, lr_max)
toolbox.register("attr_min_angle", random.randint, min_angle_low, min_angle_max)
toolbox.register("attr_max_angle", random.randint, max_angle_low, max_angle_max)

N_CYCLES = 1
# create individual and population
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_lr, toolbox.attr_min_angle,
                                                                     toolbox.attr_max_angle,
                                                                     ), n=N_CYCLES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# mutate function which randomly regenerates within the specified boundaries if a gene is selected for mutation
def mutate(individual):
    gene = random.randint(0, 2)  # select which parameter to mutate
    if gene == 0:
        individual[0] = random.uniform(lr_low, lr_max)
    elif gene == 1:
        individual[1] = random.randint(min_angle_low, min_angle_max)
    elif gene == 2:
        individual[2] = random.randint(max_angle_low, max_angle_max)
    return individual,


# runs the pruning model and gets the final training and testing accuracy using GA parameters
def evaluate(individual):
    # extract the values from the individual chromosome
    r_lr = individual[0]
    min_angle = individual[1]
    max_angle = individual[2]

    train_accuracy, test_accuracy = retrain_model(r_lr, min_angle, max_angle)
    combined_value = ((train_accuracy / 100) / 2) + ((test_accuracy / 100) / 2)

    # total accuracy minus a penalty for requiring more hidden neurons when evaluating fitness
    return combined_value,  # penalty for not pruning many neurons


# registering all functions in the toolbox
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)

# define parameters of GA
population_size = 50
crossover_probability = 0.8
mutation_probability = 0.2
number_of_generations = 50

pop = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)

# stats to provide evolution information across generations
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats=stats, mutpb=mutation_probability,
                               ngen=number_of_generations, halloffame=hof, verbose=True)
best_parameters = hof[0]  # saving the best parameters

# save best parameters to text file to call for retraining the pruned model
with open('saved params/prune_parameters.txt', 'w') as f:
    f.write(str(best_parameters))
