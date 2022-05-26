import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from NN import run_NN

# this code was constructed with the aid of an article on hyper-parameter optimization by Conor Rothwell
# information on DEAP GA construction found at
# https://www.linkedin.com/pulse/hyper-parameter-optimisation-using-genetic-algorithms-conor-rothwell

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximise the fitness function value
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# the parameters that we tune for the base model
# hn for hidden neurons and lr for learning rate
hn_min, hn_max = 2, 50
lr_low, lr_max = 0.001, 1.0

# registering these as attributes and how they will be generated
# hidden neurons is a random int in the specified range
# learning rate is a random float in the specified range
toolbox.register("attr_hn", random.randint, hn_min, hn_max)
toolbox.register("attr_lr", random.uniform, lr_low, lr_max)

N_CYCLES = 1
# create individual and population
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_hn, toolbox.attr_lr), n=N_CYCLES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# mutate function which randomly regenerates within the specified boundaries if a gene is selected for mutation
def mutate(individual):
    gene = random.randint(0, 1)  # select which parameter to mutate
    if gene == 0:
        individual[0] = random.randint(hn_min, hn_max)
    elif gene == 1:
        individual[1] = random.uniform(lr_low, lr_max)
    return individual,


# runs the base model and gets the final training, val and testing accuracy from the base model using GA parameters
def evaluate(individual):
    # extract the values from the individual chromosome
    hn = individual[0]
    lr = individual[1]

    train_accuracy, val_accuracy, test_accuracy, final_val_loss, val_loss_penalty = run_NN(hn, lr)

    # all train, val, test accuracy combined which weight up to 1.0
    combined_value = ((train_accuracy/100)/3) + ((val_accuracy/100)/3) + ((test_accuracy/100)/3)

    # total accuracy minus a penalty for requiring more hidden neurons and high val loss when evaluating fitness
    # val loss penalty is a penalty added to prevent loss curve fluctuations which indicate poor learning
    return combined_value - (hn/1000) - (final_val_loss/10) - val_loss_penalty,


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
with open('saved params/base_parameters.txt', 'w') as f:
    f.write(str(best_parameters))
