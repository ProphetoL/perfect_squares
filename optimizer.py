import numpy as np
from numpy.random import randint

from numba import njit

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time 

import json

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

@njit()
def getPart(M, i, i_type):
    """function to get the rows, colums, and diagonals"""
    global ORDER

    #get the rows
    if i_type == 0:
        return M[i]

    #get the colums
    elif i_type == 1:
        return M[:,i]

    #get the diagonals
    elif i_type == 2:
        if i == 1:
            M = np.fliplr(M)

        return np.array([M[i][i] for i in range(ORDER)])


@njit()
def loss_func(M):
    """Function to measure how bad is the square"""
    global ORDER
    
    indexes = np.array([ORDER, ORDER, 2])
    Sums = np.zeros((np.sum(indexes)), dtype=np.int64)
    
    #compute all the sums
    for i_t in range(len(indexes)):
        t = indexes[i_t]

        for i in range(t):
            Sums[i + i_t*ORDER] = np.sum(getPart(M, i, i_t))
    
    return np.std(Sums)

  

@njit()
def Optizer(M, old_loss, lrn, Opti_step):
    """function to make a step of optimaziation"""
    global ORDER

    new_M, new_loss = improve_square(M, old_loss, lrn)

    return new_M, new_loss


@njit()   
def improve_square(M, loss, lrn):
    """function to modify the square"""
    i, j = [np.int64(randint(0, ORDER-1)) for i in range(2)]

    change_factor = 0
    while change_factor == 0 :
        change_factor = randint(-Opti_step, Opti_step+1)
    
    new_M = M.copy()

    #change selected number
    new_M[i,j] += change_factor
    new_loss = loss_func(new_M)
    
    if not(new_loss < loss or np.random.random() <= lrn):

        new_M = M.copy()
        new_M[i,j] -= change_factor
        new_loss = loss_func(new_M)

    
    return new_M, new_loss

def print_perfects(perfects):
    perfects_log = open('perfects.log', 'a')
    for p in perfects:
        print(p, '\n')
        perfects_log.write(str(p))
        perfects_log.write('\n \n')
    perfects_log.close()

def print_plots(perfect_losses):
    for p_loss in perfect_losses:
        plt.plot(p_loss)
        plt.savefig(f"perfect_loss_graphs\\perfect_loss_{int(time())}.png")
    
    
    
extend = 1000
#range for the initial matrix
ORDER = 3
#order of the square
square_TYPE = np.int64
#type of the value in the square

Opti_step = 10
#step taked by the optymizer
Opti_step_init = Opti_step

lrn = 0.5
#probapility to keep a bad square
lrn_init = lrn

nb_step = 2000
generations = 200#int(1E6)

overall_best_M = randint(-extend, extend, (ORDER, ORDER),square_TYPE)
overall_best_loss = loss_func(overall_best_M)

perfects = []
nb_perfects = 0
running_best_loss = []
running_best_loss_change = []
best_loss_i = []
perfect_losses = []

"""loss_improvement_file = open(f"losses\\loss_improve_{int(time())}.log", "a")
#file to log if the squre improve"""
for i_gen in tqdm(range(generations)):
    
    #Create initial square
    M = randint(1, extend, (ORDER, ORDER),square_TYPE)
    
    #Arrays to store squares and losses
    losses = []
    loss = loss_func(M)
    Ms = []
    Opti_step = Opti_step_init
    lrn = lrn_init
    
    #loss_improvement_file.write(f"next gen\n{i_gen}\n{loss}\n")

    #Optimization loop
    i_step = 0
    while i_step <nb_step and loss >= 0:#to stop when a perfect squred is find
        losses.append(loss)
        Ms.append(M.copy())
        
        #Optimize

        M, loss = Optizer(M, loss, lrn, Opti_step)
        
        #smart Opti_step and lrn
        h = 200
        if loss < h and not losses[-1] < h:
            Opti_step = 1
            lrn = 0.05

        #loss_improvement_file.write(f"{loss}, {loss < losses[-1]}\n")

        i_step += 1

    #Find best of generation
    min_arg = np.argmin(losses)
    min_loss = loss_func(Ms[min_arg])

    #print(min_loss)
    if loss < h:
        running_best_loss_change.append(min_loss)
    else :
        running_best_loss.append(min_loss)
    best_loss_i.append(min_arg)


    #Check if overall best
    if min_loss < overall_best_loss:
        overall_best_loss = loss_func(Ms[min_arg])
        overall_best_M = Ms[min_arg]
        
    #Check for perfection
    if min_loss == 0:
        print("\nPerfect !")
        perfects.append(Ms[min_arg])
        perfect_losses.append(losses)
        nb_perfects += 1
    

    #Save the perfects
    if i_gen% 1000 == 0 and not len(perfects) == 0:
        for i in range(len(perfects)):
            perfects[i] = perfects[i].tolist()

        #Read json
        f = open('perfects.json', 'r')
        f_json  = json.loads(f.read())
        f.close()

        #Write Json
        f = open('perfects.json', 'w')
        json.dump(f_json + perfects, f , indent=4)
        f.close()

        #print(f"\n{len(perfects)} squares saved !")
        
        plt.plot(losses)
        plt.savefig(f"{extend}_{Opti_step_init}-{int(time())}.png")
        perfects = []



strd_loss, mean_loss  = np.std(running_best_loss), np.mean(running_best_loss)

#Analytics
print('\n Overall Best Square: \n', overall_best_M,
      '\n Overall Best Loss: ', overall_best_loss,
      '\n \nNumber of Perfects: ',nb_perfects,
      '\n\n mean of index of best losses :', np.mean(best_loss_i),
      '\n mean of best losses :', mean_loss, 
      '\n mean after change', np.mean(running_best_loss_change)
      )

loss_improvement_file.close()

print_perfects(perfects)
print_plots(perfect_losses)

