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
    Sums = np.zeros((np.sum(indexes)), dtype=np.int_)
    
    #compute all the sums
    for i_t in range(len(indexes)):
        t = indexes[i_t]

        for i in range(t):
            Sums[i + i_t*ORDER] = np.sum(getPart(M, i, i_t))
    
    return np.std(Sums), np.mean(Sums)

  

#@njit()
def Optizer(M, old_loss, lrn, Opti_step):
    """function to make a step of optimaziation"""
    global ORDER

    new_M, new_loss = improve_square(M, old_loss, lrn)

    return new_M, new_loss

# old algo
#@njit()   
# def improve_square(M, loss, lrn):
#     """function to modify the square"""
#     global loss_improvement_file
#     i, j = randint(0, ORDER-1, (2), np.int_)
#     choose_bad = np.random.random() <= lrn

#     change_factor = None
#     while change_factor == 0 or change_factor is None:
#         change_factor = randint(1, Opti_step+1)
    
#     large_M = M.copy()
#     small_M = M.copy()

#     #change selected number
#     large_M[i,j] += change_factor
#     small_M[i,j] -= change_factor
#     large_loss = loss_func(large_M)[0]
#     small_loss = loss_func(small_M)[0]
    
#     if large_loss > loss:
#         if choose_bad:
#             return large_M, large_loss
#         return small_M, small_loss
#     else:
#         if choose_bad:
#             return small_M, small_loss
#         return large_M, large_loss

# new algo
#@njit()   
def improve_square(M, loss, lrn):
    """function to modify the square"""
    
    #select random number
    global loss_improvement_file
    i, j = randint(0, ORDER-1, (2), np.int_)
    choose_bad = np.random.random() <= lrn

    change_factor = None
    while change_factor == 0 or change_factor is None:
        change_factor = randint(1, Opti_step+1)
    
    large_M = M.copy()
    small_M = M.copy()

    #change selected number
    large_M[i,j] += change_factor
    small_M[i,j] -= change_factor
    
    #calculate losses of steps
    large_loss = loss_func(large_M)[0]
    small_loss = loss_func(small_M)[0]
    
    if choose_bad:
        if large_loss > small_loss:
            return large_M, large_loss
        else:
            return small_M, small_loss
    else:
        if large_loss < small_loss:
            return large_M, large_loss
        else:
            return small_M, small_loss
            
            


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
    
    
    
extend = 20
#range for the initial matrix
ORDER = 3
#order of the square
square_TYPE = np.int
#type of the value in the square

Opti_step = 1
#step taked by the optymizer
lrn = 0.3
#probapility to keep a bad square

nb_step = 250
generations = 5000

overall_best_M = randint(-extend, extend, (ORDER, ORDER),square_TYPE)
overall_best_loss = loss_func(overall_best_M)[0]

perfects = []
running_best_loss = []
perfect_losses = []

loss_improvement_file = open(f"losses\\loss_improve_{int(time())}.log", "a")
#file to log if the squre improve
for i_gen in tqdm(range(generations)):
    
    #Create initial square
    M = randint(1, extend, (ORDER, ORDER),square_TYPE)
    
    #Arrays to store squares and losses
    losses = []
    loss = loss_func(M)[0]
    Ms = []
    
    
    loss_improvement_file.write(f"next gen\n{i_gen}\n{loss}\n")

    #Optimization loop
    i_step = 0
    while i_step < nb_step:#to stop when a perfect squred is find
        losses.append(loss)
        Ms.append(M.copy())
        
        #Optimize
        M, loss = Optizer(M, loss, lrn, Opti_step)
        
        loss_improvement_file.write(f"{loss}, {loss < losses[-1]}\n")
        
        if loss == 0:
            perfects.append(M.copy())
            perfect_losses.append(losses)
            break
        
        i_step += 1


#Analytics
print('\n \n Perfects: \n')

loss_improvement_file.close()

print_perfects(perfects)
print_plots(perfect_losses)
