from matplotlib.text import OffsetFrom
import numpy as np
from numpy.random import randint

import matplotlib.pyplot as plt
from tqdm import tqdm

import numba

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def loss_func(M):
    """Function to measure how bad is the square"""
    global ORDER
    
    indexes = [ORDER, ORDER, 2]
    Sums = np.zeros((sum(indexes)), np.int)
    
    #compute all the sums
    for i_t in range(len(indexes)):
        t = indexes[i_t]

        for i in range(t):
            Sums[i + i_t*ORDER] = sum(getPart(M, i, i_t))
    
    """
    Sums = abs(Sums)
    Sums[Sums == 1] = 2
    Sums[Sums < 1] = 1
    loss = np.product(Sums)"""#loss by product


    return np.std(Sums), np.mean(Sums)

  
def getPart(M, i, i_type):
    """function to get the rows, colums, and diagonals"""
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

        return M.diagonal()


def Optizer(M, old_loss, lrn_=None, Opti_step_=None):
    """function to make a step of optimaziation"""
    #Bad way to import lrn and Opti_step
    if lrn_ is None:
        global lrn
    else :
        lrn = lrn_

    if Opti_step_ is None:
        global Opti_step
    else :
        Opti_step = Opti_step_

    global ORDER

    new_M, new_loss = improve_square(M, old_loss, lrn)

    return new_M, new_loss
    
def improve_square(M, loss, lrn):
    new_M = M.copy()
    i, j = randint(0, ORDER-1, (2), np.int)
    change_factor = randint(1, Opti_step+1)
    
    #change selected number
    new_M[i,j] += change_factor
    new_loss = loss_func(new_M)[0]
    
    if new_loss < loss or np.random.random() <= lrn:
        return new_M, new_loss
    else:
        new_M[i,j] -= 2*change_factor
        new_loss = loss_func(new_M)[0]
    
    return new_M, new_loss

extend = 20
#range for the initial matrix
ORDER = 3
#order of the square
square_TYPE = np.int
#type of the value in the square

Opti_step = 1
#step taked by the optymizer
lrn = 0
#probapility to keep a bad square

nb_step = 250
generations = 5000

overall_best_M = randint(1, extend, (ORDER, ORDER),square_TYPE)
overall_best_loss = loss_func(overall_best_M)[0]

perfects = []
running_best_loss = []
for i in range(generations):
    
    #Create initial square
    M = randint(1, extend, (ORDER, ORDER),square_TYPE)
    
    #Arrays to store squares and losses
    losses = []
    loss = loss_func(M)[0]
    Ms = []
    
    #Pytorch go brrrrrrr
    for i_step in tqdm(range(int(nb_step))):
        if i_step == 200:
            lrn = 0.1
        
        losses.append(loss)
        Ms.append(M.copy())
        
        #Optimize
        M, loss = Optizer(M, loss)
    
    #Find best of generation
    min_arg = np.argmin(losses)
    min_loss = loss_func(Ms[min_arg])[0]
    
    #Check if overall best
    if min_loss < overall_best_loss:
        overall_best_loss = loss_func(Ms[min_arg])[0]
        overall_best_M = Ms[min_arg]
        
    #Check for perfection
    if min_loss == 0:
        perfects.append(Ms[min_arg])
    
    running_best_loss.append(overall_best_loss)
    print('Generation: ', i)
    """print('\n Best Square: \n',Ms[min_arg], '\n Best sd / mean: ',
          np.round(loss_func(Ms[min_arg]),2), '\n Avg Loss: ', np.mean(losses),
          '\n Iteration: ', min_arg)"""

#Analytics
print('\n Overall Best Square: \n', overall_best_M,
      '\n Overall Best Loss: ', overall_best_loss,
      '\n \n Perfects: \n')

f = open('perfects.txt', 'a')
for n in perfects:
    print(n, '\n')
    f.write(str(n))
    f.write('\n \n')
f.close()

plt.plot(running_best_loss)