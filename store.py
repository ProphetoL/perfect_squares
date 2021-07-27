from numba.core.errors import reset_terminal
import numpy as np
import json
from tqdm import tqdm

from numba import njit


#@njit()
def appToArr(arr=[], toApp=[]):
    """function to append all elements of 'toApp' to arr"""
    if len(arr) == 0:
        return toApp
    else :
        return np.append(arr, toApp, axis=0)

def getPosStore(i, Ms, M_d):
    coo = np.array(np.where(Ms == M_d[i]))
    coo = np.dstack(coo)[0] 
    return coo[0]

#@njit()
def squre_transf(M):
    """function to transforme the square in a regular form"""
    global toStudy
    M -= np.int64(np.sum(M[0])/ORDER)
    #make the sums 0

    M_d = np.sort(M.reshape((ORDER**2)))
    a_0 = getPosStore(0, M, M_d)
    a_1 = getPosStore(1, M, M_d)
        
    #Rotate and flip the square the square
    diff = np.abs(a_1-a_0)
    vectToComp = np.array([[2,1], [1,2]])

    if (diff == vectToComp[0]).all() or (diff == vectToComp[1]).all():#test if the squre as a general proprety
        if a_0[1] == 0:
            M = np.flip(M)

        M = np.rot90(M, a_0[0])

        a_1 = getPosStore(1, M, M_d)
        if a_1[1] == 2:
            M = np.fliplr(M)

    else :
        toStudy = appToArr(toStudy, [M])

    return M, M_d

@njit()
def drawPath(M, M_d):
    global ORDER

    path = np.zeros((ORDER, ORDER), np.int64)
    for i_el in range(len(M_d)):
        el = M_d[i_el]
        coo = np.where(M == el)
        path[coo[0][0]][coo[1][0]] = i_el

    return path



ORDER = 3
toStudy = np.array([])

#import the magic squares
squares_file = open("perfects.json", "r")
Ms = np.array(json.loads(squares_file.read()))
squares_file.close()

Ms = appToArr(
    Ms,
    [[
        [29**2, 1**2, 47**2],
        [41**2, 37**2, 1**2],
        [32**2, 41**2, 29**2]
    ]]
    )

nb_squares = len(Ms)

M_ds = np.array([None for i in range(nb_squares)])

all_path = {}
all_path_cont = {}
#Transforme all the square
for i in tqdm(range(nb_squares)):
    Ms[i], M_ds[i] = squre_transf(Ms[i])
    path = drawPath(Ms[i], M_ds[i])
    path_str = str(path)


    if not path_str in all_path:
        all_path_cont[path_str] = 1
        all_path[path_str] = np.array([Ms[i]])

    else :
        all_path_cont[path_str] += 1
        all_path[path_str] = appToArr(all_path[path_str], [Ms[i]])

        
def print_allPath(dict_):
    for key in list(dict_.keys()):
        print(f"\n{key}: {dict_[key]}", end="\n\n")

print_allPath(all_path_cont)