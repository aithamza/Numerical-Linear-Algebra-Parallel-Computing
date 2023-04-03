#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:20:22 2020

@author: kissami
"""
import numpy as np
import scipy
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI


''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)

def matrixVectorMult(A, b, x, size):
    
    row, col = A.shape
    start = RANK * (row // size)
    end = start + (row // size)
   #x = np.zeros(col)
    for i in range(start, end):
       # a = A[i]
        for j in range(col):
            x[i] += A[i,j] * b[j]

    return x

########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000
#Local_size = 

# counts = block of each proc
#counts = 

if RANK == 0:
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
else :
    A = None
    b = None


#########Send b to all procs and scatter A (each proc has its own local matrix#####
LocalMatrix = lil_matrix((SIZE, SIZE))
# Scatter the matrix A
#COMM.Scatter(A, LocalMatrix, root=0)
#####################Compute A*b locally#######################################
b = np.zeros(SIZE)
Localb = np.zeros(SIZE) 
#LocalX = np.zeros(SIZE)
#COMM.Scatter(b, Localb, root=0)

LocalX = np.zeros(SIZE)
LocalX = matrixVectorMult(LocalMatrix, b, LocalX, size)
if RANK == 0:
    X = np.zeros(SIZE)

#recvcounts = np.full(4, SIZE // 4)
#displs = np.arange(4) * (SIZE // 4)
#COMM.Gatherv(LocalX, [X, recvcounts, displs, MPI.DOUBLE], root = 0)
# start = MPI.Wtime()
# matrixVectorMult(LocalMatrix, b, LocalX)
# stop = MPI.Wtime()

if RANK == 0:
    start = MPI.Wtime()
    matrixVectorMult(LocalMatrix, b, LocalX, size)
    stop = MPI.Wtime()
    print("CPU time of parallel multiplication is ", (stop - start))

##################Gather te results ###########################################
# sendcouns = local size of result
#sendcounts = 
if RANK == 0:
    X = matrixVectorMult(A, b, LocalX, size) 
else :
    X = None

# Gather the result into X


##################Print the results ###########################################

if RANK == 0 :
    start = MPI.Wtime()
    X_ = A.dot(b)
    stop = MPI.Wtime()
    print("The time to calculate A*b using dot is :", stop - start)
    # print("The result of A*b using parallel version is :", X)
'''if RANK == 0:
    import matplotlib.pyplot as plt
    plt.plot(X)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Parallel Matrix-Vector Multiplication')
    plt.show()'''
