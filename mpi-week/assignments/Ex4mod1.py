'''from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# define matrix dimensions
n = 8
m = 8

# create the matrix on process 0
if rank == 0:
    matrix = np.arange(n*m, dtype='float64').reshape(n, m)
    print("Original matrix on processor 0:")
    print(matrix)
else:
    matrix = None
# divide the matrix into parts and scatter to the other processes
sendcounts = [n//2 * m//2, n//2 * m//2, n//2 * m//2]
displs = [0, n//2 * m//2, 3*n//4 * m//2]
submatrix = np.empty((n//2, m//2), dtype='float64')
comm.Scatterv([matrix, sendcounts, displs, MPI.DOUBLE], submatrix, root=0)

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Create a 4x4 matrix
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print("Original matrix on processor 0:")
    print(matrix)

    # Define how many rows each processor should receive
    rows_per_proc = [2, 1, 1]

    # Define the displacements for each processor
    displacements = [0, 8, 12]

else:
    # Initialize empty matrix
    matrix = None
# Define rows_per_proc and displacements for non-zero processors
    rows_per_proc = None
    displacements = None
# Scatter parts of the matrix to different processors
d1_local = np.zeros(3)
rows = comm.Scatterv([matrix, rows_per_proc, displacements, MPI.INT],d1_local, root=0)
print(f"Received matrix on processor {rank}:")
print(rows)
'''
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n = 8
m = 8
if rank == 0:
    n = 8
    m = 8
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i, j] = i * m + j + 1
    print("Original matrix on processor 0:")
    print(A)
# Divide the matrix into parts to send to each processor
    sendcounts = np.zeros(size, dtype=int)
    displs = np.zeros(size, dtype=int)
    sendcounts[1] = (n // 2) * (m - m // 2)
    sendcounts[2] = (n - n // 2) * (m // 2)
    sendcounts[3] = (n - n // 2) * (m - m // 2)
    displs[1] = (n // 2)
    displs[2] = (m //2) 
    displs[3] = (n - n // 2) * (m - m // 2)
else:
    A = None
    sendcounts = None
    displs = None
# Scatter the matrix parts to each processor
recvA = np.zeros((n // 2, m // 2))
recvcounts = (n // 2) * (m // 2)
print(np.transpose(A))
comm.Scatterv([np.transpose(A) , sendcounts, displs, MPI.DOUBLE], recvA, root=0)
if rank == 1:
    print("Received matrix on processor 1:")
    print(recvA)
elif rank == 2:
    print("Received matrix on processor 2:")
    print(recvA)
elif rank == 3:
    print("Received matrix on processor 2:")
    print(recvA)
