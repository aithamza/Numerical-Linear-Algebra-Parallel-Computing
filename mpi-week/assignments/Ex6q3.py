from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 848 # number of terms in the series
a = rank * N // size
b = (rank + 1) * N // size

pi_partial = 0.0

# Each process computes its part of the sum
for i in range(a, b):
    x = (i + 0.5) / N
    pi_partial += 1.0 / (1.0 + x * x)

# Send partial sum to controller
pi_sum = np.array(pi_partial, dtype='d')
pi_total = np.zeros(1, dtype='d')
comm.Reduce(pi_sum, pi_total, op=MPI.SUM, root=0)

# Controller adds up all the partial sums
if rank == 0:
    pi_total *= 4.0 / N
    print(f"Computed pi: {pi_total[0]:.10f}")

