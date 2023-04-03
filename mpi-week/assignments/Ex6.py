from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 840  # number of terms in the series
pi = 0.0


# Each process computes a part of the sum
                    
for i in range(rank, N, size):
    x = (i + 0.5) / (N)
    pi += 1.0 / (1.0 + x**2) 

# Sum up all results
sum_pi = comm.reduce(pi, op=MPI.SUM, root=0)

# Print the result in the root process
if rank == 0:
    pi_estimated = 4.0 * sum_pi / N
    print(f"Estimated pi: {pi_estimated:.10f}")
                                
# MPI.Finalize()
