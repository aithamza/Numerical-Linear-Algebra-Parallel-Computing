from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 840  # number of terms in the series
'''if rank == 0:
    start = 0
    end = N // size
else:
    start = N // 2
    end = N
'''
start = int(rank * N /size)
end = int((rank + 1)* N / size)

partial_sum = 0.0
for i in range(start, end):
    x = (i + 0.5) / N
    partial_sum += 1.0 / (1.0 + x * x)

print(f"Rank {rank} partial sum: {partial_sum:.10f}")

total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum: {total_sum:.10f}")
