from mpi4py import MPI

#Initialize MPI environment
comm = MPI.COMM_WORLD

#Get the total number of processes
world_size = comm.Get_size()

#Get the rank of the current process 
rank = comm.Get_rank()

#print "Hello World " message from each proce
print(f"Hello World from process {rank} of {world_size}")

### Q3 

if rank == 0:
    print("***"*10,"Q4","***"*10)
    print(f"I am the process {rank} of {world_size}")


