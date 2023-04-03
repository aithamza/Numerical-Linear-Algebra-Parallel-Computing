from mpi4py import MPI
import numpy as np
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

while True:
    if RANK == 0:
        sendbuf = int(input("Enter a number: "))
    else:
        sendbuf = 0
    sendbuf = COMM.bcast(sendbuf, root=0)
    if sendbuf <= 0:
        break
    if RANK == 0:
        print("I am the process 0")
    else:
        print("I am the process {rank}, I received data {data} from 0".format(rank=RANK, data=sendbuf))
