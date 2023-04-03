from mpi4py import MPI
import numpy as np
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
size = COMM.Get_size()

while True:
    if RANK == 0:
        x = int(input("Put a number: "))
        COMM.send(x, RANK +1)
    else:
        x = COMM.recv(source = RANK-1)
        print("processus recepteur est : ",RANK)
        if RANK < size -1 :
            if x <0 : x-= RANK
            COMM.send(x + RANK, RANK +1)
    if x < 0 :
        break
    print("rank : ", RANK, "data ",x)
                                                                                    
