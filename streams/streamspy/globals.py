from mpi4py import rc
rc.initialize = False
rc.finalize = False
from mpi4py import MPI

def init():
    print("initializing globals")
    global is_master, rank, comm

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_master = rank == 0;
