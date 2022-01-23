# from mpi4py import MPI
# from trapezoid import Trap
#
# comm = MPI.COMM_WORLD
# print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
#
# size = comm.Get_size()
# rank = comm.Get_rank()
# if rank == 0:
#     msg = "Hello, world"
#     comm.send(msg, dest=1)
#     comm.send("hello world", dest=2)
# elif rank == 1:
#     s = comm.recv()
#     print("rank %d: %s" % (rank, s))
# elif rank == 2:
#     s = comm.recv()
#     print("rank %d: %s" % (rank, s))
#
# if rank != 0:
#     message = "Hello from " + str(rank)
#     comm.send(message, dest=0)
# else:
#     for procid in range(1, size):
#         message = comm.recv(source=procid)
#         print("process 0 receives message from process", procid, ":", message)
#
# a = 0.0
# b = 1.0
# n = 1024
# dest = 0
# total = -1.0
#
# h = (b-a)/n
# local_n = n/size
#
# local_a = a + rank * local_n * h
# local_b = local_a + local_n * h
# integral = Trap(local_a, local_b, local_n, h)
#
# if rank == 0:
#     total = integral
#     for source in range(1, size):
#         integral = comm.recv(source=source)
#         print("PE ", rank, "<-", source, ",", integral, "\n")
#         total = total + integral
# else:
#     print("PE", rank, "->", dest, ",", integral, "\n")
#     comm.send(integral, dest=0)
#
# if rank == 0:
#     print("With n=", n, ", trapezoids, \n")
#     print("integral from", a, "to", b, "=", total, "\n")
#
# MPI.Finalize

import os


for filename in os.listdir('test_images/apples'):
    f = os.path.join('test_images/apples', filename)

    if os.path.isfile(f):
        print(f)