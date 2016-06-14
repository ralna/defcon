from dolfin import *

def empty_vector(model):
  # should be able to do return Vector(model.size()) but it doesn't work in parallel
  # dolfin's Vector API is terrible.
  b = Vector(model)
  b.zero()
  return b

def prevent_MPI_output():
    # prevent output from rank > 0 processes
    if MPI.rank(mpi_comm_world()) > 0:
        info_blue("Turning off output from rank > 0 processes...")
        sys.stdout = open("/dev/null", "w")
        # and handle C++ as well
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        stdout = libc.fdopen(1, "w")
        libc.freopen("/dev/null", "w", stdout)

