#
# Mostly harvested from
# https://github.com/Fluid-Dynamics-Group/selective-modification/blob/763f1b6369a851b374bb918270ac8a80f72f5738/solver/io_utils.py
#
import h5py
from mpi4py import rc
rc.initialize = False
rc.finalize = False
from mpi4py import MPI
import numpy as np
from typing import Tuple, Optional, List, Any
from abc import ABC, abstractmethod

class Group():
    def __init__(self, group: h5py.Group, rank: int):
        self.group = group
        self.rank = rank

    def write_attr(self, attr: str, value: Any):
        self.group[attr] = value

# Write a h5 file using MPI
#
# for a numpy array of shape = (3, nx, ny, nz), with num_proc number of mpi processesses,
# this class will write a giant array to disk in a 6 dimensional array in the shape of 
# (num_writes, 3, nx, ny, nz)
# class IoFile:
#     # params
#     # filename: the path to the .h5 file that you wish to write to
#     # num_writes: the total number of times that you will call `write_array`
#     def __init__(self, filename: str):
#         comm = MPI.COMM_WORLD
#         self.rank = comm.rank
# 
#         self.file = h5py.File(filename, 'w', driver='mpio', comm = MPI.COMM_WORLD)
class IoFile:
    def __init__(self, filename: str, mode: str = 'w', comm=None, libver: Optional[str] = 'latest'):
        self.rank = 0
        if comm is None:
            # Serial HDF5 open. Using modern file format helps reduce metadata overhead.
            self.file = h5py.File(filename, mode, libver=libver)
        else:
            # Parallel HDF5 open. This is collective across 'comm'.
            self.rank = comm.Get_rank() if hasattr(comm, "Get_rank") else MPI.COMM_WORLD.Get_rank()
            self.file = h5py.File(filename, mode, driver='mpio', comm=comm)

    def create_group(self, name: str) -> Group:
        group = self.file.create_group(name)
        return Group(group, self.rank)

    # close the underlying h5 file handle
    def close(self):
        self.file.close()

class ExportDataset(ABC):
    @abstractmethod
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        self._step_number = 0
        self._num_writes = num_writes
        self._dim = len(shape)

        self._name = name

    @abstractmethod
    def write_array(self, array: np.ndarray):
        pass

    def step_number(self) -> int:
        return self._step_number

    def inc_step_number(self):
        self._step_number += 1

    def check_can_write(self, array: np.ndarray) -> bool:
        if self._step_number >= self._num_writes:
            import warnings
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(f"attempted to write {self._step_number+1} values to {self._name} dataset h5 file, when constructed for {self._num_writes} writes - skipping this write", RuntimeWarning)
            return False
        
        if self._dim != len(array.shape):
            raise ValueError(f"reported dimension ({self._dim}) in initialization is different from argument `array` {len(array.shape)}")

        return True
        
    # Attempted solution for h5 error (Post evaluation loop)
    def close(self):
        """Release the underlying HDF5 dataset handle if present."""
        try:
            self.dset.id.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    # End of attempted solution for h5 error (Post evaluation loop)


# Used when exporting a vector field HDF5 file with the MPI split along exclusively the x-axis
#
# input arrays to `.write_array()` should be dimension 4 (<vector component>, x, ,y, z)
# and the output arrays from this class will be dimension 5 (<timestep write>, <vector component>, x, y, z)
class VectorField3D(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes, *shape))
        self.rank = rank

        assert(len(shape) == 4)

        # initialize base class so we can use their stepper functionality
        super(VectorField3D, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        # the input array has the form
        # [
        #       <u,v,w,etc>
        #       x position <------ IDX 1
        #       y position
        #       z position
        # ]
        # so we should only write to the slice of the HDF5 file that contains
        # our data range in this area
        MPI_SPLIT_IDX = 1

        split_size = np.size(array,MPI_SPLIT_IDX)
        start_slice = split_size * self.rank
        end_slice = start_slice + split_size

        # (write step number, vector components ex: (u,v,w,), x data, y data, z data)
        self.dset[self.step_number(), :, start_slice:end_slice, :, :] = array

        self.inc_step_number()

# Used when exporting a 2D vector field HDF5 file with the MPI split along exclusively the x-axis
#
# input arrays to `.write_array()` should be dimension 3 (<vector component>, x, ,y)
# and the output arrays from this class will be dimension 4 (<timestep write>, <vector component>, x, y)
class VectorFieldXY2D(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes, *shape))
        self.rank = rank

        assert(len(shape) == 3)

        # initialize base class so we can use their stepper functionality
        super(VectorFieldXY2D, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        # the input array has the form
        # [
        #       <u,v,w,etc>
        #       x position <------ IDX 1
        #       y position
        # ]
        # so we should only write to the slice of the HDF5 file that contains
        # our data range in this area
        MPI_SPLIT_IDX = 1

        split_size = np.size(array,MPI_SPLIT_IDX)
        start_slice = split_size * self.rank
        end_slice = start_slice + split_size

        # (write step number, vector components ex: (u,v,w,), x data, y data, )
        self.dset[self.step_number(), :, start_slice:end_slice, :] = array

        self.inc_step_number()

# Used when exporting a 1D scalar field HDF5 file with the MPI split along exclusively the x-axis
#
# input arrays to `.write_array()` should be dimension 3 (<vector component>, x, ,y)
# and the output arrays from this class will be dimension 4 (<timestep write>, <vector component>, x, y)
class ScalarFieldX1D(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes, *shape))
        self.rank = rank

        assert(len(shape) == 1)

        # initialize base class so we can use their stepper functionality
        super(ScalarFieldX1D, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        # the input array has the form
        # [
        #       x <------ IDX 0
        # ]
        # so we should only write to the slice of the HDF5 file that contains
        # our data range in this area
        MPI_SPLIT_IDX = 0

        split_size = np.size(array,MPI_SPLIT_IDX)
        start_slice = split_size * self.rank
        end_slice = start_slice + split_size

        # (write step number, vector components ex: (u,v,w,), x data, y data, )
        self.dset[self.step_number(), start_slice:end_slice] = array

        self.inc_step_number()

# Used when exporting a scalar to an HDF5 file with no MPI x split
#
# input arrays to `.write_array()` should be dimension 1 (ideally dim 0, but the interface requires an array) (<value>)
# and the output arrays from this class will be dimension 1 (<value at each time step>)
class Scalar0D(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes))
        self.rank = rank

        assert(len(shape) == 1)

        # initialize base class so we can use their stepper functionality
        super(Scalar0D, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        # write the single scalar value to the output
        self.dset[self.step_number()] = array

        self.inc_step_number()

# Used when exporting a 1D scalar vector to an HDF5 file with an mpi split along the x axis
#
# input arrays to `.write_array()` should be dimension 1 (<value list>)
# and the output arrays from this class will be dimension 2 (<timestep write>, <value list>)
class Scalar1DX(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes, *shape))
        self.rank = rank

        assert(len(shape) == 1)

        # initialize base class so we can use their stepper functionality
        super(Scalar1DX, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        MPI_SPLIT_IDX = 0
        split_size = np.size(array,MPI_SPLIT_IDX)
        start_slice = split_size * self.rank
        end_slice = start_slice + split_size

        # write the single scalar value to the output
        self.dset[self.step_number(), start_slice:end_slice] = array

        self.inc_step_number()

# Used when exporting a 1D scalar vector to an HDF5 file with no mpi splits
#
# input arrays to `.write_array()` should be dimension 1 (<value list>)
# and the output arrays from this class will be dimension 2 (<timestep write>, <value list>)
class Scalar1D(ExportDataset):
    # the shape here must be for the overall field, not just the data contained on this process
    def __init__(self, file: IoFile, shape: List[int], num_writes: int, name: str, rank: int):
        pass
        self.dset = file.file.create_dataset(name, (num_writes, *shape))
        self.rank = rank

        assert(len(shape) == 1)

        # initialize base class so we can use their stepper functionality
        super(Scalar1D, self).__init__(file, shape, num_writes, name, rank)

    def write_array(self, array: np.ndarray):
        if not self.check_can_write(array):
            return None

        # write the the list of values for the current timestep
        self.dset[self.step_number(), :] = array

        self.inc_step_number()
