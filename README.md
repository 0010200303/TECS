**TECS** **E**xecutes **C**ell **S**imulations

docs at: https://0010200303.github.io/TECS/

# Requirements
- CMake and compiler with support for C++20
- HDF5 (libhdf5-dev)
- python3 for build script

## optional
- Kokkos (pulled automatically if not installed)

# TODO
- use Kokkos::ViewAllocateWithoutInitializing
- implement Bowyer-Watson for delaunay pair finder
- implement approximate gabriel pair finder (maybe use cones)

# known issues
- older hdf5 versions cause small a memory leak when using the HDF5_writer on CPU (very observable when running branching on CPU)
