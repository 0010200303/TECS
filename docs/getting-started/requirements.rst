Requirements
============

TECS is designed with portability in mind and can be built across a wide range of hardware and software environments, including modern GPUs and even CPUs.

To ensure a consistent build and runtime environment, the following minimum requirements must be satisfied:

- **C++ Standard**: A compiler with support for C++20 or newer
- **CMake**: Version 4.0 or higher
- **Kokkos**: Version 5.1 or higher (automatically fetched if not available on the system)
- **HighFive**: Version 3.3 or higher (C++ interface for HDF5; automatically fetched if not available)
- **HDF5**: Required system library (on Debian/Ubuntu systems, install via ``apt install libhdf5-dev``)
- **Python**: Version 3.7 or higher (required by the build script)

For supported compilers and build systems check the `Kokkos requirements <https://kokkos.org/kokkos-core-wiki/get-started/requirements.html>`__.
