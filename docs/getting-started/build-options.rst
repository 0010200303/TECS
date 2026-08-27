Build Options
=============

The ``tecs.sh`` build script provides several command-line options for customizing the build process.

Usage
-----

.. code-block:: bash

  ./tecs.sh <source_file> [options]

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Option
     - Values
     - Description
   * - ``-o``, ``--output``
     - Filename
     - Specify the name of the compiled executable (default: ``tecs``)
   * - ``-e``, ``--execute``
     - (flag)
     - Automatically run the executable after compilation
   * - ``-B``, ``--backend``
     - ``SERIAL``, ``OPENMP``, ``THREADS``, ``CUDA``, ``HIP``, ``SYCL``, ``OPENMPTARGET``, ``OPENACC``
     - Select the Kokkos backend for parallel execution
   * - ``-t``, ``--test``
     - (flag)
     - Build and run the test suite

Examples
--------

.. code-block:: bash

  # Build with CUDA backend and run
  ./tecs.sh main.cpp -B CUDA -e

  # Build with OpenMP backend
  ./tecs.sh main.cpp -B OPENMP -o my_sim

  # Run tests
  ./tecs.sh main.cpp -t

Kokkos Compilation
------------------

When a source file is compiled using the build script, the framework automatically includes the necessary Kokkos headers and links against the appropriate libraries. The ``-B`` option selects which Kokkos execution space to use.

For more advanced configuration, see the :doc:`requirements` page.
