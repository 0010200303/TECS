I/O
===

TECS provides output writing and input reading capabilities through its I/O module.

Writers
-------

Writers are responsible for saving simulation state at each output interval.

Dummy Writer
^^^^^^^^^^^^

A no-op writer that discards all output. Useful for benchmarking or when only final state matters.

.. code-block:: cpp

  CREATE_SIMULATION_CONFIG(MySimulationConfig,
    CONFIG_WRITER(io::Dummy)
  )

HDF5 Writer
^^^^^^^^^^^

The default writer stores simulation data in the HDF5 format, enabling efficient storage and compatibility with analysis tools like ParaView and Python though the h5py package.

Output files are automatically generated with ``.h5`` and ``.xmf`` extensions. The ``.xmf`` file enables **ParaView** visualization using the XDMF model.

.. code-block:: cpp

  CREATE_SIMULATION_CONFIG(MySimulationConfig,
    CONFIG_WRITER(io::HDF5_Writer)
  )

Readers
-------

HDF5 Reader
^^^^^^^^^^^

Reads simulation data from HDF5 files produced by the HDF5 Writer:

.. code-block:: cpp

  io::HDF5_Reader positions_reader(input_path);
  auto positions_view = sim.get_view<FIELD(Vector, position)>();
  serosa_reader.read_dataset("POINTS", positions_view);

Visualization
-------------

TECS output files (``.h5`` + ``.xmf``) can be visualized using **ParaView** by loading the ``.xmf`` file. Both files must be in the same directory - the ``.xmf`` file references the ``.h5`` data by relative path.

This workflow enables interactive exploration of simulation results.

For full API details, see the :doc:`../api/io` reference.
