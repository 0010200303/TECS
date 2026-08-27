Utilities
=========

TECS provides several utility components to support simulation development.

ArgParser
---------

A lightweight argument parser for command-line configuration:

.. code-block:: cpp

  tecs::Arguments args("simulation");
  int n_cells = 100;
  std::string output = "result.h5";

  args.add_argument("-n", "--ncells", n_cells, 100, "Number of agents");
  args.add_argument("-o", "--output", output, "result.h5", "Output path");
  args.add_flag("-v", "--verbose", verbose, "Enable verbose output");

  if (!args.parse(argc, argv))
    return 1;

For more examples check out: :doc:`../examples/rotating-organoid`.

Bounds
------

Axis-aligned bounding box utilities for spatial queries. ``Bounds`` stores the per-dimension minimum and maximum of a set of points and is computed via a single parallel reduction over the agent positions.

.. code-block:: cpp

  auto bounds = utils::get_bounds<Vector>(positions_view);

  Vector min = bounds.min;  // Minimum coordinate in each dimension
  Vector max = bounds.max;  // Maximum coordinate in each dimension

  // Separate passes also available if only one is needed
  Vector min_only = utils::get_min_bounds<Vector>(positions_view);
  Vector max_only = utils::get_max_bounds<Vector>(positions_view);


Grid
----

A spatial acceleration structure for efficient neighbour lookups. Used internally by the ``BinnedAllPairs`` and ``BinnedGabriel`` pair finders, but can also be used directly for custom spatial queries.

.. code-block:: cpp

  tecs::acceleration::Grid<Vector> grid(bounds, bin_size);
  grid.build(positions_view, agent_count);

  // Query neighbours in the same and adjacent bins
  auto neighbours = grid.get_neighbours(i);

For more examples check out: :doc:`../examples/tribolium`.

(Internal) RuntimeGuard
-----------------------

.. warning::

  ``RuntimeGuard`` is used internally by the ``Simulation`` object and should not be used directly unless you really know what you are doing. Manually managing Kokkos initialization when a ``Simulation`` is also active can lead to undefined behaviour.

.. code-block:: cpp

  tecs::detail::RuntimeGuard guard(argc, argv);
  // Kokkos is now initialized and ready for use

