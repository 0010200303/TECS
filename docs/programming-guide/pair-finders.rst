Pair Finders
============

Pair finders define the neighbourhood structure of the simulation by determining which agents interact with each other during force evaluation. Different pair finders implement different neighbourhood models, each with distinct computational performance characteristics.

TECS provides several pair finder implementations with different performance, from simple all-pairs approaches suitable for small systems to spatially-binned methods that scale to large simulations.

Selecting a Pair Finder
-----------------------

Pair finders are configured at compile-time in the simulation config:

.. code-block:: cpp

  CREATE_SIMULATION_CONFIG(MySimulationConfig,
    CONFIG_PAIR_FINDER(pair_finders::BinnedGabriel)
  )

Naive All Pairs
---------------

The simplest pair finder: every agent is checked against every other agent. This is the most general option and works well for small systems.

Binned All Pairs
----------------

Uses a spatial grid to bin agents, then evaluates all pairs within each bin. This can significantly reduce the number of pair evaluations compared to the naive :math:`O(n^2)` approach, especially for large, sparse systems.

Naive Delaunay
--------------

The Delaunay triangulation connects agents such that no other agent lies inside the circumcircle of the triangle formed by any pair of neighbours. This produces a well-defined neighbourhood structure suitable for epithelial tissues. Currently this is only implemented for 2D and 3D.

Naive Gabriel
-------------

The Gabriel graph restricts interactions to agents whose shared neighbourhood contains no other agents. This produces a well-defined neighbourhood structure suitable for cell-cell adhesion and repulsion.

Binned Gabriel
--------------

Combines spatial binning with the Gabriel graph criterion for efficient neighbourhood computation.

Settings
--------

Some pair finders expose configurable settings:

.. code-block:: cpp

  // only rebuild grid every 20 steps (for binned pair finders)
  PairFinder::Settings pair_finder_settings;
  pair_finder_settings.rebuild_every_n = 20;

These can be specialized to tune performance for specific simulation conditions.
