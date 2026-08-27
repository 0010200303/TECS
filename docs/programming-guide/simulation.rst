The Simulation
==============

The ``Simulation`` class is the central component of the framework. It coordinates agent data management, neighbourhood discovery, integration and output generation within a unified interface. In practice, nearly all high-level workflows are driven through ``Simulation``.

Internally, the class manages the configured PairFinder, the selected integration scheme, the selected writer, com fixing and all registered simulation fields. While these components can also be used independently, ``Simulation`` provides the primary abstraction for building and running complete simulations.

Instantiating
-------------

A ``Simulation`` is parameterized through a ``SimulationConfig`` type, which defines the core behaviour and underlying implementation details of the simulation. This includes the ``Scalar`` datatype, number of dimensions, neighbourhood discovery, integration backend and additional compiletime configuration options.

For convenience, TECS provides a simple ``CREATE_SIMULATION_CONFIG``` macro that can be used to configure the simulation as needed: 

.. code-block:: cpp

  CREATE_SIMULATION_CONFIG(MySimulationConfig,
    CONFIG_SCALAR(double)
    CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  );

Additional details about configuration options are available in :doc:`simulation-config`.

Constructing a ``Simulation`` additionally requires several runtime parameters, which are collected in a ``Simulation::Settings`` struct:

- ``agent_count`` - Total number of agents (**required**)
- ``output_path`` - Base path and filename for generated output files (**required**)
- ``capacity`` (default: ``agent_count``) - Allocated capacity, can be larger than agent count
- ``cutoff_distance`` (default: 1'000'000) - Maximum interaction distance used for pairwise evaluations
- ``seed`` (default: 2807) - Seed used for all random number generation
- ``pair_finder_settings`` - Configuration for the pair finder
- ``writer_settings`` - Configuration for the output writer
- ``link_capacity`` (default: 0) - Allocate space for links between agents; ``0`` means no links
- ``link_active_count`` (default: 0) - Initial number of active links (must be ``<= link_capacity``)

.. attention::

  Determinism is not guaranteed across different hardware architectures or compiler implementations, even when using the same seed.

Once the configuration and runtime parameters are chosen, a new ``Simulation`` instance can be created using the ``Settings`` struct:

.. code-block:: cpp

  Simulation<MySimulationConfig> sim({
    .agent_count = 1024,
    .output_path = "./my/path/my_file",
    .cutoff_distance = 5.0,
    .seed = 42
  });

.. attention::
  
  Do **not** include a file extension in ``output_path``. Output files are automatically generated as ``.h5`` and ``.xmf`` files. 

Initialization
--------------

``Simulation`` provides multiple built-in initialization routines for generating initial agent configurations, including random spheres, cuboids, hexagons and other common geometries.
Initialization routines can be invoked directly on the simulation instance:

.. code-block:: cpp

  sim.init_random_filled_sphere(radius);

In many cases initialization requires more than just assigning positions. To support this, TECS provides the ``INIT_FUNC`` macro for defining custom initialization kernels that directly modify simulation fields immediately after position generation.

.. code-block:: cpp

  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  auto initial_conditions = INIT_FUNC() {
    polarities_view(i) = Polarity(positions_view(i));
  };
  sim.init_random_filled_sphere(radius, initial_conditions);

Further details on ``INIT_FUNC`` and field access patterns are available in :doc:`forces`.

For fully custom initialization workflows, ``Simulation`` also provides a generic ``init`` function that accepts arbitrary initialization kernels directly.

Taking a step
-------------

A simulation step is advanced through the ``take_step`` function by supplying the delta time together with all active forces for that iteration:

.. code-block::

  sim.take_step(dt, generic_force, pairwise_force_1, pairwise_force_2);

.. note::

  Supplying all forces in a single ``take_step`` call is recommended, as it allows the framework to apply internal optimizations.

Writing
-------

Simulation output is written through the ``write`` function. When called without arguments, the framework automatically handles data conversion, file generation and timestep bookkeeping. You can also supply a custom time or pass additional data views to include in the output:

.. code-block:: cpp

  sim.write();
  sim.write(dt * i, custom_view);

Static fields that are not updated by the integrator - such as masses or agent types - can be written once using ``write_static``. This should be called before the first ``write()`` call:

.. code-block:: cpp

  sim.write_static(masses, types);
