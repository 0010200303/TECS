Initializers
============

Initializers define the initial configuration of agents in a simulation. They generate positions and optionally set up additional fields.

TECS provides several built-in initializers for common geometries, accessible through the ``Simulation`` instance:

.. code-block:: cpp

  sim.init_random_filled_sphere(radius);
  sim.init_relaxed_sphere(radius);
  sim.init_random_hollow_sphere(radius);
  sim.init_random_cuboid(min, max);
  sim.init_relaxed_cuboid(min, max);
  sim.init_regular_hexagon(distance_to_neighbour);
  sim.init_regular_rectangle(distance_to_neighbour, nx);
  sim.init_random_disk(distance_to_neighbour);
  sim.init_line(distance);

Each initializer is also available as a standalone functor in the ``tecs::initializers`` namespace, allowing direct use outside the ``Simulation`` class.

Custom Initialization
---------------------

In many cases initialization requires more than just position assignment. TECS supports custom field setup by passing ``INIT_FUNC`` lambdas as additional arguments to any built-in initializer:

.. code-block:: cpp

  auto my_init = INIT_FUNC(
    GENERIC_REF(Polarity, polarity)
  ) {
    // Access current agent index via `i`
    polarity.self = Polarity(positions_view(i));
  };
  sim.init_random_filled_sphere(radius, my_init);

The init function is executed immediately after position generation for each agent, allowing you to set up field values based on the generated position.

Multiple init functions can be chained:

.. code-block:: cpp

  sim.init_random_cuboid(min, max, my_init_1, my_init_2);

Available Initializers
----------------------

Random Filled Sphere
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RandomFilledSphere
   :project: TECS
   :members:
   :undoc-members:

Random Hollow Sphere
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RandomHollowSphere
   :project: TECS
   :members:
   :undoc-members:

Relaxed Sphere
^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RelaxedSphere
   :project: TECS
   :members:
   :undoc-members:

Random Cuboid
^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RandomCuboid
   :project: TECS
   :members:
   :undoc-members:

Relaxed Cuboid
^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RelaxedCuboid
   :project: TECS
   :members:
   :undoc-members:

Regular Hexagon
^^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RegularHexagon
   :project: TECS
   :members:
   :undoc-members:

Regular Rectangle
^^^^^^^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RegularRectangle
   :project: TECS
   :members:
   :undoc-members:

Random Disk
^^^^^^^^^^^

.. doxygenstruct:: tecs::initializers::RandomDisk
   :project: TECS
   :members:
   :undoc-members:

Line
^^^^

.. doxygenstruct:: tecs::initializers::Line
   :project: TECS
   :members:
   :undoc-members:
