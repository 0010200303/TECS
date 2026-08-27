Types
=====

TECS provides a set of core types used throughout the framework. These types are parameterized by the simulation configuration's scalar type and dimensions.

Vector
------

The fundamental spatial data type. ``VectorN`` is an :math:`n`-dimensional vector with component-wise operations.

.. code-block:: cpp

  Vector position = {1.0, 2.0, 3.0};
  Vector direction = pos.normalized();
  Scalar speed = direction.dot({0.0, 1.0, 0.0});

Polarity
--------

Represents a cell's polarisation direction using spherical coordinates :math:`(\theta, \phi)`. ``Polarity`` inherits from ``VectorN<Scalar, 2>`` and stores the polar angle :math:`\theta` (from the :math:`z`-axis) and the azimuthal angle :math:`\phi` (in the :math:`xy`-plane).

.. code-block:: cpp

  Polarity p = {0.5, 1.2};                 // theta = 0.5, phi = 1.2
  Vector3<Scalar> v = p.to_vector3();       // Convert to Cartesian unit vector

  Polarity q = {0.8, -0.3};                // theta = 0.8, phi = -0.3
  Scalar alignment = p.dot(q);              // Cosine of angle between them

  Polarity force = p.bidirectional_polarization_force(q);

Polarity is used throughout TECS examples to model orientation-dependent forces, including polarisation, bending, apical constriction, and migration.

View
----

A wrapper around ``Kokkos::DualView`` that provides a simplified interface for managing data on both host and device. ``View`` handles synchronization, resizing, and memory management.

Views track which side (host or device) last modified the data. Four access methods are available:

* ``operator(i)`` - read and write access that sets the modified flag on the active side (host or device).
* ``write(i)`` - explicit write that also sets the modified flag; equivalent to ``operator(i)``.
* ``read(i)`` - read-only access that **never** sets the modified flag. Use this inside force or update kernels where only the force output view needs write tracking, avoiding unnecessary device-to-host syncs.
* ``access(i)`` - raw access that **never** sets the modified flag, on either host or device. Unlike ``read()``, which also avoids the flag, ``access()`` is intended for manual sync scenarios where you handle synchronization explicitly and want zero flag overhead.

.. code-block:: cpp

  auto positions_view = sim.get_view<FIELD(Vector, position)>();
  positions_view(0) = { 0.0, 0.0, 0.0 }; // Easy host access
  positions_view.auto_sync();            // Automatically sync in the correct direction

  // Easy device access - read() avoids flag overhead for input data
  auto force = UPDATE_FORCE(
    Vector pos = positions_view.read(i);  // Read without setting modified flag
  );

DeviceVar
---------

A single value accessible from both host and device without explicit synchronization. ``DeviceVar`` wraps a ``Kokkos::View<T>`` of size 1 and handles all synchronization automatically. It is very useful for implementing counters (e.g. in proliferation and links).

.. code-block:: cpp

  DeviceVar<int> counter = sim.get_agent_count();

  auto proliferate = UPDATE_FUNC(
    int new_idx = Kokkos::atomic_fetch_add(counter.data(), 1);
    positions(new_idx) = positions_view(i);
  );

  // On the host: reads back automatically
  sim.set_agent_count(counter);

For more examples check out: :doc:`../examples/passive-growth`.

Link
----

Represents a connection between two agents storing their indices (:cpp:member:`a` and :cpp:member:`b`). Links are useful for modelling protrusions and lineage tracking. They are created and updated inside ``UPDATE_FUNC`` and the corresponding forces are evaluated using ``LINK_FORCE``.

.. code-block:: cpp

  // Create random links inside an UPDATE_FUNC
  auto update_protrusions = UPDATE_FUNC(
    Link link = {rng.urand(agent_count), rng.urand(agent_count)};
    links(i) = link;
  );

  // Evaluate spring forces between linked pairs
  auto prot_forces = LINK_FORCE(
    Vector displacement = ctx.position.b - ctx.position.a;
    Scalar distance = displacement.length();
    ctx.position.delta_a += strength * displacement / distance;
    ctx.position.delta_b -= strength * displacement / distance;
  );

The example above shows the typical link lifecycle: links are first populated in an ``UPDATE_FUNC``, then a ``LINK_FORCE`` computes pairwise interactions between the linked agent pairs. Unlike ``PAIRWISE_FORCE`` which checks all agents, ``LINK_FORCE`` only evaluates forces for explicitly defined connections, making it efficient for sparse interactions like protrusions.

For more examples check out: :doc:`../examples/sorting-by-protrusions` or :doc:`../examples/lineage-tracing`.

Plane
-----

An :math:`n`-dimensional plane defined by a normal vector and an offset.

For more examples check out: :doc:`../examples/growth-with-wall` or :doc:`../examples/growth-with-squeeze`.
