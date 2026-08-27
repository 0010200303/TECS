#ifndef TECS_INITIALIZERS_CUBOID_INIT_HPP
#define TECS_INITIALIZERS_CUBOID_INIT_HPP

#include <numbers>
#include <utility>

#include <Kokkos_Core.hpp>

#include "../utils/utils.hpp"
#include "relax_force.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents at uniformly random positions inside a cuboid.
   *
   * Each agent gets a position uniformly sampled from the box
   * @f$ [\text{min}, \text{max}] @f$.
   */
  template<typename SimulationConfig>
  struct RandomCuboid {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    /// Lower corner of the cuboid.
    Vector min;
    /// The extent (width, height, depth) of the cuboid.
    Vector cuboid_dimensions;

    /**
     * @param positions View to fill with positions.
     * @param minimum   Lower corner of the cuboid.
     * @param maximum   Upper corner of the cuboid.
     */
    template<typename ViewType>
    RandomCuboid(
      ViewType positions,
      const Vector& minimum,
      const Vector& maximum
    ) : positions_view(positions)
      , min(minimum)
      , cuboid_dimensions(maximum - minimum) { }

    /// @brief Sample a random position inside the cuboid for agent @p i.
    INIT_OP {
      for (int j = 0; j < dimensions; ++j)
        positions_view(i)[j] = min[j] + generator.drand(0.0, cuboid_dimensions[j]);
    }
  };

  /**
   * @brief Like RandomCuboid, followed by a relaxation step to reduce overlaps.
   *
   * After placing agents randomly, runs several time steps with a repulsive
   * force to spread them out into a more uniform distribution.
   */
  template<typename SimulationConfig>
  struct RelaxedCuboid : public RandomCuboid<SimulationConfig> {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    
    /// Number of relaxation steps.
    unsigned int steps;

    /**
     * @param positions          View to fill with positions.
     * @param minimum            Lower corner of the cuboid.
     * @param maximum            Upper corner of the cuboid.
     * @param relaxation_steps   How many relaxation steps to run (default: 2000).
     */
    template<typename ViewType>
    RelaxedCuboid(
      ViewType positions,
      const Vector& minimum,
      const Vector& maximum,
      const unsigned int relaxation_steps = 2000
    ) : RandomCuboid<SimulationConfig>(positions, minimum, maximum)
      , steps(relaxation_steps) { }

    /// @brief Run the relaxation simulation to spread agents evenly.
    template<typename Simulation>
    void relax(Simulation& simulation) {
      for (int i = 0; i < steps; ++i)
        simulation.take_step(0.1, details::RelaxForce<SimulationConfig>(0.8));
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_CUBOID_INIT_HPP
