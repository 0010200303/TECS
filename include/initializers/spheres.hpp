#ifndef TECS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP
#define TECS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP

#include <Kokkos_MathematicalConstants.hpp>

#include "../utils/utils.hpp"
#include "relax_force.hpp"

namespace tecs::initializers {

  /**
   * Random positions on the surface of a sphere (min 3D).
   */
  template<typename SimulationConfig>
  struct RandomHollowSphere {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    static_assert(dimensions >= 3, "RandomHollowSphere requires at least 3-dimensional vectors");

    /// Radius of the sphere.
    Scalar radius;
    VectorView positions_view;

    /**
     * @param positions View to fill with positions.
     * @param radius_   Radius of the sphere.
     */
    RandomHollowSphere(VectorView positions, Scalar radius_)
      : positions_view(positions), radius(radius_) { }

    /// @brief Sample a random point on the sphere surface for agent @p i.
    INIT_OP {
      const Scalar u = static_cast<Scalar>(generator.drand());
      const Scalar v = static_cast<Scalar>(generator.drand());

      const Scalar z = Scalar(2.0) * u - Scalar(1.0);
      const Scalar theta = Scalar(2.0) * Scalar(Kokkos::numbers::pi_v<Scalar>) * v;
      const Scalar rxy = Kokkos::sqrt(Scalar(1.0) - z * z) * radius;

      positions_view(i)[0] = rxy * Kokkos::cos(theta);
      positions_view(i)[1] = rxy * Kokkos::sin(theta);
      positions_view(i)[2] = z * radius;
    }
  };

  /**
   * Random positions inside a solid sphere (min 3D).
   */
  template<typename SimulationConfig>
  struct RandomFilledSphere {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    static_assert(dimensions >= 3, "RandomFilledSphere requires at least 3-dimensional vectors");

    /// Radius of the sphere.
    Scalar radius;
    VectorView positions_view;

    /**
     * @param positions View to fill with positions.
     * @param radius_   Radius of the sphere.
     */
    RandomFilledSphere(VectorView positions, Scalar radius_)
      : positions_view(positions), radius(radius_) { }

    /// @brief Sample a random point inside the sphere for agent @p i.
    INIT_OP {
      const Scalar r = radius * Kokkos::cbrt(static_cast<Scalar>(generator.drand()));

      const Scalar z = static_cast<Scalar>(generator.drand(0.0, 2.0)) - Scalar(1.0);
      const Scalar theta = static_cast<Scalar>(generator.drand(0.0, 2.0 * Kokkos::numbers::pi_v<double>));
      const Scalar rxy = Kokkos::sqrt(Scalar(1.0) - z * z) * r;

      positions_view(i)[0] = rxy * Kokkos::cos(theta);
      positions_view(i)[1] = rxy * Kokkos::sin(theta);
      positions_view(i)[2] = z * r;
    }
  };

  /**
   * RandomFilledSphere followed by relaxation to spread agents apart.
   */
  template<typename SimulationConfig>
  struct RelaxedSphere : public RandomFilledSphere<SimulationConfig> {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    /// Number of relaxation steps.
    unsigned int steps;

    /**
     * @param positions          View to fill with positions.
     * @param initial_radius     Radius of the sphere for random placement.
     * @param relaxation_steps   How many relaxation steps to run (default: 2000).
     */
    template<typename ViewType>
    RelaxedSphere(
      ViewType positions,
      const Scalar initial_radius,
      const unsigned int relaxation_steps = 2000
    ) : RandomFilledSphere<SimulationConfig>(positions, initial_radius)
      , steps(relaxation_steps) { }
    
    /// @brief Run the relaxation simulation to spread agents evenly.
    template<typename Simulation>
    void relax(Simulation& simulation) {
      for (int i = 0; i < steps; ++i)
        simulation.take_step(0.1, details::RelaxForce<SimulationConfig>(0.8));
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP
