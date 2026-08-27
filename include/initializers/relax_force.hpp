#ifndef TECS_INITIALIZERS_RELAX_FORCE_HPP
#define TECS_INITIALIZERS_RELAX_FORCE_HPP

#include "../forces/predefined.hpp"
#include "../utils/utils.hpp"

namespace tecs::details {

  /**
   * @brief Internal repulsive force used during initialisation relaxation.
   *
   * Applies a piecewise-linear repulsion to push overlapping agents apart so
   * that the initial configuration is closer to a uniform distribution.
   * This is not intended for use by the user.
   */
  template<typename SimulationConfig>
  struct RelaxForce {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    /// Equilibrium (homeostatic) distance between neighbouring agents.
    Scalar equilibrium_distance;

    KOKKOS_INLINE_FUNCTION
    RelaxForce(Scalar equilibrium_distance_ = 0.8) : equilibrium_distance(equilibrium_distance_) { }

    // TODO: this should not be using hard coded names (ctx.position)
    PAIRWISE_FORCE_OP {
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, equilibrium_distance, Scalar(2), Scalar(1));
    }
  };
} // namespace tecs::details

#endif // TECS_INITIALIZERS_RELAX_FORCE_HPP
