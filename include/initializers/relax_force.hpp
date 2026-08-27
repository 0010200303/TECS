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

    /// Minimum force scaling factor.
    Scalar min;
    /// Maximum force scaling factor.
    Scalar max;

    KOKKOS_INLINE_FUNCTION
    RelaxForce(Scalar min_ = 0.8, Scalar max_ = 0.8) : min(min_), max(max_) { }

    // TODO: this should not be using hard coded names (ctx.position)
    PAIRWISE_FORCE_OP {
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, min, max, Scalar(2), Scalar(1));
    }
  };
} // namespace tecs::details

#endif // TECS_INITIALIZERS_RELAX_FORCE_HPP
